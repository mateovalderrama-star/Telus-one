"""VerifierAgent — Pass 2.5: critically reviews the VisionAgent's draft answer.

The verifier sees the chart image AND the draft answer/explanation and decides
whether to CONFIRM or REVISE the answer. This teaches multi-agent critique patterns.

Unlike VisionAgent (which uses CrewAI + tool-use to explore the chart), the
verifier makes a single direct VLM call — showing that multi-agent critique
does not always require a full orchestration framework.

Key teaching point
------------------
Two agents looking at the same chart image can disagree. When the second model
(the verifier) has explicit access to the first model's reasoning, it can catch
errors the first model missed — axis misreads, arithmetic mistakes, etc.
"""

import base64
import os
from pathlib import Path
from typing import Any, Optional, Tuple

from ..opik_integration.tracing import close_span, open_llm_span
from ..utils.json_strict import parse_strict


VERIFIER_REQUIRED_KEYS = ["verdict", "answer", "reasoning"]

_VERIFIER_PROMPT = """\
You are a critical chart QA verifier. A vision agent has already attempted to answer
the question below. Your job: look at the chart image carefully and audit the work.

Question         : {question}
Question Type    : {question_type}

Inspection plan the agent was supposed to follow:
{plan_steps}

Vision Agent's Draft Answer     : {draft_answer}
Vision Agent's Draft Explanation: {draft_explanation}

Examine the chart image. Then decide:
  CONFIRM — the draft answer is correct (output the same answer unchanged)
  REVISE  — you can see a clear, specific error; output the corrected answer

Rules:
- Only REVISE when you are confident you can point to a concrete error in the chart
- If uncertain, CONFIRM — do not second-guess without visual evidence
- For MCQ questions: the answer must be one of the stated choices
- If the answer is truly unanswerable from the chart, say exactly "UNANSWERABLE"
- Keep answers concise — numbers, short phrases, or single words where appropriate

Output ONLY JSON, no markdown, no extra text:
{{"verdict": "confirmed" | "revised", "answer": "<final answer>", "reasoning": "<one sentence grounded in what you see in the chart>"}}"""


# ---------------------------------------------------------------------------
# VLM helpers (same pattern as error_taxonomy.py)
# ---------------------------------------------------------------------------


def _encode_image(image_path: str) -> tuple:
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = {
        "jpg": "jpeg",
        "jpeg": "jpeg",
        "png": "png",
        "gif": "gif",
        "webp": "webp",
    }.get(ext, "jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime


def _call_vlm_openai(
    prompt: str, image_path: str, model: str, api_key: Optional[str]
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
    b64, mime = _encode_image(image_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{b64}"},
                    },
                ],
            }
        ],
        max_completion_tokens=256,
        temperature=0,
    )
    return response.choices[0].message.content or ""


def _call_vlm_gemini(
    prompt: str, image_path: str, model: str, api_key: Optional[str]
) -> str:
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=api_key or os.environ.get("GEMINI_API_KEY", ""))
    gemini_model = genai.GenerativeModel(model)
    image = Image.open(image_path)
    response = gemini_model.generate_content(
        [image, prompt],
        generation_config=genai.types.GenerationConfig(
            temperature=0, max_output_tokens=256
        ),
    )
    return response.text or ""


# ---------------------------------------------------------------------------
# VerifierAgent
# ---------------------------------------------------------------------------


class VerifierAgent:
    """
    Pass 2.5: critiques the VisionAgent's draft answer against the chart image.

    Makes a single direct VLM call — no orchestration framework needed for
    pure critique tasks. Produces a verdict of "confirmed" or "revised" plus
    the final answer and a one-sentence visual rationale.
    """

    def __init__(
        self,
        backend: str = "gemini",
        model: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
    ):
        self.backend = backend
        self.model = model
        self.api_key = api_key

    def run(
        self,
        sample,  # PerceivedSample
        plan: dict,
        vision_parsed: dict,
        opik_trace: Any = None,
    ) -> Tuple[str, dict, bool, str]:
        """
        Verify the VisionAgent's draft answer.

        Returns
        -------
            prompt       – rendered verifier prompt
            parsed       – {verdict, answer, reasoning}
            parse_error  – True if JSON parsing failed or fell back
            raw_text     – raw VLM output
        """
        plan_steps = plan.get("steps", [])
        steps_text = (
            "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan_steps)) or "  (none)"
        )

        draft_answer = vision_parsed.get("answer", "(none)")
        draft_explanation = vision_parsed.get("explanation", "(none)")
        question_type = getattr(
            getattr(sample, "question_type", None),
            "value",
            str(getattr(sample, "question_type", "standard")),
        )

        prompt = _VERIFIER_PROMPT.format(
            question=sample.question,
            question_type=question_type,
            plan_steps=steps_text,
            draft_answer=draft_answer,
            draft_explanation=draft_explanation,
        )

        span = open_llm_span(
            opik_trace,
            name="verifier",
            input_data={"prompt": prompt, "draft_answer": draft_answer},
            model=self.model,
            metadata={"backend": self.backend},
        )

        image_path = getattr(sample, "image_path", "") or ""
        has_image = image_path and Path(image_path).exists()

        try:
            if has_image:
                if self.backend == "openai":
                    raw = _call_vlm_openai(prompt, image_path, self.model, self.api_key)
                elif self.backend == "gemini":
                    raw = _call_vlm_gemini(prompt, image_path, self.model, self.api_key)
                else:
                    raise ValueError(f"Unknown backend: {self.backend!r}")
            else:
                # Image missing — cannot verify visually; default to confirm
                raw = (
                    '{"verdict": "confirmed", "answer": "'
                    + draft_answer.replace('"', '\\"')
                    + '", "reasoning": "Image unavailable; cannot verify visually."}'
                )

            parsed, parse_ok = parse_strict(raw, required_keys=VERIFIER_REQUIRED_KEYS)
            if not parsed:
                parsed = {
                    "verdict": "confirmed",
                    "answer": draft_answer,
                    "reasoning": f"Parse error — defaulting to confirm. Raw: {raw[:120]}",
                }
                parse_ok = False

            # Normalise verdict to known values
            if parsed.get("verdict", "").lower() not in ("confirmed", "revised"):
                parsed["verdict"] = "confirmed"

            close_span(span, output=parsed)
            return prompt, parsed, not parse_ok, raw

        except Exception as exc:
            fallback = {
                "verdict": "confirmed",
                "answer": draft_answer,
                "reasoning": f"Verifier error: {exc}",
            }
            close_span(span, output=fallback, error=str(exc))
            return prompt, fallback, True, ""
