"""LLM-as-judge evaluator for MEP outputs.

Scores five rubric dimensions (0.0–1.0) using a text LLM.
"""

import os
from typing import Optional

from ..utils.json_strict import parse_strict


_JUDGE_PROMPT = """\
You are an expert chart question-answering evaluator.

Question        : {question}
Expected Answer : {expected_answer}
Agent Answer    : {agent_answer}
Agent Explanation: {agent_explanation}

Inspection plan the agent should have followed:
{plan_steps}

Score each dimension from 0.0 to 1.0 (floats):

1. explanation_quality   – Is the explanation specific and grounded in chart elements
                           (axes, legend, series, data values)?
2. hallucination_rate    – Does the explanation claim things NOT supported by the chart?
                           (0 = no hallucination, 1 = severe hallucination)
3. plan_coverage         – Does the explanation address each required inspection step?
4. plan_adherence        – Did the agent follow the plan steps in order without skipping?
5. faithfulness_alignment – Does the explanation logically support the given answer?

Output ONLY JSON, no markdown, no extra text:
{{
  "explanation_quality": 0.0,
  "hallucination_rate": 0.0,
  "plan_coverage": 0.0,
  "plan_adherence": 0.0,
  "faithfulness_alignment": 0.0,
  "reasoning": "<one-sentence rationale>"
}}"""

_JUDGE_KEYS = [
    "explanation_quality",
    "hallucination_rate",
    "plan_coverage",
    "plan_adherence",
    "faithfulness_alignment",
]


def _default_scores() -> dict:
    return dict.fromkeys(_JUDGE_KEYS, 0.0)


def _call_llm(prompt: str, backend: str, model: str, api_key: Optional[str]) -> str:
    if backend == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=512,
        )
        return resp.choices[0].message.content or ""

    if backend == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=api_key or os.environ.get("GEMINI_API_KEY", ""))
        m = genai.GenerativeModel(model)
        return m.generate_content(prompt).text or ""

    raise ValueError(f"Unknown judge backend: {backend!r}")


def judge_mep(
    mep: dict,
    backend: str = "gemini",
    model: str = "gemini-2.5-flash-lite",
    api_key: Optional[str] = None,
) -> dict:
    """Run LLM judge on a single MEP dict. Returns a scores dict."""
    sample = mep.get("sample", {})
    plan = mep.get("plan", {}).get("parsed", {})
    vision = mep.get("vision", {})
    parsed_vision = vision.get("parsed", {})

    question = sample.get("question", "")
    expected = sample.get("expected_output", "")
    agent_answer = parsed_vision.get("answer", "")
    agent_explanation = parsed_vision.get("explanation", "")
    plan_steps = plan.get("steps", [])

    steps_text = (
        "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan_steps)) or "  (none)"
    )

    prompt = _JUDGE_PROMPT.format(
        question=question or "unknown",
        expected_answer=expected,
        agent_answer=agent_answer,
        agent_explanation=agent_explanation,
        plan_steps=steps_text,
    )

    try:
        raw = _call_llm(prompt, backend, model, api_key)
        scores, ok = parse_strict(raw, required_keys=_JUDGE_KEYS)
        if not scores:
            scores = _default_scores()
            scores["judge_parse_error"] = True
        return scores
    except Exception as exc:
        s = _default_scores()
        s["judge_error"] = str(exc)
        return s
