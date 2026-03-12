"""vision_qa_tool — CrewAI tool wrapping OpenAI and Gemini VLM backends.

The tool takes an image path + question + plan steps and returns the raw VLM
response text. Trace metadata is stored in a private attribute and accessible
via `tool.pop_traces()` after the Crew finishes.
"""

import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


class VisionQAInput(BaseModel):
    """Input schema for VisionQATool."""

    image_path: str = Field(
        description="Absolute or relative path to the chart image file"
    )
    question: str = Field(description="The question to answer about the chart")
    plan_steps: List[str] = Field(
        description="Ordered inspection steps from the planner"
    )
    choices: Optional[List[str]] = Field(
        default=None, description="MCQ answer choices if applicable"
    )
    context: Optional[List[dict]] = Field(
        default=None, description="Prior conversation turns"
    )


class VisionQATool(BaseTool):
    """Calls a VLM backend (OpenAI or Gemini) to analyze a chart image."""

    name: str = "vision_qa_tool"
    description: str = (
        "Analyze a chart image using a vision language model to answer a question. "
        "Provide the image path, question, and ordered inspection plan steps. "
        "Returns a JSON string with 'answer' and 'explanation' fields."
    )
    args_schema: Type[BaseModel] = VisionQAInput

    # Public config fields
    backend: str = "gemini"  # "openai" | "gemini"
    model: str = "gemini-2.5-flash-lite"
    api_key: str = ""
    opik_trace: Optional[Any] = None  # Opik Trace object for span creation

    # Private mutable trace storage (not a Pydantic field)
    _traces: list = PrivateAttr(default_factory=list)

    def pop_traces(self) -> list:
        """Return collected traces and clear the buffer."""
        traces = list(self._traces)
        self._traces.clear()
        return traces

    # ------------------------------------------------------------------
    # CrewAI entry point
    # ------------------------------------------------------------------

    def _run(
        self,
        image_path: str,
        question: str,
        plan_steps: List[str],
        choices: Optional[List[str]] = None,
        context: Optional[List[dict]] = None,
    ) -> str:
        """Run the vision QA tool with the given inputs.

        Handles tracing and error capture.
        """
        from ..opik_integration.tracing import close_span, open_llm_span

        start_ts = datetime.now(timezone.utc).isoformat()
        t0 = time.time()

        opik_span = open_llm_span(
            self.opik_trace,
            name="vision_qa_tool",
            input_data={
                "image_path": image_path,
                "question": question,
                "plan_steps": plan_steps,
            },
            model=self.model,
            metadata={"backend": self.backend},
        )

        provider_meta: dict = {}
        error_str: Optional[str] = None
        try:
            if self.backend == "openai":
                raw_text, provider_meta = self._call_openai(
                    image_path, question, plan_steps, choices, context
                )
            elif self.backend == "gemini":
                raw_text, provider_meta = self._call_gemini(
                    image_path, question, plan_steps, choices, context
                )
            else:
                raise ValueError(f"Unknown backend: {self.backend!r}")
        except Exception as exc:
            raw_text = json.dumps(
                {"answer": "ERROR", "explanation": f"Tool error: {exc}"}
            )
            provider_meta = {"error": str(exc)}
            error_str = str(exc)

        end_ts = datetime.now(timezone.utc).isoformat()
        elapsed_ms = (time.time() - t0) * 1000.0

        model_used = provider_meta.pop("model", self.model)
        usage = provider_meta.get("usage", {})

        close_span(
            opik_span,
            output={"raw_text": raw_text},
            usage=usage if usage else None,
            error=error_str,
        )

        self._traces.append(
            {
                "tool": "vision_qa_tool",
                "backend": self.backend,
                "model": model_used,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "elapsed_ms": elapsed_ms,
                "provider_metadata": provider_meta,
            }
        )

        return raw_text

    # ------------------------------------------------------------------
    # Shared prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        question: str,
        plan_steps: List[str],
        choices: Optional[List[str]],
        context: Optional[List[dict]],
    ) -> str:
        """Construct a VLM prompt.

        Uses the question, plan steps, choices, and context.
        """
        parts: list = []

        if context:
            parts.append("Conversation context:")
            for turn in context:
                parts.append(f"  {turn.get('role', 'user')}: {turn.get('content', '')}")
            parts.append("")

        parts.append(f"Question: {question}")

        if choices:
            parts.append(f"Choices: {', '.join(choices)}")

        parts.append("\nInspection plan — follow in order:")
        for i, step in enumerate(plan_steps, 1):
            parts.append(f"  {i}. {step}")

        parts.append(
            "\nOutput ONLY this JSON (no markdown, no extra text):\n"
            '{"answer": "...", "explanation": "..."}\n'
            "If the question cannot be answered from the chart: "
            '{"answer": "UNANSWERABLE", "explanation": "..."}'
        )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # OpenAI backend
    # ------------------------------------------------------------------

    def _encode_image(self, image_path: str) -> tuple:
        """Encode an image as base64 and determine its MIME type."""
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

    def _call_openai(
        self,
        image_path: str,
        question: str,
        plan_steps: List[str],
        choices: Optional[List[str]],
        context: Optional[List[dict]],
    ) -> tuple:
        """Call the OpenAI vision API and return raw text and provider metadata."""
        import os

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key or os.environ.get("OPENAI_API_KEY", ""))
        prompt = self._build_prompt(question, plan_steps, choices, context)
        b64, mime = self._encode_image(image_path)

        response = client.chat.completions.create(
            model=self.model,
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
            max_completion_tokens=1024,
            temperature=0,
        )

        raw_text = response.choices[0].message.content or ""
        provider_meta = {
            "model": response.model,
            "request_id": response.id,
            "usage": response.usage.model_dump() if response.usage else {},
        }
        return raw_text, provider_meta

    # ------------------------------------------------------------------
    # Gemini backend
    # ------------------------------------------------------------------

    def _call_gemini(
        self,
        image_path: str,
        question: str,
        plan_steps: List[str],
        choices: Optional[List[str]],
        context: Optional[List[dict]],
    ) -> tuple:
        """Call the Gemini vision API and return raw text and provider metadata."""
        import os

        import google.generativeai as genai
        from PIL import Image

        genai.configure(api_key=self.api_key or os.environ.get("GEMINI_API_KEY", ""))
        gemini_model = genai.GenerativeModel(self.model)
        prompt = self._build_prompt(question, plan_steps, choices, context)
        image = Image.open(image_path)

        response = gemini_model.generate_content(
            [image, prompt],
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=1024,
            ),
        )

        raw_text = response.text or ""
        finish = (
            str(response.candidates[0].finish_reason)
            if response.candidates
            else "unknown"
        )
        provider_meta = {
            "model": self.model,
            "finish_reason": finish,
        }
        return raw_text, provider_meta
