"""ocr_reader_tool — CrewAI tool that extracts all visible text from a chart image.

Runs a single VLM call focused purely on text transcription (no reasoning).
Returns structured JSON with axis labels, legend entries, title, data labels, and
annotations. The output grounds the downstream VisionAgent in observed chart text,
separating perception from reasoning.
"""

import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


_OCR_PROMPT = """\
Extract ALL visible text from this chart image into structured JSON.

Transcribe only what is visually present. Do NOT interpret, infer, or reason.
Do NOT answer any question. Only read and record text.

Output ONLY this JSON (no markdown, no extra text):
{
  "chart_type": "<bar|line|pie|scatter|table|dashboard|other>",
  "title": "<chart title, or empty string if absent>",
  "x_axis": {
    "label": "<x-axis label, or empty string>",
    "ticks": ["<tick1>", "<tick2>", "..."]
  },
  "y_axis": {
    "label": "<y-axis label, or empty string>",
    "ticks": ["<tick1>", "<tick2>", "..."]
  },
  "legend": ["<entry1>", "<entry2>", "..."],
  "data_labels": ["<label1>", "<label2>", "..."],
  "annotations": ["<note1>", "<note2>", "..."]
}

Rules:
- chart_type: the primary chart type you can see
- title: full title text exactly as written
- x_axis.ticks / y_axis.ticks: every visible tick label, in order
- legend: every legend item label, in order
- data_labels: numeric or text labels attached directly to bars/points/slices
- annotations: footnotes, source lines, asterisks, or any other floating text
- Use empty string "" for absent text fields; use [] for absent list fields
- JSON only — no markdown code fences, no preamble, no trailing text
"""


class OcrReaderInput(BaseModel):
    """Input schema for OcrReaderTool."""

    image_path: str = Field(
        description="Absolute or relative path to the chart image file"
    )


class OcrReaderTool(BaseTool):
    """Extract all visible text from a chart image as structured JSON."""

    name: str = "ocr_reader_tool"
    description: str = (
        "Extract all visible text from a chart image — axis labels, tick values, "
        "legend entries, title, data labels, and annotations — as structured JSON. "
        "Use this BEFORE answering any question to ground your response in observed "
        "text."
    )
    args_schema: Type[BaseModel] = OcrReaderInput

    backend: str = "gemini"
    model: str = "gemini-2.5-flash-lite"
    api_key: str = ""
    opik_trace: Optional[Any] = None

    _traces: list = PrivateAttr(default_factory=list)

    def pop_traces(self) -> list:
        """Return collected traces and clear the buffer."""
        traces = list(self._traces)
        self._traces.clear()
        return traces

    # ------------------------------------------------------------------
    # CrewAI entry point
    # ------------------------------------------------------------------

    def _run(self, image_path: str) -> str:
        from ..opik_integration.tracing import close_span, open_llm_span

        start_ts = datetime.now(timezone.utc).isoformat()
        t0 = time.time()

        opik_span = open_llm_span(
            self.opik_trace,
            name="ocr_reader_tool",
            input_data={"image_path": image_path},
            model=self.model,
            metadata={"backend": self.backend},
        )

        provider_meta: dict = {}
        error_str: Optional[str] = None
        try:
            if self.backend == "openai":
                raw_text, provider_meta = self._call_openai(image_path)
            elif self.backend == "gemini":
                raw_text, provider_meta = self._call_gemini(image_path)
            else:
                raise ValueError(f"Unknown backend: {self.backend!r}")
        except Exception as exc:
            raw_text = json.dumps(
                {
                    "chart_type": "unknown",
                    "title": "",
                    "x_axis": {"label": "", "ticks": []},
                    "y_axis": {"label": "", "ticks": []},
                    "legend": [],
                    "data_labels": [],
                    "annotations": [],
                    "error": str(exc),
                }
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
                "tool": "ocr_reader_tool",
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
    # OpenAI backend
    # ------------------------------------------------------------------

    def _encode_image(self, image_path: str) -> tuple:
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

    def _call_openai(self, image_path: str) -> tuple:
        import os

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key or os.environ.get("OPENAI_API_KEY", ""))
        b64, mime = self._encode_image(image_path)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _OCR_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            max_completion_tokens=512,
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

    def _call_gemini(self, image_path: str) -> tuple:
        import os

        import google.generativeai as genai
        from PIL import Image

        genai.configure(api_key=self.api_key or os.environ.get("GEMINI_API_KEY", ""))
        gemini_model = genai.GenerativeModel(self.model)
        image = Image.open(image_path)

        response = gemini_model.generate_content(
            [image, _OCR_PROMPT],
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=512,
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
