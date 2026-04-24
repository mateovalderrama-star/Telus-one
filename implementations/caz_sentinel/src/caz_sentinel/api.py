"""FastAPI app for CAZ Sentinel."""
from __future__ import annotations

import os
from typing import Any
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from caz_sentinel.model_loader import load_model_and_tokenizer
from caz_sentinel.probe_library import ProbeLibrary
from caz_sentinel.scorer import Scorer


class AuditRequest(BaseModel):
    input_text: str | None = None
    messages: list[dict[str, Any]] | None = None

    def text(self) -> str:
        if self.input_text is not None:
            return self.input_text
        if self.messages:
            return "\n".join(m.get("content", "") for m in self.messages)
        raise ValueError("one of input_text or messages must be provided")


def _resolve_dtype(device: str) -> torch.dtype:
    return torch.bfloat16 if device == "cuda" else torch.float32


def build_app() -> FastAPI:
    probe_dir = os.environ["CAZ_SENTINEL_PROBE_DIR"]
    model_id = os.environ["CAZ_SENTINEL_MODEL_ID"]
    device = os.environ.get("CAZ_SENTINEL_DEVICE", "cuda")
    refusal = os.environ.get("CAZ_SENTINEL_REFUSAL_MESSAGE",
                             "This request was blocked by the CAZ Sentinel policy.")

    library = ProbeLibrary.load(probe_dir)
    model, tok = load_model_and_tokenizer(model_id, dtype=_resolve_dtype(device), device=device)
    scorer = Scorer(model=model, tokenizer=tok, library=library, device=device)

    app = FastAPI(title="CAZ Sentinel")
    app.state.scorer = scorer
    app.state.refusal = refusal
    app.state.model_id = model_id

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_id": model_id,
            "concepts": library.concepts,
            "d_model": library.d_model,
        }

    @app.post("/v1/audit")
    def audit(req: AuditRequest) -> dict[str, Any]:
        audit_result, _ = scorer.score(req.text())
        return audit_result.to_dict()

    return app


app = build_app() if os.environ.get("CAZ_SENTINEL_EAGER") == "1" else None  # lazy by default
