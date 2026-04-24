"""FastAPI app for CAZ Sentinel."""
from __future__ import annotations

import os
from typing import Any
import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from caz_sentinel.model_loader import load_model_and_tokenizer
from caz_sentinel.probe_library import ProbeLibrary
from caz_sentinel.scorer import Scorer
from caz_sentinel.openai_shapes import (
    ChatCompletionRequest, build_pass_response, build_suppressed_response,
)
from caz_sentinel.types import Decision


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
        try:
            text = req.text()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        audit_result, _ = scorer.score(text)
        return audit_result.to_dict()

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest, response: Response) -> dict[str, Any]:
        prompt = "\n".join(m.content for m in req.messages)
        audit_result, past_kv = scorer.score(prompt)
        response.headers["x-sentinel-request-id"] = audit_result.request_id
        response.headers["x-sentinel-decision"] = audit_result.decision.value

        prompt_inputs = tok(prompt, return_tensors="pt")
        prompt_tokens = prompt_inputs.input_ids.shape[1]

        if audit_result.decision == Decision.SUPPRESSED:
            return build_suppressed_response(
                request_id=audit_result.request_id, model=req.model,
                refusal=refusal, prompt_tokens=prompt_tokens,
            ).model_dump()

        # Pass path: generate with KV cache reuse.
        input_ids = prompt_inputs.input_ids.to(device)
        cache_position = torch.arange(input_ids.shape[1], device=device)
        gen_kwargs: dict[str, Any] = dict(
            past_key_values=past_kv,
            cache_position=cache_position,
            max_new_tokens=req.max_tokens or 128,
            do_sample=req.temperature > 0.0,
            temperature=max(req.temperature, 1e-5),
            top_p=req.top_p,
            pad_token_id=tok.eos_token_id,
        )
        with torch.no_grad():
            out_ids = model.generate(input_ids, **gen_kwargs)
        new_ids = out_ids[0, input_ids.shape[1]:]
        completion_text = tok.decode(new_ids, skip_special_tokens=True)
        return build_pass_response(
            request_id=audit_result.request_id, model=req.model,
            completion=completion_text, prompt_tokens=prompt_tokens,
            completion_tokens=int(new_ids.shape[0]),
        ).model_dump()

    return app


app = build_app() if os.environ.get("CAZ_SENTINEL_EAGER") == "1" else None  # lazy by default
