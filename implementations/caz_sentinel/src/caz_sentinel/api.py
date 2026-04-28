"""FastAPI app for CAZ Sentinel."""
from __future__ import annotations

import os
import threading
import uuid
import time
from contextlib import asynccontextmanager
from typing import Any
import torch
from transformers import TextIteratorStreamer
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from caz_sentinel.audit_store import AuditStore
from caz_sentinel.model_loader import load_model_and_tokenizer
from caz_sentinel.probe_library import ProbeLibrary
from caz_sentinel.scorer import Scorer
from caz_sentinel.openai_shapes import (
    ChatCompletionRequest, build_pass_response, build_suppressed_response,
)
from caz_sentinel.streaming import suppressed_stream, pass_stream
from caz_sentinel.types import Decision
from caz_sentinel.chronicle_sink import ChronicleSink, NoopSink, build_udm_event


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
    audit_db = os.environ.get("CAZ_SENTINEL_AUDIT_DB", "caz_sentinel_audit.db")

    library = ProbeLibrary.load(probe_dir)
    model, tok = load_model_and_tokenizer(model_id, dtype=_resolve_dtype(device), device=device)
    scorer = Scorer(model=model, tokenizer=tok, library=library, device=device)
    store = AuditStore(audit_db)

    chronicle_endpoint = os.environ.get("CHRONICLE_ENDPOINT")
    chronicle_customer_id = os.environ.get("CHRONICLE_CUSTOMER_ID", "")
    sink: ChronicleSink | NoopSink = (
        ChronicleSink(endpoint=chronicle_endpoint)
        if chronicle_endpoint
        else NoopSink()
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await sink.start()
        yield
        await sink.close()

    app = FastAPI(title="CAZ Sentinel", lifespan=lifespan)
    app.state.scorer = scorer
    app.state.refusal = refusal
    app.state.model_id = model_id
    app.state.store = store
    app.state.sink = sink
    app.state.chronicle_customer_id = chronicle_customer_id

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
        store.append(audit_result)
        return audit_result.to_dict()

    @app.post("/v1/chat/completions", response_model=None)
    def chat_completions(req: ChatCompletionRequest, response: Response, raw_request: Request) -> dict[str, Any] | StreamingResponse:
        prompt = "\n".join(m.content for m in req.messages)
        prompt_inputs = tok(prompt, return_tensors="pt")
        prompt_tokens = prompt_inputs.input_ids.shape[1]

        if raw_request.headers.get("x-sentinel-bypass") == "1":
            response.headers["x-sentinel-decision"] = "bypass"
            input_ids = prompt_inputs.input_ids.to(device)
            gen_kwargs: dict[str, Any] = dict(
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
                request_id=str(uuid.uuid4()),
                model=req.model,
                completion=completion_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=int(new_ids.shape[0]),
            ).model_dump()

        audit_result, past_kv = scorer.score(prompt)
        response.headers["x-sentinel-request-id"] = audit_result.request_id
        response.headers["x-sentinel-decision"] = audit_result.decision.value
        store.append(audit_result)
        if audit_result.decision == Decision.SUPPRESSED:
            sink.emit(build_udm_event(audit_result, model_id=model_id, customer_id=chronicle_customer_id))

        if req.stream:
            if audit_result.decision == Decision.SUPPRESSED:
                return StreamingResponse(
                    suppressed_stream(req.model, refusal),
                    media_type="text/event-stream",
                )
            # Pass + stream: iterate token-by-token via TextIteratorStreamer.
            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            input_ids = prompt_inputs.input_ids.to(device)
            # Note: streaming doesn't use cache_position, so we must not use past_kv either
            gen_kwargs = dict(
                max_new_tokens=req.max_tokens or 128,
                do_sample=req.temperature > 0.0, temperature=max(req.temperature, 1e-5),
                top_p=req.top_p, pad_token_id=tok.eos_token_id, streamer=streamer,
            )
            thread = threading.Thread(target=model.generate, args=(input_ids,), kwargs=gen_kwargs, daemon=True)
            thread.start()
            return StreamingResponse(pass_stream(req.model, streamer), media_type="text/event-stream")

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

    @app.get("/v1/audit/{request_id}")
    def get_audit(request_id: str) -> dict[str, Any]:
        result = store.get(request_id)
        if result is None:
            raise HTTPException(status_code=404, detail="not found")
        return result.to_dict()

    @app.get("/v1/audit")
    def list_audits(limit: int = 100) -> list[dict[str, Any]]:
        return [a.to_dict() for a in store.list(limit=limit)]

    return app


app = build_app() if os.environ.get("CAZ_SENTINEL_EAGER") == "1" else None  # lazy by default
