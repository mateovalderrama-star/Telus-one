"""SSE chunk builders for OpenAI-compatible streaming."""
from __future__ import annotations

import json
import time
import uuid
from typing import Iterator


def _chunk_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _sse(data: dict | str) -> str:
    if isinstance(data, dict):
        data = json.dumps(data)
    return f"data: {data}\n\n"


def suppressed_stream(model: str, refusal: str) -> Iterator[str]:
    chunk_id = _chunk_id()
    created = int(time.time())
    yield _sse({"id": chunk_id, "object": "chat.completion.chunk", "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": refusal},
                             "finish_reason": "content_filter"}]})
    yield _sse("[DONE]")


def pass_stream(model: str, token_iter: Iterator[str]) -> Iterator[str]:
    chunk_id = _chunk_id()
    created = int(time.time())
    # First chunk establishes role.
    yield _sse({"id": chunk_id, "object": "chat.completion.chunk", "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
    try:
        for tok_text in token_iter:
            yield _sse({"id": chunk_id, "object": "chat.completion.chunk", "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": tok_text}, "finish_reason": None}]})
    except GeneratorExit:
        # Client closed connection; stop gracefully
        return
    yield _sse({"id": chunk_id, "object": "chat.completion.chunk", "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    yield _sse("[DONE]")
