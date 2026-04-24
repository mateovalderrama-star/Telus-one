"""Integration tests for SSE streaming endpoints."""
from __future__ import annotations

import json
import pytest


@pytest.mark.integration_test
def test_streaming_suppressed_emits_done(client_suppress):
    with client_suppress.stream("POST", "/v1/chat/completions", json={
        "model": "pythia-70m",
        "messages": [{"role": "user", "content": "x"}],
        "stream": True,
    }) as r:
        chunks = [line for line in r.iter_lines() if line]
    assert chunks[-1] == "data: [DONE]"
    data_chunks = [json.loads(c[len("data: "):]) for c in chunks[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in data_chunks)
    assert any(c["choices"][0].get("finish_reason") == "content_filter" for c in data_chunks)


@pytest.mark.integration_test
def test_streaming_pass_emits_done(client):
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "pythia-70m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "max_tokens": 3,
    }) as r:
        chunks = [line for line in r.iter_lines() if line]
    assert chunks[-1] == "data: [DONE]"
    data_chunks = [json.loads(c[len("data: "):]) for c in chunks[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in data_chunks)
    assert data_chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    assert data_chunks[-1]["choices"][0].get("finish_reason") == "stop"
