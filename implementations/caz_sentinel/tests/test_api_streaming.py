"""Streaming tests for CAZ Sentinel API."""
import pytest


@pytest.mark.integration_test
def test_streaming_suppressed_emits_done(client_suppress):
    with client_suppress.stream("POST", "/v1/chat/completions", json={
        "model": "pythia-70m",
        "messages": [{"role": "user", "content": "x"}],
        "stream": True,
    }) as r:
        chunks = [line for line in r.iter_lines() if line]
    assert any("content_filter" in c for c in chunks)
    assert chunks[-1] == "data: [DONE]"


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
