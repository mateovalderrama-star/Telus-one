"""Integration tests for forced-suppress behavior in CAZ Sentinel API."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_suppress(monkeypatch, always_suppress_probe_dir):
    """Client configured with always-suppress probes."""
    monkeypatch.setenv("CAZ_SENTINEL_PROBE_DIR", str(always_suppress_probe_dir))
    monkeypatch.setenv("CAZ_SENTINEL_MODEL_ID", "EleutherAI/pythia-70m")
    monkeypatch.setenv("CAZ_SENTINEL_DEVICE", "cpu")
    monkeypatch.setenv("CAZ_SENTINEL_REFUSAL_MESSAGE", "BLOCKED")
    from caz_sentinel.api import build_app

    return TestClient(build_app())


@pytest.mark.integration_test
def test_chat_completions_suppresses(client_suppress):
    """Test that chat completions are suppressed with always-suppress probes."""
    r = client_suppress.post(
        "/v1/chat/completions",
        json={
            "model": "pythia-70m",
            "messages": [{"role": "user", "content": "anything"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["finish_reason"] == "content_filter"
    assert body["choices"][0]["message"]["content"] == "BLOCKED"
    assert body["usage"]["completion_tokens"] == 0
    assert r.headers["x-sentinel-decision"] == "suppressed"
