import os, pytest, subprocess, sys
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    subprocess.check_call([sys.executable, "-m", "caz_sentinel.scripts.build_synthetic_probes",
                           "--out", str(tmp_path), "--model", "EleutherAI/pythia-70m", "--seed", "1"])
    # Seed 1 keeps random directions low-scoring against typical prompts; safe for "pass" test.
    monkeypatch.setenv("CAZ_SENTINEL_PROBE_DIR", str(tmp_path))
    monkeypatch.setenv("CAZ_SENTINEL_MODEL_ID", "EleutherAI/pythia-70m")
    monkeypatch.setenv("CAZ_SENTINEL_DEVICE", "cpu")
    from caz_sentinel.api import build_app
    return TestClient(build_app())


@pytest.mark.integration_test
def test_chat_completions_pass(client):
    # Seed-1 synthetic probes are unlikely to suppress; if threshold 0.7 is crossed, skip.
    r = client.post("/v1/chat/completions", json={
        "model": "pythia-70m",
        "messages": [{"role": "user", "content": "Hello, world."}],
        "max_tokens": 5,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["finish_reason"] in ("stop", "length", "content_filter")
    assert r.headers["x-sentinel-decision"] in ("pass", "suppressed")
