import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Build synthetic probes in tmp dir for this test
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "caz_sentinel.scripts.build_synthetic_probes",
                           "--out", str(tmp_path), "--model", "EleutherAI/pythia-70m"])
    monkeypatch.setenv("CAZ_SENTINEL_PROBE_DIR", str(tmp_path))
    monkeypatch.setenv("CAZ_SENTINEL_MODEL_ID", "EleutherAI/pythia-70m")
    monkeypatch.setenv("CAZ_SENTINEL_DEVICE", "cpu")
    from caz_sentinel.api import build_app
    return TestClient(build_app())


@pytest.mark.integration_test
def test_health(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert len(body["concepts"]) == 9
    assert body["model_id"] == "EleutherAI/pythia-70m"


@pytest.mark.integration_test
def test_audit_returns_nine_scores(client):
    r = client.post("/v1/audit", json={"input_text": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["per_concept_scores"]) == 9
    assert body["decision"] in ("pass", "suppressed")
