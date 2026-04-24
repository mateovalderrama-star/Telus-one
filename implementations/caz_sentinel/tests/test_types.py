import numpy as np
import pytest
from caz_sentinel.types import Probe, AuditResult, Decision


def test_probe_validates_unit_direction():
    d = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    p = Probe(concept="jailbreak", layer_idx=4, direction=d, threshold=0.7,
              calibration={"mu": 0.0, "sigma": 1.0}, pool_method="last")
    assert p.d_model == 3
    assert abs(np.linalg.norm(p.direction) - 1.0) < 1e-5


def test_probe_rejects_zero_direction():
    with pytest.raises(ValueError, match="non-zero"):
        Probe(concept="x", layer_idx=0, direction=np.zeros(3, dtype=np.float32),
              threshold=0.5, calibration={}, pool_method="last")


def test_probe_rejects_out_of_range_threshold():
    d = np.array([1.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="threshold"):
        Probe(concept="x", layer_idx=0, direction=d, threshold=1.5,
              calibration={}, pool_method="last")


def test_audit_result_serializes_roundtrip():
    a = AuditResult(
        request_id="req-1", timestamp_ns=123, input_text="hi",
        per_concept_scores={"c1": 0.1, "c2": 0.9},
        alerts=["c2"], decision=Decision.SUPPRESSED, latency_ms=42.0,
    )
    payload = a.to_dict()
    assert payload["decision"] == "suppressed"
    assert payload["alerts"] == ["c2"]
    assert AuditResult.from_dict(payload) == a
