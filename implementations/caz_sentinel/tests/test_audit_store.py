from pathlib import Path
from caz_sentinel.audit_store import AuditStore
from caz_sentinel.types import AuditResult, Decision


def _mk(rid: str, decision: Decision, ts_ns: int = 1) -> AuditResult:
    return AuditResult(request_id=rid, timestamp_ns=ts_ns, input_text="x",
                       per_concept_scores={"c": 0.1}, alerts=[],
                       decision=decision, latency_ms=1.0)


def test_append_and_query(tmp_path: Path):
    store = AuditStore(tmp_path / "a.db")
    store.append(_mk("r1", Decision.PASS, ts_ns=1))
    store.append(_mk("r2", Decision.SUPPRESSED, ts_ns=2))
    all_ = store.list(limit=10)
    assert [a.request_id for a in all_] == ["r2", "r1"]
    assert store.get("r1").decision == Decision.PASS
