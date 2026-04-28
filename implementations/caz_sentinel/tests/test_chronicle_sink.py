import asyncio
import pytest
from caz_sentinel.chronicle_sink import build_udm_event, ChronicleSink, NoopSink
from caz_sentinel.types import AuditResult, Decision


def test_udm_event_has_required_fields():
    a = AuditResult(request_id="r", timestamp_ns=0, input_text="x",
                    per_concept_scores={"jailbreak": 0.9, "negation": 0.1},
                    alerts=["jailbreak"], decision=Decision.SUPPRESSED, latency_ms=1.0)
    ev = build_udm_event(a, model_id="pythia-6.9b", customer_id="cust-1")
    assert ev["metadata"]["event_type"] == "GENERIC_EVENT"
    assert ev["metadata"]["product_name"] == "caz_sentinel"
    assert ev["security_result"][0]["rule_name"] == "jailbreak"
    assert ev["security_result"][0]["severity"] in ("LOW", "MEDIUM", "HIGH")
    assert ev["about"][0]["labels"]["score_jailbreak"] == "0.9000"


def test_udm_only_built_for_suppressed():
    a = AuditResult(request_id="r", timestamp_ns=0, input_text="x",
                    per_concept_scores={}, alerts=[], decision=Decision.PASS, latency_ms=1.0)
    with pytest.raises(ValueError, match="suppressed"):
        build_udm_event(a, model_id="m", customer_id="c")


@pytest.mark.asyncio
async def test_sink_enqueues_and_flushes():
    seen: list[dict] = []

    async def transport(event: dict) -> None:
        seen.append(event)

    sink = ChronicleSink(transport=transport, max_queue=10)
    await sink.start()
    sink.emit({"metadata": {"event_type": "X"}})
    await sink.drain()
    assert len(seen) == 1
    await sink.close()


@pytest.mark.asyncio
async def test_noop_sink_is_safe():
    sink = NoopSink()
    sink.emit({"x": 1})
    await sink.drain()
