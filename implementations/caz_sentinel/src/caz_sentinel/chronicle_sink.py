"""Chronicle UDM event builder and async sink."""
from __future__ import annotations

from typing import Any
from caz_sentinel.types import AuditResult, Decision


def _severity(score: float) -> str:
    if score >= 0.9:
        return "HIGH"
    if score >= 0.8:
        return "MEDIUM"
    return "LOW"


def build_udm_event(a: AuditResult, *, model_id: str, customer_id: str) -> dict[str, Any]:
    if a.decision != Decision.SUPPRESSED:
        raise ValueError("UDM events are only built for suppressed results")
    labels = {f"score_{c}": f"{s:.4f}" for c, s in a.per_concept_scores.items()}
    labels["model_id"] = model_id
    labels["request_id"] = a.request_id
    labels["latency_ms"] = f"{a.latency_ms:.2f}"
    return {
        "metadata": {
            "event_timestamp": {"seconds": a.timestamp_ns // 1_000_000_000},
            "event_type": "GENERIC_EVENT",
            "product_name": "caz_sentinel",
            "vendor_name": "TELUS",
        },
        "principal": {"resource": {"name": f"caz-sentinel/{customer_id}"}},
        "security_result": [
            {
                "rule_name": concept,
                "severity": _severity(a.per_concept_scores[concept]),
                "summary": f"CAZ probe '{concept}' exceeded threshold",
            }
            for concept in a.alerts
        ],
        "about": [{"labels": labels}],
    }
