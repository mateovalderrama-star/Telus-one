"""Chronicle UDM event builder and async sink."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable
import httpx
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


log = logging.getLogger(__name__)


class NoopSink:
    def emit(self, event: dict[str, Any]) -> None:
        pass

    async def drain(self) -> None:
        pass

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass


class ChronicleSink:
    def __init__(
        self, *, transport: Callable[[dict], Awaitable[None]] | None = None,
        endpoint: str | None = None, max_queue: int = 1000,
    ) -> None:
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task | None = None
        self._endpoint = endpoint
        self._transport = transport or self._http_transport
        self._client: httpx.AsyncClient | None = None

    async def _http_transport(self, event: dict) -> None:
        assert self._endpoint is not None
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=5.0)
        r = await self._client.post(self._endpoint, json={"events": [event]})
        if r.status_code >= 400:
            log.warning("chronicle emit failed: %s %s", r.status_code, r.text[:200])

    async def start(self) -> None:
        self._task = asyncio.create_task(self._worker())

    async def _worker(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await self._transport(event)
            except Exception:
                log.exception("chronicle transport error (dropped)")
            finally:
                self._queue.task_done()

    def emit(self, event: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            log.warning("chronicle queue full; dropping event")

    async def drain(self) -> None:
        await self._queue.join()

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
        if self._client:
            await self._client.aclose()
