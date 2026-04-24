"""Measure added latency from the Sentinel path."""
from __future__ import annotations

import statistics
import time

import httpx


URL = "http://localhost:8000/v1/chat/completions"
PROMPTS = ["Explain neural networks.", "Write a limerick.", "What's the capital of France?"]


def run(bypass: bool) -> list[float]:
    lat: list[float] = []
    headers = {"x-sentinel-bypass": "1"} if bypass else {}
    for p in PROMPTS * 10:
        t0 = time.perf_counter()
        httpx.post(
            URL,
            headers=headers,
            json={
                "model": "pythia-6.9b",
                "messages": [{"role": "user", "content": p}],
                "max_tokens": 64,
            },
            timeout=60,
        )
        lat.append((time.perf_counter() - t0) * 1000)
    return lat


if __name__ == "__main__":
    off = run(bypass=True)
    on = run(bypass=False)
    print(f"OFF p50={statistics.median(off):.1f}ms  p99={sorted(off)[int(0.99 * len(off))]:.1f}ms")
    print(f"ON  p50={statistics.median(on):.1f}ms  p99={sorted(on)[int(0.99 * len(on))]:.1f}ms")
    print(f"Delta p50={statistics.median(on) - statistics.median(off):.1f}ms")
