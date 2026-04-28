"""Scoring math and decision logic (pure functions, no model state)."""
from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np
import torch

from caz_sentinel.model_loader import get_transformer_layers
from caz_sentinel.probe_library import ProbeLibrary
from caz_sentinel.types import AuditResult, Decision


def cosine_score_normalized(hidden: np.ndarray, direction: np.ndarray) -> float:
    """Compute cosine similarity remapped from [-1, 1] to [0, 1].

    Parameters
    ----------
    hidden : np.ndarray
        Residual-stream vector at the last token position (any norm).
    direction : np.ndarray
        Unit-normalised probe direction (||direction|| == 1.0).

    Returns
    -------
    float
        Score in [0, 1]; 1.0 means perfectly aligned, 0.0 means antiparallel,
        0.5 means orthogonal.
    """
    h_norm = float(np.linalg.norm(hidden))
    if h_norm < 1e-12:
        return 0.5
    cos = float(np.dot(hidden, direction) / h_norm)  # direction is unit-norm
    return 0.5 * (cos + 1.0)


def decide(
    scores: dict[str, float],
    thresholds: dict[str, float],
) -> tuple[list[str], bool]:
    """Return (alerts, suppressed). Alerts are concepts at or above threshold.

    Parameters
    ----------
    scores : dict[str, float]
        Per-concept cosine scores in [0, 1].
    thresholds : dict[str, float]
        Per-concept threshold values in [0, 1].

    Returns
    -------
    tuple[list[str], bool]
        Sorted list of alert concept names and whether the request is suppressed.
    """
    alerts = sorted(c for c, s in scores.items() if s >= thresholds[c])
    return alerts, bool(alerts)


class Scorer:
    """Single-forward-pass scorer over a probe library.

    Captures residual-stream activations at the unique set of layers required by
    the probe library, computes per-concept cosine scores on the last-token
    activation, and returns an AuditResult + the KV cache for handoff.
    """

    def __init__(self, model: Any, tokenizer: Any, library: ProbeLibrary, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.library = library
        self.device = device
        self._layers = get_transformer_layers(model)
        self._target_layers = sorted({p.layer_idx for p in library.probes.values()})
        n = len(self._layers)
        for idx in self._target_layers:
            if idx < 0 or idx >= n:
                raise ValueError(f"probe layer_idx={idx} out of range for model with {n} layers")

    def _install_hooks(self, buf: dict[int, torch.Tensor]) -> list[Any]:
        handles = []
        for idx in self._target_layers:
            def make_hook(i: int):
                def hook(_mod: Any, _inp: Any, out: Any) -> None:
                    # Block modules return a tuple (hidden_state, ...) on GPT-NeoX/Llama/GPT-2.
                    h = out[0] if isinstance(out, tuple) else out
                    buf[i] = h.detach()
                return hook
            handles.append(self._layers[idx].register_forward_hook(make_hook(idx)))
        return handles

    def score(self, prompt: str) -> tuple[AuditResult, Any]:
        """Forward-pass the prompt, collect hidden states, score against library.

        Parameters
        ----------
        prompt : str
            The input text to score.

        Returns
        -------
        tuple[AuditResult, Any]
            (AuditResult, past_key_values) — KV cache ready for handoff to model.generate().
        """
        t0 = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        buf: dict[int, torch.Tensor] = {}
        handles = self._install_hooks(buf)
        try:
            with torch.no_grad():
                out = self.model(**inputs, use_cache=True)
        finally:
            for h in handles:
                h.remove()

        scores: dict[str, float] = {}
        for concept, probe in self.library.probes.items():
            hidden = buf[probe.layer_idx][0, -1, :].to(torch.float32).cpu().numpy()
            scores[concept] = cosine_score_normalized(hidden, probe.direction)

        thresholds = {c: p.threshold for c, p in self.library.probes.items()}
        alerts, suppressed = decide(scores, thresholds)
        audit = AuditResult(
            request_id=f"req-{uuid.uuid4().hex[:12]}",
            timestamp_ns=time.time_ns(),
            input_text=prompt,
            per_concept_scores=scores,
            alerts=alerts,
            decision=Decision.SUPPRESSED if suppressed else Decision.PASS,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
        return audit, out.past_key_values
