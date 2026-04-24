"""Scoring math and decision logic (pure functions, no model state)."""
from __future__ import annotations

import numpy as np


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
