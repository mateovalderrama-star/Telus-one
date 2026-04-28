"""Core dataclasses for the CAZ Sentinel runtime."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any

import numpy as np


class Decision(str, Enum):
    """Decision emitted by the Sentinel for each scored prompt."""

    PASS = "pass"
    SUPPRESSED = "suppressed"


@dataclass(frozen=True)
class Probe:
    """A single calibrated CAZ direction used for runtime scoring.

    Parameters
    ----------
    concept : str
        Human-readable concept name (e.g. "jailbreak-intent").
    layer_idx : int
        Transformer layer index where this direction was extracted.
    direction : np.ndarray
        Unit-normalised float32 direction vector in the residual stream.
    threshold : float
        Cosine similarity threshold in [0, 1]; at or above means suppressed.
    calibration : dict[str, float]
        Calibration statistics from offline fitting (mu, sigma, percentiles).
    pool_method : str
        Token pooling method used during extraction ("last" by default).
    """

    concept: str
    layer_idx: int
    direction: np.ndarray
    threshold: float
    calibration: dict[str, float]
    pool_method: str = "last"

    def __post_init__(self) -> None:
        if not isinstance(self.direction, np.ndarray):
            raise TypeError("direction must be a numpy array")
        if self.direction.ndim != 1:
            raise ValueError("direction must be 1-D")
        norm = float(np.linalg.norm(self.direction))
        if norm < 1e-8:
            raise ValueError("direction must be non-zero")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold {self.threshold} must lie in [0, 1]")
        # Defensive re-normalization (offline pipeline should already unit-norm).
        object.__setattr__(self, "direction", (self.direction / norm).astype(np.float32))

    @property
    def d_model(self) -> int:
        """Dimensionality of the residual stream (inferred from direction)."""
        return int(self.direction.shape[0])


@dataclass(frozen=True)
class AuditResult:
    """Immutable record of one scored request.

    Parameters
    ----------
    request_id : str
        Unique identifier for the request.
    timestamp_ns : int
        Unix timestamp in nanoseconds at scoring time.
    input_text : str
        The prompt text that was scored.
    per_concept_scores : dict[str, float]
        Cosine similarity score in [0, 1] per concept.
    alerts : list[str]
        Concepts whose score met or exceeded their threshold.
    decision : Decision
        Final decision: PASS or SUPPRESSED.
    latency_ms : float
        Total scoring latency in milliseconds.
    """

    request_id: str
    timestamp_ns: int
    input_text: str
    per_concept_scores: dict[str, float]
    alerts: list[str]
    decision: Decision
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d = asdict(self)
        d["decision"] = self.decision.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AuditResult":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        data = dict(d)
        data["decision"] = Decision(data["decision"])
        return cls(**data)
