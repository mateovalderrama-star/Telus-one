"""Shared pytest fixtures for CAZ Sentinel tests."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def synthetic_probe_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with 3 synthetic probe .npz files."""
    d = tmp_path / "probes"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i, concept in enumerate(["c1", "c2", "c3"]):
        direction = rng.standard_normal(8).astype(np.float32)
        direction /= np.linalg.norm(direction)
        np.savez(
            d / f"{concept}.npz",
            concept=concept,
            layer_idx=i + 1,
            direction=direction,
            threshold=np.float32(0.6),
            calibration_mu=np.float32(0.0),
            calibration_sigma=np.float32(1.0),
            pool_method="last",
            model_fingerprint="test-fp",
            d_model=np.int64(8),
        )
    return d


@pytest.fixture
def always_suppress_probe_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with 9 probes all with threshold=0.0 to guarantee suppression."""
    import hashlib

    fp = hashlib.sha256(b"EleutherAI/pythia-70m").hexdigest()[:16]
    d = tmp_path / "probes"
    d.mkdir()
    # A single probe with threshold=0.0 guarantees suppression.
    direction = np.zeros(512, dtype=np.float32)
    direction[0] = 1.0
    for i in range(9):
        np.savez(
            d / f"c{i}.npz",
            concept=f"c{i}",
            layer_idx=np.int64(i % 6),
            direction=direction,
            threshold=np.float32(0.0),
            calibration_mu=np.float32(0),
            calibration_sigma=np.float32(1),
            pool_method="last",
            model_fingerprint=fp,
            d_model=np.int64(512),
        )
    return d
