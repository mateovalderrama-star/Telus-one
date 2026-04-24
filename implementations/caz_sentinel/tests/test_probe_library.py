"""Tests for ProbeLibrary loader."""

import numpy as np
import pytest
from caz_sentinel.probe_library import ProbeLibrary, ProbeLibraryError


def test_loads_all_probes(synthetic_probe_dir):
    """Test that ProbeLibrary.load reads all probes and extracts metadata."""
    lib = ProbeLibrary.load(synthetic_probe_dir)
    assert set(lib.concepts) == {"c1", "c2", "c3"}
    assert lib.d_model == 8
    assert lib.model_fingerprint == "test-fp"


def test_threshold_lookup(synthetic_probe_dir):
    """Test that ProbeLibrary.get returns the correct Probe with threshold."""
    lib = ProbeLibrary.load(synthetic_probe_dir)
    assert lib.get("c1").threshold == pytest.approx(0.6)


def test_rejects_mixed_d_model(tmp_path):
    """Test that ProbeLibrary rejects probes with inconsistent d_model."""
    d = tmp_path / "probes"
    d.mkdir()
    for i, dim in enumerate([8, 16]):
        direction = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        np.savez(
            d / f"c{i}.npz",
            concept=f"c{i}",
            layer_idx=0,
            direction=direction,
            threshold=np.float32(0.5),
            calibration_mu=np.float32(0),
            calibration_sigma=np.float32(1),
            pool_method="last",
            model_fingerprint="fp",
            d_model=np.int64(dim),
        )
    with pytest.raises(ProbeLibraryError, match="d_model"):
        ProbeLibrary.load(d)


def test_rejects_mixed_fingerprints(tmp_path):
    """Test that ProbeLibrary rejects probes with inconsistent fingerprints."""
    d = tmp_path / "probes"
    d.mkdir()
    for i, fp in enumerate(["fp-a", "fp-b"]):
        direction = np.array([1.0, 0.0], dtype=np.float32)
        np.savez(
            d / f"c{i}.npz",
            concept=f"c{i}",
            layer_idx=0,
            direction=direction,
            threshold=np.float32(0.5),
            calibration_mu=np.float32(0),
            calibration_sigma=np.float32(1),
            pool_method="last",
            model_fingerprint=fp,
            d_model=np.int64(2),
        )
    with pytest.raises(ProbeLibraryError, match="fingerprint"):
        ProbeLibrary.load(d)
