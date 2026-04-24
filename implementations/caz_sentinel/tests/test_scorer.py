"""Tests for scorer math and decision logic."""
import numpy as np
import pytest

from caz_sentinel.scorer import cosine_score_normalized, decide


def test_cosine_parallel_is_one():
    h = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    d = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine_score_normalized(h, d) == pytest.approx(1.0)


def test_cosine_antiparallel_is_zero():
    h = np.array([-2.0, 0.0, 0.0], dtype=np.float32)
    d = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine_score_normalized(h, d) == pytest.approx(0.0, abs=1e-6)


def test_cosine_orthogonal_is_half():
    h = np.array([0.0, 1.0], dtype=np.float32)
    d = np.array([1.0, 0.0], dtype=np.float32)
    assert cosine_score_normalized(h, d) == pytest.approx(0.5)


def test_decide_suppress_on_any_over_threshold():
    scores = {"c1": 0.3, "c2": 0.8, "c3": 0.4}
    thresholds = {"c1": 0.7, "c2": 0.7, "c3": 0.7}
    alerts, suppressed = decide(scores, thresholds)
    assert alerts == ["c2"]
    assert suppressed is True


def test_decide_pass_when_all_below():
    scores = {"c1": 0.3, "c2": 0.5}
    thresholds = {"c1": 0.7, "c2": 0.7}
    alerts, suppressed = decide(scores, thresholds)
    assert alerts == []
    assert suppressed is False
