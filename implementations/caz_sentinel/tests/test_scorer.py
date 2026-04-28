"""Tests for scorer math and decision logic."""
import numpy as np
import pytest
import torch

from caz_sentinel.model_loader import load_model_and_tokenizer
from caz_sentinel.probe_library import ProbeLibrary
from caz_sentinel.scorer import cosine_score_normalized, decide, Scorer
from caz_sentinel.types import Decision


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


@pytest.fixture(scope="module")
def tiny_stack(tmp_path_factory):
    """Build synthetic probes and load pythia-70m for integration tests."""
    import subprocess
    import sys
    out = tmp_path_factory.mktemp("probes")
    subprocess.check_call([
        sys.executable, "-m", "caz_sentinel.scripts.build_synthetic_probes",
        "--out", str(out),
        "--model", "EleutherAI/pythia-70m",
    ])
    model, tok = load_model_and_tokenizer("EleutherAI/pythia-70m", dtype=torch.float32, device="cpu")
    lib = ProbeLibrary.load(out)
    return Scorer(model=model, tokenizer=tok, library=lib, device="cpu")


@pytest.mark.integration_test
def test_scorer_returns_nine_scores(tiny_stack):
    scorer = tiny_stack
    audit, kv = scorer.score("hello world")
    assert set(audit.per_concept_scores.keys()) == set(scorer.library.concepts)
    for v in audit.per_concept_scores.values():
        assert 0.0 <= v <= 1.0


@pytest.mark.integration_test
def test_scorer_decision_shape(tiny_stack):
    audit, kv = tiny_stack.score("hello")
    assert audit.decision in (Decision.PASS, Decision.SUPPRESSED)
    assert kv is not None
