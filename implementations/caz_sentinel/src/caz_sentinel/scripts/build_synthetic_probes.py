"""Produce a 9-probe synthetic library on pythia-70m for fast CI + local dev.

Usage: uv run --group caz-sentinel python -m caz_sentinel.scripts.build_synthetic_probes \
    --out implementations/caz_sentinel/tests/fixtures/synthetic_probes
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CONCEPTS = [
    "concept_1", "concept_2", "concept_3", "concept_4", "concept_5",
    "concept_6", "concept_7", "concept_8", "concept_9",
]


def model_fingerprint(model_id: str) -> str:
    """Compute a short fingerprint for the model ID."""
    return hashlib.sha256(model_id.encode()).hexdigest()[:16]


def main() -> None:
    """Build and write 9 synthetic probe files to --out directory."""
    ap = argparse.ArgumentParser(description="Build synthetic probe library on a small model.")
    ap.add_argument("--model", default="EleutherAI/pythia-70m")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    rng = np.random.default_rng(args.seed)

    fp = model_fingerprint(args.model)
    for i, concept in enumerate(CONCEPTS):
        direction = rng.standard_normal(d_model).astype(np.float32)
        direction /= np.linalg.norm(direction)
        layer_idx = i % n_layers
        np.savez(
            args.out / f"{concept}.npz",
            concept=concept,
            layer_idx=np.int64(layer_idx),
            direction=direction,
            threshold=np.float32(0.7),
            calibration_mu=np.float32(0.0),
            calibration_sigma=np.float32(1.0),
            pool_method="last",
            model_fingerprint=fp,
            d_model=np.int64(d_model),
        )
    print(f"Wrote {len(CONCEPTS)} probes to {args.out}")


if __name__ == "__main__":
    main()
