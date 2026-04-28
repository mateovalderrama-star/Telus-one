"""Load and validate probe files produced by the offline CAZ pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from caz_sentinel.types import Probe


class ProbeLibraryError(RuntimeError):
    """Raised when the probe library is invalid or inconsistent."""


@dataclass(frozen=True)
class ProbeLibrary:
    """In-memory registry of calibrated CAZ probes.

    Parameters
    ----------
    probes : dict[str, Probe]
        Mapping from concept name to its fitted Probe.
    d_model : int
        Residual-stream dimensionality (must be identical across all probes).
    model_fingerprint : str
        Hash of the model config+weights used during probe extraction.
    """

    probes: dict[str, Probe]
    d_model: int
    model_fingerprint: str

    @property
    def concepts(self) -> list[str]:
        """Return concept names in load order."""
        return list(self.probes.keys())

    def get(self, concept: str) -> Probe:
        """Return the Probe for *concept*; raises KeyError if absent."""
        return self.probes[concept]

    @classmethod
    def load(cls, probe_dir: str | Path) -> "ProbeLibrary":
        """Load all .npz probe files from *probe_dir*.

        Parameters
        ----------
        probe_dir : str | Path
            Directory containing one .npz file per concept.

        Returns
        -------
        ProbeLibrary
            Validated registry ready for runtime scoring.

        Raises
        ------
        ProbeLibraryError
            If the directory is empty, or probes disagree on d_model or
            model fingerprint.
        """
        probe_dir = Path(probe_dir)
        files = sorted(probe_dir.glob("*.npz"))
        if not files:
            raise ProbeLibraryError(f"No .npz probe files in {probe_dir}")

        probes: dict[str, Probe] = {}
        d_models: set[int] = set()
        fingerprints: set[str] = set()

        for f in files:
            data = np.load(f, allow_pickle=False)
            concept = str(data["concept"])
            try:
                probe = Probe(
                    concept=concept,
                    layer_idx=int(data["layer_idx"]),
                    direction=np.asarray(data["direction"], dtype=np.float32),
                    threshold=float(data["threshold"]),
                    calibration={
                        "mu": float(data["calibration_mu"]),
                        "sigma": float(data["calibration_sigma"]),
                    },
                    pool_method=str(data["pool_method"]),
                )
            except (TypeError, ValueError) as exc:
                raise ProbeLibraryError(f"Invalid probe data in {f.name}: {exc}") from exc
            if concept in probes:
                raise ProbeLibraryError(f"duplicate concept: {concept}")
            probes[concept] = probe
            d_models.add(int(data["d_model"]))
            fingerprints.add(str(data["model_fingerprint"]))

        if len(d_models) != 1:
            raise ProbeLibraryError(f"inconsistent d_model across probes: {d_models}")
        if len(fingerprints) != 1:
            raise ProbeLibraryError(f"inconsistent model fingerprint across probes: {fingerprints}")

        return cls(probes=probes, d_model=next(iter(d_models)), model_fingerprint=next(iter(fingerprints)))
