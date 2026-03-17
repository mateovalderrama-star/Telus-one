"""Utility functions."""

import gc

import torch


def get_device() -> torch.device:
    """Get the appropriate device (GPU, MPS, or CPU) for PyTorch operations.

    Returns
    -------
    torch.device
        The device to be used for PyTorch operations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Check for Apple Silicon (MPS) support
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        pass

    # Fallback to CPU if no GPU is available
    return torch.device("cpu")


def release_memory() -> None:
    """Release memory occupied by a PyTorch module."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except AttributeError:
        pass
