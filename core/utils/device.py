"""Torch device selection."""

import torch


def get_device(name: str = "auto") -> torch.device:
    """'auto' -> CUDA if available, else CPU (MPS is slower than CPU for nets this small)."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
