"""Device selection helpers."""

from __future__ import annotations

import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return a CUDA device when available, otherwise CPU."""

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
