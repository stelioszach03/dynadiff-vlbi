"""I/O helpers for dataset artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_npz(path: str | Path, arrays: dict[str, Any]) -> None:
    """Save a dictionary of arrays to a compressed NPZ file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a compressed NPZ file into a plain dictionary."""

    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
