"""Filesystem-backed logging helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from dynadiff_vlbi.utils.config import ExperimentConfig


def prepare_output_dirs(output_root: str, run_name: str, config: ExperimentConfig) -> dict[str, Path]:
    """Create the standard output directory layout for a run."""

    root = Path(output_root) / run_name
    subdirs = {
        "root": root,
        "checkpoints": root / config.outputs.checkpoint_subdir,
        "figures": root / config.outputs.figure_subdir,
        "logs": root / config.outputs.log_subdir,
        "predictions": root / config.outputs.prediction_subdir,
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def save_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a dictionary to YAML."""

    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a dictionary to JSON."""

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_csv_row(path: str | Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    """Append a row to a CSV log file, writing a header when needed."""

    csv_path = Path(path)
    is_new = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
