"""Helpers for cross-model comparison summaries."""

from __future__ import annotations

import csv
from pathlib import Path


def save_comparison_csv(path: str | Path, summary: dict[str, dict[str, float]]) -> None:
    """Save per-model metrics to a simple CSV table."""

    rows = []
    for model_name, metrics in summary.items():
        if not isinstance(metrics, dict) or not metrics or model_name == "uncertainty":
            continue
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
