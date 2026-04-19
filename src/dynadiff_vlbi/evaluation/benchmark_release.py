"""Benchmark-release helpers for deterministic split and manifest export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dynadiff_vlbi.data.measurement_holdout import (
    HOLDOUT_STRATEGY_DESCRIPTIONS,
    HOLDOUT_STRATEGY_LABELS,
    build_structured_holdout_split,
)
from dynadiff_vlbi.utils.config import ExperimentConfig
from dynadiff_vlbi.utils.logging_utils import save_json


def load_dataset_arrays(dataset_dir: str | Path, split: str = "test") -> dict[str, np.ndarray]:
    """Load one generated dataset split into memory."""

    with np.load(Path(dataset_dir) / f"{split}.npz") as payload:
        return {key: payload[key] for key in payload.files}


def export_split_manifests(
    *,
    config: ExperimentConfig,
    dataset_dir: str | Path,
    output_dir: str | Path,
    split_name: str = "test",
) -> dict[str, Any]:
    """Export deterministic support/target manifests for one EMC protocol."""

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset_arrays(dataset_dir, split=split_name)
    baseline_pairs = dataset.get("baseline_pairs")
    frame_uv_indices = dataset.get("frame_uv_indices")
    frame_uv_coords = dataset.get("frame_uv_coords")
    station_positions = dataset.get("station_positions")
    if frame_uv_indices is None or frame_uv_coords is None:
        raise KeyError(f"Dataset {dataset_dir} is missing frame_uv_indices or frame_uv_coords.")

    support_tags: list[str] = []
    manifest: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "split_name": split_name,
        "strategy": config.holdout.strategy,
        "strategy_label": HOLDOUT_STRATEGY_LABELS.get(config.holdout.strategy, config.holdout.strategy),
        "strategy_description": HOLDOUT_STRATEGY_DESCRIPTIONS.get(config.holdout.strategy, ""),
        "support_fractions": {},
    }
    sample_indices = np.arange(dataset["ground_truth"].shape[0], dtype=np.int64)

    for support_fraction in config.holdout.eval_support_fractions:
        fraction_tag = f"{int(round(100.0 * float(support_fraction))):02d}"
        support_tags.append(fraction_tag)
        support_masks: list[np.ndarray] = []
        target_masks: list[np.ndarray] = []
        target_unit_counts: list[int] = []
        support_unit_counts: list[int] = []
        for sample_index in sample_indices.tolist():
            measurements = (
                dataset["vis_real"][sample_index] + 1j * dataset["vis_imag"][sample_index]
            ).astype(np.complex64)
            split = build_structured_holdout_split(
                measurements=measurements,
                observed_mask=dataset["mask"][sample_index].astype(np.float32),
                frame_uv_indices=frame_uv_indices,
                frame_uv_coords=frame_uv_coords,
                baseline_pairs=baseline_pairs,
                station_positions=station_positions,
                base_seed=config.project.seed,
                sample_index=sample_index,
                support_fraction=float(support_fraction),
                strategy=config.holdout.strategy,
            )
            support_masks.append(split.support_mask.astype(np.float32))
            target_masks.append(split.target_mask.astype(np.float32))
            target_unit_counts.append(int(split.target_unit_count))
            support_unit_counts.append(int(split.support_unit_count))

        np.savez_compressed(
            output_dir / f"support_{fraction_tag}_split_manifest.npz",
            sample_index=sample_indices,
            support_mask=np.stack(support_masks).astype(np.float32),
            target_mask=np.stack(target_masks).astype(np.float32),
            target_unit_count=np.asarray(target_unit_counts, dtype=np.int32),
            support_unit_count=np.asarray(support_unit_counts, dtype=np.int32),
        )
        manifest["support_fractions"][fraction_tag] = {
            "support_fraction": float(support_fraction),
            "sample_count": int(sample_indices.shape[0]),
            "mean_target_unit_count": float(np.mean(target_unit_counts)),
            "mean_support_unit_count": float(np.mean(support_unit_counts)),
            "manifest_npz": str((output_dir / f"support_{fraction_tag}_split_manifest.npz").resolve()),
        }

    manifest["support_tags"] = support_tags
    save_json(output_dir / "split_manifest.json", manifest)
    return manifest


def write_benchmark_output_manifest(
    *,
    output_path: str | Path,
    payload: dict[str, Any],
) -> None:
    """Write one benchmark-release manifest with deterministic formatting."""

    save_json(Path(output_path), payload)
