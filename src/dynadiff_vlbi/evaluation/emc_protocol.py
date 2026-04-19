"""Evaluation helpers for the Earned Measurement Consistency paper path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dynadiff_vlbi.data.feature_formatting import format_dirty_input, format_visibility_tensor
from dynadiff_vlbi.data.measurement_holdout import (
    HOLDOUT_STRATEGY_DESCRIPTIONS,
    HOLDOUT_STRATEGY_LABELS,
    build_structured_holdout_split,
    closure_triangle_support_counts,
)
from dynadiff_vlbi.evaluation.metrics import (
    closure_phase_mae,
    compute_reconstruction_metrics,
    observed_visibility_rmse,
)
from dynadiff_vlbi.emc.baselines import load_dps_baseline
from dynadiff_vlbi.models.factory import build_model
from dynadiff_vlbi.physics.classical_reconstruction import tikhonov_iterative_reconstruction
from dynadiff_vlbi.utils.config import ExperimentConfig, ModelConfig
from dynadiff_vlbi.utils.device import get_device
from dynadiff_vlbi.utils.logging_utils import save_json


@dataclass(frozen=True)
class ComparatorSpec:
    """One comparator checkpoint or classical method in the EMC protocol."""

    key: str
    label: str
    kind: str
    checkpoint_path: Path | None = None


@dataclass(frozen=True)
class LoadedComparator:
    """One loaded comparator model."""

    key: str
    label: str
    kind: str
    model: Any
    model_config: Any


def aggregate_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-sample metrics with NaN-aware means."""

    if not metric_rows:
        return {}
    keys = list(metric_rows[0].keys())
    aggregated: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in metric_rows], dtype=np.float64)
        aggregated[key] = float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")
    return aggregated


def _load_dataset_arrays(dataset_dir: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(dataset_dir) / "test.npz") as payload:
        return {key: payload[key] for key in payload.files}


def _load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, ModelConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = ModelConfig(**checkpoint["config"]["model"])
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_config


def load_comparators(specs: list[ComparatorSpec], device: torch.device) -> dict[str, LoadedComparator]:
    """Load the requested comparator models once for protocol evaluation."""

    loaded: dict[str, LoadedComparator] = {}
    for spec in specs:
        if spec.kind in {"dirty", "tikhonov", "ehtim_bridge"}:
            loaded[spec.key] = LoadedComparator(spec.key, spec.label, spec.kind, None, None)
            continue
        if spec.kind == "dps":
            if spec.checkpoint_path is None or not spec.checkpoint_path.exists():
                raise FileNotFoundError(f"Missing DPS checkpoint for comparator '{spec.key}': {spec.checkpoint_path}")
            loaded[spec.key] = LoadedComparator(
                spec.key,
                spec.label,
                spec.kind,
                load_dps_baseline(spec.checkpoint_path, device=device),
                None,
            )
            continue
        if spec.checkpoint_path is None or not spec.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for comparator '{spec.key}': {spec.checkpoint_path}")
        model, model_config = _load_checkpoint_model(spec.checkpoint_path, device=device)
        loaded[spec.key] = LoadedComparator(spec.key, spec.label, spec.kind, model, model_config)
    return loaded


def _predict_baseline(
    model: torch.nn.Module,
    dirty_sequence: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    input_tensor = torch.from_numpy(dirty_sequence).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        predictive_mean, _, _ = model.predict_with_uncertainty(x=input_tensor, n_samples=1)
    return predictive_mean.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def _predict_phase2(
    model: torch.nn.Module,
    model_config: ModelConfig,
    support_vis_real: np.ndarray,
    support_vis_imag: np.ndarray,
    support_mask: np.ndarray,
    support_dirty: np.ndarray,
    uv_coords: np.ndarray,
    frame_uv_coords: np.ndarray | None,
    frame_uv_indices: np.ndarray | None,
    measurements: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    visibility_input_array = format_visibility_tensor(
        vis_real=support_vis_real,
        vis_imag=support_vis_imag,
        mask=support_mask,
        representation=model_config.visibility_representation,
        include_mask_channel=model_config.include_mask_channel,
        include_uv_coords=model_config.include_uv_coords,
        uv_coords=uv_coords,
        include_observation_metadata=model_config.include_observation_metadata,
        frame_uv_coords=frame_uv_coords,
        frame_uv_indices=frame_uv_indices,
    )
    visibility_input = torch.from_numpy(visibility_input_array).unsqueeze(0).to(device)
    dirty_input = torch.from_numpy(format_dirty_input(support_dirty)).unsqueeze(0).to(device)
    model_measurements = torch.from_numpy((measurements * support_mask).astype(np.complex64)).unsqueeze(0).to(device)
    model_mask = torch.from_numpy(support_mask.astype(np.float32)).unsqueeze(0).to(device)
    baseline_pairs_tensor = None
    if frame_uv_indices is not None:
        frame_uv_indices_tensor = torch.from_numpy(frame_uv_indices.astype(np.int64)).unsqueeze(0).to(device)
    else:
        frame_uv_indices_tensor = None
    if frame_uv_coords is not None:
        _ = frame_uv_coords
    with torch.no_grad():
        outputs = model(
            visibility_input=visibility_input,
            dirty_input=dirty_input,
            measurements=model_measurements,
            mask=model_mask,
            baseline_pairs=baseline_pairs_tensor,
            frame_uv_indices=frame_uv_indices_tensor,
        )
    result = {
        "mean": outputs.mean.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32),
    }
    if hasattr(outputs, "pre_dc_prediction"):
        result["pre_dc_prediction"] = outputs.pre_dc_prediction.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    return result


def _predict_dps(
    baseline,
    support_measurements: np.ndarray,
    support_mask: np.ndarray,
    support_dirty: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        baseline.sample(
            support_vis=support_measurements.astype(np.complex64),
            support_mask=support_mask.astype(np.float32),
            dirty_recon=support_dirty.astype(np.float32),
        ),
        dtype=np.float32,
    )


def _prediction_metrics(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    ring_radius_px: float,
    hotspot_coords_px: np.ndarray,
    support_measurements: np.ndarray,
    support_mask: np.ndarray,
    target_measurements: np.ndarray,
    target_mask: np.ndarray,
    baseline_pairs: np.ndarray | None,
    frame_uv_indices: np.ndarray | None,
    min_eval_closure_triangles: int,
    max_triangles: int,
) -> dict[str, float]:
    metrics = compute_reconstruction_metrics(
        prediction=prediction,
        target=target,
        target_ring_radius_px=ring_radius_px,
        target_hotspot_coords_px=hotspot_coords_px,
    )
    metrics["support_visibility_rmse"] = observed_visibility_rmse(
        prediction=prediction,
        measurements=support_measurements,
        mask=support_mask,
    )
    metrics["heldout_visibility_rmse"] = observed_visibility_rmse(
        prediction=prediction,
        measurements=target_measurements,
        mask=target_mask,
    )
    metrics["support_closure_phase_mae"] = closure_phase_mae(
        prediction=prediction,
        measurements=support_measurements,
        mask=support_mask,
        baseline_pairs=baseline_pairs,
        frame_uv_indices=frame_uv_indices,
        max_triangles=max_triangles,
    )
    triangle_counts = closure_triangle_support_counts(
        target_mask=target_mask,
        support_mask=support_mask,
        baseline_pairs=np.asarray(baseline_pairs if baseline_pairs is not None else np.zeros((0, 2), dtype=np.int64)),
        frame_uv_indices=np.asarray(frame_uv_indices if frame_uv_indices is not None else np.zeros((0, 0, 2), dtype=np.int64)),
        max_triangles=max_triangles,
    )
    metrics["heldout_all_target_triangles"] = float(triangle_counts["all_target"])
    metrics["heldout_mixed_triangles"] = float(triangle_counts["mixed"])
    if triangle_counts["all_target"] >= min_eval_closure_triangles:
        metrics["heldout_closure_phase_mae"] = closure_phase_mae(
            prediction=prediction,
            measurements=target_measurements,
            mask=target_mask,
            baseline_pairs=baseline_pairs,
            frame_uv_indices=frame_uv_indices,
            max_triangles=max_triangles,
        )
    else:
        metrics["heldout_closure_phase_mae"] = float("nan")
    return metrics


def _support_fraction_tag(support_fraction: float) -> str:
    return f"{int(round(100.0 * support_fraction)):02d}"


def evaluate_emc_condition(
    *,
    config: ExperimentConfig,
    dataset_dir: Path,
    comparators: dict[str, LoadedComparator],
    output_dir: Path,
    support_fractions: tuple[float, ...],
) -> dict[str, Any]:
    """Evaluate a shared-split EMC support-fraction protocol for one condition."""

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    figures_dir = output_dir / "figures"
    predictions_dir = output_dir / "predictions"
    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset_arrays(dataset_dir)
    device = get_device()
    uv_coords = dataset.get("uv_coords")
    if uv_coords is None:
        raise KeyError(f"Dataset {dataset_dir} is missing uv_coords required for EMC evaluation.")
    baseline_pairs = dataset.get("baseline_pairs")
    frame_uv_indices = dataset.get("frame_uv_indices")
    frame_uv_coords = dataset.get("frame_uv_coords")

    summary: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "holdout": {
            "strategy": config.holdout.strategy,
            "label": HOLDOUT_STRATEGY_LABELS.get(config.holdout.strategy, config.holdout.strategy),
            "description": HOLDOUT_STRATEGY_DESCRIPTIONS.get(config.holdout.strategy, ""),
        },
        "support_fractions": {},
        "comparator_labels": {key: value.label for key, value in comparators.items()},
    }
    rows: list[dict[str, Any]] = []

    for support_fraction in support_fractions:
        fraction_key = _support_fraction_tag(float(support_fraction))
        per_model_metrics: dict[str, list[dict[str, float]]] = {key: [] for key in comparators}
        per_sample_rows: list[dict[str, Any]] = []
        prediction_payload: dict[str, list[np.ndarray]] = {
            "ground_truth": [],
            "support_mask": [],
            "target_mask": [],
        }
        for key in comparators:
            prediction_payload[key] = []

        for sample_index in range(dataset["ground_truth"].shape[0]):
            measurements = (dataset["vis_real"][sample_index] + 1j * dataset["vis_imag"][sample_index]).astype(
                np.complex64
            )
            observed_mask = dataset["mask"][sample_index].astype(np.float32)
            split = build_structured_holdout_split(
                measurements=measurements,
                observed_mask=observed_mask,
                frame_uv_indices=frame_uv_indices,
                frame_uv_coords=frame_uv_coords,
                baseline_pairs=baseline_pairs,
                station_positions=dataset.get("station_positions"),
                base_seed=config.project.seed,
                sample_index=sample_index,
                support_fraction=float(support_fraction),
                strategy=config.holdout.strategy,
            )
            target = dataset["ground_truth"][sample_index].astype(np.float32)
            ring_radius_px = float(dataset["ring_radius_px"][sample_index])
            hotspot_coords_px = dataset["hotspot_coords_px"][sample_index].astype(np.float32)
            support_vis_real = (measurements.real * split.support_mask).astype(np.float32)
            support_vis_imag = (measurements.imag * split.support_mask).astype(np.float32)

            predictions: dict[str, np.ndarray] = {
                "dirty": split.support_dirty.astype(np.float32),
                "tikhonov": tikhonov_iterative_reconstruction(
                    measurements=measurements,
                    mask=split.support_mask,
                    lambda_reg=config.evaluation.tikhonov_lambda,
                    iterations=config.evaluation.tikhonov_iterations,
                    step_size=config.evaluation.tikhonov_step_size,
                ).astype(np.float32),
            }

            for key, comparator in comparators.items():
                if comparator.kind == "dirty":
                    continue
                if comparator.kind == "tikhonov":
                    continue
                if comparator.kind == "dps":
                    predictions[key] = _predict_dps(
                        comparator.model,
                        split.support_measurements,
                        split.support_mask,
                        split.support_dirty.astype(np.float32),
                    )
                    continue
                if comparator.kind == "baseline":
                    predictions[key] = _predict_baseline(
                        model=comparator.model,  # type: ignore[arg-type]
                        dirty_sequence=split.support_dirty.astype(np.float32),
                        device=device,
                    )
                    continue
                phase2_prediction = _predict_phase2(
                    model=comparator.model,  # type: ignore[arg-type]
                    model_config=comparator.model_config,  # type: ignore[arg-type]
                    support_vis_real=support_vis_real,
                    support_vis_imag=support_vis_imag,
                    support_mask=split.support_mask,
                    support_dirty=split.support_dirty.astype(np.float32),
                    uv_coords=uv_coords,
                    frame_uv_coords=frame_uv_coords,
                    frame_uv_indices=frame_uv_indices,
                    measurements=measurements,
                    device=device,
                )
                predictions[key] = phase2_prediction["mean"]

            for key, prediction in predictions.items():
                metrics = _prediction_metrics(
                    prediction=prediction,
                    target=target,
                    ring_radius_px=ring_radius_px,
                    hotspot_coords_px=hotspot_coords_px,
                    support_measurements=split.support_measurements,
                    support_mask=split.support_mask,
                    target_measurements=split.target_measurements,
                    target_mask=split.target_mask,
                    baseline_pairs=baseline_pairs,
                    frame_uv_indices=frame_uv_indices,
                    min_eval_closure_triangles=config.holdout.min_eval_closure_triangles,
                    max_triangles=config.holdout.max_triangles,
                )
                per_model_metrics[key].append(metrics)
                sample_row = {
                    "sample_index": sample_index,
                    "support_fraction": float(support_fraction),
                    "support_fraction_tag": fraction_key,
                    "holdout_strategy": split.strategy,
                    "holdout_strategy_label": HOLDOUT_STRATEGY_LABELS.get(split.strategy, split.strategy),
                    "target_unit_count": float(split.target_unit_count),
                    "support_unit_count": float(split.support_unit_count),
                    "model": key,
                    "model_label": comparators[key].label,
                }
                sample_row.update(metrics)
                per_sample_rows.append(sample_row)

            prediction_payload["ground_truth"].append(target.astype(np.float32))
            prediction_payload["support_mask"].append(split.support_mask.astype(np.float32))
            prediction_payload["target_mask"].append(split.target_mask.astype(np.float32))
            for key in comparators:
                prediction_payload[key].append(predictions[key].astype(np.float32))

        aggregate = {key: aggregate_metrics(value) for key, value in per_model_metrics.items()}
        summary["support_fractions"][fraction_key] = {
            "support_fraction": float(support_fraction),
            "holdout_strategy": config.holdout.strategy,
            "holdout_strategy_label": HOLDOUT_STRATEGY_LABELS.get(config.holdout.strategy, config.holdout.strategy),
            "mean_target_unit_count": float(
                np.mean([row["target_unit_count"] for row in per_sample_rows if row["model"] == "dirty"])
            ),
            "mean_support_unit_count": float(
                np.mean([row["support_unit_count"] for row in per_sample_rows if row["model"] == "dirty"])
            ),
            "models": aggregate,
        }
        for key, metrics in aggregate.items():
            row = {
                "support_fraction": float(support_fraction),
                "support_fraction_tag": fraction_key,
                "model": key,
                "model_label": comparators[key].label,
            }
            row.update(metrics)
            rows.append(row)
        np.savez_compressed(
            predictions_dir / f"support_{fraction_key}.npz",
            **{key: np.stack(value).astype(np.float32) for key, value in prediction_payload.items()},
        )
        import csv

        per_sample_fields: list[str] = []
        for row in per_sample_rows:
            for key in row.keys():
                if key not in per_sample_fields:
                    per_sample_fields.append(key)
        with (logs_dir / f"per_sample_support_{fraction_key}.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=per_sample_fields)
            writer.writeheader()
            writer.writerows(per_sample_rows)

    save_json(logs_dir / "emc_protocol_summary.json", summary)
    import csv

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with (logs_dir / "support_fraction_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary
