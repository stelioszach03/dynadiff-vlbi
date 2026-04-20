"""Observation-domain EMC evaluation on real interferometric measurements."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from dynadiff_vlbi.data.measurement_holdout import (
    HOLDOUT_STRATEGY_DESCRIPTIONS,
    HOLDOUT_STRATEGY_LABELS,
    build_structured_holdout_split,
    resolve_partition_strategy,
    closure_triangle_support_counts,
)
from dynadiff_vlbi.evaluation.emc_protocol import (
    LoadedComparator,
    _predict_baseline,
    _predict_phase2,
    _support_fraction_tag,
    aggregate_metrics,
)
from dynadiff_vlbi.evaluation.ehtim_bridge import predict_ehtim_bridge_sequence
from dynadiff_vlbi.evaluation.metrics import (
    closure_phase_mae,
    observed_visibility_rmse,
    reduced_weighted_visibility_chi2,
    weighted_visibility_chi2,
)
from dynadiff_vlbi.physics.classical_reconstruction import tikhonov_iterative_reconstruction
from dynadiff_vlbi.utils.config import ExperimentConfig
from dynadiff_vlbi.utils.device import get_device
from dynadiff_vlbi.utils.logging_utils import save_json


def _load_dataset_arrays(dataset_dir: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(dataset_dir) / "test.npz") as payload:
        return {key: payload[key] for key in payload.files}


def _load_dataset_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(dataset_dir) / "real_data_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _real_prediction_metrics(
    *,
    prediction: np.ndarray,
    measurements: np.ndarray,
    observed_mask: np.ndarray,
    sigma: np.ndarray | None,
    support_measurements: np.ndarray,
    support_mask: np.ndarray,
    target_measurements: np.ndarray,
    target_mask: np.ndarray,
    baseline_pairs: np.ndarray | None,
    frame_uv_indices: np.ndarray | None,
    min_eval_closure_triangles: int,
    max_triangles: int,
) -> dict[str, float]:
    metrics = {
        "support_visibility_rmse": observed_visibility_rmse(
            prediction=prediction,
            measurements=support_measurements,
            mask=support_mask,
        ),
        "heldout_visibility_rmse": observed_visibility_rmse(
            prediction=prediction,
            measurements=target_measurements,
            mask=target_mask,
        ),
        "observed_visibility_rmse": observed_visibility_rmse(
            prediction=prediction,
            measurements=measurements,
            mask=observed_mask,
        ),
        "support_weighted_chi2": weighted_visibility_chi2(
            prediction=prediction,
            measurements=support_measurements,
            mask=support_mask,
            sigma=(sigma * support_mask) if sigma is not None else None,
        ),
        "heldout_weighted_chi2": weighted_visibility_chi2(
            prediction=prediction,
            measurements=target_measurements,
            mask=target_mask,
            sigma=(sigma * target_mask) if sigma is not None else None,
        ),
        "observed_weighted_chi2": weighted_visibility_chi2(
            prediction=prediction,
            measurements=measurements,
            mask=observed_mask,
            sigma=sigma,
        ),
        "support_reduced_chi2": reduced_weighted_visibility_chi2(
            prediction=prediction,
            measurements=support_measurements,
            mask=support_mask,
            sigma=(sigma * support_mask) if sigma is not None else None,
        ),
        "heldout_reduced_chi2": reduced_weighted_visibility_chi2(
            prediction=prediction,
            measurements=target_measurements,
            mask=target_mask,
            sigma=(sigma * target_mask) if sigma is not None else None,
        ),
        "observed_reduced_chi2": reduced_weighted_visibility_chi2(
            prediction=prediction,
            measurements=measurements,
            mask=observed_mask,
            sigma=sigma,
        ),
        "support_closure_phase_mae": closure_phase_mae(
            prediction=prediction,
            measurements=support_measurements,
            mask=support_mask,
            baseline_pairs=baseline_pairs,
            frame_uv_indices=frame_uv_indices,
            max_triangles=max_triangles,
        ),
        "observed_closure_phase_mae": closure_phase_mae(
            prediction=prediction,
            measurements=measurements,
            mask=observed_mask,
            baseline_pairs=baseline_pairs,
            frame_uv_indices=frame_uv_indices,
            max_triangles=max_triangles,
        ),
        "observed_coefficient_count": float(np.count_nonzero(observed_mask)),
        "support_coefficient_count": float(np.count_nonzero(support_mask)),
        "heldout_coefficient_count": float(np.count_nonzero(target_mask)),
    }
    triangle_counts = closure_triangle_support_counts(
        target_mask=target_mask,
        support_mask=support_mask,
        baseline_pairs=np.asarray(
            baseline_pairs if baseline_pairs is not None else np.zeros((0, 2), dtype=np.int64)
        ),
        frame_uv_indices=np.asarray(
            frame_uv_indices if frame_uv_indices is not None else np.zeros((0, 0, 2), dtype=np.int64)
        ),
        max_triangles=max_triangles,
    )
    metrics["heldout_all_target_triangles"] = float(triangle_counts["all_target"])
    metrics["heldout_mixed_triangles"] = float(triangle_counts["mixed"])
    metrics["support_only_triangles"] = float(triangle_counts["support_only"])
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


def evaluate_real_data_condition(
    *,
    config: ExperimentConfig,
    dataset_dir: Path,
    comparators: dict[str, LoadedComparator],
    output_dir: Path,
    support_fractions: tuple[float, ...],
) -> dict[str, Any]:
    """Evaluate EMC-style support/target protocols on real measurement products."""

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    predictions_dir = output_dir / "predictions"
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset_arrays(dataset_dir)
    manifest = _load_dataset_manifest(dataset_dir)
    device = get_device()
    uv_coords = dataset.get("uv_coords")
    if uv_coords is None:
        raise KeyError(f"Dataset {dataset_dir} is missing uv_coords required for real-data EMC evaluation.")
    sigma = dataset.get("vis_sigma")
    baseline_pairs = dataset.get("baseline_pairs")
    station_labels = dataset.get("station_labels")
    station_positions = dataset.get("station_positions")
    frame_uv_indices_all = dataset.get("frame_uv_indices")
    frame_uv_coords_all = dataset.get("frame_uv_coords")
    if frame_uv_indices_all is None or frame_uv_coords_all is None:
        raise KeyError(f"Dataset {dataset_dir} is missing frame_uv_indices/frame_uv_coords.")

    sample_count = int(dataset["vis_real"].shape[0])
    summary: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "sample_count": sample_count,
        "sample_ids": dataset.get("sample_id", np.arange(sample_count)).tolist(),
        "release_code": manifest.get("release_code", dataset.get("release_code", np.asarray([""])).tolist()[0] if "release_code" in dataset else ""),
        "target": manifest.get("target", dataset.get("target_name", np.asarray([""])).tolist()[0] if "target_name" in dataset else ""),
        "campaign_year": int(
            manifest.get(
                "campaign_year",
                int(dataset.get("campaign_year", np.asarray([0], dtype=np.int32))[0]) if "campaign_year" in dataset else 0,
            )
        ),
        "pipeline": manifest.get("pipeline", dataset.get("pipeline_name", np.asarray([""])).tolist()[0] if "pipeline_name" in dataset else ""),
        "source_repo": manifest.get("source_repo", ""),
        "files": manifest.get("files", []),
        "dataset_description": manifest.get("dataset_description", ""),
        "sample_groups": manifest.get("sample_groups", []),
        "holdout": {
            "strategy": config.holdout.strategy,
            "label": HOLDOUT_STRATEGY_LABELS.get(config.holdout.strategy, config.holdout.strategy),
            "description": HOLDOUT_STRATEGY_DESCRIPTIONS.get(config.holdout.strategy, ""),
        },
        "notes": (
            "Real-data validation is observation-domain only: there is no image-domain ground truth, "
            "so the protocol reports support-set, held-out-set, and full observed measurement agreement. "
            "Sigma-weighted diagnostics use released Isigma values as observation-domain residual weights, "
            "not as a full calibrated likelihood model."
        ),
        "support_fractions": {},
        "comparator_labels": {key: value.label for key, value in comparators.items()},
    }
    rows: list[dict[str, Any]] = []

    for support_fraction in support_fractions:
        fraction_key = _support_fraction_tag(float(support_fraction))
        per_model_metrics: dict[str, list[dict[str, float]]] = {key: [] for key in comparators}
        per_sample_rows: list[dict[str, Any]] = []
        prediction_payload: dict[str, list[np.ndarray]] = {
            "support_mask": [],
            "target_mask": [],
        }
        for key in comparators:
            prediction_payload[key] = []

        for sample_index in range(sample_count):
            frame_uv_indices = frame_uv_indices_all[sample_index].astype(np.int64)
            frame_uv_coords = frame_uv_coords_all[sample_index].astype(np.float32)
            measurements = (
                dataset["vis_real"][sample_index] + 1j * dataset["vis_imag"][sample_index]
            ).astype(np.complex64)
            sample_sigma = sigma[sample_index].astype(np.float32) if sigma is not None else None
            observed_mask = dataset["mask"][sample_index].astype(np.float32)
            strategy, oracle_model = resolve_partition_strategy(config.holdout)
            split = build_structured_holdout_split(
                measurements=measurements,
                observed_mask=observed_mask,
                frame_uv_indices=frame_uv_indices,
                frame_uv_coords=frame_uv_coords,
                baseline_pairs=baseline_pairs,
                station_positions=station_positions,
                base_seed=config.project.seed,
                sample_index=sample_index,
                support_fraction=float(support_fraction),
                strategy=strategy,
                oracle_model=oracle_model,
            )

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
                if comparator.kind == "baseline":
                    predictions[key] = _predict_baseline(
                        model=comparator.model,  # type: ignore[arg-type]
                        dirty_sequence=split.support_dirty.astype(np.float32),
                        device=device,
                    )
                    continue
                if comparator.kind == "ehtim_bridge":
                    if station_labels is None or station_positions is None:
                        raise KeyError(
                            f"Dataset {dataset_dir} is missing station_labels/station_positions required for the ehtim bridge."
                        )
                    predictions[key] = predict_ehtim_bridge_sequence(
                        measurements=measurements,
                        support_mask=split.support_mask,
                        sigma=sample_sigma,
                        frame_uv_indices=frame_uv_indices,
                        baseline_pairs=baseline_pairs if baseline_pairs is not None else np.zeros((0, 2), dtype=np.int64),
                        station_labels=station_labels,
                        station_positions=station_positions,
                        rf_hz=float(dataset.get("freq_ghz", np.asarray([230.0], dtype=np.float32))[sample_index]) * 1.0e9,
                        source_name=str(
                            dataset.get("source_name", np.asarray([summary["target"]], dtype="<U16"))[sample_index]
                        ),
                        mjd=float(dataset.get("mjd", np.asarray([0.0], dtype=np.float32))[sample_index]),
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
                metrics = _real_prediction_metrics(
                    prediction=prediction,
                    measurements=measurements,
                    observed_mask=observed_mask,
                    sigma=sample_sigma,
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
                    "sample_id": str(dataset.get("sample_id", np.asarray([sample_index]))[sample_index]),
                    "support_fraction": float(support_fraction),
                    "support_fraction_tag": fraction_key,
                    "holdout_strategy": split.strategy,
                    "holdout_strategy_label": HOLDOUT_STRATEGY_LABELS.get(split.strategy, split.strategy),
                    "target_unit_count": float(split.target_unit_count),
                    "support_unit_count": float(split.support_unit_count),
                    "model": key,
                    "model_label": comparators[key].label,
                    "release_code": summary["release_code"],
                    "target": summary["target"],
                    "campaign_year": summary["campaign_year"],
                    "pipeline": summary["pipeline"],
                }
                if "day_of_year" in dataset:
                    sample_row["day_of_year"] = int(dataset["day_of_year"][sample_index])
                if "band" in dataset:
                    sample_row["band"] = str(dataset["band"][sample_index])
                sample_row.update(metrics)
                per_sample_rows.append(sample_row)

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
            "mean_target_coefficients": float(
                np.mean([row["heldout_coefficient_count"] for row in per_sample_rows if row["model"] == "dirty"])
            ),
            "mean_support_coefficients": float(
                np.mean([row["support_coefficient_count"] for row in per_sample_rows if row["model"] == "dirty"])
            ),
            "mean_all_target_triangles": float(
                np.mean([row["heldout_all_target_triangles"] for row in per_sample_rows if row["model"] == "dirty"])
            ),
            "mean_mixed_triangles": float(
                np.mean([row["heldout_mixed_triangles"] for row in per_sample_rows if row["model"] == "dirty"])
            ),
            "days_present": sorted(
                {int(row["day_of_year"]) for row in per_sample_rows if row["model"] == "dirty" and "day_of_year" in row}
            ),
            "bands_present": sorted(
                {str(row["band"]) for row in per_sample_rows if row["model"] == "dirty" and "band" in row}
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

    save_json(logs_dir / "real_data_protocol_summary.json", summary)
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
