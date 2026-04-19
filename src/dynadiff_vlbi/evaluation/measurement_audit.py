"""Measurement-attribution audit helpers for the CCRR paper path."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from dynadiff_vlbi.data.feature_formatting import format_visibility_tensor
from dynadiff_vlbi.data.visibility_dataset import VisibilityConditionedDataset
from dynadiff_vlbi.evaluation.ccrr_artifacts import summarize_prediction_bundle
from dynadiff_vlbi.evaluation.metrics import closure_phase_mae, observed_visibility_rmse
from dynadiff_vlbi.evaluation.paper_artifacts import format_value, save_json, write_csv
from dynadiff_vlbi.evaluation.paper_visuals import load_predictions
from dynadiff_vlbi.models.factory import build_model
from dynadiff_vlbi.physics.classical_reconstruction import dirty_image_reconstruction
from dynadiff_vlbi.physics.sampling import conjugate_index
from dynadiff_vlbi.utils.config import ModelConfig
from dynadiff_vlbi.utils.device import get_device


DEFAULT_HOLDOUT_FRACTION = 0.20
DEFAULT_MAX_TRIANGLES = 24
DEFAULT_MIN_TOTAL_HELDOUT_TRIANGLES = 24
DEFAULT_MIN_VALID_CLOSURE_SAMPLES = 16


class MeasurementAuditError(RuntimeError):
    """Raised when required CCRR audit inputs are missing or inconsistent."""


@dataclass(frozen=True)
class MeasurementAuditSpec:
    """One saved CCRR run to audit."""

    key: str
    title: str
    seed: int
    run_name: str
    dataset_dir: Path
    output_root: Path

    @property
    def prediction_path(self) -> Path:
        return self.output_root / self.run_name / "predictions" / "test_predictions.npz"

    @property
    def checkpoint_path(self) -> Path:
        return self.output_root / self.run_name / "checkpoints" / "best.pt"


@dataclass(frozen=True)
class ClosureTopology:
    """Reusable closure-triangle topology for one dataset."""

    triangles: tuple[tuple[int, int, int], ...]
    pair_lookup: dict[tuple[int, int], int]


def _load_dataset_arrays(dataset_dir: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(dataset_dir) / "test.npz") as payload:
        return {key: payload[key] for key in payload.files}


def _float_mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if not np.all(np.isnan(array)) else float("nan")


def _float_std(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanstd(array)) if not np.all(np.isnan(array)) else float("nan")


def _load_ccrr_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, ModelConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = ModelConfig(**checkpoint["config"]["model"])
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_config


def _build_closure_topology(baseline_pairs: np.ndarray, max_triangles: int) -> ClosureTopology:
    pair_lookup = {
        tuple(pair.tolist()): pair_index for pair_index, pair in enumerate(np.asarray(baseline_pairs, dtype=np.int64))
    }
    if not pair_lookup:
        return ClosureTopology(triangles=tuple(), pair_lookup={})
    station_count = int(np.max(baseline_pairs)) + 1
    triangles = tuple(combinations(range(station_count), 3))[:max_triangles]
    return ClosureTopology(triangles=triangles, pair_lookup=pair_lookup)


def validate_prediction_bundle(
    prediction_path: str | Path,
    dataset_dir: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Validate that an archived CCRR prediction bundle is audit-ready."""

    bundle = load_predictions(prediction_path)
    dataset = _load_dataset_arrays(dataset_dir)
    required_keys = ("ground_truth", "baseline_prediction", "pre_dc_prediction", "ccrr")
    missing = [key for key in required_keys if key not in bundle]
    if missing:
        missing_text = ", ".join(missing)
        raise MeasurementAuditError(
            f"Missing required keys in {prediction_path}: {missing_text}. "
            "The measurement audit requires archived pre-DC and post-DC predictions."
        )

    expected_shape = dataset["ground_truth"].shape
    for key in required_keys:
        if bundle[key].shape != expected_shape:
            raise MeasurementAuditError(
                f"Inconsistent shape for '{key}' in {prediction_path}: "
                f"expected {expected_shape}, got {bundle[key].shape}."
            )

    if "baseline_pairs" not in dataset or "frame_uv_indices" not in dataset:
        raise MeasurementAuditError(
            f"Dataset {dataset_dir} is missing closure metadata required for the CCRR measurement audit."
        )
    return bundle, dataset


def compute_pre_post_dc_audit(
    prediction_path: str | Path,
    dataset_dir: str | Path,
) -> dict[str, Any]:
    """Measure full-observed metrics before and after the CCRR DC layer."""

    bundle, dataset = validate_prediction_bundle(prediction_path=prediction_path, dataset_dir=dataset_dir)
    measurements = dataset["vis_real"] + 1j * dataset["vis_imag"]
    mask = dataset["mask"]
    baseline_pairs = dataset["baseline_pairs"]
    frame_uv_indices = dataset["frame_uv_indices"]

    stages = {
        "baseline_prediction": bundle["baseline_prediction"],
        "pre_dc_prediction": bundle["pre_dc_prediction"],
        "ccrr_post_dc": bundle["ccrr"],
    }
    metric_lists: dict[str, dict[str, list[float]]] = {
        stage_key: {"observed_visibility_rmse": [], "closure_phase_mae": []} for stage_key in stages
    }

    for sample_index in range(dataset["ground_truth"].shape[0]):
        sample_measurements = measurements[sample_index]
        sample_mask = mask[sample_index]
        for stage_key, predictions in stages.items():
            prediction = predictions[sample_index]
            metric_lists[stage_key]["observed_visibility_rmse"].append(
                observed_visibility_rmse(
                    prediction=prediction,
                    measurements=sample_measurements,
                    mask=sample_mask,
                )
            )
            metric_lists[stage_key]["closure_phase_mae"].append(
                closure_phase_mae(
                    prediction=prediction,
                    measurements=sample_measurements,
                    mask=sample_mask,
                    baseline_pairs=baseline_pairs,
                    frame_uv_indices=frame_uv_indices,
                )
            )

    summary: dict[str, Any] = {"stages": {}}
    for stage_key, metrics in metric_lists.items():
        summary["stages"][stage_key] = {
            "observed_visibility_rmse_mean": _float_mean(metrics["observed_visibility_rmse"]),
            "observed_visibility_rmse_std": _float_std(metrics["observed_visibility_rmse"]),
            "closure_phase_mae_mean": _float_mean(metrics["closure_phase_mae"]),
            "closure_phase_mae_std": _float_std(metrics["closure_phase_mae"]),
            "sample_count": len(metrics["observed_visibility_rmse"]),
            "closure_valid_samples": int(np.sum(~np.isnan(np.asarray(metrics["closure_phase_mae"], dtype=np.float64)))),
        }

    baseline_stage = summary["stages"]["baseline_prediction"]
    pre_stage = summary["stages"]["pre_dc_prediction"]
    post_stage = summary["stages"]["ccrr_post_dc"]
    summary["deltas"] = {
        "pre_dc_vs_baseline_visrmse_gain": float(
            baseline_stage["observed_visibility_rmse_mean"] - pre_stage["observed_visibility_rmse_mean"]
        ),
        "post_dc_vs_pre_dc_visrmse_gain": float(
            pre_stage["observed_visibility_rmse_mean"] - post_stage["observed_visibility_rmse_mean"]
        ),
        "post_dc_vs_baseline_visrmse_gain": float(
            baseline_stage["observed_visibility_rmse_mean"] - post_stage["observed_visibility_rmse_mean"]
        ),
        "pre_dc_vs_baseline_closure_gain": float(
            baseline_stage["closure_phase_mae_mean"] - pre_stage["closure_phase_mae_mean"]
        ),
        "post_dc_vs_pre_dc_closure_gain": float(
            pre_stage["closure_phase_mae_mean"] - post_stage["closure_phase_mae_mean"]
        ),
        "post_dc_vs_baseline_closure_gain": float(
            baseline_stage["closure_phase_mae_mean"] - post_stage["closure_phase_mae_mean"]
        ),
    }
    return summary


def build_deterministic_heldout_masks(
    mask: np.ndarray,
    frame_uv_indices: np.ndarray,
    base_seed: int,
    sample_index: int,
    fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Reserve a deterministic subset of observed coefficients for held-out auditing."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("Hold-out fraction must lie strictly between 0 and 1.")

    heldout_mask = np.zeros_like(mask, dtype=np.float32)
    enforced_mask = np.asarray(mask, dtype=np.float32).copy()
    baseline_counts: list[int] = []
    image_size = int(mask.shape[-1])

    for frame_index in range(mask.shape[0]):
        observed_baselines = [
            (baseline_index, int(row), int(col))
            for baseline_index, (row, col) in enumerate(frame_uv_indices[frame_index])
            if mask[frame_index, row, col] > 0.0
        ]
        if not observed_baselines:
            baseline_counts.append(0)
            continue

        holdout_count = max(1, int(round(len(observed_baselines) * fraction)))
        rng = np.random.default_rng(base_seed * 1000003 + sample_index * 1009 + frame_index)
        chosen_indices = rng.choice(
            len(observed_baselines),
            size=min(holdout_count, len(observed_baselines)),
            replace=False,
        )
        baseline_counts.append(int(len(np.atleast_1d(chosen_indices))))

        for chosen_index in np.atleast_1d(chosen_indices).tolist():
            _, row, col = observed_baselines[int(chosen_index)]
            heldout_mask[frame_index, row, col] = 1.0
            enforced_mask[frame_index, row, col] = 0.0
            conj_row = conjugate_index(row, image_size)
            conj_col = conjugate_index(col, image_size)
            if mask[frame_index, conj_row, conj_col] > 0.0:
                heldout_mask[frame_index, conj_row, conj_col] = 1.0
                enforced_mask[frame_index, conj_row, conj_col] = 0.0

    metadata = {
        "grid_coefficients_total": float(heldout_mask.sum()),
        "grid_coefficients_mean": float(heldout_mask.sum() / max(mask.shape[0], 1)),
        "baseline_coefficients_total": float(np.sum(baseline_counts)),
        "baseline_coefficients_mean": float(np.mean(baseline_counts)) if baseline_counts else 0.0,
    }
    return heldout_mask, enforced_mask, metadata


def summarize_triangle_support(
    observed_mask: np.ndarray,
    heldout_mask: np.ndarray,
    baseline_pairs: np.ndarray,
    frame_uv_indices: np.ndarray,
    max_triangles: int = DEFAULT_MAX_TRIANGLES,
) -> dict[str, int]:
    """Count all-heldout and mixed closure triangles for audit support reporting."""

    topology = _build_closure_topology(baseline_pairs=baseline_pairs, max_triangles=max_triangles)
    all_heldout = 0
    mixed = 0
    enforced_only = 0
    for frame_index in range(observed_mask.shape[0]):
        frame_observed = observed_mask[frame_index].astype(bool)
        frame_heldout = heldout_mask[frame_index].astype(bool)
        frame_indices = frame_uv_indices[frame_index]
        for first, second, third in topology.triangles:
            pair_ab = topology.pair_lookup.get((first, second))
            pair_bc = topology.pair_lookup.get((second, third))
            pair_ac = topology.pair_lookup.get((first, third))
            if pair_ab is None or pair_bc is None or pair_ac is None:
                continue
            row_ab, col_ab = frame_indices[pair_ab]
            row_bc, col_bc = frame_indices[pair_bc]
            row_ac, col_ac = frame_indices[pair_ac]
            observed_edges = [
                frame_observed[row_ab, col_ab],
                frame_observed[row_bc, col_bc],
                frame_observed[row_ac, col_ac],
            ]
            if not all(observed_edges):
                continue
            heldout_edges = [
                frame_heldout[row_ab, col_ab],
                frame_heldout[row_bc, col_bc],
                frame_heldout[row_ac, col_ac],
            ]
            if all(heldout_edges):
                all_heldout += 1
            elif any(heldout_edges):
                mixed += 1
            else:
                enforced_only += 1
    return {
        "all_heldout": int(all_heldout),
        "mixed": int(mixed),
        "enforced_only": int(enforced_only),
    }


def _predict_ccrr_with_measurement_override(
    model: torch.nn.Module,
    model_config: ModelConfig,
    sample: dict[str, torch.Tensor],
    reduced_measurements: np.ndarray,
    enforced_mask: np.ndarray,
    reduced_dirty: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    visibility_input_array = format_visibility_tensor(
        vis_real=reduced_measurements.real.astype(np.float32),
        vis_imag=reduced_measurements.imag.astype(np.float32),
        mask=enforced_mask.astype(np.float32),
        representation=model_config.visibility_representation,
        include_mask_channel=model_config.include_mask_channel,
        include_uv_coords=model_config.include_uv_coords,
        uv_coords=sample["uv_coords"].cpu().numpy() if "uv_coords" in sample else None,
        include_observation_metadata=model_config.include_observation_metadata,
        frame_uv_coords=sample["frame_uv_coords"].cpu().numpy() if "frame_uv_coords" in sample else None,
        frame_uv_indices=sample["frame_uv_indices"].cpu().numpy() if "frame_uv_indices" in sample else None,
    )
    visibility_input = torch.from_numpy(visibility_input_array).unsqueeze(0).to(device)
    dirty_input = torch.from_numpy(reduced_dirty[None]).unsqueeze(0).to(device)
    measurements = torch.from_numpy(reduced_measurements).unsqueeze(0).to(device).to(torch.complex64)
    mask_tensor = torch.from_numpy(enforced_mask).unsqueeze(0).to(device).to(torch.float32)
    baseline_pairs = sample["baseline_pairs"].unsqueeze(0).to(device) if "baseline_pairs" in sample else None
    frame_uv_indices = sample["frame_uv_indices"].unsqueeze(0).to(device) if "frame_uv_indices" in sample else None

    with torch.no_grad():
        outputs = model(
            visibility_input=visibility_input,
            dirty_input=dirty_input,
            measurements=measurements,
            mask=mask_tensor,
            baseline_pairs=baseline_pairs,
            frame_uv_indices=frame_uv_indices,
        )
    return {
        "baseline_prediction": outputs.baseline_prediction.squeeze(0).squeeze(0).cpu().numpy(),
        "pre_dc_prediction": outputs.pre_dc_prediction.squeeze(0).squeeze(0).cpu().numpy(),
        "ccrr_post_dc": outputs.mean.squeeze(0).squeeze(0).cpu().numpy(),
    }


def run_heldout_measurement_audit(
    spec: MeasurementAuditSpec,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    max_triangles: int = DEFAULT_MAX_TRIANGLES,
    min_total_heldout_triangles: int = DEFAULT_MIN_TOTAL_HELDOUT_TRIANGLES,
    min_valid_closure_samples: int = DEFAULT_MIN_VALID_CLOSURE_SAMPLES,
) -> dict[str, Any]:
    """Run the deterministic held-out attribution audit on one saved CCRR checkpoint."""

    device = get_device(prefer_gpu=False)
    model, model_config = _load_ccrr_checkpoint(checkpoint_path=spec.checkpoint_path, device=device)
    dataset = VisibilityConditionedDataset(spec.dataset_dir / "test.npz", model_config=model_config)

    stage_metric_lists = {
        stage_key: {"heldout_visibility_rmse": [], "heldout_closure_phase_mae": []}
        for stage_key in ("baseline_prediction", "pre_dc_prediction", "ccrr_post_dc")
    }
    support_rows: list[dict[str, int | float]] = []

    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        observed_mask = sample["mask"].cpu().numpy().astype(np.float32)
        measurements = sample["vis_real"].cpu().numpy() + 1j * sample["vis_imag"].cpu().numpy()
        frame_uv_indices = sample["frame_uv_indices"].cpu().numpy()
        baseline_pairs = sample["baseline_pairs"].cpu().numpy()

        heldout_mask, enforced_mask, holdout_metadata = build_deterministic_heldout_masks(
            mask=observed_mask,
            frame_uv_indices=frame_uv_indices,
            base_seed=spec.seed,
            sample_index=sample_index,
            fraction=holdout_fraction,
        )
        reduced_measurements = measurements * enforced_mask
        reduced_dirty = dirty_image_reconstruction(measurements=reduced_measurements, mask=enforced_mask)
        stage_predictions = _predict_ccrr_with_measurement_override(
            model=model,
            model_config=model_config,
            sample=sample,
            reduced_measurements=reduced_measurements,
            enforced_mask=enforced_mask,
            reduced_dirty=reduced_dirty,
            device=device,
        )
        support = summarize_triangle_support(
            observed_mask=observed_mask,
            heldout_mask=heldout_mask,
            baseline_pairs=baseline_pairs,
            frame_uv_indices=frame_uv_indices,
            max_triangles=max_triangles,
        )
        support_rows.append(
            {
                "sample_index": sample_index,
                "grid_coefficients_total": int(holdout_metadata["grid_coefficients_total"]),
                "baseline_coefficients_total": int(holdout_metadata["baseline_coefficients_total"]),
                "all_heldout_triangles": support["all_heldout"],
                "mixed_triangles": support["mixed"],
                "enforced_only_triangles": support["enforced_only"],
            }
        )

        for stage_key, prediction in stage_predictions.items():
            stage_metric_lists[stage_key]["heldout_visibility_rmse"].append(
                observed_visibility_rmse(
                    prediction=prediction,
                    measurements=measurements,
                    mask=heldout_mask,
                )
            )
            stage_metric_lists[stage_key]["heldout_closure_phase_mae"].append(
                closure_phase_mae(
                    prediction=prediction,
                    measurements=measurements,
                    mask=heldout_mask,
                    baseline_pairs=baseline_pairs,
                    frame_uv_indices=frame_uv_indices,
                    max_triangles=max_triangles,
                )
            )

    support_array = {
        key: np.asarray([row[key] for row in support_rows], dtype=np.float64) for key in support_rows[0].keys() if key != "sample_index"
    }
    valid_closure_samples = int(
        np.sum(~np.isnan(np.asarray(stage_metric_lists["ccrr_post_dc"]["heldout_closure_phase_mae"], dtype=np.float64)))
    )
    total_heldout_triangles = int(np.sum(support_array["all_heldout_triangles"]))
    closure_reported = (
        total_heldout_triangles >= min_total_heldout_triangles and valid_closure_samples >= min_valid_closure_samples
    )

    summary: dict[str, Any] = {
        "support": {
            "holdout_fraction": holdout_fraction,
            "mean_grid_coefficients": float(np.mean(support_array["grid_coefficients_total"])),
            "mean_baseline_coefficients": float(np.mean(support_array["baseline_coefficients_total"])),
            "mean_all_heldout_triangles": float(np.mean(support_array["all_heldout_triangles"])),
            "mean_mixed_triangles": float(np.mean(support_array["mixed_triangles"])),
            "total_all_heldout_triangles": total_heldout_triangles,
            "total_mixed_triangles": int(np.sum(support_array["mixed_triangles"])),
            "valid_closure_samples": valid_closure_samples,
            "insufficient_closure_samples": int(len(dataset) - valid_closure_samples),
            "closure_reported": closure_reported,
            "min_total_heldout_triangles": min_total_heldout_triangles,
            "min_valid_closure_samples": min_valid_closure_samples,
        },
        "stages": {},
    }

    for stage_key, metrics in stage_metric_lists.items():
        summary["stages"][stage_key] = {
            "heldout_visibility_rmse_mean": _float_mean(metrics["heldout_visibility_rmse"]),
            "heldout_visibility_rmse_std": _float_std(metrics["heldout_visibility_rmse"]),
            "heldout_closure_phase_mae_mean": _float_mean(metrics["heldout_closure_phase_mae"]) if closure_reported else float("nan"),
            "heldout_closure_phase_mae_std": _float_std(metrics["heldout_closure_phase_mae"]) if closure_reported else float("nan"),
            "sample_count": len(metrics["heldout_visibility_rmse"]),
            "closure_valid_samples": int(np.sum(~np.isnan(np.asarray(metrics["heldout_closure_phase_mae"], dtype=np.float64)))),
        }

    baseline_stage = summary["stages"]["baseline_prediction"]
    pre_stage = summary["stages"]["pre_dc_prediction"]
    post_stage = summary["stages"]["ccrr_post_dc"]
    summary["deltas"] = {
        "pre_dc_vs_baseline_heldout_visrmse_gain": float(
            baseline_stage["heldout_visibility_rmse_mean"] - pre_stage["heldout_visibility_rmse_mean"]
        ),
        "post_dc_vs_pre_dc_heldout_visrmse_gain": float(
            pre_stage["heldout_visibility_rmse_mean"] - post_stage["heldout_visibility_rmse_mean"]
        ),
        "post_dc_vs_baseline_heldout_visrmse_gain": float(
            baseline_stage["heldout_visibility_rmse_mean"] - post_stage["heldout_visibility_rmse_mean"]
        ),
        "pre_dc_vs_baseline_heldout_closure_gain": float(
            baseline_stage["heldout_closure_phase_mae_mean"] - pre_stage["heldout_closure_phase_mae_mean"]
        )
        if closure_reported
        else float("nan"),
        "post_dc_vs_pre_dc_heldout_closure_gain": float(
            pre_stage["heldout_closure_phase_mae_mean"] - post_stage["heldout_closure_phase_mae_mean"]
        )
        if closure_reported
        else float("nan"),
        "post_dc_vs_baseline_heldout_closure_gain": float(
            baseline_stage["heldout_closure_phase_mae_mean"] - post_stage["heldout_closure_phase_mae_mean"]
        )
        if closure_reported
        else float("nan"),
    }
    return summary


def build_attribution_interpretation(
    observed_summary: dict[str, Any],
    heldout_summary: dict[str, Any],
    no_dc_metrics: dict[str, float],
    no_closure_metrics: dict[str, float],
) -> dict[str, Any]:
    """Classify the attribution pattern for the CCRR observation-domain gains."""

    observed_deltas = observed_summary["deltas"]
    heldout_deltas = heldout_summary["deltas"]

    closure_secondary = (
        abs(no_closure_metrics["observed_visibility_rmse"] - observed_summary["stages"]["ccrr_post_dc"]["observed_visibility_rmse_mean"]) <= 5e-3
        and abs(no_closure_metrics["closure_phase_mae"] - observed_summary["stages"]["ccrr_post_dc"]["closure_phase_mae_mean"]) <= 5e-2
    )

    meaningful_vis_gain = 5e-4
    meaningful_closure_gain = 2e-2
    heldout_vis_supports_learning = heldout_deltas["post_dc_vs_baseline_heldout_visrmse_gain"] > meaningful_vis_gain
    heldout_closure_supports_learning = (
        not math.isnan(heldout_deltas["post_dc_vs_baseline_heldout_closure_gain"])
        and heldout_deltas["post_dc_vs_baseline_heldout_closure_gain"] > meaningful_closure_gain
    )

    if (
        observed_deltas["pre_dc_vs_baseline_visrmse_gain"] > meaningful_vis_gain
        and heldout_deltas["pre_dc_vs_baseline_heldout_visrmse_gain"] > meaningful_vis_gain
        and (
            math.isnan(heldout_deltas["pre_dc_vs_baseline_heldout_closure_gain"])
            or heldout_deltas["pre_dc_vs_baseline_heldout_closure_gain"] > meaningful_closure_gain
        )
    ):
        label = "independent learned contribution"
    elif heldout_vis_supports_learning or heldout_closure_supports_learning:
        label = "mixed-attribution"
    else:
        label = "DC-dominant"

    rationale = [
        (
            "Observed visibility gains are dominated by the in-model DC projection because "
            f"the mean VisRMSE gain from baseline to pre-DC is {format_value(observed_deltas['pre_dc_vs_baseline_visrmse_gain'])}, "
            f"while the additional pre-DC to post-DC gain is {format_value(observed_deltas['post_dc_vs_pre_dc_visrmse_gain'])}."
        ),
        (
            "The held-out audit does not show a strong independent gain on coefficients removed from both the visibility input and the DC mask: "
            f"baseline-to-final held-out VisRMSE changes by {format_value(heldout_deltas['post_dc_vs_baseline_heldout_visrmse_gain'])} "
            "and the held-out closure change is "
            f"{format_value(heldout_deltas['post_dc_vs_baseline_heldout_closure_gain']) if not math.isnan(heldout_deltas['post_dc_vs_baseline_heldout_closure_gain']) else 'n/a'}."
        ),
    ]
    if heldout_summary["support"]["closure_reported"]:
        rationale.append(
            "Held-out closure is reported only on all-heldout triangles; mixed triangles are excluded. "
            f"The total all-heldout support is {heldout_summary['support']['total_all_heldout_triangles']} triangles, "
            f"with {heldout_summary['support']['total_mixed_triangles']} mixed triangles documented separately."
        )
    else:
        rationale.append(
            "Held-out closure is treated as insufficiently supported for formal interpretation in this audit, "
            "so closure attribution falls back to the full-mask pre-/post-DC decomposition and the no-closure ablation."
        )
    if closure_secondary:
        rationale.append(
            "The no-closure ablation changes default32 seed-7 VisRMSE and closure metrics only marginally, "
            "so closure-aware supervision should be described as a secondary ingredient rather than the dominant driver."
        )
    else:
        rationale.append(
            "The no-closure ablation shows a measurable change in observation-domain metrics, so closure-aware supervision remains a material contributor."
        )

    rubric = {
        "DC-dominant": "Most observation-domain gains appear only after the DC projection, with little or no held-out improvement.",
        "mixed-attribution": "The DC layer drives the largest observed-coefficient gains, but held-out coefficients or pre-DC metrics still improve measurably.",
        "independent learned contribution": "Substantial observation-domain gains are already present before the DC projection and remain visible on held-out coefficients.",
    }

    return {
        "label": label,
        "closure_secondary": closure_secondary,
        "rationale": rationale,
        "rubric": rubric,
    }


def build_claim_to_evidence_rows(
    observed_summary: dict[str, Any],
    heldout_summary: dict[str, Any],
    no_dc_metrics: dict[str, float],
    no_closure_metrics: dict[str, float],
) -> list[dict[str, str]]:
    """Map manuscript claims to exact audit outputs before paper regeneration."""

    rows = [
        {
            "claim": "A meaningful share of the observation-domain gain is DC-driven.",
            "evidence": (
                "Observed pre-/post-DC audit: baseline->pre-DC VisRMSE gain "
                f"{format_value(observed_summary['deltas']['pre_dc_vs_baseline_visrmse_gain'])}, "
                "pre-DC->post-DC VisRMSE gain "
                f"{format_value(observed_summary['deltas']['post_dc_vs_pre_dc_visrmse_gain'])}; "
                "no-DC ablation VisRMSE "
                f"{format_value(no_dc_metrics['observed_visibility_rmse'])} vs CCRR "
                f"{format_value(observed_summary['stages']['ccrr_post_dc']['observed_visibility_rmse_mean'])}."
            ),
            "location": "Abstract / Results / Discussion",
        },
        {
            "claim": "The held-out audit does not show a strong independent measurement gain beyond direct coefficient enforcement.",
            "evidence": (
                "Held-out audit on withheld coefficients: baseline->final held-out VisRMSE change "
                f"{format_value(heldout_summary['deltas']['post_dc_vs_baseline_heldout_visrmse_gain'])}, "
                "held-out closure change "
                f"{format_value(heldout_summary['deltas']['post_dc_vs_baseline_heldout_closure_gain']) if not math.isnan(heldout_summary['deltas']['post_dc_vs_baseline_heldout_closure_gain']) else 'n/a'}; "
                "support "
                f"{heldout_summary['support']['mean_baseline_coefficients']:.2f} held-out baseline coefficients "
                "and "
                f"{heldout_summary['support']['mean_all_heldout_triangles']:.2f} all-heldout triangles per sample."
            ),
            "location": "Abstract / Results / Discussion",
        },
        {
            "claim": "Closure-aware supervision is secondary rather than the dominant driver in the current default32 regime.",
            "evidence": (
                "No-closure ablation: VisRMSE "
                f"{format_value(no_closure_metrics['observed_visibility_rmse'])} vs CCRR "
                f"{format_value(observed_summary['stages']['ccrr_post_dc']['observed_visibility_rmse_mean'])}; "
                "ClosurePhase "
                f"{format_value(no_closure_metrics['closure_phase_mae'])} vs CCRR "
                f"{format_value(observed_summary['stages']['ccrr_post_dc']['closure_phase_mae_mean'])}."
            ),
            "location": "Method / Results / Limitations",
        },
        {
            "claim": "The remaining open problem is preserving morphology while retaining measurement consistency.",
            "evidence": (
                "Main protocol tables already show lower SSIM and worse temporal consistency for CCRR, while the audit shows that "
                "measurement-facing gains persist even when structural smoothness does not."
            ),
            "location": "Discussion / Limitations / Conclusion",
        },
    ]
    return rows


def _audit_rows_for_csv(seed_summaries: list[dict[str, Any]], overall_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_summary in seed_summaries:
        for audit_view_key, audit_view in (
            ("observed_pre_post", seed_summary["observed_pre_post"]),
            ("heldout", seed_summary["heldout"]),
        ):
            for stage_key, metrics in audit_view["stages"].items():
                row: dict[str, Any] = {
                    "seed_key": seed_summary["seed_key"],
                    "seed": seed_summary["seed"],
                    "audit_view": audit_view_key,
                    "stage": stage_key,
                }
                row.update(metrics)
                if audit_view_key == "heldout":
                    row.update(audit_view["support"])
                rows.append(row)

    for audit_view_key, audit_view in (
        ("observed_pre_post", overall_summary["observed_pre_post"]),
        ("heldout", overall_summary["heldout"]),
    ):
        for stage_key, metrics in audit_view["stages"].items():
            row = {
                "seed_key": "overall_mean",
                "seed": -1,
                "audit_view": audit_view_key,
                "stage": stage_key,
            }
            row.update(metrics)
            if audit_view_key == "heldout":
                row.update(audit_view["support"])
            rows.append(row)
    return rows


def _aggregate_seed_metric(seed_summaries: list[dict[str, Any]], audit_view_key: str, stage_key: str, metric_key: str) -> tuple[float, float]:
    values = [
        float(seed_summary[audit_view_key]["stages"][stage_key][metric_key]) for seed_summary in seed_summaries
    ]
    return _float_mean(values), _float_std(values)


def _aggregate_support_metric(seed_summaries: list[dict[str, Any]], metric_key: str) -> tuple[float, float]:
    values = [float(seed_summary["heldout"]["support"][metric_key]) for seed_summary in seed_summaries]
    return _float_mean(values), _float_std(values)


def aggregate_audit_summaries(
    seed_summaries: list[dict[str, Any]],
    interpretation: dict[str, Any],
    claim_rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Aggregate per-seed audit summaries into one paper-facing overall summary."""

    observed_stage_keys = ("baseline_prediction", "pre_dc_prediction", "ccrr_post_dc")
    heldout_stage_keys = ("baseline_prediction", "pre_dc_prediction", "ccrr_post_dc")
    overall = {
        "observed_pre_post": {"stages": {}, "deltas": {}},
        "heldout": {"stages": {}, "support": {}},
        "interpretation": interpretation,
        "claim_to_evidence": claim_rows,
    }

    for stage_key in observed_stage_keys:
        vis_mean, vis_std = _aggregate_seed_metric(
            seed_summaries, "observed_pre_post", stage_key, "observed_visibility_rmse_mean"
        )
        clo_mean, clo_std = _aggregate_seed_metric(
            seed_summaries, "observed_pre_post", stage_key, "closure_phase_mae_mean"
        )
        overall["observed_pre_post"]["stages"][stage_key] = {
            "observed_visibility_rmse_mean": vis_mean,
            "observed_visibility_rmse_std": vis_std,
            "closure_phase_mae_mean": clo_mean,
            "closure_phase_mae_std": clo_std,
        }

    observed_baseline = overall["observed_pre_post"]["stages"]["baseline_prediction"]
    observed_pre = overall["observed_pre_post"]["stages"]["pre_dc_prediction"]
    observed_post = overall["observed_pre_post"]["stages"]["ccrr_post_dc"]
    overall["observed_pre_post"]["deltas"] = {
        "pre_dc_vs_baseline_visrmse_gain": float(
            observed_baseline["observed_visibility_rmse_mean"] - observed_pre["observed_visibility_rmse_mean"]
        ),
        "post_dc_vs_pre_dc_visrmse_gain": float(
            observed_pre["observed_visibility_rmse_mean"] - observed_post["observed_visibility_rmse_mean"]
        ),
        "post_dc_vs_baseline_visrmse_gain": float(
            observed_baseline["observed_visibility_rmse_mean"] - observed_post["observed_visibility_rmse_mean"]
        ),
        "pre_dc_vs_baseline_closure_gain": float(
            observed_baseline["closure_phase_mae_mean"] - observed_pre["closure_phase_mae_mean"]
        ),
        "post_dc_vs_pre_dc_closure_gain": float(
            observed_pre["closure_phase_mae_mean"] - observed_post["closure_phase_mae_mean"]
        ),
        "post_dc_vs_baseline_closure_gain": float(
            observed_baseline["closure_phase_mae_mean"] - observed_post["closure_phase_mae_mean"]
        ),
    }

    closure_reported = all(bool(seed_summary["heldout"]["support"]["closure_reported"]) for seed_summary in seed_summaries)
    for stage_key in heldout_stage_keys:
        vis_mean, vis_std = _aggregate_seed_metric(
            seed_summaries, "heldout", stage_key, "heldout_visibility_rmse_mean"
        )
        clo_mean, clo_std = _aggregate_seed_metric(
            seed_summaries, "heldout", stage_key, "heldout_closure_phase_mae_mean"
        )
        overall["heldout"]["stages"][stage_key] = {
            "heldout_visibility_rmse_mean": vis_mean,
            "heldout_visibility_rmse_std": vis_std,
            "heldout_closure_phase_mae_mean": clo_mean if closure_reported else float("nan"),
            "heldout_closure_phase_mae_std": clo_std if closure_reported else float("nan"),
        }

    heldout_baseline = overall["heldout"]["stages"]["baseline_prediction"]
    heldout_pre = overall["heldout"]["stages"]["pre_dc_prediction"]
    heldout_post = overall["heldout"]["stages"]["ccrr_post_dc"]
    overall["heldout"]["deltas"] = {
        "pre_dc_vs_baseline_heldout_visrmse_gain": float(
            heldout_baseline["heldout_visibility_rmse_mean"] - heldout_pre["heldout_visibility_rmse_mean"]
        ),
        "post_dc_vs_pre_dc_heldout_visrmse_gain": float(
            heldout_pre["heldout_visibility_rmse_mean"] - heldout_post["heldout_visibility_rmse_mean"]
        ),
        "post_dc_vs_baseline_heldout_visrmse_gain": float(
            heldout_baseline["heldout_visibility_rmse_mean"] - heldout_post["heldout_visibility_rmse_mean"]
        ),
        "pre_dc_vs_baseline_heldout_closure_gain": float(
            heldout_baseline["heldout_closure_phase_mae_mean"] - heldout_pre["heldout_closure_phase_mae_mean"]
        )
        if closure_reported
        else float("nan"),
        "post_dc_vs_pre_dc_heldout_closure_gain": float(
            heldout_pre["heldout_closure_phase_mae_mean"] - heldout_post["heldout_closure_phase_mae_mean"]
        )
        if closure_reported
        else float("nan"),
        "post_dc_vs_baseline_heldout_closure_gain": float(
            heldout_baseline["heldout_closure_phase_mae_mean"] - heldout_post["heldout_closure_phase_mae_mean"]
        )
        if closure_reported
        else float("nan"),
    }
    overall["heldout"]["support"] = {
        "closure_reported": closure_reported,
        "mean_grid_coefficients": _aggregate_support_metric(seed_summaries, "mean_grid_coefficients")[0],
        "std_grid_coefficients": _aggregate_support_metric(seed_summaries, "mean_grid_coefficients")[1],
        "mean_baseline_coefficients": _aggregate_support_metric(seed_summaries, "mean_baseline_coefficients")[0],
        "std_baseline_coefficients": _aggregate_support_metric(seed_summaries, "mean_baseline_coefficients")[1],
        "mean_all_heldout_triangles": _aggregate_support_metric(seed_summaries, "mean_all_heldout_triangles")[0],
        "std_all_heldout_triangles": _aggregate_support_metric(seed_summaries, "mean_all_heldout_triangles")[1],
        "mean_mixed_triangles": _aggregate_support_metric(seed_summaries, "mean_mixed_triangles")[0],
        "std_mixed_triangles": _aggregate_support_metric(seed_summaries, "mean_mixed_triangles")[1],
        "total_all_heldout_triangles": int(
            np.sum([seed_summary["heldout"]["support"]["total_all_heldout_triangles"] for seed_summary in seed_summaries])
        ),
        "total_mixed_triangles": int(
            np.sum([seed_summary["heldout"]["support"]["total_mixed_triangles"] for seed_summary in seed_summaries])
        ),
        "valid_closure_samples": int(
            np.sum([seed_summary["heldout"]["support"]["valid_closure_samples"] for seed_summary in seed_summaries])
        ),
        "insufficient_closure_samples": int(
            np.sum([seed_summary["heldout"]["support"]["insufficient_closure_samples"] for seed_summary in seed_summaries])
        ),
        "holdout_fraction": float(seed_summaries[0]["heldout"]["support"]["holdout_fraction"]),
        "min_total_heldout_triangles": int(seed_summaries[0]["heldout"]["support"]["min_total_heldout_triangles"]),
        "min_valid_closure_samples": int(seed_summaries[0]["heldout"]["support"]["min_valid_closure_samples"]),
    }
    return overall


def save_measurement_audit_figure(
    audit_summary: dict[str, Any],
    output_png: str | Path,
    output_svg: str | Path | None = None,
) -> None:
    """Draw a compact two-panel figure clarifying the DC-versus-learned attribution story."""

    stage_labels = ["Baseline", "Pre-DC", "Post-DC"]
    observed_vis = [
        audit_summary["observed_pre_post"]["stages"]["baseline_prediction"]["observed_visibility_rmse_mean"],
        audit_summary["observed_pre_post"]["stages"]["pre_dc_prediction"]["observed_visibility_rmse_mean"],
        audit_summary["observed_pre_post"]["stages"]["ccrr_post_dc"]["observed_visibility_rmse_mean"],
    ]
    observed_vis_std = [
        audit_summary["observed_pre_post"]["stages"]["baseline_prediction"]["observed_visibility_rmse_std"],
        audit_summary["observed_pre_post"]["stages"]["pre_dc_prediction"]["observed_visibility_rmse_std"],
        audit_summary["observed_pre_post"]["stages"]["ccrr_post_dc"]["observed_visibility_rmse_std"],
    ]
    heldout_vis = [
        audit_summary["heldout"]["stages"]["baseline_prediction"]["heldout_visibility_rmse_mean"],
        audit_summary["heldout"]["stages"]["pre_dc_prediction"]["heldout_visibility_rmse_mean"],
        audit_summary["heldout"]["stages"]["ccrr_post_dc"]["heldout_visibility_rmse_mean"],
    ]
    heldout_vis_std = [
        audit_summary["heldout"]["stages"]["baseline_prediction"]["heldout_visibility_rmse_std"],
        audit_summary["heldout"]["stages"]["pre_dc_prediction"]["heldout_visibility_rmse_std"],
        audit_summary["heldout"]["stages"]["ccrr_post_dc"]["heldout_visibility_rmse_std"],
    ]
    observed_closure = [
        audit_summary["observed_pre_post"]["stages"]["baseline_prediction"]["closure_phase_mae_mean"],
        audit_summary["observed_pre_post"]["stages"]["pre_dc_prediction"]["closure_phase_mae_mean"],
        audit_summary["observed_pre_post"]["stages"]["ccrr_post_dc"]["closure_phase_mae_mean"],
    ]
    observed_closure_std = [
        audit_summary["observed_pre_post"]["stages"]["baseline_prediction"]["closure_phase_mae_std"],
        audit_summary["observed_pre_post"]["stages"]["pre_dc_prediction"]["closure_phase_mae_std"],
        audit_summary["observed_pre_post"]["stages"]["ccrr_post_dc"]["closure_phase_mae_std"],
    ]
    heldout_closure = [
        audit_summary["heldout"]["stages"]["baseline_prediction"]["heldout_closure_phase_mae_mean"],
        audit_summary["heldout"]["stages"]["pre_dc_prediction"]["heldout_closure_phase_mae_mean"],
        audit_summary["heldout"]["stages"]["ccrr_post_dc"]["heldout_closure_phase_mae_mean"],
    ]
    heldout_closure_std = [
        audit_summary["heldout"]["stages"]["baseline_prediction"]["heldout_closure_phase_mae_std"],
        audit_summary["heldout"]["stages"]["pre_dc_prediction"]["heldout_closure_phase_mae_std"],
        audit_summary["heldout"]["stages"]["ccrr_post_dc"]["heldout_closure_phase_mae_std"],
    ]

    x_positions = np.arange(len(stage_labels))
    width = 0.34
    observed_color = "#0f766e"
    heldout_color = "#b45309"

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    fig.patch.set_facecolor("white")

    axes[0].bar(
        x_positions - width / 2,
        observed_vis,
        width=width,
        yerr=observed_vis_std,
        capsize=4,
        color=observed_color,
        label="Observed coefficients",
    )
    axes[0].bar(
        x_positions + width / 2,
        heldout_vis,
        width=width,
        yerr=heldout_vis_std,
        capsize=4,
        color=heldout_color,
        label="Held-out coefficients",
    )
    axes[0].set_xticks(x_positions, stage_labels)
    axes[0].set_ylabel("Visibility RMSE")
    axes[0].set_title("Measurement consistency")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].bar(
        x_positions - width / 2,
        observed_closure,
        width=width,
        yerr=observed_closure_std,
        capsize=4,
        color=observed_color,
        label="Observed triangles",
    )
    axes[1].bar(
        x_positions + width / 2,
        heldout_closure,
        width=width,
        yerr=heldout_closure_std,
        capsize=4,
        color=heldout_color,
        label="Held-out triangles",
    )
    axes[1].set_xticks(x_positions, stage_labels)
    axes[1].set_ylabel("Closure-phase MAE")
    axes[1].set_title("Closure agreement")
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)

    holdout_support = audit_summary["heldout"]["support"]
    support_text = (
        f"Held-out support: {holdout_support['mean_baseline_coefficients']:.1f} baselines/sample, "
        f"{holdout_support['mean_all_heldout_triangles']:.1f} all-heldout triangles/sample.\n"
        "Mixed triangles are excluded from the held-out closure metric."
    )
    fig.text(0.5, -0.02, support_text, ha="center", va="top", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    if output_svg is not None:
        output_svg = Path(output_svg)
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)


def write_measurement_audit_markdown(
    output_path: str | Path,
    seed_summaries: list[dict[str, Any]],
    overall_summary: dict[str, Any],
    failures: list[dict[str, str]],
) -> None:
    """Write the paper-facing measurement-audit summary."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# CCRR Measurement Audit", ""]
    lines.append("## Overall Interpretation")
    lines.append("")
    lines.append(f"- Label: `{overall_summary['interpretation']['label']}`")
    lines.append(f"- Closure-aware supervision secondary: `{str(overall_summary['interpretation']['closure_secondary']).lower()}`")
    for item in overall_summary["interpretation"]["rationale"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Interpretation Rubric")
    lines.append("")
    for label, description in overall_summary["interpretation"]["rubric"].items():
        lines.append(f"- `{label}`: {description}")
    lines.append("")

    observed = overall_summary["observed_pre_post"]["stages"]
    heldout = overall_summary["heldout"]["stages"]
    lines.append("# CCRR Measurement Audit (overall mean across seeds)")
    lines.append("")
    lines.append("| Stage | Observed VisRMSE | Observed ClosurePhase | Held-out VisRMSE | Held-out ClosurePhase |")
    lines.append("|---|---|---|---|---|")
    for stage_key, stage_label in (
        ("baseline_prediction", "Baseline prediction"),
        ("pre_dc_prediction", "Pre-DC prediction"),
        ("ccrr_post_dc", "Post-DC CCRR"),
    ):
        heldout_closure = (
            f"{format_value(heldout[stage_key]['heldout_closure_phase_mae_mean'])} ± {format_value(heldout[stage_key]['heldout_closure_phase_mae_std'])}"
            if overall_summary["heldout"]["support"]["closure_reported"]
            else "n/a"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    stage_label,
                    f"{format_value(observed[stage_key]['observed_visibility_rmse_mean'])} ± {format_value(observed[stage_key]['observed_visibility_rmse_std'])}",
                    f"{format_value(observed[stage_key]['closure_phase_mae_mean'])} ± {format_value(observed[stage_key]['closure_phase_mae_std'])}",
                    f"{format_value(heldout[stage_key]['heldout_visibility_rmse_mean'])} ± {format_value(heldout[stage_key]['heldout_visibility_rmse_std'])}",
                    heldout_closure,
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Observed metrics use the archived full-mask pre-DC and post-DC prediction bundles.")
    lines.append("- Held-out metrics reserve 20% of observed coefficients per frame, remove them from the visibility input and DC mask, and evaluate only on that held-out subset.")
    lines.append("- Held-out closure uses only triangles composed entirely of held-out coefficients; mixed triangles are excluded and reported separately.")
    lines.append("")
    lines.append("## Support Counts")
    lines.append("")
    support = overall_summary["heldout"]["support"]
    lines.extend(
        [
            f"- Mean held-out grid coefficients per sample: {format_value(support['mean_grid_coefficients'])} ± {format_value(support['std_grid_coefficients'])}",
            f"- Mean held-out baseline coefficients per sample: {format_value(support['mean_baseline_coefficients'])} ± {format_value(support['std_baseline_coefficients'])}",
            f"- Mean all-heldout triangles per sample: {format_value(support['mean_all_heldout_triangles'])} ± {format_value(support['std_all_heldout_triangles'])}",
            f"- Mean mixed triangles per sample: {format_value(support['mean_mixed_triangles'])} ± {format_value(support['std_mixed_triangles'])}",
            f"- Total all-heldout triangles: {support['total_all_heldout_triangles']}",
            f"- Total mixed triangles: {support['total_mixed_triangles']}",
            f"- Held-out closure valid samples: {support['valid_closure_samples']}",
            f"- Held-out closure insufficient-support samples: {support['insufficient_closure_samples']}",
        ]
    )
    lines.append("")
    lines.append("## Claim-to-Evidence Check")
    lines.append("")
    lines.append("| Claim | Supporting evidence | Manuscript locations |")
    lines.append("|---|---|---|")
    for row in overall_summary["claim_to_evidence"]:
        lines.append(f"| {row['claim']} | {row['evidence']} | {row['location']} |")
    lines.append("")
    lines.append("## Seed-Level Notes")
    lines.append("")
    if failures:
        for failure in failures:
            lines.append(f"- `{failure['seed_key']}` failed: {failure['message']}")
    else:
        lines.append("- All default32 seed-repeat archives contained valid `pre_dc_prediction` bundles and consistent shapes.")
    lines.append("")
    lines.append("## Seed-Level Verdicts")
    lines.append("")
    for seed_summary in seed_summaries:
        lines.append(
            "- "
            f"{seed_summary['seed_key']}: observed post-DC VisRMSE gain "
            f"{format_value(seed_summary['observed_pre_post']['deltas']['post_dc_vs_baseline_visrmse_gain'])}, "
            "held-out post-DC VisRMSE gain "
            f"{format_value(seed_summary['heldout']['deltas']['post_dc_vs_baseline_heldout_visrmse_gain'])}, "
            "held-out closure support "
            f"{seed_summary['heldout']['support']['total_all_heldout_triangles']} triangles."
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_measurement_audit_artifacts(
    seed_specs: Sequence[MeasurementAuditSpec],
    no_dc_spec: MeasurementAuditSpec,
    no_closure_spec: MeasurementAuditSpec,
    artifact_root: str | Path,
    paper_root: str | Path,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    max_triangles: int = DEFAULT_MAX_TRIANGLES,
    min_total_heldout_triangles: int = DEFAULT_MIN_TOTAL_HELDOUT_TRIANGLES,
    min_valid_closure_samples: int = DEFAULT_MIN_VALID_CLOSURE_SAMPLES,
) -> dict[str, Any]:
    """Run the CCRR reviewer-risk measurement audit and export paper-facing artifacts."""

    artifact_root = Path(artifact_root)
    paper_root = Path(paper_root)
    tables_dir = artifact_root / "tables"
    summaries_dir = artifact_root / "summaries"
    figures_dir = paper_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    seed_summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for spec in seed_specs:
        try:
            observed_summary = compute_pre_post_dc_audit(
                prediction_path=spec.prediction_path,
                dataset_dir=spec.dataset_dir,
            )
            heldout_summary = run_heldout_measurement_audit(
                spec=spec,
                holdout_fraction=holdout_fraction,
                max_triangles=max_triangles,
                min_total_heldout_triangles=min_total_heldout_triangles,
                min_valid_closure_samples=min_valid_closure_samples,
            )
        except MeasurementAuditError as error:
            failures.append({"seed_key": spec.key, "message": str(error)})
            continue
        seed_summaries.append(
            {
                "seed_key": spec.key,
                "title": spec.title,
                "seed": spec.seed,
                "prediction_path": str(spec.prediction_path),
                "checkpoint_path": str(spec.checkpoint_path),
                "dataset_dir": str(spec.dataset_dir),
                "observed_pre_post": observed_summary,
                "heldout": heldout_summary,
            }
        )

    if len(seed_summaries) != len(seed_specs):
        placeholder = {
            "seed_summaries": seed_summaries,
            "failures": failures,
            "status": "failed",
        }
        save_json(summaries_dir / "ccrr_measurement_audit.json", placeholder)
        write_measurement_audit_markdown(
            output_path=tables_dir / "ccrr_measurement_audit.md",
            seed_summaries=seed_summaries,
            overall_summary={
                "interpretation": {
                    "label": "audit-failed",
                    "closure_secondary": False,
                    "rationale": ["One or more seed archives are missing required CCRR audit fields."],
                    "rubric": {},
                },
                "observed_pre_post": {"stages": {}},
                "heldout": {"stages": {}, "support": {"closure_reported": False}},
                "claim_to_evidence": [],
            },
            failures=failures,
        )
        raise MeasurementAuditError(
            "Measurement audit failed for one or more seeds. "
            "See outputs/ccrr_paper_artifacts/tables/ccrr_measurement_audit.md for details."
        )

    no_dc_summary = summarize_prediction_bundle(no_dc_spec.prediction_path, no_dc_spec.dataset_dir)
    no_closure_summary = summarize_prediction_bundle(no_closure_spec.prediction_path, no_closure_spec.dataset_dir)
    no_dc_metrics = no_dc_summary["aggregate"]["ccrr"]
    no_closure_metrics = no_closure_summary["aggregate"]["ccrr"]

    observed_mean_summary = aggregate_audit_summaries(
        seed_summaries=seed_summaries,
        interpretation={},
        claim_rows=[],
    )
    interpretation = build_attribution_interpretation(
        observed_summary=observed_mean_summary["observed_pre_post"],
        heldout_summary=observed_mean_summary["heldout"],
        no_dc_metrics=no_dc_metrics,
        no_closure_metrics=no_closure_metrics,
    )
    claim_rows = build_claim_to_evidence_rows(
        observed_summary=observed_mean_summary["observed_pre_post"],
        heldout_summary=observed_mean_summary["heldout"],
        no_dc_metrics=no_dc_metrics,
        no_closure_metrics=no_closure_metrics,
    )
    overall_summary = aggregate_audit_summaries(
        seed_summaries=seed_summaries,
        interpretation=interpretation,
        claim_rows=claim_rows,
    )
    overall_summary["ablation_context"] = {
        "no_dc": {
            "observed_visibility_rmse": float(no_dc_metrics["observed_visibility_rmse"]),
            "closure_phase_mae": float(no_dc_metrics["closure_phase_mae"]),
            "ssim": float(no_dc_metrics["ssim"]),
        },
        "no_closure": {
            "observed_visibility_rmse": float(no_closure_metrics["observed_visibility_rmse"]),
            "closure_phase_mae": float(no_closure_metrics["closure_phase_mae"]),
            "ssim": float(no_closure_metrics["ssim"]),
        },
    }

    csv_rows = _audit_rows_for_csv(seed_summaries=seed_summaries, overall_summary=overall_summary)
    write_csv(tables_dir / "ccrr_measurement_audit.csv", csv_rows)
    save_json(
        summaries_dir / "ccrr_measurement_audit.json",
        {
            "seed_summaries": seed_summaries,
            "overall": overall_summary,
            "failures": failures,
            "status": "ok",
        },
    )
    write_measurement_audit_markdown(
        output_path=tables_dir / "ccrr_measurement_audit.md",
        seed_summaries=seed_summaries,
        overall_summary=overall_summary,
        failures=failures,
    )
    save_measurement_audit_figure(
        audit_summary=overall_summary,
        output_png=figures_dir / "fig06_ccrr_measurement_audit.png",
        output_svg=figures_dir / "fig06_ccrr_measurement_audit.svg",
    )
    save_json(summaries_dir / "ccrr_claim_to_evidence.json", {"rows": claim_rows})

    claim_md_path = summaries_dir / "ccrr_claim_to_evidence.md"
    claim_lines = ["# CCRR Claim-to-Evidence Check", "", "| Claim | Evidence | Location |", "|---|---|---|"]
    for row in claim_rows:
        claim_lines.append(f"| {row['claim']} | {row['evidence']} | {row['location']} |")
    claim_md_path.write_text("\n".join(claim_lines) + "\n", encoding="utf-8")

    return {
        "seed_summaries": seed_summaries,
        "overall": overall_summary,
        "csv_path": str(tables_dir / "ccrr_measurement_audit.csv"),
        "markdown_path": str(tables_dir / "ccrr_measurement_audit.md"),
        "json_path": str(summaries_dir / "ccrr_measurement_audit.json"),
        "figure_png": str(figures_dir / "fig06_ccrr_measurement_audit.png"),
        "figure_svg": str(figures_dir / "fig06_ccrr_measurement_audit.svg"),
        "claim_markdown_path": str(claim_md_path),
    }
