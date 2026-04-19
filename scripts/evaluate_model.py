#!/usr/bin/env python3
"""Evaluate baseline or phase 2 reconstruction models on the synthetic test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_BASE_CONFIG = "configs/base.yaml"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.data.dataset import DynamicVLBIDataset
from dynadiff_vlbi.data.feature_formatting import format_visibility_tensor
from dynadiff_vlbi.data.visibility_dataset import VisibilityConditionedDataset
from dynadiff_vlbi.evaluation.comparison import save_comparison_csv
from dynadiff_vlbi.evaluation.metrics import (
    compute_reconstruction_metrics,
    empirical_coverage,
    risk_coverage_auc,
    topk_error_recall,
    uncertainty_error_correlation,
)
from dynadiff_vlbi.evaluation.visualization import save_phase2_comparison_panel, save_reconstruction_panel
from dynadiff_vlbi.models.factory import build_model
from dynadiff_vlbi.utils.config import DEFAULT_BASE_CONFIG_PATH, ExperimentConfig, ModelConfig, load_experiment_config
from dynadiff_vlbi.utils.device import get_device
from dynadiff_vlbi.utils.logging_utils import prepare_output_dirs, save_json
from dynadiff_vlbi.utils.seed import set_seed
from dynadiff_vlbi.physics.classical_reconstruction import (
    tikhonov_iterative_reconstruction,
    visibility_data_consistency_projection,
)


def _explicit_arg(name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in sys.argv[1:])


def _resolve_preset(raw_preset: str | None, base_config_explicit: bool) -> str | None:
    if raw_preset is not None:
        return raw_preset
    if base_config_explicit:
        return None
    return "smoke"


def _experiment_label(base_config_path: Path, preset: str | None) -> str:
    is_default_base = base_config_path.resolve() == (ROOT / DEFAULT_BASE_CONFIG_PATH).resolve()
    base_name = base_config_path.stem
    if preset is not None and is_default_base:
        return preset
    if preset is not None:
        return f"{preset}_{base_name}"
    return base_name


def aggregate_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = list(metric_rows[0].keys())
    aggregated: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in metric_rows], dtype=np.float64)
        aggregated[key] = float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")
    return aggregated


def _load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = ModelConfig(**checkpoint["config"]["model"])
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_config


def _resolve_reference_baseline_checkpoint(
    explicit_checkpoint: str | None,
    config: ExperimentConfig,
    output_root: Path,
) -> Path | None:
    if explicit_checkpoint is not None:
        checkpoint_path = Path(explicit_checkpoint)
        return checkpoint_path if checkpoint_path.exists() else None
    if config.evaluation.reference_baseline_run_name:
        checkpoint_path = output_root / config.evaluation.reference_baseline_run_name / "checkpoints" / "best.pt"
        return checkpoint_path if checkpoint_path.exists() else None
    return None


def _resolve_reference_visibility_checkpoint(
    explicit_checkpoint: str | None,
    config: ExperimentConfig,
    output_root: Path,
) -> Path | None:
    if explicit_checkpoint is not None:
        checkpoint_path = Path(explicit_checkpoint)
        return checkpoint_path if checkpoint_path.exists() else None
    if config.evaluation.reference_visibility_run_name:
        checkpoint_path = output_root / config.evaluation.reference_visibility_run_name / "checkpoints" / "best.pt"
        return checkpoint_path if checkpoint_path.exists() else None
    return None


def _resolve_reference_residual_checkpoint(
    explicit_checkpoint: str | None,
    config: ExperimentConfig,
    output_root: Path,
) -> Path | None:
    if explicit_checkpoint is not None:
        checkpoint_path = Path(explicit_checkpoint)
        return checkpoint_path if checkpoint_path.exists() else None
    if config.evaluation.reference_residual_run_name:
        checkpoint_path = output_root / config.evaluation.reference_residual_run_name / "checkpoints" / "best.pt"
        return checkpoint_path if checkpoint_path.exists() else None
    return None


def _phase2_dirty_input(
    sample: dict[str, torch.Tensor],
    model_config: ModelConfig,
    device: torch.device,
) -> torch.Tensor | None:
    dirty_input = sample["dirty_input"].unsqueeze(0).to(device)
    if not model_config.include_dirty_input and model_config.model_type not in {
        "residual_visibility_refinement",
        "ccrr",
    }:
        return None
    return dirty_input


def _predict_phase2_model(
    model: torch.nn.Module,
    model_config: ModelConfig,
    sample: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    visibility_input_array = format_visibility_tensor(
        vis_real=sample["vis_real"].cpu().numpy(),
        vis_imag=sample["vis_imag"].cpu().numpy(),
        mask=sample["mask"].cpu().numpy(),
        representation=model_config.visibility_representation,
        include_mask_channel=model_config.include_mask_channel,
        include_uv_coords=model_config.include_uv_coords,
        uv_coords=sample.get("uv_coords").cpu().numpy() if sample.get("uv_coords") is not None else None,
        include_observation_metadata=model_config.include_observation_metadata,
        frame_uv_coords=sample.get("frame_uv_coords").cpu().numpy()
        if sample.get("frame_uv_coords") is not None
        else None,
        frame_uv_indices=sample.get("frame_uv_indices").cpu().numpy()
        if sample.get("frame_uv_indices") is not None
        else None,
    )
    visibility_input = torch.from_numpy(visibility_input_array).unsqueeze(0).to(device)
    dirty_input = _phase2_dirty_input(sample=sample, model_config=model_config, device=device)
    measurements = (
        sample["vis_real"].unsqueeze(0).to(device).to(torch.float32)
        + 1j * sample["vis_imag"].unsqueeze(0).to(device).to(torch.float32)
    ).to(torch.complex64)
    mask = sample["mask"].unsqueeze(0).to(device).to(torch.float32)
    baseline_pairs = sample.get("baseline_pairs")
    if baseline_pairs is not None:
        baseline_pairs = baseline_pairs.unsqueeze(0).to(device)
    frame_uv_indices = sample.get("frame_uv_indices")
    if frame_uv_indices is not None:
        frame_uv_indices = frame_uv_indices.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(
            visibility_input=visibility_input,
            dirty_input=dirty_input,
            measurements=measurements,
            mask=mask,
            baseline_pairs=baseline_pairs,
            frame_uv_indices=frame_uv_indices,
        )
        mean = outputs.mean.squeeze(0).squeeze(0).detach().cpu().numpy()
        if outputs.log_variance is not None:
            std = torch.exp(0.5 * outputs.log_variance).squeeze(0).squeeze(0).detach().cpu().numpy()
        else:
            std = np.zeros_like(mean, dtype=np.float32)

    extras: dict[str, np.ndarray] = {}
    if hasattr(outputs, "baseline_prediction"):
        extras["baseline_prediction"] = outputs.baseline_prediction.squeeze(0).squeeze(0).detach().cpu().numpy()
    if hasattr(outputs, "residual_correction"):
        extras["residual_correction"] = outputs.residual_correction.squeeze(0).squeeze(0).detach().cpu().numpy()
    if hasattr(outputs, "pre_dc_prediction"):
        extras["pre_dc_prediction"] = outputs.pre_dc_prediction.squeeze(0).squeeze(0).detach().cpu().numpy()
    if hasattr(outputs, "dc_weight"):
        extras["dc_weight"] = np.asarray(outputs.dc_weight.detach().cpu().numpy(), dtype=np.float32)
    return mean, std, extras


def run_baseline_evaluation(
    config: ExperimentConfig,
    data_dir: Path,
    checkpoint_path: Path,
    output_root: Path,
    run_name: str,
) -> dict[str, dict[str, float]]:
    """Run the original baseline evaluation path."""

    output_dirs = prepare_output_dirs(str(output_root), run_name=run_name, config=config)
    dataset = DynamicVLBIDataset(data_dir / "test.npz")
    device = get_device()
    model, _ = _load_model_from_checkpoint(checkpoint_path=checkpoint_path, device=device)

    dirty_metrics_rows = []
    tikhonov_metrics_rows = []
    learned_metrics_rows = []
    coverage_scores = []
    uncertainty_correlations = []
    risk_coverage_scores = []
    topk_recall_scores = []

    prediction_payload: dict[str, list[np.ndarray]] = {
        "ground_truth": [],
        "dirty": [],
        "tikhonov": [],
        "learned_mean": [],
        "uncertainty": [],
    }

    for index in range(len(dataset)):
        sample = dataset[index]
        ground_truth = sample["target"].numpy()
        dirty = sample["dirty"].numpy()
        measurements = sample["vis_real"].numpy() + 1j * sample["vis_imag"].numpy()
        mask = sample["mask"].numpy()
        ring_radius_px = float(sample["ring_radius_px"].item())
        hotspot_coords_px = sample["hotspot_coords_px"].numpy()
        baseline_pairs = sample["baseline_pairs"].numpy() if "baseline_pairs" in sample else None
        frame_uv_indices = sample["frame_uv_indices"].numpy() if "frame_uv_indices" in sample else None

        tikhonov = tikhonov_iterative_reconstruction(
            measurements=measurements,
            mask=mask,
            lambda_reg=config.evaluation.tikhonov_lambda,
            iterations=config.evaluation.tikhonov_iterations,
            step_size=config.evaluation.tikhonov_step_size,
        )
        input_tensor = sample["input"].unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            predictive_mean, predictive_std, _ = model.predict_with_uncertainty(
                x=input_tensor,
                n_samples=config.evaluation.mc_samples,
            )
        learned_mean = predictive_mean.squeeze(0).squeeze(0).cpu().numpy()
        uncertainty = predictive_std.squeeze(0).squeeze(0).cpu().numpy()

        dirty_metrics_rows.append(
            compute_reconstruction_metrics(
                prediction=dirty,
                target=ground_truth,
                target_ring_radius_px=ring_radius_px,
                target_hotspot_coords_px=hotspot_coords_px,
                measurements=measurements,
                mask=mask,
                baseline_pairs=baseline_pairs,
                frame_uv_indices=frame_uv_indices,
            )
        )
        tikhonov_metrics_rows.append(
            compute_reconstruction_metrics(
                prediction=tikhonov,
                target=ground_truth,
                target_ring_radius_px=ring_radius_px,
                target_hotspot_coords_px=hotspot_coords_px,
                measurements=measurements,
                mask=mask,
                baseline_pairs=baseline_pairs,
                frame_uv_indices=frame_uv_indices,
            )
        )
        learned_metrics_rows.append(
            compute_reconstruction_metrics(
                prediction=learned_mean,
                target=ground_truth,
                target_ring_radius_px=ring_radius_px,
                target_hotspot_coords_px=hotspot_coords_px,
                measurements=measurements,
                mask=mask,
                baseline_pairs=baseline_pairs,
                frame_uv_indices=frame_uv_indices,
            )
        )

        coverage_scores.append(empirical_coverage(ground_truth, learned_mean, uncertainty))
        uncertainty_correlations.append(uncertainty_error_correlation(ground_truth, learned_mean, uncertainty))
        risk_coverage_scores.append(risk_coverage_auc(ground_truth, learned_mean, uncertainty))
        topk_recall_scores.append(topk_error_recall(ground_truth, learned_mean, uncertainty))

        if config.evaluation.save_prediction_arrays:
            prediction_payload["ground_truth"].append(ground_truth.astype(np.float32))
            prediction_payload["dirty"].append(dirty.astype(np.float32))
            prediction_payload["tikhonov"].append(tikhonov.astype(np.float32))
            prediction_payload["learned_mean"].append(learned_mean.astype(np.float32))
            prediction_payload["uncertainty"].append(uncertainty.astype(np.float32))

        if index < config.evaluation.num_visualizations:
            save_reconstruction_panel(
                path=output_dirs["figures"] / f"sample_{index:03d}.png",
                ground_truth=ground_truth,
                dirty=dirty,
                tikhonov=tikhonov,
                learned_mean=learned_mean,
                uncertainty=uncertainty,
                frames_to_plot=config.evaluation.frames_to_plot,
                dpi=config.evaluation.figure_dpi,
            )

    summary = {
        "dirty": aggregate_metrics(dirty_metrics_rows),
        "tikhonov": aggregate_metrics(tikhonov_metrics_rows),
        "learned": aggregate_metrics(learned_metrics_rows),
        "uncertainty": {
            "empirical_95_coverage": float(np.mean(coverage_scores)),
            "error_uncertainty_correlation": float(np.mean(uncertainty_correlations)),
            "risk_coverage_auc": float(np.mean(risk_coverage_scores)),
            "top10_error_recall": float(np.mean(topk_recall_scores)),
        },
    }
    save_json(output_dirs["logs"] / "evaluation_summary.json", summary)
    if config.evaluation.save_comparison_csv:
        save_comparison_csv(output_dirs["logs"] / "comparison_metrics.csv", summary)
    if config.evaluation.save_prediction_arrays and prediction_payload["ground_truth"]:
        np.savez_compressed(
            output_dirs["predictions"] / "test_predictions.npz",
            **{key: np.stack(value).astype(np.float32) for key, value in prediction_payload.items()},
        )
    return summary


def run_phase2_evaluation(
    config: ExperimentConfig,
    data_dir: Path,
    checkpoint_path: Path,
    output_root: Path,
    run_name: str,
    reference_baseline_checkpoint: Path | None,
    reference_visibility_checkpoint: Path | None,
    reference_residual_checkpoint: Path | None,
) -> dict[str, dict[str, float] | str | None]:
    """Run evaluation for the visibility-conditioned phase 2 model."""

    output_dirs = prepare_output_dirs(str(output_root), run_name=run_name, config=config)
    device = get_device()
    phase2_model, phase2_model_config = _load_model_from_checkpoint(checkpoint_path=checkpoint_path, device=device)
    dataset = VisibilityConditionedDataset(data_dir / "test.npz", model_config=phase2_model_config)
    current_model_key = (
        "emc"
        if phase2_model_config.model_type == "emc_ccrr"
        else (
            "ccrr"
            if phase2_model_config.model_type == "ccrr"
            else (
                "residual_refinement"
                if phase2_model_config.model_type == "residual_visibility_refinement"
                else "visibility_conditioned"
            )
        )
    )

    reference_baseline_model = None
    if reference_baseline_checkpoint is not None and reference_baseline_checkpoint.exists():
        reference_baseline_model, _ = _load_model_from_checkpoint(
            checkpoint_path=reference_baseline_checkpoint,
            device=device,
        )
    reference_visibility_model = None
    reference_visibility_model_config = None
    if (
        current_model_key in {"residual_refinement", "ccrr", "emc"}
        and reference_visibility_checkpoint is not None
        and reference_visibility_checkpoint.exists()
    ):
        reference_visibility_model, reference_visibility_model_config = _load_model_from_checkpoint(
            checkpoint_path=reference_visibility_checkpoint,
            device=device,
        )
    reference_residual_model = None
    reference_residual_model_config = None
    if (
        current_model_key in {"ccrr", "emc"}
        and reference_residual_checkpoint is not None
        and reference_residual_checkpoint.exists()
    ):
        reference_residual_model, reference_residual_model_config = _load_model_from_checkpoint(
            checkpoint_path=reference_residual_checkpoint,
            device=device,
        )

    dirty_metrics_rows = []
    tikhonov_metrics_rows = []
    baseline_metrics_rows = []
    baseline_data_consistent_metrics_rows = []
    reference_visibility_metrics_rows = []
    reference_residual_metrics_rows = []
    phase2_metrics_rows = []
    coverage_scores = []
    uncertainty_correlations = []
    risk_coverage_scores = []
    topk_recall_scores = []

    prediction_payload: dict[str, list[np.ndarray]] = {
        "ground_truth": [],
        "dirty": [],
        "tikhonov": [],
        current_model_key: [],
        "uncertainty": [],
    }
    if reference_baseline_model is not None:
        prediction_payload["baseline_learned"] = []
        prediction_payload["baseline_data_consistent"] = []
    if reference_visibility_model is not None:
        prediction_payload["visibility_conditioned"] = []
    if reference_residual_model is not None:
        prediction_payload["residual_refinement"] = []
    if current_model_key in {"residual_refinement", "ccrr", "emc"}:
        prediction_payload["baseline_prediction"] = []
        prediction_payload["residual_correction"] = []
    if current_model_key in {"ccrr", "emc"}:
        prediction_payload["pre_dc_prediction"] = []

    for index in range(len(dataset)):
        sample = dataset[index]
        ground_truth = sample["target"].numpy()
        dirty = sample["dirty"].numpy()
        measurements = sample["vis_real"].numpy() + 1j * sample["vis_imag"].numpy()
        mask = sample["mask"].numpy()
        ring_radius_px = float(sample["ring_radius_px"].item())
        hotspot_coords_px = sample["hotspot_coords_px"].numpy()
        baseline_pairs = sample["baseline_pairs"].numpy() if "baseline_pairs" in sample else None
        frame_uv_indices = sample["frame_uv_indices"].numpy() if "frame_uv_indices" in sample else None

        tikhonov = tikhonov_iterative_reconstruction(
            measurements=measurements,
            mask=mask,
            lambda_reg=config.evaluation.tikhonov_lambda,
            iterations=config.evaluation.tikhonov_iterations,
            step_size=config.evaluation.tikhonov_step_size,
        )

        baseline_learned = None
        baseline_data_consistent = None
        if reference_baseline_model is not None:
            baseline_input = sample["input"].unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                baseline_mean, _, _ = reference_baseline_model.predict_with_uncertainty(
                    x=baseline_input,
                    n_samples=config.evaluation.mc_samples,
                )
            baseline_learned = baseline_mean.squeeze(0).squeeze(0).detach().cpu().numpy()
            baseline_metrics_rows.append(
                compute_reconstruction_metrics(
                    prediction=baseline_learned,
                    target=ground_truth,
                    target_ring_radius_px=ring_radius_px,
                    target_hotspot_coords_px=hotspot_coords_px,
                    measurements=measurements,
                    mask=mask,
                    baseline_pairs=baseline_pairs,
                    frame_uv_indices=frame_uv_indices,
                )
            )
            baseline_data_consistent = visibility_data_consistency_projection(
                prediction=baseline_learned,
                measurements=measurements,
                mask=mask,
            )
            baseline_data_consistent_metrics_rows.append(
                compute_reconstruction_metrics(
                    prediction=baseline_data_consistent,
                    target=ground_truth,
                    target_ring_radius_px=ring_radius_px,
                    target_hotspot_coords_px=hotspot_coords_px,
                    measurements=measurements,
                    mask=mask,
                    baseline_pairs=baseline_pairs,
                    frame_uv_indices=frame_uv_indices,
                )
            )

        reference_visibility_mean = None
        if reference_visibility_model is not None and reference_visibility_model_config is not None:
            reference_visibility_mean, _, _ = _predict_phase2_model(
                model=reference_visibility_model,
                model_config=reference_visibility_model_config,
                sample=sample,
                device=device,
            )
            reference_visibility_metrics_rows.append(
                compute_reconstruction_metrics(
                    prediction=reference_visibility_mean,
                    target=ground_truth,
                    target_ring_radius_px=ring_radius_px,
                    target_hotspot_coords_px=hotspot_coords_px,
                    measurements=measurements,
                    mask=mask,
                    baseline_pairs=baseline_pairs,
                    frame_uv_indices=frame_uv_indices,
                )
            )

        reference_residual_mean = None
        if reference_residual_model is not None and reference_residual_model_config is not None:
            reference_residual_mean, _, _ = _predict_phase2_model(
                model=reference_residual_model,
                model_config=reference_residual_model_config,
                sample=sample,
                device=device,
            )
            reference_residual_metrics_rows.append(
                compute_reconstruction_metrics(
                    prediction=reference_residual_mean,
                    target=ground_truth,
                    target_ring_radius_px=ring_radius_px,
                    target_hotspot_coords_px=hotspot_coords_px,
                    measurements=measurements,
                    mask=mask,
                    baseline_pairs=baseline_pairs,
                    frame_uv_indices=frame_uv_indices,
                )
            )

        phase2_mean, phase2_std, extras = _predict_phase2_model(
            model=phase2_model,
            model_config=phase2_model_config,
            sample=sample,
            device=device,
        )

        dirty_metrics_rows.append(
            compute_reconstruction_metrics(
                prediction=dirty,
                target=ground_truth,
                target_ring_radius_px=ring_radius_px,
                target_hotspot_coords_px=hotspot_coords_px,
                measurements=measurements,
                mask=mask,
                baseline_pairs=baseline_pairs,
                frame_uv_indices=frame_uv_indices,
            )
        )
        tikhonov_metrics_rows.append(
            compute_reconstruction_metrics(
                prediction=tikhonov,
                target=ground_truth,
                target_ring_radius_px=ring_radius_px,
                target_hotspot_coords_px=hotspot_coords_px,
                measurements=measurements,
                mask=mask,
                baseline_pairs=baseline_pairs,
                frame_uv_indices=frame_uv_indices,
            )
        )
        phase2_metrics_rows.append(
            compute_reconstruction_metrics(
                prediction=phase2_mean,
                target=ground_truth,
                target_ring_radius_px=ring_radius_px,
                target_hotspot_coords_px=hotspot_coords_px,
                measurements=measurements,
                mask=mask,
                baseline_pairs=baseline_pairs,
                frame_uv_indices=frame_uv_indices,
            )
        )

        coverage_scores.append(empirical_coverage(ground_truth, phase2_mean, phase2_std))
        uncertainty_correlations.append(uncertainty_error_correlation(ground_truth, phase2_mean, phase2_std))
        risk_coverage_scores.append(risk_coverage_auc(ground_truth, phase2_mean, phase2_std))
        topk_recall_scores.append(topk_error_recall(ground_truth, phase2_mean, phase2_std))

        if config.evaluation.save_prediction_arrays:
            prediction_payload["ground_truth"].append(ground_truth.astype(np.float32))
            prediction_payload["dirty"].append(dirty.astype(np.float32))
            prediction_payload["tikhonov"].append(tikhonov.astype(np.float32))
            prediction_payload[current_model_key].append(phase2_mean.astype(np.float32))
            prediction_payload["uncertainty"].append(phase2_std.astype(np.float32))
            if baseline_learned is not None:
                prediction_payload["baseline_learned"].append(baseline_learned.astype(np.float32))
            if baseline_data_consistent is not None:
                prediction_payload["baseline_data_consistent"].append(baseline_data_consistent.astype(np.float32))
            if reference_visibility_mean is not None:
                prediction_payload["visibility_conditioned"].append(reference_visibility_mean.astype(np.float32))
            if reference_residual_mean is not None:
                prediction_payload["residual_refinement"].append(reference_residual_mean.astype(np.float32))
            if "baseline_prediction" in extras:
                prediction_payload["baseline_prediction"].append(extras["baseline_prediction"].astype(np.float32))
            if "residual_correction" in extras:
                prediction_payload["residual_correction"].append(extras["residual_correction"].astype(np.float32))
            if "pre_dc_prediction" in extras:
                prediction_payload["pre_dc_prediction"].append(extras["pre_dc_prediction"].astype(np.float32))

        if index < config.evaluation.num_visualizations:
            save_phase2_comparison_panel(
                path=output_dirs["figures"] / f"sample_{index:03d}.png",
                ground_truth=ground_truth,
                dirty=dirty,
                tikhonov=tikhonov,
                baseline_learned=baseline_learned,
                baseline_data_consistent=baseline_data_consistent,
                visibility_conditioned=reference_visibility_mean
                if current_model_key in {"residual_refinement", "ccrr"}
                else phase2_mean,
                residual_refinement=phase2_mean if current_model_key == "residual_refinement" else None,
                ccrr=phase2_mean if current_model_key == "ccrr" else None,
                uncertainty=phase2_std,
                frames_to_plot=config.evaluation.frames_to_plot,
                dpi=config.evaluation.figure_dpi,
            )

    summary: dict[str, dict[str, float] | str | None] = {
        "dirty": aggregate_metrics(dirty_metrics_rows),
        "tikhonov": aggregate_metrics(tikhonov_metrics_rows),
        current_model_key: aggregate_metrics(phase2_metrics_rows),
        "uncertainty": {
            "empirical_95_coverage": float(np.mean(coverage_scores)),
            "error_uncertainty_correlation": float(np.mean(uncertainty_correlations)),
            "risk_coverage_auc": float(np.mean(risk_coverage_scores)),
            "top10_error_recall": float(np.mean(topk_recall_scores)),
        },
        "reference_baseline_checkpoint": str(reference_baseline_checkpoint) if reference_baseline_checkpoint else None,
        "reference_visibility_checkpoint": str(reference_visibility_checkpoint) if reference_visibility_checkpoint else None,
        "reference_residual_checkpoint": str(reference_residual_checkpoint) if reference_residual_checkpoint else None,
    }
    if baseline_metrics_rows:
        summary["baseline_learned"] = aggregate_metrics(baseline_metrics_rows)
    if baseline_data_consistent_metrics_rows:
        summary["baseline_data_consistent"] = aggregate_metrics(baseline_data_consistent_metrics_rows)
    if reference_visibility_metrics_rows:
        summary["visibility_conditioned"] = aggregate_metrics(reference_visibility_metrics_rows)
    if reference_residual_metrics_rows:
        summary["residual_refinement"] = aggregate_metrics(reference_residual_metrics_rows)

    save_json(output_dirs["logs"] / "evaluation_summary.json", summary)
    if config.evaluation.save_comparison_csv:
        save_comparison_csv(output_dirs["logs"] / "comparison_metrics.csv", summary)
    if config.evaluation.save_prediction_arrays and prediction_payload["ground_truth"]:
        np.savez_compressed(
            output_dirs["predictions"] / "test_predictions.npz",
            **{key: np.stack(value).astype(np.float32) for key, value in prediction_payload.items()},
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--preset", default=None, choices=["smoke", "default32", "default64", "exp64"])
    parser.add_argument(
        "--model-type",
        default=None,
        choices=["baseline", "visibility_conditioned", "residual_visibility_refinement", "ccrr", "emc_ccrr"],
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--reference-baseline-checkpoint", default=None)
    parser.add_argument("--reference-visibility-checkpoint", default=None)
    parser.add_argument("--reference-residual-checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config_explicit = _explicit_arg("--base-config")
    preset = _resolve_preset(args.preset, base_config_explicit=base_config_explicit)
    base_config_path = ROOT / args.base_config
    config = load_experiment_config(
        base_path=base_config_path,
        train_path=ROOT / args.train_config,
        eval_path=ROOT / args.eval_config,
        preset=preset,
        default_base_path=ROOT / DEFAULT_BASE_CONFIG_PATH,
    )
    if args.model_type is not None:
        config.model.model_type = args.model_type
    set_seed(config.project.seed)
    experiment_label = _experiment_label(base_config_path=base_config_path, preset=preset)
    data_dir = Path(args.data_dir) if args.data_dir else ROOT / config.paths.data_root / experiment_label
    output_root = Path(args.output_root) if args.output_root else ROOT / config.paths.output_root
    run_name = args.run_name or f"train_{experiment_label}"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_root / run_name / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if config.model.model_type == "baseline":
        summary = run_baseline_evaluation(
            config=config,
            data_dir=data_dir,
            checkpoint_path=checkpoint_path,
            output_root=output_root,
            run_name=run_name,
        )
    elif config.model.model_type in {"visibility_conditioned", "residual_visibility_refinement", "ccrr", "emc_ccrr"}:
        reference_baseline_checkpoint = _resolve_reference_baseline_checkpoint(
            explicit_checkpoint=args.reference_baseline_checkpoint,
            config=config,
            output_root=output_root,
        )
        reference_visibility_checkpoint = _resolve_reference_visibility_checkpoint(
            explicit_checkpoint=args.reference_visibility_checkpoint,
            config=config,
            output_root=output_root,
        )
        reference_residual_checkpoint = _resolve_reference_residual_checkpoint(
            explicit_checkpoint=args.reference_residual_checkpoint,
            config=config,
            output_root=output_root,
        )
        summary = run_phase2_evaluation(
            config=config,
            data_dir=data_dir,
            checkpoint_path=checkpoint_path,
            output_root=output_root,
            run_name=run_name,
            reference_baseline_checkpoint=reference_baseline_checkpoint,
            reference_visibility_checkpoint=reference_visibility_checkpoint,
            reference_residual_checkpoint=reference_residual_checkpoint,
        )
    else:
        raise ValueError(f"Unsupported model_type '{config.model.model_type}'.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
