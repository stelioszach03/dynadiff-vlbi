#!/usr/bin/env python3
"""Evaluate classical and learned baselines on the synthetic test set."""

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
from dynadiff_vlbi.evaluation.metrics import (
    compute_reconstruction_metrics,
    empirical_coverage,
    uncertainty_error_correlation,
)
from dynadiff_vlbi.evaluation.visualization import save_reconstruction_panel
from dynadiff_vlbi.models.temporal_unet import TemporalUNet3D
from dynadiff_vlbi.physics.classical_reconstruction import tikhonov_iterative_reconstruction
from dynadiff_vlbi.utils.config import DEFAULT_BASE_CONFIG_PATH, ExperimentConfig, load_experiment_config
from dynadiff_vlbi.utils.device import get_device
from dynadiff_vlbi.utils.logging_utils import prepare_output_dirs, save_json
from dynadiff_vlbi.utils.seed import set_seed


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
    keys = metric_rows[0].keys()
    return {key: float(np.mean([row[key] for row in metric_rows])) for key in keys}


def run_evaluation(
    config: ExperimentConfig,
    data_dir: Path,
    checkpoint_path: Path,
    output_root: Path,
    run_name: str,
) -> dict[str, dict[str, float]]:
    """Run evaluation and save figures, predictions, and metrics."""

    output_dirs = prepare_output_dirs(str(output_root), run_name=run_name, config=config)
    dataset = DynamicVLBIDataset(data_dir / "test.npz")
    device = get_device()
    model = TemporalUNet3D(config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dirty_metrics_rows = []
    tikhonov_metrics_rows = []
    learned_metrics_rows = []
    coverage_scores = []
    uncertainty_correlations = []

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

        tikhonov = tikhonov_iterative_reconstruction(
            measurements=measurements,
            mask=mask,
            lambda_reg=config.evaluation.tikhonov_lambda,
            iterations=config.evaluation.tikhonov_iterations,
            step_size=config.evaluation.tikhonov_step_size,
        )
        input_tensor = sample["input"].unsqueeze(0).unsqueeze(0).to(device)
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
            )
        )
        tikhonov_metrics_rows.append(
            compute_reconstruction_metrics(
                prediction=tikhonov,
                target=ground_truth,
                target_ring_radius_px=ring_radius_px,
                target_hotspot_coords_px=hotspot_coords_px,
            )
        )
        learned_metrics_rows.append(
            compute_reconstruction_metrics(
                prediction=learned_mean,
                target=ground_truth,
                target_ring_radius_px=ring_radius_px,
                target_hotspot_coords_px=hotspot_coords_px,
            )
        )

        coverage_scores.append(empirical_coverage(ground_truth, learned_mean, uncertainty))
        uncertainty_correlations.append(uncertainty_error_correlation(ground_truth, learned_mean, uncertainty))

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
        },
    }
    save_json(output_dirs["logs"] / "evaluation_summary.json", summary)
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
    parser.add_argument("--preset", default=None, choices=["smoke", "default32", "exp64"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint", default=None)
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
    set_seed(config.project.seed)
    experiment_label = _experiment_label(base_config_path=base_config_path, preset=preset)
    data_dir = Path(args.data_dir) if args.data_dir else ROOT / config.paths.data_root / experiment_label
    output_root = Path(args.output_root) if args.output_root else ROOT / config.paths.output_root
    run_name = args.run_name or f"train_{experiment_label}"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_root / run_name / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    summary = run_evaluation(
        config=config,
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        output_root=output_root,
        run_name=run_name,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
