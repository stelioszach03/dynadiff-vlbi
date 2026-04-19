#!/usr/bin/env python3
"""Generate MNRAS-facing tables and figures from verified experiment outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.evaluation.paper_artifacts import (
    format_mean_std,
    format_value,
    save_json,
    write_csv,
    write_markdown_table,
)
from dynadiff_vlbi.evaluation.paper_visuals import (
    PredictionCondition,
    save_realism_bridge_figure,
    save_sparse_uv_killer_figure,
    save_uncertainty_risk_coverage_figure,
)


MODEL_LABELS = {
    "dirty": "Dirty",
    "tikhonov": "Tikhonov",
    "baseline_learned": "Baseline 3D U-Net",
    "baseline_data_consistent": "Baseline + Data Consistency",
    "visibility_conditioned": "Standalone Visibility",
    "residual_refinement": "Residual Refinement",
}

MODEL_ORDER = [
    "dirty",
    "tikhonov",
    "baseline_learned",
    "baseline_data_consistent",
    "visibility_conditioned",
    "residual_refinement",
]

CORE_METRICS = [
    "mse",
    "psnr",
    "ssim",
    "temporal_consistency",
    "ring_radius_error",
    "hotspot_localization_error",
]
ASTRO_METRICS = [
    "arc_profile_correlation",
    "hotspot_track_velocity_error",
    "observed_visibility_rmse",
]
BRIDGE_METRICS = ASTRO_METRICS + ["closure_phase_mae"]
UNCERTAINTY_METRICS = [
    "empirical_95_coverage",
    "error_uncertainty_correlation",
    "risk_coverage_auc",
    "top10_error_recall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--artifact-root", default="outputs/mnras_artifacts")
    parser.add_argument("--paper-root", default="paper")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _summary_path(output_root: Path, run_name: str) -> Path:
    return output_root / run_name / "logs" / "evaluation_summary.json"


def _prediction_path(output_root: Path, run_name: str) -> Path:
    return output_root / run_name / "predictions" / "test_predictions.npz"


def _metric_direction(metric_key: str) -> str:
    if metric_key in {"psnr", "ssim", "arc_profile_correlation"}:
        return "higher"
    return "lower"


def _format_metric(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return format_value(float(value))


def _aggregate_seed_repeats(summaries: list[dict[str, Any]], model_order: list[str], metric_keys: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key in model_order:
        row: dict[str, Any] = {"model": model_key, "model_label": MODEL_LABELS[model_key]}
        for metric_key in metric_keys:
            values = np.asarray([summary[model_key][metric_key] for summary in summaries], dtype=np.float64)
            row[f"{metric_key}_mean"] = float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")
            row[f"{metric_key}_std"] = float(np.nanstd(values)) if not np.all(np.isnan(values)) else float("nan")
        rows.append(row)
    return rows


def _residual_verdict(reference_value: float, candidate_value: float, metric_key: str) -> str:
    if math.isnan(reference_value) or math.isnan(candidate_value):
        return "n/a"
    if math.isclose(reference_value, candidate_value, rel_tol=0.0, abs_tol=1e-12):
        return "tie"
    direction = _metric_direction(metric_key)
    if direction == "higher":
        return "win" if candidate_value > reference_value else "loss"
    return "win" if candidate_value < reference_value else "loss"


def _condition_rows(
    summaries: dict[str, dict[str, Any]],
    condition_order: list[str],
    model_order: list[str],
    metric_keys: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition_key in condition_order:
        summary = summaries[condition_key]
        for model_key in model_order:
            metrics = summary.get(model_key)
            if metrics is None:
                continue
            row = {
                "condition": condition_key,
                "model": model_key,
                "model_label": MODEL_LABELS[model_key],
            }
            for metric_key in metric_keys:
                row[metric_key] = metrics[metric_key]
            rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    output_root = (ROOT / args.output_root).resolve()
    artifact_root = (ROOT / args.artifact_root).resolve()
    paper_root = (ROOT / args.paper_root).resolve()
    tables_dir = artifact_root / "tables"
    summaries_dir = artifact_root / "summaries"
    figures_dir = paper_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    seed_runs = {
        "default32_seed7": "compare_default32_residual_refine_fair8",
        "default32_seed19": "paper_default32_seed19_residual_refine",
        "default32_seed31": "paper_default32_seed31_residual_refine",
    }
    condition_runs = {
        "noise_high": "paper_noise_high_residual_refine",
        "sparse_uv": "paper_sparse_uv_residual_refine",
        "exp64": "paper_exp64_residual_refine_clean",
        "realism_bridge": "mnras_bridge_default32_residual_refine",
    }

    all_summaries = {
        key: load_json(_summary_path(output_root, run_name))
        for key, run_name in {**seed_runs, **condition_runs}.items()
    }

    seed_summaries = [all_summaries[key] for key in seed_runs]
    default32_core_rows = _aggregate_seed_repeats(seed_summaries, MODEL_ORDER, CORE_METRICS)
    default32_astro_rows = _aggregate_seed_repeats(seed_summaries, MODEL_ORDER, ASTRO_METRICS)

    write_csv(tables_dir / "mnras_default32_core_table.csv", default32_core_rows)
    save_json(tables_dir / "mnras_default32_core_table.json", {"rows": default32_core_rows})
    write_markdown_table(
        tables_dir / "mnras_default32_core_table.md",
        headers=["Model", "MSE", "PSNR", "SSIM", "Temporal", "Ring", "Hotspot"],
        rows=[
            [
                str(row["model_label"]),
                format_mean_std(float(row["mse_mean"]), float(row["mse_std"])),
                format_mean_std(float(row["psnr_mean"]), float(row["psnr_std"])),
                format_mean_std(float(row["ssim_mean"]), float(row["ssim_std"])),
                format_mean_std(float(row["temporal_consistency_mean"]), float(row["temporal_consistency_std"])),
                format_mean_std(float(row["ring_radius_error_mean"]), float(row["ring_radius_error_std"])),
                format_mean_std(float(row["hotspot_localization_error_mean"]), float(row["hotspot_localization_error_std"])),
            ]
            for row in default32_core_rows
        ],
        title="Default32 Seed-Repeat Core Metrics",
        notes=["Mean ± std over seeds 7, 19, and 31."],
    )

    write_csv(tables_dir / "mnras_default32_astro_table.csv", default32_astro_rows)
    save_json(tables_dir / "mnras_default32_astro_table.json", {"rows": default32_astro_rows})
    write_markdown_table(
        tables_dir / "mnras_default32_astro_table.md",
        headers=["Model", "ArcCorr", "Track", "VisRMSE"],
        rows=[
            [
                str(row["model_label"]),
                format_mean_std(float(row["arc_profile_correlation_mean"]), float(row["arc_profile_correlation_std"])),
                format_mean_std(float(row["hotspot_track_velocity_error_mean"]), float(row["hotspot_track_velocity_error_std"])),
                format_mean_std(float(row["observed_visibility_rmse_mean"]), float(row["observed_visibility_rmse_std"])),
            ]
            for row in default32_astro_rows
        ],
        title="Default32 Astronomy-Facing Metrics",
        notes=["ArcCorr is higher-is-better. Track and VisRMSE are lower-is-better."],
    )

    robustness_order = ["noise_high", "sparse_uv", "exp64", "realism_bridge"]
    robustness_rows = _condition_rows(
        summaries=all_summaries,
        condition_order=robustness_order,
        model_order=["baseline_learned", "baseline_data_consistent", "residual_refinement"],
        metric_keys=["mse", "psnr", "ssim", "temporal_consistency", "arc_profile_correlation"],
    )
    write_csv(tables_dir / "mnras_robustness_table.csv", robustness_rows)
    save_json(tables_dir / "mnras_robustness_table.json", {"rows": robustness_rows})
    write_markdown_table(
        tables_dir / "mnras_robustness_table.md",
        headers=["Condition", "Model", "MSE", "PSNR", "SSIM", "Temporal", "ArcCorr"],
        rows=[
            [
                row["condition"],
                str(row["model_label"]),
                _format_metric(float(row["mse"])),
                _format_metric(float(row["psnr"])),
                _format_metric(float(row["ssim"])),
                _format_metric(float(row["temporal_consistency"])),
                _format_metric(float(row["arc_profile_correlation"])),
            ]
            for row in robustness_rows
        ],
        title="Robustness Summary Across Hard Conditions",
        notes=[
            "This table focuses on the baseline backbone, the added data-consistency comparator, and the final residual-refinement model.",
        ],
    )

    realism_bridge = all_summaries["realism_bridge"]
    bridge_rows = [
        {
            "model": model_key,
            "model_label": MODEL_LABELS[model_key],
            **{metric_key: realism_bridge[model_key][metric_key] for metric_key in CORE_METRICS + BRIDGE_METRICS},
        }
        for model_key in MODEL_ORDER
        if model_key in realism_bridge
    ]
    write_csv(tables_dir / "mnras_realism_bridge_table.csv", bridge_rows)
    save_json(tables_dir / "mnras_realism_bridge_table.json", {"rows": bridge_rows})
    write_markdown_table(
        tables_dir / "mnras_realism_bridge_table.md",
        headers=["Model", "MSE", "PSNR", "SSIM", "Temporal", "ArcCorr", "Track", "VisRMSE", "ClosurePhase"],
        rows=[
            [
                str(row["model_label"]),
                _format_metric(float(row["mse"])),
                _format_metric(float(row["psnr"])),
                _format_metric(float(row["ssim"])),
                _format_metric(float(row["temporal_consistency"])),
                _format_metric(float(row["arc_profile_correlation"])),
                _format_metric(float(row["hotspot_track_velocity_error"])),
                _format_metric(float(row["observed_visibility_rmse"])),
                _format_metric(float(row["closure_phase_mae"])),
            ]
            for row in bridge_rows
        ],
        title="Realism-Bridge Station-Track Evaluation",
        notes=[
            "ClosurePhase is only defined for the station-inspired realism-bridge path.",
            "Baseline + Data Consistency is a non-learned projection baseline that enforces agreement with measured visibilities.",
        ],
    )

    uncertainty_rows = []
    for condition_key in [*seed_runs.keys(), *condition_runs.keys()]:
        uncertainty = all_summaries[condition_key]["uncertainty"]
        uncertainty_rows.append({"condition": condition_key, **uncertainty})
    write_csv(tables_dir / "mnras_uncertainty_table.csv", uncertainty_rows)
    save_json(tables_dir / "mnras_uncertainty_table.json", {"rows": uncertainty_rows})
    write_markdown_table(
        tables_dir / "mnras_uncertainty_table.md",
        headers=["Condition", "Coverage", "Err/Unc Corr.", "RiskCov AUC", "Top10 Recall"],
        rows=[
            [
                row["condition"],
                _format_metric(float(row["empirical_95_coverage"])),
                _format_metric(float(row["error_uncertainty_correlation"])),
                _format_metric(float(row["risk_coverage_auc"])),
                _format_metric(float(row["top10_error_recall"])),
            ]
            for row in uncertainty_rows
        ],
        title="Uncertainty Diagnostics",
        notes=["Coverage is conservative; lower RiskCov AUC and higher Top10 Recall are better."],
    )

    verdict_rows = []
    for condition_key, summary in all_summaries.items():
        residual = summary["residual_refinement"]
        baseline = summary["baseline_learned"]
        data_consistent = summary.get("baseline_data_consistent")
        for metric_key in ["mse", "psnr", "ssim", "temporal_consistency", "arc_profile_correlation", "observed_visibility_rmse"]:
            row = {
                "condition": condition_key,
                "metric": metric_key,
                "vs_baseline_unet": _residual_verdict(float(baseline[metric_key]), float(residual[metric_key]), metric_key),
            }
            if data_consistent is not None:
                row["vs_baseline_data_consistent"] = _residual_verdict(
                    float(data_consistent[metric_key]),
                    float(residual[metric_key]),
                    metric_key,
                )
            verdict_rows.append(row)
    write_csv(tables_dir / "mnras_verdicts.csv", verdict_rows)
    save_json(tables_dir / "mnras_verdicts.json", {"rows": verdict_rows})
    write_markdown_table(
        tables_dir / "mnras_verdicts.md",
        headers=["Condition", "Metric", "Residual vs Baseline", "Residual vs Baseline+DC"],
        rows=[
            [
                row["condition"],
                row["metric"],
                str(row["vs_baseline_unet"]),
                str(row.get("vs_baseline_data_consistent", "n/a")),
            ]
            for row in verdict_rows
        ],
        title="Residual-Refinement Verdict Summary",
        notes=["Win/loss/tie is computed directly from the saved evaluation summaries for each condition."],
    )

    sparse_uv_run = condition_runs["sparse_uv"]
    realism_run = condition_runs["realism_bridge"]
    sparse_uv_predictions = _prediction_path(output_root, sparse_uv_run)
    realism_predictions = _prediction_path(output_root, realism_run)
    realism_dataset = ROOT / "data/generated/mnras_bridge_default32_shared/test.npz"

    sparse_selection = save_sparse_uv_killer_figure(
        prediction_path=sparse_uv_predictions,
        output_png=figures_dir / "fig05_sparse_uv_killer.png",
        output_svg=figures_dir / "fig05_sparse_uv_killer.svg",
        selection_manifest=figures_dir / "fig05_sparse_uv_killer.selection.json",
    )
    bridge_selection = save_realism_bridge_figure(
        prediction_path=realism_predictions,
        dataset_path=realism_dataset,
        output_png=figures_dir / "fig06_realism_bridge.png",
        output_svg=figures_dir / "fig06_realism_bridge.svg",
        selection_manifest=figures_dir / "fig06_realism_bridge.selection.json",
    )
    save_uncertainty_risk_coverage_figure(
        conditions=[
            PredictionCondition("default32", "Default32", _prediction_path(output_root, seed_runs["default32_seed7"])),
            PredictionCondition("high_noise", "High noise", _prediction_path(output_root, condition_runs["noise_high"])),
            PredictionCondition("sparse_uv", "Sparse uv", sparse_uv_predictions),
            PredictionCondition("exp64", "Exp64", _prediction_path(output_root, condition_runs["exp64"])),
            PredictionCondition("bridge", "Realism bridge", realism_predictions),
        ],
        output_png=figures_dir / "fig07_uncertainty_risk_coverage.png",
        output_svg=figures_dir / "fig07_uncertainty_risk_coverage.svg",
    )

    summary_payload = {
        "seed_runs": seed_runs,
        "condition_runs": condition_runs,
        "default32_core_rows": default32_core_rows,
        "default32_astro_rows": default32_astro_rows,
        "sparse_uv_selection": sparse_selection,
        "realism_bridge_selection": bridge_selection,
    }
    save_json(summaries_dir / "mnras_artifact_manifest.json", summary_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
