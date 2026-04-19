#!/usr/bin/env python3
"""Generate CCRR-focused paper tables, statistics, and figures."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.evaluation.ccrr_artifacts import (
    ASTRO_METRICS,
    CORE_METRICS,
    MODEL_LABELS,
    UNCERTAINTY_METRICS,
    ExperimentSpec,
    aggregate_seed_summaries,
    build_paired_rows,
    save_ccrr_pareto_figure,
    save_ccrr_qualitative_figure,
    save_ccrr_risk_coverage_figure,
    save_ccrr_schematic,
    save_ccrr_supplementary_gif,
    save_json,
    summarize_prediction_bundle,
    write_csv,
    write_markdown_table,
    format_mean_std,
    format_value,
)
from dynadiff_vlbi.evaluation.measurement_audit import (
    MeasurementAuditSpec,
    generate_measurement_audit_artifacts,
)


MODEL_ORDER = [
    "dirty",
    "tikhonov",
    "baseline_learned",
    "baseline_data_consistent",
    "visibility_conditioned",
    "residual_refinement",
    "ccrr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--data-root", default="data/generated")
    parser.add_argument("--artifact-root", default="outputs/ccrr_paper_artifacts")
    parser.add_argument("--paper-root", default="paper")
    return parser.parse_args()


def _fmt(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return format_value(float(value))


def _fmt_p(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if value <= 1e-4:
        return "<1e-4"
    return format_value(float(value))


def main() -> int:
    args = parse_args()
    output_root = (ROOT / args.output_root).resolve()
    data_root = (ROOT / args.data_root).resolve()
    artifact_root = (ROOT / args.artifact_root).resolve()
    paper_root = (ROOT / args.paper_root).resolve()
    tables_dir = artifact_root / "tables"
    summaries_dir = artifact_root / "summaries"
    figures_dir = paper_root / "figures"
    media_dir = paper_root / "supplementary_media"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    seed_specs = [
        ExperimentSpec("seed7", "Default32 seed 7", "ccrr_seed7_main", data_root / "ccrr_default32_seed7_shared", output_root),
        ExperimentSpec("seed19", "Default32 seed 19", "ccrr_seed19_main", data_root / "ccrr_default32_seed19_shared", output_root),
        ExperimentSpec("seed31", "Default32 seed 31", "ccrr_seed31_main", data_root / "ccrr_default32_seed31_shared", output_root),
        ExperimentSpec("seed43", "Default32 seed 43", "ccrr_seed43_main", data_root / "ccrr_default32_seed43_shared", output_root),
        ExperimentSpec("seed59", "Default32 seed 59", "ccrr_seed59_main", data_root / "ccrr_default32_seed59_shared", output_root),
    ]
    condition_specs = [
        ExperimentSpec("noise_high", "High noise", "ccrr_noise_high_main", data_root / "ccrr_noise_high_shared", output_root),
        ExperimentSpec("sparse_uv", "Sparse uv", "ccrr_sparse_uv_main", data_root / "ccrr_sparse_uv_shared", output_root),
        ExperimentSpec("exp64", "Exp64", "ccrr_exp64_main", data_root / "ccrr_exp64_shared", output_root),
        ExperimentSpec("realism_bridge2", "Realism bridge 2.0", "ccrr_realism_bridge2_main", data_root / "ccrr_realism_bridge2_shared", output_root),
    ]
    ablation_specs = [
        ExperimentSpec("ccrr", "CCRR", "ccrr_seed7_main", data_root / "ccrr_default32_seed7_shared", output_root),
        ExperimentSpec("no_dc", "No DC layer", "ccrr_ablation_no_dc", data_root / "ccrr_default32_seed7_shared", output_root),
        ExperimentSpec("no_closure", "No closure loss", "ccrr_ablation_no_closure", data_root / "ccrr_default32_seed7_shared", output_root),
        ExperimentSpec("no_metadata", "No metadata conditioning", "ccrr_ablation_no_metadata", data_root / "ccrr_default32_seed7_shared", output_root),
        ExperimentSpec("no_uncertainty", "No uncertainty head", "ccrr_ablation_no_uncertainty", data_root / "ccrr_default32_seed7_shared", output_root),
    ]

    all_specs = seed_specs + condition_specs + ablation_specs
    run_summaries: dict[str, dict[str, Any]] = {}
    for spec in all_specs:
        run_summaries[spec.key] = summarize_prediction_bundle(spec.prediction_path, spec.dataset_dir)

    seed_rows = aggregate_seed_summaries(run_summaries, [spec.key for spec in seed_specs], MODEL_ORDER)
    write_csv(tables_dir / "ccrr_main_results.csv", seed_rows)
    write_markdown_table(
        tables_dir / "ccrr_main_results.md",
        headers=["Model", "MSE", "PSNR", "SSIM", "Temporal", "Ring", "Thickness", "SectorAngle"],
        rows=[
            [
                row["model_label"],
                format_mean_std(float(row["mse_mean"]), float(row["mse_std"])),
                format_mean_std(float(row["psnr_mean"]), float(row["psnr_std"])),
                format_mean_std(float(row["ssim_mean"]), float(row["ssim_std"])),
                format_mean_std(float(row["temporal_consistency_mean"]), float(row["temporal_consistency_std"])),
                format_mean_std(float(row["ring_radius_error_mean"]), float(row["ring_radius_error_std"])),
                format_mean_std(float(row["ring_thickness_error_mean"]), float(row["ring_thickness_error_std"])),
                format_mean_std(float(row["bright_sector_angle_error_mean"]), float(row["bright_sector_angle_error_std"])),
            ]
            for row in seed_rows
        ],
        title="CCRR Default32 Seed-Repeat Results",
        notes=["Mean ± std over seeds 7, 19, 31, 43, and 59."],
    )

    astro_rows = []
    for row in seed_rows:
        astro_rows.append(
            {
                "model_label": row["model_label"],
                "arc_profile_correlation": row["arc_profile_correlation_mean"],
                "hotspot_track_velocity_error": row["hotspot_track_velocity_error_mean"],
                "observed_visibility_rmse": row["observed_visibility_rmse_mean"],
                "closure_phase_mae": row["closure_phase_mae_mean"],
            }
        )
    write_csv(tables_dir / "ccrr_astronomy_metrics.csv", astro_rows)
    write_markdown_table(
        tables_dir / "ccrr_astronomy_metrics.md",
        headers=["Model", "ArcCorr", "Track", "VisRMSE", "ClosurePhase"],
        rows=[
            [
                row["model_label"],
                _fmt(float(row["arc_profile_correlation"])),
                _fmt(float(row["hotspot_track_velocity_error"])),
                _fmt(float(row["observed_visibility_rmse"])),
                _fmt(float(row["closure_phase_mae"])),
            ]
            for row in astro_rows
        ],
        title="CCRR Default32 Astronomy-Facing Metrics",
        notes=["ArcCorr is higher-is-better. Track, VisRMSE, and ClosurePhase are lower-is-better."],
    )

    paired_rows = build_paired_rows(
        run_summaries=run_summaries,
        seed_keys=[spec.key for spec in seed_specs],
        metric_keys=["mse", "psnr", "ssim", "temporal_consistency", "observed_visibility_rmse", "closure_phase_mae"],
        comparisons=[
            ("ccrr", "baseline_learned", "CCRR vs Baseline 3D U-Net"),
            ("ccrr", "residual_refinement", "CCRR vs Residual Refinement"),
            ("ccrr", "baseline_data_consistent", "CCRR vs Baseline + DC"),
        ],
    )
    write_csv(tables_dir / "ccrr_paired_statistics.csv", paired_rows)
    write_markdown_table(
        tables_dir / "ccrr_paired_statistics.md",
        headers=["Comparison", "Metric", "Mean delta", "95% CI", "Win rate", "p"],
        rows=[
            [
                row["comparison"],
                row["metric_label"],
                _fmt(float(row["mean_delta"])),
                f"[{_fmt(float(row['ci_low']))}, {_fmt(float(row['ci_high']))}]",
                _fmt(float(row["win_rate"])),
                _fmt_p(float(row["p_value"])),
            ]
            for row in paired_rows
        ],
        title="CCRR Paired Per-Sample Statistics",
        notes=[
            "Positive mean delta favors CCRR after metric-direction normalization.",
            "Confidence intervals are paired bootstrap intervals over pooled seed-repeat test samples.",
        ],
    )

    condition_rows: list[dict[str, Any]] = []
    for spec in condition_specs:
        summary = run_summaries[spec.key]["aggregate"]
        for model_key in MODEL_ORDER:
            if model_key not in summary:
                continue
            row = {"condition": spec.key, "condition_title": spec.title, "model": model_key, "model_label": MODEL_LABELS[model_key]}
            row.update(summary[model_key])
            condition_rows.append(row)
    write_csv(tables_dir / "ccrr_robustness.csv", condition_rows)
    write_markdown_table(
        tables_dir / "ccrr_robustness.md",
        headers=["Condition", "Model", "MSE", "PSNR", "SSIM", "Temporal", "VisRMSE", "ClosurePhase"],
        rows=[
            [
                row["condition_title"],
                row["model_label"],
                _fmt(float(row["mse"])),
                _fmt(float(row["psnr"])),
                _fmt(float(row["ssim"])),
                _fmt(float(row["temporal_consistency"])),
                _fmt(float(row["observed_visibility_rmse"])),
                _fmt(float(row["closure_phase_mae"])),
            ]
            for row in condition_rows
        ],
        title="CCRR Robustness Across Hard Conditions",
        notes=["All conditions are evaluated on shared splits with the same comparator set."],
    )

    ablation_rows = []
    for spec in ablation_specs:
        aggregate = run_summaries[spec.key]["aggregate"]["ccrr"]
        ablation_rows.append({"ablation": spec.title, **aggregate})
    write_csv(tables_dir / "ccrr_ablations.csv", ablation_rows)
    write_markdown_table(
        tables_dir / "ccrr_ablations.md",
        headers=["Ablation", "MSE", "PSNR", "SSIM", "Temporal", "VisRMSE", "ClosurePhase"],
        rows=[
            [
                row["ablation"],
                _fmt(float(row["mse"])),
                _fmt(float(row["psnr"])),
                _fmt(float(row["ssim"])),
                _fmt(float(row["temporal_consistency"])),
                _fmt(float(row["observed_visibility_rmse"])),
                _fmt(float(row["closure_phase_mae"])),
            ]
            for row in ablation_rows
        ],
        title="CCRR Ablations on Default32 Seed 7",
        notes=["Each ablation is evaluated on the same shared default32 seed-7 split."],
    )

    uncertainty_rows = []
    for spec in seed_specs + condition_specs:
        uncertainty = run_summaries[spec.key]["uncertainty"]
        uncertainty_rows.append({"condition": spec.title, **uncertainty})
    write_csv(tables_dir / "ccrr_uncertainty.csv", uncertainty_rows)
    write_markdown_table(
        tables_dir / "ccrr_uncertainty.md",
        headers=["Condition", "Coverage", "Err/Unc Corr.", "RiskCov AUC", "Top10 Recall"],
        rows=[
            [
                row["condition"],
                _fmt(float(row["empirical_95_coverage"])),
                _fmt(float(row["error_uncertainty_correlation"])),
                _fmt(float(row["risk_coverage_auc"])),
                _fmt(float(row["top10_error_recall"])),
            ]
            for row in uncertainty_rows
        ],
        title="CCRR Uncertainty Diagnostics",
        notes=["Coverage is conservative; lower RiskCov AUC and higher Top10 Recall are better."],
    )

    save_ccrr_schematic(
        figures_dir / "fig01_ccrr_schematic.png",
        figures_dir / "fig01_ccrr_schematic.svg",
    )
    seed_gains = []
    for spec in seed_specs:
        aggregate = run_summaries[spec.key]["aggregate"]
        seed_gains.append(
            (
                aggregate["residual_refinement"]["mse"] - aggregate["ccrr"]["mse"],
                spec,
            )
        )
    seed_gains.sort(key=lambda item: item[0])
    representative_spec = seed_gains[len(seed_gains) // 2][1]
    representative_selection = save_ccrr_qualitative_figure(
        prediction_path=representative_spec.prediction_path,
        output_png=figures_dir / "fig02_ccrr_representative_default32.png",
        output_svg=figures_dir / "fig02_ccrr_representative_default32.svg",
        selection_manifest=figures_dir / "fig02_ccrr_representative_default32.selection.json",
        title="Representative default32 example selected reproducibly from the median CCRR gain over residual refinement",
        selection="representative",
    )

    candidate_conditions = [row for row in condition_rows if row["model"] in {"residual_refinement", "ccrr"}]
    condition_scores: dict[str, float] = {}
    for spec in condition_specs:
        ccrr_row = next(row for row in candidate_conditions if row["condition"] == spec.key and row["model"] == "ccrr")
        residual_row = next(
            row for row in candidate_conditions if row["condition"] == spec.key and row["model"] == "residual_refinement"
        )
        score = (residual_row["mse"] - ccrr_row["mse"]) + (ccrr_row["ssim"] - residual_row["ssim"])
        score += residual_row["observed_visibility_rmse"] - ccrr_row["observed_visibility_rmse"]
        condition_scores[spec.key] = float(score)
    killer_spec = max(condition_specs, key=lambda spec: condition_scores[spec.key])
    killer_selection = save_ccrr_qualitative_figure(
        prediction_path=killer_spec.prediction_path,
        output_png=figures_dir / "fig03_ccrr_hard_condition.png",
        output_svg=figures_dir / "fig03_ccrr_hard_condition.svg",
        selection_manifest=figures_dir / "fig03_ccrr_hard_condition.selection.json",
        title=f"{killer_spec.title}: high-gain hard example selected reproducibly by CCRR improvement over residual refinement",
        selection="hard",
    )

    save_ccrr_pareto_figure(
        condition_rows,
        figures_dir / "fig04_ccrr_pareto.png",
        figures_dir / "fig04_ccrr_pareto.svg",
    )
    save_ccrr_risk_coverage_figure(
        seed_specs[:1] + condition_specs,
        figures_dir / "fig05_ccrr_risk_coverage.png",
        figures_dir / "fig05_ccrr_risk_coverage.svg",
    )
    save_ccrr_supplementary_gif(
        prediction_path=representative_spec.prediction_path,
        output_path=media_dir / "supp_ccrr_default32_sequence.gif",
        sample_index=int(representative_selection["sample_index"]),
    )

    measurement_audit_summary = generate_measurement_audit_artifacts(
        seed_specs=[
            MeasurementAuditSpec("seed7", "Default32 seed 7", 7, "ccrr_seed7_main", data_root / "ccrr_default32_seed7_shared", output_root),
            MeasurementAuditSpec("seed19", "Default32 seed 19", 19, "ccrr_seed19_main", data_root / "ccrr_default32_seed19_shared", output_root),
            MeasurementAuditSpec("seed31", "Default32 seed 31", 31, "ccrr_seed31_main", data_root / "ccrr_default32_seed31_shared", output_root),
            MeasurementAuditSpec("seed43", "Default32 seed 43", 43, "ccrr_seed43_main", data_root / "ccrr_default32_seed43_shared", output_root),
            MeasurementAuditSpec("seed59", "Default32 seed 59", 59, "ccrr_seed59_main", data_root / "ccrr_default32_seed59_shared", output_root),
        ],
        no_dc_spec=MeasurementAuditSpec(
            "no_dc",
            "No DC layer",
            7,
            "ccrr_ablation_no_dc",
            data_root / "ccrr_default32_seed7_shared",
            output_root,
        ),
        no_closure_spec=MeasurementAuditSpec(
            "no_closure",
            "No closure loss",
            7,
            "ccrr_ablation_no_closure",
            data_root / "ccrr_default32_seed7_shared",
            output_root,
        ),
        artifact_root=artifact_root,
        paper_root=paper_root,
    )

    def _spec_dict(spec: ExperimentSpec) -> dict[str, str]:
        return {
            "key": spec.key,
            "title": spec.title,
            "run_name": spec.run_name,
            "dataset_dir": str(spec.dataset_dir),
            "prediction_path": str(spec.prediction_path),
        }

    manifest = {
        "seed_specs": [_spec_dict(spec) for spec in seed_specs],
        "condition_specs": [_spec_dict(spec) for spec in condition_specs],
        "ablation_specs": [_spec_dict(spec) for spec in ablation_specs],
        "representative_selection": representative_selection,
        "representative_seed": representative_spec.key,
        "killer_condition": killer_spec.key,
        "killer_selection": killer_selection,
        "measurement_audit": measurement_audit_summary,
        "tables": sorted(str(path) for path in tables_dir.glob("*")),
        "figures": sorted(str(path) for path in figures_dir.glob("fig0[1-6]_ccrr*")),
    }
    save_json(summaries_dir / "ccrr_artifact_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
