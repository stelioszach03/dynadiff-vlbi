#!/usr/bin/env python3
"""Generate paper-facing tables, summaries, and figures from completed experiment outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.evaluation.paper_artifacts import (
    METRIC_SPECS,
    MODEL_LABELS,
    ConditionSpec,
    aggregate_seed_repeats,
    build_condition_verdict_rows,
    collect_metric_rows,
    compute_verdict,
    format_mean_std,
    format_value,
    save_json,
    save_methods_figure,
    save_qualitative_panel,
    write_csv,
    write_markdown_table,
)


MODEL_ORDER = [
    "dirty",
    "tikhonov",
    "baseline_learned",
    "visibility_conditioned",
    "residual_refinement",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default="outputs/paper_final_artifacts")
    parser.add_argument("--output-root", default="outputs")
    return parser.parse_args()


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"None of the expected paths exist: {joined}")


def build_condition_specs(output_root: Path) -> list[ConditionSpec]:
    exp64_summary = _first_existing(
        output_root / "paper_exp64_residual_refine_clean/logs/evaluation_summary.json",
        output_root / "paper_exp64_residual_refine/logs/evaluation_summary.json",
    )
    exp64_figure = _first_existing(
        output_root / "paper_exp64_residual_refine_clean/figures/sample_000.png",
        output_root / "paper_exp64_residual_refine/figures/sample_000.png",
    )
    return [
        ConditionSpec(
            key="default32_seed7",
            title="Default32 (seed 7)",
            summary_path=output_root / "compare_default32_residual_refine_fair8/logs/evaluation_summary.json",
            figure_path=output_root / "compare_default32_residual_refine_fair8/figures/sample_000.png",
        ),
        ConditionSpec(
            key="default32_seed19",
            title="Default32 (seed 19)",
            summary_path=output_root / "paper_default32_seed19_residual_refine/logs/evaluation_summary.json",
            figure_path=output_root / "paper_default32_seed19_residual_refine/figures/sample_000.png",
        ),
        ConditionSpec(
            key="default32_seed31",
            title="Default32 (seed 31)",
            summary_path=output_root / "paper_default32_seed31_residual_refine/logs/evaluation_summary.json",
            figure_path=output_root / "paper_default32_seed31_residual_refine/figures/sample_000.png",
        ),
        ConditionSpec(
            key="noise_high",
            title="High noise",
            summary_path=output_root / "paper_noise_high_residual_refine/logs/evaluation_summary.json",
            figure_path=output_root / "paper_noise_high_residual_refine/figures/sample_000.png",
        ),
        ConditionSpec(
            key="sparse_uv",
            title="Sparse uv",
            summary_path=output_root / "paper_sparse_uv_residual_refine/logs/evaluation_summary.json",
            figure_path=output_root / "paper_sparse_uv_residual_refine/figures/sample_000.png",
        ),
        ConditionSpec(
            key="exp64",
            title="Exp64",
            summary_path=exp64_summary,
            figure_path=exp64_figure,
        ),
    ]


def _condition_row(condition_key: str, condition_title: str, model_key: str, metrics: dict[str, float]) -> list[str]:
    return [
        condition_title,
        MODEL_LABELS[model_key],
        format_value(float(metrics["mse"])),
        format_value(float(metrics["psnr"])),
        format_value(float(metrics["ssim"])),
        format_value(float(metrics["temporal_consistency"])),
        format_value(float(metrics["ring_radius_error"])),
        format_value(float(metrics["hotspot_localization_error"])),
    ]


def write_condition_summary(
    artifact_root: Path,
    spec: ConditionSpec,
    summary: dict[str, dict[str, float]],
    verdict_rows: list[dict[str, object]],
) -> None:
    summary_dir = artifact_root / "summaries"
    relevant_verdicts = [row for row in verdict_rows if row["condition"] == spec.key]
    verdict_payload = {
        row["metric"]: {
            "baseline_3d_unet": row["baseline_3d_unet"],
            "residual_refinement": row["residual_refinement"],
            "verdict_vs_baseline": row["verdict_vs_baseline"],
        }
        for row in relevant_verdicts
    }
    export_payload = {
        "condition": spec.key,
        "title": spec.title,
        "summary": summary,
        "residual_vs_baseline": verdict_payload,
    }
    save_json(summary_dir / f"{spec.key}_summary.json", export_payload)

    csv_rows: list[dict[str, object]] = []
    for model_key in MODEL_ORDER:
        metrics = summary.get(model_key)
        if not metrics:
            continue
        row: dict[str, object] = {
            "model": MODEL_LABELS[model_key],
            **metrics,
        }
        if model_key == "residual_refinement":
            for metric_key, payload in verdict_payload.items():
                row[f"{metric_key}_verdict_vs_baseline"] = payload["verdict_vs_baseline"]
        csv_rows.append(row)
    write_csv(summary_dir / f"{spec.key}_summary.csv", csv_rows)

    markdown_rows = [
        _condition_row(spec.key, spec.title, model_key, summary[model_key])
        for model_key in MODEL_ORDER
        if model_key in summary
    ]
    notes = [
        f"{METRIC_SPECS[metric_key][0]}: {payload['verdict_vs_baseline']}"
        for metric_key, payload in verdict_payload.items()
    ]
    if "uncertainty" in summary:
        notes.append(
            "Residual uncertainty: "
            f"coverage={format_value(float(summary['uncertainty']['empirical_95_coverage']))}, "
            f"error/uncertainty correlation="
            f"{format_value(float(summary['uncertainty']['error_uncertainty_correlation']))}"
        )
    write_markdown_table(
        summary_dir / f"{spec.key}_summary.md",
        headers=["Condition", "Model", "MSE", "PSNR", "SSIM", "Temporal", "Ring", "Hotspot"],
        rows=markdown_rows,
        title=f"{spec.title} Summary",
        notes=notes,
    )


def write_seed_repeat_summary(
    artifact_root: Path,
    aggregate_rows: list[dict[str, object]],
    verdict_counts: dict[str, dict[str, int]],
) -> None:
    summary_dir = artifact_root / "summaries"
    save_json(
        summary_dir / "default32_seed_repeats_summary.json",
        {
            "condition": "default32_seed_repeats",
            "rows": aggregate_rows,
            "residual_vs_baseline_verdict_counts": verdict_counts,
        },
    )
    write_csv(summary_dir / "default32_seed_repeats_summary.csv", aggregate_rows)

    markdown_rows = []
    for row in aggregate_rows:
        markdown_rows.append(
            [
                str(row["model_label"]),
                format_mean_std(float(row["mse_mean"]), float(row["mse_std"])),
                format_mean_std(float(row["psnr_mean"]), float(row["psnr_std"])),
                format_mean_std(float(row["ssim_mean"]), float(row["ssim_std"])),
                format_mean_std(
                    float(row["temporal_consistency_mean"]),
                    float(row["temporal_consistency_std"]),
                ),
                format_mean_std(
                    float(row["ring_radius_error_mean"]),
                    float(row["ring_radius_error_std"]),
                ),
                format_mean_std(
                    float(row["hotspot_localization_error_mean"]),
                    float(row["hotspot_localization_error_std"]),
                ),
            ]
        )
    notes = [
        (
            f"Residual vs baseline on {METRIC_SPECS[metric_key][0]} across seeds: "
            f"{counts['win']} wins, {counts['tie']} ties, {counts['loss']} losses"
        )
        for metric_key, counts in verdict_counts.items()
    ]
    residual_row = next(row for row in aggregate_rows if row["model"] == "residual_refinement")
    notes.append(
        "Residual uncertainty across seed repeats: "
        f"coverage={format_mean_std(float(residual_row['empirical_95_coverage_mean']), float(residual_row['empirical_95_coverage_std']))}, "
        f"error/uncertainty correlation="
        f"{format_mean_std(float(residual_row['error_uncertainty_correlation_mean']), float(residual_row['error_uncertainty_correlation_std']))}"
    )
    write_markdown_table(
        summary_dir / "default32_seed_repeats_summary.md",
        headers=["Model", "MSE", "PSNR", "SSIM", "Temporal", "Ring", "Hotspot"],
        rows=markdown_rows,
        title="Default32 Seed Repeats Summary",
        notes=notes,
    )


def main() -> None:
    args = parse_args()
    output_root = ROOT / args.output_root
    artifact_root = ROOT / args.artifact_root
    tables_dir = artifact_root / "tables"
    figures_dir = artifact_root / "figures"
    summaries_dir = artifact_root / "summaries"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    condition_specs = build_condition_specs(output_root=output_root)
    loaded, all_rows = collect_metric_rows(condition_specs=condition_specs, model_order=MODEL_ORDER)

    seed_keys = ["default32_seed7", "default32_seed19", "default32_seed31"]
    aggregate_rows, verdict_counts = aggregate_seed_repeats(
        loaded=loaded,
        seed_keys=seed_keys,
        model_order=MODEL_ORDER,
    )
    write_seed_repeat_summary(
        artifact_root=artifact_root,
        aggregate_rows=aggregate_rows,
        verdict_counts=verdict_counts,
    )

    save_json(tables_dir / "main_results_table.json", {"rows": aggregate_rows})
    write_csv(tables_dir / "main_results_table.csv", aggregate_rows)
    write_markdown_table(
        tables_dir / "main_results_table.md",
        headers=["Model", "MSE", "PSNR", "SSIM", "Temporal", "Ring", "Hotspot"],
        rows=[
            [
                str(row["model_label"]),
                format_mean_std(float(row["mse_mean"]), float(row["mse_std"])),
                format_mean_std(float(row["psnr_mean"]), float(row["psnr_std"])),
                format_mean_std(float(row["ssim_mean"]), float(row["ssim_std"])),
                format_mean_std(
                    float(row["temporal_consistency_mean"]),
                    float(row["temporal_consistency_std"]),
                ),
                format_mean_std(
                    float(row["ring_radius_error_mean"]),
                    float(row["ring_radius_error_std"]),
                ),
                format_mean_std(
                    float(row["hotspot_localization_error_mean"]),
                    float(row["hotspot_localization_error_std"]),
                ),
            ]
            for row in aggregate_rows
        ],
        title="Main Results Table",
        notes=["Mean ± std over default32 seeds 7, 19, and 31."],
    )

    robustness_specs = [spec for spec in condition_specs if spec.key in {"noise_high", "sparse_uv", "exp64"}]
    robustness_rows: list[dict[str, object]] = []
    for spec in robustness_specs:
        summary = loaded[spec.key]
        for model_key in MODEL_ORDER:
            if model_key not in summary:
                continue
            row = {
                "condition": spec.key,
                "condition_title": spec.title,
                "model": model_key,
                "model_label": MODEL_LABELS[model_key],
                **summary[model_key],
            }
            if model_key == "residual_refinement":
                baseline_metrics = summary["baseline_learned"]
                for metric_key, (_, direction) in METRIC_SPECS.items():
                    row[f"{metric_key}_verdict_vs_baseline"] = compute_verdict(
                        baseline_value=float(baseline_metrics[metric_key]),
                        candidate_value=float(summary[model_key][metric_key]),
                        direction=direction,
                    )
            robustness_rows.append(row)
    save_json(tables_dir / "robustness_table.json", {"rows": robustness_rows})
    write_csv(tables_dir / "robustness_table.csv", robustness_rows)
    write_markdown_table(
        tables_dir / "robustness_table.md",
        headers=["Condition", "Model", "MSE", "PSNR", "SSIM", "Temporal", "Ring", "Hotspot"],
        rows=[
            _condition_row(str(row["condition"]), str(row["condition_title"]), str(row["model"]), row)
            for row in robustness_rows
        ],
        title="Robustness Table",
        notes=["Rows cover the harder conditions: high noise, sparse uv, and exp64."],
    )

    uncertainty_rows = []
    residual_seed_row = next(row for row in aggregate_rows if row["model"] == "residual_refinement")
    uncertainty_rows.append(
        {
            "condition": "default32_seed_repeats",
            "condition_title": "Default32 seed repeats",
            "empirical_95_coverage_mean": residual_seed_row["empirical_95_coverage_mean"],
            "empirical_95_coverage_std": residual_seed_row["empirical_95_coverage_std"],
            "error_uncertainty_correlation_mean": residual_seed_row["error_uncertainty_correlation_mean"],
            "error_uncertainty_correlation_std": residual_seed_row["error_uncertainty_correlation_std"],
        }
    )
    for spec in robustness_specs:
        uncertainty = loaded[spec.key]["uncertainty"]
        uncertainty_rows.append(
            {
                "condition": spec.key,
                "condition_title": spec.title,
                "empirical_95_coverage_mean": float(uncertainty["empirical_95_coverage"]),
                "empirical_95_coverage_std": 0.0,
                "error_uncertainty_correlation_mean": float(uncertainty["error_uncertainty_correlation"]),
                "error_uncertainty_correlation_std": 0.0,
            }
        )
    save_json(tables_dir / "uncertainty_summary_table.json", {"rows": uncertainty_rows})
    write_csv(tables_dir / "uncertainty_summary_table.csv", uncertainty_rows)
    write_markdown_table(
        tables_dir / "uncertainty_summary_table.md",
        headers=["Condition", "Coverage", "Error/Uncertainty Correlation"],
        rows=[
            [
                str(row["condition_title"]),
                format_mean_std(
                    float(row["empirical_95_coverage_mean"]),
                    float(row["empirical_95_coverage_std"]),
                ),
                format_mean_std(
                    float(row["error_uncertainty_correlation_mean"]),
                    float(row["error_uncertainty_correlation_std"]),
                ),
            ]
            for row in uncertainty_rows
        ],
        title="Uncertainty Summary Table",
        notes=["Uncertainty is useful but conservative; near-saturated coverage should not be interpreted as full calibration."],
    )

    verdict_rows = build_condition_verdict_rows(
        loaded=loaded,
        condition_keys=[spec.key for spec in condition_specs],
    )
    save_json(tables_dir / "residual_vs_baseline_verdicts.json", {"rows": verdict_rows})
    write_csv(tables_dir / "residual_vs_baseline_verdicts.csv", verdict_rows)
    write_markdown_table(
        tables_dir / "residual_vs_baseline_verdicts.md",
        headers=["Condition", "Metric", "Baseline 3D U-Net", "Residual Refinement", "Verdict"],
        rows=[
            [
                row["condition"],
                row["metric_label"],
                format_value(float(row["baseline_3d_unet"])),
                format_value(float(row["residual_refinement"])),
                str(row["verdict_vs_baseline"]),
            ]
            for row in verdict_rows
        ],
        title="Residual Refinement vs Baseline Verdicts",
    )

    for spec in condition_specs:
        write_condition_summary(
            artifact_root=artifact_root,
            spec=spec,
            summary=loaded[spec.key],
            verdict_rows=verdict_rows,
        )

    qualitative_specs = [
        next(spec for spec in condition_specs if spec.key == key)
        for key in ["default32_seed7", "noise_high", "sparse_uv", "exp64"]
    ]
    save_qualitative_panel(
        condition_specs=qualitative_specs,
        output_path=figures_dir / "qualitative_comparison_panel.png",
    )
    save_methods_figure(
        png_path=figures_dir / "residual_refinement_schematic.png",
        svg_path=figures_dir / "residual_refinement_schematic.svg",
    )

    print(f"Paper artifacts written to {artifact_root}")


if __name__ == "__main__":
    main()
