"""Artifact generation for the final MNRAS-strengthening EMC paper path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import csv
import json
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from dynadiff_vlbi.evaluation.ccrr_artifacts import paired_bootstrap_stats
from dynadiff_vlbi.evaluation.paper_artifacts import format_value, save_json, write_csv, write_markdown_table


MODEL_LABELS: dict[str, str] = {
    "dirty": "Dirty",
    "tikhonov": "Tikhonov",
    "baseline_learned": "Baseline 3D U-Net",
    "visibility_conditioned": "Standalone Visibility",
    "residual_refinement": "Residual Refinement",
    "ccrr": "CCRR",
    "emc": "EMC",
    "emc_no_dc": "EMC without DC",
    "emc_with_closure": "EMC + closure auxiliary",
    "emc_no_metadata": "EMC without metadata",
    "emc_no_uncertainty": "EMC without uncertainty",
}

MODEL_COLORS: dict[str, str] = {
    "dirty": "#6b7280",
    "tikhonov": "#b45309",
    "baseline_learned": "#111827",
    "visibility_conditioned": "#0f766e",
    "residual_refinement": "#2563eb",
    "ccrr": "#7c3aed",
    "emc": "#dc2626",
    "emc_no_dc": "#0284c7",
    "emc_with_closure": "#65a30d",
    "emc_no_metadata": "#ea580c",
    "emc_no_uncertainty": "#be185d",
}

LEARNED_MODEL_ORDER = [
    "baseline_learned",
    "residual_refinement",
    "ccrr",
    "emc",
]

REAL_MODEL_ORDER = [
    "dirty",
    "tikhonov",
    "baseline_learned",
    "visibility_conditioned",
    "residual_refinement",
    "ccrr",
    "emc",
]

METRIC_DIRECTIONS = {
    "heldout_visibility_rmse": "lower",
    "observed_visibility_rmse": "lower",
    "support_visibility_rmse": "lower",
    "heldout_closure_phase_mae": "lower",
    "ssim": "higher",
    "temporal_consistency": "lower",
}


@dataclass(frozen=True)
class SummarySpec:
    """One named summary file used to assemble MNRAS-facing artifacts."""

    key: str
    title: str
    summary_path: Path


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON artifact."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Load one CSV file into dictionaries."""

    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: Any) -> float:
    if value in {None, "", "nan", "NaN"}:
        return float("nan")
    return float(value)


def _fmt(value: float) -> str:
    return "n/a" if math.isnan(float(value)) else format_value(float(value))


def _descending_support_tags(summary: dict[str, Any]) -> list[str]:
    return sorted(summary["support_fractions"].keys(), key=lambda item: int(item), reverse=True)


def _model_metrics(summary: dict[str, Any], support_tag: str, model_key: str) -> dict[str, float]:
    resolved_key = model_key
    models = summary["support_fractions"][support_tag]["models"]
    if resolved_key not in models and resolved_key == "emc_with_closure" and "emc_no_closure" in models:
        resolved_key = "emc_no_closure"
    return {
        key: _safe_float(value)
        for key, value in models[resolved_key].items()
    }


def _mean_support_metric(summary: dict[str, Any], model_key: str, metric_key: str) -> float:
    values = [
        _model_metrics(summary, support_tag, model_key).get(metric_key, float("nan"))
        for support_tag in _descending_support_tags(summary)
    ]
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if not np.all(np.isnan(array)) else float("nan")


def build_real_data_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one compact row per support fraction for public real-data validation."""

    rows: list[dict[str, Any]] = []
    for support_tag in _descending_support_tags(summary):
        support_payload = summary["support_fractions"][support_tag]
        model_metrics = support_payload["models"]
        best_heldout_key = min(
            REAL_MODEL_ORDER,
            key=lambda key: _safe_float(model_metrics[key]["heldout_visibility_rmse"]),
        )
        best_observed_key = min(
            REAL_MODEL_ORDER,
            key=lambda key: _safe_float(model_metrics[key]["observed_visibility_rmse"]),
        )
        rows.append(
            {
                "support_fraction_tag": support_tag,
                "support_fraction": float(support_payload["support_fraction"]),
                "mean_support_coefficients": float(support_payload["mean_support_coefficients"]),
                "mean_target_coefficients": float(support_payload["mean_target_coefficients"]),
                "mean_all_target_triangles": float(support_payload["mean_all_target_triangles"]),
                "mean_mixed_triangles": float(support_payload["mean_mixed_triangles"]),
                "best_heldout_model": best_heldout_key,
                "best_heldout_model_label": MODEL_LABELS[best_heldout_key],
                "best_heldout_visibility_rmse": _safe_float(model_metrics[best_heldout_key]["heldout_visibility_rmse"]),
                "best_observed_model": best_observed_key,
                "best_observed_model_label": MODEL_LABELS[best_observed_key],
                "best_observed_visibility_rmse": _safe_float(model_metrics[best_observed_key]["observed_visibility_rmse"]),
                "baseline_heldout_visibility_rmse": _safe_float(model_metrics["baseline_learned"]["heldout_visibility_rmse"]),
                "residual_heldout_visibility_rmse": _safe_float(model_metrics["residual_refinement"]["heldout_visibility_rmse"]),
                "ccrr_heldout_visibility_rmse": _safe_float(model_metrics["ccrr"]["heldout_visibility_rmse"]),
                "emc_heldout_visibility_rmse": _safe_float(model_metrics["emc"]["heldout_visibility_rmse"]),
                "dirty_heldout_visibility_rmse": _safe_float(model_metrics["dirty"]["heldout_visibility_rmse"]),
                "tikhonov_heldout_visibility_rmse": _safe_float(model_metrics["tikhonov"]["heldout_visibility_rmse"]),
                "emc_observed_visibility_rmse": _safe_float(model_metrics["emc"]["observed_visibility_rmse"]),
                "emc_support_visibility_rmse": _safe_float(model_metrics["emc"]["support_visibility_rmse"]),
                "emc_heldout_closure_phase_mae": _safe_float(model_metrics["emc"]["heldout_closure_phase_mae"]),
            }
        )
    return rows


def write_real_data_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write the public M87 validation table."""

    markdown_rows = [
        [
            str(row["support_fraction_tag"]),
            _fmt(float(row["mean_target_coefficients"])),
            _fmt(float(row["mean_all_target_triangles"])),
            str(row["best_heldout_model_label"]),
            _fmt(float(row["best_heldout_visibility_rmse"])),
            _fmt(float(row["emc_heldout_visibility_rmse"])),
            _fmt(float(row["ccrr_heldout_visibility_rmse"])),
            _fmt(float(row["baseline_heldout_visibility_rmse"])),
            _fmt(float(row["tikhonov_heldout_visibility_rmse"])),
            str(row["best_observed_model_label"]),
        ]
        for row in rows
    ]
    write_markdown_table(
        path,
        headers=[
            "Support (%)",
            "Held-out coeffs",
            "All-target triangles",
            "Best held-out model",
            "Best held-out VisRMSE",
            "EMC",
            "CCRR",
            "Baseline",
            "Tikhonov",
            "Best observed model",
        ],
        rows=markdown_rows,
        title="Public M87 Observation-Domain Validation",
        notes=[
            "The public validation uses the official EHT 2017 M87 calibrated Stokes I release and reports observation-domain metrics only; there is no image-domain ground truth.",
            "Held-out closure is reported in the main summary JSON, but the table also exposes all-target triangle support because closure support is sparse at 80 per cent support.",
        ],
    )


def save_real_data_support_figure(
    *,
    summary: dict[str, Any],
    output_png: str | Path,
    output_svg: str | Path,
) -> None:
    """Save the public M87 support-fraction figure."""

    x_values = [int(tag) for tag in _descending_support_tags(summary)]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    fig.patch.set_facecolor("white")

    for model_key in REAL_MODEL_ORDER:
        heldout_values = [_model_metrics(summary, str(tag), model_key)["heldout_visibility_rmse"] for tag in x_values]
        observed_values = [_model_metrics(summary, str(tag), model_key)["observed_visibility_rmse"] for tag in x_values]
        axes[0].plot(
            x_values,
            heldout_values,
            marker="o",
            linewidth=2.0,
            color=MODEL_COLORS[model_key],
            label=MODEL_LABELS[model_key],
        )
        axes[1].plot(
            x_values,
            observed_values,
            marker="o",
            linewidth=2.0,
            color=MODEL_COLORS[model_key],
            label=MODEL_LABELS[model_key],
        )

    axes[0].set_title("Held-out real measurements", fontsize=12)
    axes[1].set_title("All observed measurements", fontsize=12)
    for axis in axes:
        axis.set_xlabel("Support fraction (%)", fontsize=10)
        axis.set_ylabel("Visibility RMSE", fontsize=10)
        axis.grid(alpha=0.22)
        axis.set_xticks(x_values)
    legend_handles = [
        Line2D([0], [0], color=MODEL_COLORS[model_key], lw=2.0, marker="o", label=MODEL_LABELS[model_key])
        for model_key in REAL_MODEL_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle("Public EHT M87 observation-domain validation", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(Path(output_svg), bbox_inches="tight")
    plt.close(fig)


def build_ablation_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one compact row per ablation variant, averaged over support fractions."""

    descriptions = {
        "ccrr": "No support-target holdout objective (measurement-consistent predecessor).",
        "emc": "Full EMC objective.",
        "emc_no_dc": "Remove support-set data consistency.",
        "emc_with_closure": "Reintroduce closure-aware supervision as an auxiliary loss.",
        "emc_no_metadata": "Remove station/baseline metadata conditioning.",
        "emc_no_uncertainty": "Remove uncertainty head.",
    }
    order = ["ccrr", "emc", "emc_no_dc", "emc_with_closure", "emc_no_metadata", "emc_no_uncertainty"]
    full_emc_heldout = _mean_support_metric(summary, "emc", "heldout_visibility_rmse")
    rows: list[dict[str, Any]] = []
    for model_key in order:
        heldout = _mean_support_metric(summary, model_key, "heldout_visibility_rmse")
        rows.append(
            {
                "model": model_key,
                "model_label": MODEL_LABELS[model_key],
                "description": descriptions[model_key],
                "mean_heldout_visibility_rmse": heldout,
                "delta_vs_emc_heldout_visibility_rmse": heldout - full_emc_heldout,
                "mean_heldout_closure_phase_mae": _mean_support_metric(summary, model_key, "heldout_closure_phase_mae"),
                "mean_support_visibility_rmse": _mean_support_metric(summary, model_key, "support_visibility_rmse"),
                "mean_ssim": _mean_support_metric(summary, model_key, "ssim"),
                "mean_temporal_consistency": _mean_support_metric(summary, model_key, "temporal_consistency"),
            }
        )
    return rows


def write_ablation_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write the compact EMC component ablation table."""

    markdown_rows = [
        [
            str(row["model_label"]),
            _fmt(float(row["mean_heldout_visibility_rmse"])),
            _fmt(float(row["delta_vs_emc_heldout_visibility_rmse"])),
            _fmt(float(row["mean_heldout_closure_phase_mae"])),
            _fmt(float(row["mean_support_visibility_rmse"])),
            _fmt(float(row["mean_ssim"])),
            _fmt(float(row["mean_temporal_consistency"])),
            str(row["description"]),
        ]
        for row in rows
    ]
    write_markdown_table(
        path,
        headers=[
            "Variant",
            "Mean held-out VisRMSE",
            "Δ vs EMC",
            "Mean held-out closure",
            "Mean support VisRMSE",
            "Mean SSIM",
            "Mean temporal",
            "Interpretation",
        ],
        rows=markdown_rows,
        title="EMC Component Ablations",
        notes=[
            "Averages are taken across the 80/60/40/20 support sweep on the default32 baseline-track protocol.",
            "Lower is better for visibility and temporal metrics; higher is better for SSIM.",
        ],
    )


def _collect_paired_values(
    *,
    per_sample_paths: list[Path],
    candidate_key: str,
    reference_key: str,
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_values: list[float] = []
    reference_values: list[float] = []
    for csv_path in per_sample_paths:
        rows = load_csv_rows(csv_path)
        grouped: dict[tuple[str, str], dict[str, float]] = {}
        for row in rows:
            sample_id = str(row.get("sample_id", row.get("sample_index", "")))
            support_tag = str(row.get("support_fraction_tag", ""))
            metric_value = _safe_float(row.get(metric_key))
            if math.isnan(metric_value):
                continue
            grouped.setdefault((support_tag, sample_id), {})[str(row["model"])] = metric_value
        for key, model_map in grouped.items():
            if candidate_key not in model_map or reference_key not in model_map:
                continue
            candidate_values.append(model_map[candidate_key])
            reference_values.append(model_map[reference_key])
    return np.asarray(candidate_values, dtype=np.float64), np.asarray(reference_values, dtype=np.float64)


def build_statistical_rows(
    *,
    benchmark_per_sample_paths: list[Path],
    real_data_per_sample_paths: list[Path],
) -> list[dict[str, Any]]:
    """Build paired-bootstrap attribution rows for synthetic and public real-data evidence."""

    specs = [
        ("Synthetic benchmark breadth", benchmark_per_sample_paths, "emc", "ccrr", "heldout_visibility_rmse"),
        ("Synthetic benchmark breadth", benchmark_per_sample_paths, "emc", "residual_refinement", "heldout_visibility_rmse"),
        ("Synthetic benchmark breadth", benchmark_per_sample_paths, "emc", "residual_refinement", "ssim"),
        ("Synthetic benchmark breadth", benchmark_per_sample_paths, "emc", "baseline_learned", "heldout_visibility_rmse"),
        ("Public M87 validation", real_data_per_sample_paths, "emc", "ccrr", "heldout_visibility_rmse"),
        ("Public M87 validation", real_data_per_sample_paths, "emc", "baseline_learned", "heldout_visibility_rmse"),
        ("Public M87 validation", real_data_per_sample_paths, "emc", "tikhonov", "heldout_visibility_rmse"),
        ("Public M87 validation", real_data_per_sample_paths, "emc", "ccrr", "observed_visibility_rmse"),
    ]
    rows: list[dict[str, Any]] = []
    for cohort, paths, candidate_key, reference_key, metric_key in specs:
        candidate, reference = _collect_paired_values(
            per_sample_paths=paths,
            candidate_key=candidate_key,
            reference_key=reference_key,
            metric_key=metric_key,
        )
        if candidate.size == 0:
            continue
        direction = METRIC_DIRECTIONS[metric_key]
        stats = paired_bootstrap_stats(candidate, reference, direction=direction)
        rows.append(
            {
                "cohort": cohort,
                "comparison": f"{MODEL_LABELS[candidate_key]} vs {MODEL_LABELS[reference_key]}",
                "metric": metric_key,
                "n_pairs": int(candidate.size),
                "candidate_mean": float(np.mean(candidate)),
                "reference_mean": float(np.mean(reference)),
                **stats,
            }
        )
    return rows


def write_statistical_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact paired-bootstrap robustness table."""

    markdown_rows = [
        [
            str(row["cohort"]),
            str(row["comparison"]),
            str(row["metric"]),
            str(row["n_pairs"]),
            _fmt(float(row["candidate_mean"])),
            _fmt(float(row["reference_mean"])),
            _fmt(float(row["mean_delta"])),
            f"[{_fmt(float(row['ci_low']))}, {_fmt(float(row['ci_high']))}]",
            _fmt(float(row["win_rate"])),
            _fmt(float(row["p_value"])),
        ]
        for row in rows
    ]
    write_markdown_table(
        path,
        headers=[
            "Cohort",
            "Comparison",
            "Metric",
            "Pairs",
            "Candidate mean",
            "Reference mean",
            "Mean delta",
            "95% CI",
            "Win rate",
            "p-value",
        ],
        rows=markdown_rows,
        title="Paired Bootstrap Robustness Summary",
        notes=[
            "Mean deltas are direction-aware: positive means the candidate is better under the metric direction.",
            "Synthetic rows pool sample-level comparisons across the full three-family benchmark breadth and the four support fractions; public M87 rows pool the four support fractions of the real-data validation.",
        ],
    )


def build_tradeoff_rows(specs: list[SummarySpec]) -> list[dict[str, Any]]:
    """Build a compact table for the structure-versus-measurement trade-off."""

    rows: list[dict[str, Any]] = []
    for spec in specs:
        summary = load_json(spec.summary_path)
        for model_key in LEARNED_MODEL_ORDER:
            rows.append(
                {
                    "condition": spec.key,
                    "condition_title": spec.title,
                    "model": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    "mean_heldout_visibility_rmse": _mean_support_metric(summary, model_key, "heldout_visibility_rmse"),
                    "mean_ssim": _mean_support_metric(summary, model_key, "ssim"),
                    "mean_temporal_consistency": _mean_support_metric(summary, model_key, "temporal_consistency"),
                }
            )
    return rows


def write_tradeoff_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write the compact trade-off table."""

    markdown_rows = [
        [
            str(row["condition_title"]),
            str(row["model_label"]),
            _fmt(float(row["mean_heldout_visibility_rmse"])),
            _fmt(float(row["mean_ssim"])),
            _fmt(float(row["mean_temporal_consistency"])),
        ]
        for row in rows
    ]
    write_markdown_table(
        path,
        headers=["Condition", "Model", "Mean held-out VisRMSE", "Mean SSIM", "Mean temporal"],
        rows=markdown_rows,
        title="Structure-Versus-Measurement Trade-off",
        notes=[
            "This table summarizes the central trade-off discussed in the manuscript: EMC improves held-out measurement recovery, while residual refinement often remains stronger on structural fidelity.",
            "Means are taken across the support sweep within each condition.",
        ],
    )


def save_tradeoff_figure(
    *,
    specs: list[SummarySpec],
    output_png: str | Path,
    output_svg: str | Path,
) -> None:
    """Save a Pareto-style measurement-versus-structure figure."""

    support_markers = {"80": "o", "60": "s", "40": "^", "20": "D"}
    fig, axes = plt.subplots(1, len(specs), figsize=(12.2, 4.6), sharey=True)
    if len(specs) == 1:
        axes = np.asarray([axes])
    fig.patch.set_facecolor("white")

    for axis, spec in zip(axes, specs):
        summary = load_json(spec.summary_path)
        support_tags = _descending_support_tags(summary)
        for model_key in LEARNED_MODEL_ORDER:
            x_values = [_model_metrics(summary, support_tag, model_key)["heldout_visibility_rmse"] for support_tag in support_tags]
            y_values = [_model_metrics(summary, support_tag, model_key)["ssim"] for support_tag in support_tags]
            axis.plot(
                x_values,
                y_values,
                linewidth=1.8,
                color=MODEL_COLORS[model_key],
                alpha=0.9,
            )
            for support_tag, x_value, y_value in zip(support_tags, x_values, y_values):
                axis.scatter(
                    [x_value],
                    [y_value],
                    s=56,
                    marker=support_markers[support_tag],
                    color=MODEL_COLORS[model_key],
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=3,
                )
        axis.set_title(spec.title, fontsize=12)
        axis.set_xlabel("Held-out visibility RMSE", fontsize=10)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("SSIM", fontsize=10)

    model_handles = [
        Line2D([0], [0], color=MODEL_COLORS[model_key], lw=2.0, label=MODEL_LABELS[model_key])
        for model_key in LEARNED_MODEL_ORDER
    ]
    support_handles = [
        Line2D([0], [0], marker=support_markers[support_tag], color="#111827", lw=0.0, markersize=7, label=f"{support_tag}%")
        for support_tag in ["80", "60", "40", "20"]
    ]
    fig.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.30, 0.89),
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    fig.legend(
        handles=support_handles,
        loc="upper center",
        bbox_to_anchor=(0.80, 0.89),
        ncol=4,
        fontsize=8,
        frameon=False,
        title="Support",
    )
    fig.suptitle("Measurement-versus-structure trade-off across support fractions", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(Path(output_svg), bbox_inches="tight")
    plt.close(fig)


def build_claim_to_evidence_rows(
    *,
    benchmark_matrix_path: Path,
    real_data_table_path: Path,
    bootstrap_table_path: Path,
    ablation_table_path: Path,
    tradeoff_table_path: Path,
) -> list[dict[str, str]]:
    """Map the strengthened manuscript claims to exact artifact files."""

    return [
        {
            "claim": "EMC remains the strongest learned model on held-out visibility RMSE across the synthetic structured benchmark breadth.",
            "evidence": str(benchmark_matrix_path.resolve()),
        },
        {
            "claim": "On public calibrated M87 measurements, the EMC protocol remains meaningful, but EMC is not the best held-out model and the real-data transfer gap stays visible.",
            "evidence": str(real_data_table_path.resolve()),
        },
        {
            "claim": "The benchmark-first claim is statistically supported on synthetic held-out visibility RMSE, while real-data superiority is not claimed.",
            "evidence": str(bootstrap_table_path.resolve()),
        },
        {
            "claim": "Support-target training matters more than closure-aware supervision in the current synthetic regime, and closure remains a secondary ingredient.",
            "evidence": str(ablation_table_path.resolve()),
        },
        {
            "claim": "The central scientific trade-off is between held-out measurement recovery and structural fidelity rather than universal dominance on every metric.",
            "evidence": str(tradeoff_table_path.resolve()),
        },
    ]
