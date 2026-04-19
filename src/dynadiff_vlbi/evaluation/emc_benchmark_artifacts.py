"""Benchmark-first artifact generation for the EMC paper path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import math

import matplotlib.pyplot as plt
import numpy as np

from dynadiff_vlbi.evaluation.emc_artifacts import (
    FULL_MODEL_ORDER,
    LEARNED_MODEL_ORDER,
    MODEL_COLORS,
    MODEL_LABELS,
    load_json,
    save_emc_qualitative_figure,
)
from dynadiff_vlbi.evaluation.paper_artifacts import format_value, save_json, write_csv, write_markdown_table


@dataclass(frozen=True)
class BenchmarkProtocolSpec:
    """One benchmark protocol output used to assemble benchmark-first artifacts."""

    key: str
    title: str
    family_label: str
    family_description: str
    protocol_dir: Path
    condition_group: str

    @property
    def summary_path(self) -> Path:
        return self.protocol_dir / "logs" / "emc_protocol_summary.json"

    def prediction_path(self, support_fraction_tag: str) -> Path:
        return self.protocol_dir / "predictions" / f"support_{support_fraction_tag}.npz"

    def per_sample_path(self, support_fraction_tag: str) -> Path:
        return self.protocol_dir / "logs" / f"per_sample_support_{support_fraction_tag}.csv"


def _fmt(value: float) -> str:
    return "n/a" if math.isnan(float(value)) else format_value(float(value))


def _support_tags(summary: dict[str, Any]) -> list[str]:
    return sorted(summary["support_fractions"].keys(), key=lambda item: int(item))


def _model_metrics(summary: dict[str, Any], support_tag: str, model_key: str) -> dict[str, float]:
    models = summary["support_fractions"][support_tag]["models"]
    if model_key not in models:
        reference_metrics = next(iter(models.values()))
        return {key: float("nan") for key in reference_metrics}
    return {
        key: float(value)
        for key, value in models[model_key].items()
    }


def benchmark_long_rows(specs: list[BenchmarkProtocolSpec]) -> list[dict[str, Any]]:
    """Flatten benchmark summaries into long-form rows."""

    rows: list[dict[str, Any]] = []
    for spec in specs:
        summary = load_json(spec.summary_path)
        holdout = summary.get("holdout", {})
        for support_tag in _support_tags(summary):
            support_payload = summary["support_fractions"][support_tag]
            for model_key in FULL_MODEL_ORDER:
                metrics = _model_metrics(summary, support_tag, model_key)
                rows.append(
                    {
                        "condition": spec.key,
                        "condition_title": spec.title,
                        "condition_group": spec.condition_group,
                        "family_label": spec.family_label,
                        "family_description": spec.family_description,
                        "support_fraction_tag": support_tag,
                        "support_fraction": float(support_payload["support_fraction"]),
                        "holdout_strategy": holdout.get("strategy"),
                        "holdout_strategy_label": holdout.get("label"),
                        "mean_target_unit_count": float(support_payload.get("mean_target_unit_count", float("nan"))),
                        "mean_support_unit_count": float(support_payload.get("mean_support_unit_count", float("nan"))),
                        "model": model_key,
                        "model_label": MODEL_LABELS[model_key],
                        **metrics,
                    }
                )
    return rows


def build_family_matrix_rows(specs: list[BenchmarkProtocolSpec]) -> list[dict[str, Any]]:
    """Build one matrix row per holdout family and support fraction."""

    rows: list[dict[str, Any]] = []
    for spec in specs:
        summary = load_json(spec.summary_path)
        for support_tag in _support_tags(summary):
            emc_metrics = _model_metrics(summary, support_tag, "emc")
            baseline_metrics = _model_metrics(summary, support_tag, "baseline_learned")
            residual_metrics = _model_metrics(summary, support_tag, "residual_refinement")
            ccrr_metrics = _model_metrics(summary, support_tag, "ccrr")
            dps_metrics = _model_metrics(summary, support_tag, "dps")
            rows.append(
                {
                    "condition": spec.key,
                    "family": spec.family_label,
                    "support_fraction_tag": support_tag,
                    "support_fraction": float(summary["support_fractions"][support_tag]["support_fraction"]),
                    "holdout_strategy_label": summary.get("holdout", {}).get("label", spec.family_label),
                    "mean_target_unit_count": float(summary["support_fractions"][support_tag].get("mean_target_unit_count", float("nan"))),
                    "emc_heldout_visibility_rmse": float(emc_metrics["heldout_visibility_rmse"]),
                    "emc_heldout_closure_phase_mae": float(emc_metrics["heldout_closure_phase_mae"]),
                    "emc_mse": float(emc_metrics["mse"]),
                    "emc_ssim": float(emc_metrics["ssim"]),
                    "baseline_heldout_visibility_rmse": float(baseline_metrics["heldout_visibility_rmse"]),
                    "residual_heldout_visibility_rmse": float(residual_metrics["heldout_visibility_rmse"]),
                    "ccrr_heldout_visibility_rmse": float(ccrr_metrics["heldout_visibility_rmse"]),
                    "dps_heldout_visibility_rmse": float(dps_metrics["heldout_visibility_rmse"]),
                    "emc_coverage_90": float("nan"),
                    "emc_miw": float("nan"),
                    "emc_vs_baseline": "win"
                    if float(emc_metrics["heldout_visibility_rmse"]) < float(baseline_metrics["heldout_visibility_rmse"])
                    else "loss",
                    "emc_vs_residual": "win"
                    if float(emc_metrics["heldout_visibility_rmse"]) < float(residual_metrics["heldout_visibility_rmse"])
                    else "loss",
                    "emc_vs_ccrr": "win"
                    if float(emc_metrics["heldout_visibility_rmse"]) < float(ccrr_metrics["heldout_visibility_rmse"])
                    else "loss",
                }
            )
    return rows


def build_realism_rows(spec: BenchmarkProtocolSpec) -> list[dict[str, Any]]:
    """Build compact realism-track rows."""

    summary = load_json(spec.summary_path)
    rows: list[dict[str, Any]] = []
    for support_tag in _support_tags(summary):
        for model_key in FULL_MODEL_ORDER:
            metrics = _model_metrics(summary, support_tag, model_key)
            rows.append(
                {
                    "support_fraction_tag": support_tag,
                    "model": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    "heldout_visibility_rmse": float(metrics["heldout_visibility_rmse"]),
                    "heldout_closure_phase_mae": float(metrics["heldout_closure_phase_mae"]),
                    "mse": float(metrics["mse"]),
                    "ssim": float(metrics["ssim"]),
                    "temporal_consistency": float(metrics["temporal_consistency"]),
                }
            )
    return rows


def save_family_support_figure(
    *,
    specs: list[BenchmarkProtocolSpec],
    output_png: str | Path,
    output_svg: str | Path,
) -> None:
    """Save the central benchmark support-fraction figure across holdout families."""

    fig, axes = plt.subplots(1, len(specs), figsize=(15.0, 4.6), sharey=True)
    if len(specs) == 1:
        axes = np.asarray([axes])
    fig.patch.set_facecolor("white")

    for axis, spec in zip(axes, specs):
        summary = load_json(spec.summary_path)
        x_values = [int(tag) for tag in _support_tags(summary)]
        for model_key in LEARNED_MODEL_ORDER:
            heldout_vis = [_model_metrics(summary, str(tag), model_key)["heldout_visibility_rmse"] for tag in x_values]
            if all(math.isnan(float(value)) for value in heldout_vis):
                continue
            axis.plot(
                x_values,
                heldout_vis,
                marker="o",
                linewidth=2.1,
                color=MODEL_COLORS[model_key],
                label=MODEL_LABELS[model_key],
            )
        axis.set_title(spec.family_label, fontsize=12)
        axis.set_xlabel("Support fraction (%)", fontsize=10)
        axis.grid(alpha=0.22)
        axis.set_xticks(x_values)
    axes[0].set_ylabel("Held-out visibility RMSE", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=8.5, frameon=False)
    fig.suptitle(
        "EMC benchmark: held-out measurement recovery across structured holdout families",
        fontsize=14,
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(Path(output_svg), bbox_inches="tight")
    plt.close(fig)


def save_realism_support_figure(
    *,
    spec: BenchmarkProtocolSpec,
    output_png: str | Path,
    output_svg: str | Path,
) -> None:
    """Save a realism-track summary figure."""

    summary = load_json(spec.summary_path)
    x_values = [int(tag) for tag in _support_tags(summary)]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    fig.patch.set_facecolor("white")
    for model_key in LEARNED_MODEL_ORDER:
        vis_values = [_model_metrics(summary, str(tag), model_key)["heldout_visibility_rmse"] for tag in x_values]
        closure_values = [_model_metrics(summary, str(tag), model_key)["heldout_closure_phase_mae"] for tag in x_values]
        axes[0].plot(x_values, vis_values, marker="o", linewidth=2.1, color=MODEL_COLORS[model_key], label=MODEL_LABELS[model_key])
        valid_pairs = [(x, y) for x, y in zip(x_values, closure_values) if not math.isnan(float(y))]
        if valid_pairs:
            axes[1].plot(
                [x for x, _ in valid_pairs],
                [y for _, y in valid_pairs],
                marker="o",
                linewidth=2.1,
                color=MODEL_COLORS[model_key],
                label=MODEL_LABELS[model_key],
            )
    axes[0].set_title("Held-out VisRMSE", fontsize=12)
    axes[1].set_title("Held-out closure MAE", fontsize=12)
    for axis in axes:
        axis.set_xlabel("Support fraction (%)", fontsize=10)
        axis.grid(alpha=0.22)
        axis.set_xticks(x_values)
    axes[0].set_ylabel("Lower is better", fontsize=10)
    axes[1].set_ylabel("Lower is better", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=8.5, frameon=False)
    fig.suptitle("Challenge-inspired realism track", fontsize=14, y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(Path(output_svg), bbox_inches="tight")
    plt.close(fig)


def write_family_matrix_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write the main benchmark matrix table."""

    ordered_rows = sorted(rows, key=lambda row: (row["family"], int(str(row["support_fraction_tag"]))))
    markdown_rows = [
        [
            str(row["family"]),
            str(row["support_fraction_tag"]),
            _fmt(float(row["emc_heldout_visibility_rmse"])),
            _fmt(float(row["baseline_heldout_visibility_rmse"])),
            _fmt(float(row["residual_heldout_visibility_rmse"])),
            _fmt(float(row["ccrr_heldout_visibility_rmse"])),
            _fmt(float(row.get("dps_heldout_visibility_rmse", float("nan")))),
            _fmt(float(row.get("emc_coverage_90", float("nan")))),
            _fmt(float(row.get("emc_miw", float("nan")))),
            str(row["emc_vs_baseline"]),
            str(row["emc_vs_residual"]),
            str(row["emc_vs_ccrr"]),
        ]
        for row in ordered_rows
    ]
    write_markdown_table(
        path,
        headers=[
            "Holdout family",
            "Support (%)",
            "EMC held-out VisRMSE",
            "Baseline",
            "Residual",
            "CCRR",
            "DPS",
            "EMC 90% cov.",
            "EMC MIW",
            "EMC vs baseline",
            "EMC vs residual",
            "EMC vs CCRR",
        ],
        rows=markdown_rows,
        title="EMC Benchmark Matrix",
        notes=[
            "Each family uses the same dataset split and support-fraction sweep; only the structured support-target partition changes.",
            "DPS is rerun only on the baseline-track family in this add-on cycle; untouched families remain n/a by design.",
            "The matrix is benchmark-first: the central question is earned recovery on unseen held-out measurements, not full-mask reconstruction alone.",
        ],
    )


def write_realism_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write the realism-track summary table."""

    ordered_rows = sorted(rows, key=lambda row: (int(str(row["support_fraction_tag"])), row["model_label"]))
    markdown_rows = [
        [
            str(row["support_fraction_tag"]),
            str(row["model_label"]),
            _fmt(float(row["heldout_visibility_rmse"])),
            _fmt(float(row["heldout_closure_phase_mae"])),
            _fmt(float(row["mse"])),
            _fmt(float(row["ssim"])),
            _fmt(float(row["temporal_consistency"])),
        ]
        for row in ordered_rows
    ]
    write_markdown_table(
        path,
        headers=["Support (%)", "Model", "Held-out VisRMSE", "Held-out Closure", "MSE", "SSIM", "Temporal"],
        rows=markdown_rows,
        title="Challenge-Inspired Realism Track",
        notes=[
            "This track uses only public-style corruption families already implemented in the repository: station-track sampling, scan gaps, gain corruption, and baseline-dependent noise.",
            "It is challenge-inspired and astronomy-facing, but it is not a private ngEHT Challenge #2 dataset.",
        ],
    )


def write_leaderboard_template(path: str | Path) -> None:
    """Write a compact leaderboard-style CSV template."""

    rows = [
        {
            "team_or_method": "Your method",
            "holdout_family": "baseline_track_blocks",
            "support_fraction": "0.8",
            "heldout_visibility_rmse": "",
            "heldout_closure_phase_mae": "",
            "mse": "",
            "ssim": "",
            "temporal_consistency": "",
            "config_manifest": "",
            "split_manifest": "",
            "notes": "",
        }
    ]
    write_csv(path, rows)


def save_benchmark_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    """Write the benchmark artifact manifest."""

    save_json(path, payload)
