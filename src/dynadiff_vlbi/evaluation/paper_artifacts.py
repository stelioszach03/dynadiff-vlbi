"""Utilities for exporting paper-facing tables, summaries, and figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import math

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


METRIC_SPECS: dict[str, tuple[str, str]] = {
    "mse": ("MSE", "lower"),
    "psnr": ("PSNR", "higher"),
    "ssim": ("SSIM", "higher"),
    "temporal_consistency": ("Temporal", "lower"),
    "ring_radius_error": ("Ring", "lower"),
    "hotspot_localization_error": ("Hotspot", "lower"),
    "arc_profile_correlation": ("ArcCorr", "higher"),
    "hotspot_track_velocity_error": ("Track", "lower"),
    "observed_visibility_rmse": ("VisRMSE", "lower"),
    "closure_phase_mae": ("ClosurePhase", "lower"),
}

MODEL_LABELS: dict[str, str] = {
    "dirty": "Dirty",
    "tikhonov": "Tikhonov",
    "baseline_learned": "Baseline 3D U-Net",
    "baseline_data_consistent": "Baseline + Data Consistency",
    "visibility_conditioned": "Standalone Visibility",
    "residual_refinement": "Residual Refinement",
}


@dataclass(frozen=True)
class ConditionSpec:
    """Paper-export specification for one evaluation condition."""

    key: str
    title: str
    summary_path: Path
    figure_path: Path


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON payload from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save a JSON payload with a deterministic layout."""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def repo_relative_path(path: str | Path) -> str:
    """Return a stable repo-relative POSIX path when possible."""

    resolved = Path(path)
    if not resolved.is_absolute():
        return resolved.as_posix()
    try:
        return resolved.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def relativize_payload_paths(payload: Any) -> Any:
    """Recursively convert repo-local absolute paths inside a payload."""

    if isinstance(payload, dict):
        return {key: relativize_payload_paths(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [relativize_payload_paths(value) for value in payload]
    if isinstance(payload, str) and payload.startswith("/"):
        return repo_relative_path(payload)
    return payload


def format_value(value: float) -> str:
    """Format numeric values for compact paper-facing tables."""

    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.3f}"
    return f"{value:.6f}"


def format_mean_std(mean: float, std: float) -> str:
    """Format a mean plus standard deviation entry."""

    return f"{format_value(mean)} ± {format_value(std)}"


def mean_and_std(values: list[float]) -> tuple[float, float]:
    """Return population mean and population standard deviation."""

    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def _metric_value(metrics: dict[str, Any], metric_key: str) -> float:
    """Return one metric value, falling back to NaN for older summary shapes."""

    value = metrics.get(metric_key, float("nan"))
    return float(value)


def compute_verdict(baseline_value: float, candidate_value: float, direction: str) -> str:
    """Compute win/tie/loss for a candidate model relative to a baseline."""

    if math.isclose(candidate_value, baseline_value, rel_tol=0.0, abs_tol=1e-12):
        return "tie"
    if direction == "lower":
        return "win" if candidate_value < baseline_value else "loss"
    return "win" if candidate_value > baseline_value else "loss"


def collect_metric_rows(
    condition_specs: list[ConditionSpec],
    model_order: list[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Load condition summaries and flatten them into per-model metric rows."""

    loaded: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for spec in condition_specs:
        summary = load_json(spec.summary_path)
        loaded[spec.key] = summary
        for model_key in model_order:
            metrics = summary.get(model_key)
            if not metrics:
                continue
            row: dict[str, Any] = {
                "condition": spec.key,
                "condition_title": spec.title,
                "model": model_key,
                "model_label": MODEL_LABELS[model_key],
            }
            row.update(metrics)
            if model_key == "residual_refinement" and "uncertainty" in summary:
                row["empirical_95_coverage"] = summary["uncertainty"].get("empirical_95_coverage")
                row["error_uncertainty_correlation"] = summary["uncertainty"].get("error_uncertainty_correlation")
            rows.append(row)
    return loaded, rows


def aggregate_seed_repeats(
    loaded: dict[str, dict[str, Any]],
    seed_keys: list[str],
    model_order: list[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Aggregate repeated-seed summaries into mean/std rows and verdict counts."""

    rows: list[dict[str, Any]] = []
    verdict_counts = {
        metric_key: {"win": 0, "tie": 0, "loss": 0}
        for metric_key in METRIC_SPECS
    }

    for model_key in model_order:
        row = {
            "model": model_key,
            "model_label": MODEL_LABELS[model_key],
        }
        for metric_key in METRIC_SPECS:
            values = [_metric_value(loaded[seed_key][model_key], metric_key) for seed_key in seed_keys]
            mean, std = mean_and_std(values)
            row[f"{metric_key}_mean"] = mean
            row[f"{metric_key}_std"] = std
        if model_key == "residual_refinement":
            coverage_values = [
                float(loaded[seed_key]["uncertainty"]["empirical_95_coverage"])
                for seed_key in seed_keys
            ]
            correlation_values = [
                float(loaded[seed_key]["uncertainty"]["error_uncertainty_correlation"])
                for seed_key in seed_keys
            ]
            coverage_mean, coverage_std = mean_and_std(coverage_values)
            correlation_mean, correlation_std = mean_and_std(correlation_values)
            row["empirical_95_coverage_mean"] = coverage_mean
            row["empirical_95_coverage_std"] = coverage_std
            row["error_uncertainty_correlation_mean"] = correlation_mean
            row["error_uncertainty_correlation_std"] = correlation_std

            for seed_key in seed_keys:
                baseline_metrics = loaded[seed_key]["baseline_learned"]
                residual_metrics = loaded[seed_key]["residual_refinement"]
                for metric_key, (_, direction) in METRIC_SPECS.items():
                    baseline_value = _metric_value(baseline_metrics, metric_key)
                    residual_value = _metric_value(residual_metrics, metric_key)
                    if math.isnan(baseline_value) or math.isnan(residual_value):
                        continue
                    verdict = compute_verdict(
                        baseline_value=baseline_value,
                        candidate_value=residual_value,
                        direction=direction,
                    )
                    verdict_counts[metric_key][verdict] += 1
        rows.append(row)

    return rows, verdict_counts


def build_condition_verdict_rows(
    loaded: dict[str, dict[str, Any]],
    condition_keys: list[str],
) -> list[dict[str, Any]]:
    """Build one compact row per condition and metric versus the baseline backbone."""

    rows: list[dict[str, Any]] = []
    for condition_key in condition_keys:
        baseline_metrics = loaded[condition_key]["baseline_learned"]
        residual_metrics = loaded[condition_key]["residual_refinement"]
        for metric_key, (metric_label, direction) in METRIC_SPECS.items():
            baseline_value = _metric_value(baseline_metrics, metric_key)
            residual_value = _metric_value(residual_metrics, metric_key)
            if math.isnan(baseline_value) or math.isnan(residual_value):
                continue
            rows.append(
                {
                    "condition": condition_key,
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "baseline_3d_unet": baseline_value,
                    "residual_refinement": residual_value,
                    "verdict_vs_baseline": compute_verdict(
                        baseline_value=baseline_value,
                        candidate_value=residual_value,
                        direction=direction,
                    ),
                }
            )
    return rows


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write a CSV file from a list of dictionaries."""

    if not rows:
        return
    import csv

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(
    path: str | Path,
    headers: list[str],
    rows: list[list[str]],
    title: str,
    notes: list[str] | None = None,
) -> None:
    """Write a compact markdown table."""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    if notes:
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
    resolved.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_qualitative_panel(condition_specs: list[ConditionSpec], output_path: str | Path) -> None:
    """Compose a 2x2 panel from already-generated condition figures."""

    images = [(spec.title, plt.imread(spec.figure_path)) for spec in condition_specs]
    columns = 2
    rows = int(math.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(9.5 * columns, 7.5 * rows))
    axes = np.atleast_1d(axes).reshape(rows, columns)

    for axis in axes.ravel():
        axis.axis("off")

    for axis, (title, image) in zip(axes.ravel(), images):
        axis.imshow(image)
        axis.set_title(title, fontsize=14)
        axis.axis("off")

    fig.tight_layout()
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(resolved, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_methods_figure(png_path: str | Path, svg_path: str | Path | None = None) -> None:
    """Draw a simple architecture schematic for the residual refinement path."""

    fig, axis = plt.subplots(figsize=(12, 4.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    def add_box(x: float, y: float, width: float, height: float, label: str, color: str) -> None:
        patch = patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.6,
            edgecolor="#1f2937",
            facecolor=color,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=11)

    def add_arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#1f2937"},
        )

    add_box(0.04, 0.62, 0.18, 0.18, "Dirty sequence", "#fde68a")
    add_box(0.30, 0.62, 0.20, 0.18, "Baseline 3D U-Net\n(backbone)", "#bfdbfe")
    add_box(0.58, 0.62, 0.16, 0.18, "Baseline prediction", "#dbeafe")

    add_box(0.04, 0.20, 0.18, 0.18, "Visibilities\n+ mask / uv", "#fbcfe8")
    add_box(0.30, 0.20, 0.24, 0.18, "Residual refinement branch", "#fecaca")
    add_box(0.62, 0.20, 0.12, 0.18, "Residual\ncorrection", "#fee2e2")

    add_box(0.79, 0.40, 0.08, 0.12, "+", "#e5e7eb")
    add_box(0.89, 0.40, 0.09, 0.12, "Final\nprediction", "#bbf7d0")
    add_box(0.89, 0.15, 0.09, 0.12, "Log-var\nhead", "#ddd6fe")

    add_arrow((0.22, 0.71), (0.30, 0.71))
    add_arrow((0.50, 0.71), (0.58, 0.71))
    add_arrow((0.22, 0.29), (0.30, 0.29))
    add_arrow((0.54, 0.29), (0.62, 0.29))
    add_arrow((0.74, 0.71), (0.79, 0.46))
    add_arrow((0.74, 0.29), (0.79, 0.46))
    add_arrow((0.87, 0.46), (0.89, 0.46))
    add_arrow((0.935, 0.40), (0.935, 0.27))

    axis.text(
        0.5,
        0.94,
        "Residual visibility refinement keeps the baseline 3D U-Net as the reference backbone\nand uses visibility information only as a corrective residual path.",
        ha="center",
        va="center",
        fontsize=12,
    )

    png_resolved = Path(png_path)
    png_resolved.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_resolved, dpi=240, bbox_inches="tight")
    if svg_path is not None:
        svg_resolved = Path(svg_path)
        svg_resolved.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(svg_resolved, bbox_inches="tight")
    plt.close(fig)
