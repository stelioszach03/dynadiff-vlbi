"""CCRR-focused paper artifact generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import math

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from PIL import Image

from dynadiff_vlbi.evaluation.metrics import (
    compute_reconstruction_metrics,
    empirical_coverage,
    risk_coverage_auc,
    topk_error_recall,
    uncertainty_error_correlation,
)
from dynadiff_vlbi.evaluation.paper_artifacts import (
    format_mean_std,
    format_value,
    save_json,
    write_csv,
    write_markdown_table,
)
from dynadiff_vlbi.evaluation.paper_visuals import (
    ERROR_CMAP,
    IMAGE_CMAP,
    UNCERTAINTY_CMAP,
    load_predictions,
    select_best_improvement,
    select_hard_improvement,
    select_representative_improvement,
)


MODEL_LABELS: dict[str, str] = {
    "ground_truth": "Ground truth",
    "dirty": "Dirty",
    "tikhonov": "Tikhonov",
    "baseline_learned": "Baseline 3D U-Net",
    "baseline_data_consistent": "Baseline + Data Consistency",
    "visibility_conditioned": "Standalone Visibility",
    "residual_refinement": "Residual Refinement",
    "ccrr": "CCRR",
    "uncertainty": "Uncertainty",
}

METRIC_SPECS: dict[str, tuple[str, str]] = {
    "mse": ("MSE", "lower"),
    "psnr": ("PSNR", "higher"),
    "ssim": ("SSIM", "higher"),
    "temporal_consistency": ("Temporal", "lower"),
    "ring_radius_error": ("Ring", "lower"),
    "ring_thickness_error": ("Thickness", "lower"),
    "bright_sector_angle_error": ("SectorAngle", "lower"),
    "hotspot_localization_error": ("Hotspot", "lower"),
    "arc_profile_correlation": ("ArcCorr", "higher"),
    "hotspot_track_velocity_error": ("Track", "lower"),
    "observed_visibility_rmse": ("VisRMSE", "lower"),
    "closure_phase_mae": ("ClosurePhase", "lower"),
}

CORE_METRICS = [
    "mse",
    "psnr",
    "ssim",
    "temporal_consistency",
    "ring_radius_error",
    "ring_thickness_error",
    "bright_sector_angle_error",
]
ASTRO_METRICS = [
    "arc_profile_correlation",
    "hotspot_track_velocity_error",
    "observed_visibility_rmse",
    "closure_phase_mae",
]
UNCERTAINTY_METRICS = [
    "empirical_95_coverage",
    "error_uncertainty_correlation",
    "risk_coverage_auc",
    "top10_error_recall",
]


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    title: str
    run_name: str
    dataset_dir: Path
    output_root: Path | None = None

    @property
    def prediction_path(self) -> Path:
        output_root = self.output_root or (self.dataset_dir.parents[2] / "outputs")
        return output_root / self.run_name / "predictions" / "test_predictions.npz"


def _safe_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0)


def _normalize_map(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    minimum = float(image.min())
    maximum = float(image.max())
    if maximum <= minimum:
        return np.zeros_like(image, dtype=np.float32)
    return (image - minimum) / (maximum - minimum)


def _load_dataset(dataset_dir: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(dataset_dir) / "test.npz") as payload:
        return {key: payload[key] for key in payload.files}


def _available_models(bundle: dict[str, np.ndarray]) -> list[str]:
    return [key for key in MODEL_LABELS if key in bundle]


def summarize_prediction_bundle(
    prediction_path: str | Path,
    dataset_dir: str | Path,
) -> dict[str, Any]:
    bundle = load_predictions(prediction_path)
    dataset = _load_dataset(dataset_dir)
    measurements = dataset["vis_real"] + 1j * dataset["vis_imag"]
    mask = dataset["mask"]
    ground_truth = dataset["ground_truth"]
    ring_radius_px = dataset["ring_radius_px"]
    hotspot_coords_px = dataset["hotspot_coords_px"]
    baseline_pairs = dataset.get("baseline_pairs")
    frame_uv_indices = dataset.get("frame_uv_indices")

    per_model: dict[str, dict[str, np.ndarray]] = {}
    aggregate: dict[str, dict[str, float]] = {}
    for model_key in _available_models(bundle):
        metric_lists = {metric_key: [] for metric_key in METRIC_SPECS}
        for sample_index in range(ground_truth.shape[0]):
            metrics = compute_reconstruction_metrics(
                prediction=bundle[model_key][sample_index],
                target=ground_truth[sample_index],
                target_ring_radius_px=float(ring_radius_px[sample_index]),
                target_hotspot_coords_px=hotspot_coords_px[sample_index],
                measurements=measurements[sample_index],
                mask=mask[sample_index],
                baseline_pairs=baseline_pairs,
                frame_uv_indices=frame_uv_indices,
            )
            for metric_key, value in metrics.items():
                metric_lists[metric_key].append(float(value))
        per_model[model_key] = {
            metric_key: np.asarray(values, dtype=np.float64) for metric_key, values in metric_lists.items()
        }
        aggregate[model_key] = {
            metric_key: float(np.nanmean(values)) for metric_key, values in per_model[model_key].items()
        }

    uncertainty_summary: dict[str, float] | None = None
    if "ccrr" in bundle and "uncertainty" in bundle:
        predictive_mean = bundle["ccrr"]
        predictive_std = np.maximum(bundle["uncertainty"], 1e-8)
        uncertainty_summary = {
            "empirical_95_coverage": empirical_coverage(ground_truth, predictive_mean, predictive_std),
            "error_uncertainty_correlation": uncertainty_error_correlation(ground_truth, predictive_mean, predictive_std),
            "risk_coverage_auc": risk_coverage_auc(ground_truth, predictive_mean, predictive_std),
            "top10_error_recall": topk_error_recall(ground_truth, predictive_mean, predictive_std),
        }

    return {
        "bundle": bundle,
        "dataset": dataset,
        "aggregate": aggregate,
        "per_model": per_model,
        "uncertainty": uncertainty_summary,
    }


def aggregate_seed_summaries(
    run_summaries: dict[str, dict[str, Any]],
    seed_keys: list[str],
    model_order: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key in model_order:
        row = {"model": model_key, "model_label": MODEL_LABELS[model_key]}
        for metric_key in METRIC_SPECS:
            values = np.asarray(
                [run_summaries[seed_key]["aggregate"][model_key][metric_key] for seed_key in seed_keys],
                dtype=np.float64,
            )
            row[f"{metric_key}_mean"] = float(np.nanmean(values))
            row[f"{metric_key}_std"] = float(np.nanstd(values))
        rows.append(row)
    return rows


def _normalized_better(candidate: np.ndarray, reference: np.ndarray, direction: str) -> np.ndarray:
    return candidate - reference if direction == "higher" else reference - candidate


def paired_bootstrap_stats(
    candidate: np.ndarray,
    reference: np.ndarray,
    direction: str,
    seed: int = 7,
    num_resamples: int = 4000,
) -> dict[str, float]:
    better = _normalized_better(candidate, reference, direction=direction)
    rng = np.random.default_rng(seed)
    sample_size = better.shape[0]
    boot_means = np.empty(num_resamples, dtype=np.float64)
    for index in range(num_resamples):
        sample_indices = rng.integers(0, sample_size, size=sample_size)
        boot_means[index] = float(np.mean(better[sample_indices]))
    observed_mean = float(np.mean(better))
    sign_flips = rng.choice([-1.0, 1.0], size=(2048, sample_size))
    null_means = np.mean(sign_flips * better[None, :], axis=1)
    p_value = float((np.abs(null_means) >= abs(observed_mean)).mean())
    return {
        "mean_delta": observed_mean,
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
        "win_rate": float(np.mean(better > 0.0)),
        "tie_rate": float(np.mean(np.isclose(better, 0.0, atol=1e-12))),
        "p_value": p_value,
    }


def build_paired_rows(
    run_summaries: dict[str, dict[str, Any]],
    seed_keys: list[str],
    metric_keys: list[str],
    comparisons: list[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_key, reference_key, label in comparisons:
        for metric_key in metric_keys:
            direction = METRIC_SPECS[metric_key][1]
            candidate = np.concatenate(
                [run_summaries[seed_key]["per_model"][candidate_key][metric_key] for seed_key in seed_keys]
            )
            reference = np.concatenate(
                [run_summaries[seed_key]["per_model"][reference_key][metric_key] for seed_key in seed_keys]
            )
            stats = paired_bootstrap_stats(candidate, reference, direction=direction)
            rows.append(
                {
                    "comparison": label,
                    "metric": metric_key,
                    "metric_label": METRIC_SPECS[metric_key][0],
                    **stats,
                }
            )
    return rows


def _select_example(
    bundle: dict[str, np.ndarray],
    selection: str,
    reference_key: str,
    candidate_key: str,
) -> tuple[int, int]:
    if selection == "best":
        return select_best_improvement(bundle, reference_key=reference_key, candidate_key=candidate_key)
    if selection == "representative":
        return select_representative_improvement(bundle, reference_key=reference_key, candidate_key=candidate_key)
    if selection == "hard":
        return select_hard_improvement(bundle, reference_key=reference_key, candidate_key=candidate_key)
    raise ValueError(f"Unknown selection mode: {selection}")


def save_ccrr_schematic(png_path: str | Path, svg_path: str | Path | None = None) -> None:
    fig, axis = plt.subplots(figsize=(13, 5.4))
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
            edgecolor="#111827",
            facecolor=color,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=11)

    def add_arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#111827"})

    add_box(0.03, 0.68, 0.16, 0.16, "Dirty sequence", "#fde68a")
    add_box(0.25, 0.68, 0.18, 0.16, "Baseline 3D U-Net", "#bfdbfe")
    add_box(0.49, 0.68, 0.16, 0.16, "Backbone\nprediction", "#dbeafe")

    add_box(0.03, 0.22, 0.18, 0.18, "Visibilities\n+ mask / uv\n+ metadata", "#fbcfe8")
    add_box(0.27, 0.22, 0.20, 0.18, "Residual branch", "#fecaca")
    add_box(0.53, 0.22, 0.12, 0.18, "Residual", "#fee2e2")

    add_box(0.71, 0.50, 0.12, 0.14, "Pre-DC\nprediction", "#e0f2fe")
    add_box(0.71, 0.22, 0.12, 0.14, "DC layer", "#ddd6fe")
    add_box(0.87, 0.40, 0.10, 0.14, "CCRR\noutput", "#bbf7d0")
    add_box(0.87, 0.17, 0.10, 0.12, "Log-var\nhead", "#ddd6fe")

    add_arrow((0.19, 0.76), (0.25, 0.76))
    add_arrow((0.43, 0.76), (0.49, 0.76))
    add_arrow((0.21, 0.31), (0.27, 0.31))
    add_arrow((0.47, 0.31), (0.53, 0.31))
    add_arrow((0.65, 0.76), (0.71, 0.57))
    add_arrow((0.65, 0.31), (0.71, 0.57))
    add_arrow((0.77, 0.50), (0.77, 0.36))
    add_arrow((0.83, 0.29), (0.87, 0.47))
    add_arrow((0.92, 0.40), (0.92, 0.29))

    axis.text(
        0.5,
        0.95,
        "Closure-Consistent Residual Refinement (CCRR): a strong image-domain backbone,\nvisibility-guided residual correction, and an in-model data-consistency layer.",
        ha="center",
        va="center",
        fontsize=12,
    )
    axis.text(
        0.77,
        0.12,
        "Training: image + temporal + visibility + closure + uncertainty terms",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#374151",
    )

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    if svg_path is not None:
        svg_path = Path(svg_path)
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def save_ccrr_qualitative_figure(
    prediction_path: str | Path,
    output_png: str | Path,
    output_svg: str | Path,
    selection_manifest: str | Path,
    *,
    title: str,
    selection: str,
    reference_key: str = "residual_refinement",
    candidate_key: str = "ccrr",
) -> dict[str, Any]:
    bundle = load_predictions(prediction_path)
    sample_index, frame_index = _select_example(
        bundle,
        selection=selection,
        reference_key=reference_key,
        candidate_key=candidate_key,
    )
    keys = [
        "ground_truth",
        "dirty",
        "baseline_learned",
        "baseline_data_consistent",
        "residual_refinement",
        "ccrr",
        "uncertainty",
    ]
    fig, axes = plt.subplots(2, len(keys), figsize=(16.5, 5.8))
    fig.patch.set_facecolor("white")
    for column_index, key in enumerate(keys):
        image = bundle[key][sample_index, frame_index]
        if key == "uncertainty":
            image = _normalize_map(image)
            cmap = UNCERTAINTY_CMAP
        else:
            image = _safe_image(image)
            cmap = IMAGE_CMAP
        axes[0, column_index].imshow(image, cmap=cmap, interpolation="nearest")
        axes[0, column_index].set_xticks([])
        axes[0, column_index].set_yticks([])
        axes[0, column_index].set_title(MODEL_LABELS.get(key, key.replace("_", " ").title()), fontsize=11, pad=6)
        if column_index == 0:
            axes[0, column_index].set_ylabel("Reconstruction", fontsize=11)

        if key in {"ground_truth", "uncertainty"}:
            error_map = np.zeros_like(bundle["ground_truth"][sample_index, frame_index], dtype=np.float32)
        else:
            error_map = _normalize_map(
                np.abs(bundle[key][sample_index, frame_index] - bundle["ground_truth"][sample_index, frame_index])
            )
        axes[1, column_index].imshow(error_map, cmap=ERROR_CMAP, interpolation="nearest")
        axes[1, column_index].set_xticks([])
        axes[1, column_index].set_yticks([])
        if column_index == 0:
            axes[1, column_index].set_ylabel("Abs. error", fontsize=11)
        for row_index in range(2):
            border_color = "#1d4ed8" if key == candidate_key else "#d1d5db"
            border_width = 2.0 if key == candidate_key else 0.8
            for spine in axes[row_index, column_index].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(border_width)
                spine.set_edgecolor(border_color)

    fig.suptitle(title, fontsize=14.5, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_png = Path(output_png)
    output_svg = Path(output_svg)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "prediction_path": str(prediction_path),
        "selection_mode": selection,
        "reference_key": reference_key,
        "candidate_key": candidate_key,
        "sample_index": sample_index,
        "frame_index": frame_index,
    }
    Path(selection_manifest).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def save_ccrr_pareto_figure(
    condition_rows: list[dict[str, Any]],
    output_png: str | Path,
    output_svg: str | Path,
    *,
    x_metric: str = "observed_visibility_rmse",
    y_metric: str = "ssim",
) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    fig.patch.set_facecolor("white")
    model_colors = {
        "baseline_learned": "#2563eb",
        "baseline_data_consistent": "#9333ea",
        "visibility_conditioned": "#dc2626",
        "residual_refinement": "#ea580c",
        "ccrr": "#059669",
    }
    for row in condition_rows:
        model_key = row["model"]
        if model_key not in model_colors:
            continue
        marker = "*" if model_key == "ccrr" else "o"
        size = 180 if model_key == "ccrr" else 90
        axis.scatter(
            float(row[x_metric]),
            float(row[y_metric]),
            s=size,
            marker=marker,
            color=model_colors[model_key],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.9,
        )
        axis.text(
            float(row[x_metric]) + 0.0015,
            float(row[y_metric]) + 0.0006,
            f"{row['condition']} / {MODEL_LABELS[model_key]}",
            fontsize=8.5,
            color="#111827",
        )
    axis.set_xlabel("Observed visibility RMSE (lower is better)")
    axis.set_ylabel("SSIM (higher is better)")
    axis.set_title("Structural fidelity versus observation-domain agreement")
    axis.grid(alpha=0.28, linewidth=0.6)
    axis.invert_xaxis()
    fig.tight_layout()
    output_png = Path(output_png)
    output_svg = Path(output_svg)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)


def save_ccrr_risk_coverage_figure(
    condition_specs: list[ExperimentSpec],
    output_png: str | Path,
    output_svg: str | Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    fig.patch.set_facecolor("white")
    palette = ["#1d4ed8", "#dc2626", "#059669", "#7c3aed", "#b45309", "#0891b2"]
    for color, spec in zip(palette, condition_specs):
        payload = summarize_prediction_bundle(spec.prediction_path, spec.dataset_dir)
        ground_truth = payload["bundle"]["ground_truth"]
        prediction = payload["bundle"]["ccrr"]
        uncertainty = payload["bundle"]["uncertainty"]
        errors = ((ground_truth - prediction) ** 2).reshape(-1)
        ordering = np.argsort(uncertainty.reshape(-1))
        sorted_errors = errors[ordering]
        coverages = np.linspace(0.1, 1.0, 20)
        risks = []
        for coverage in coverages:
            keep = max(1, int(round(coverage * sorted_errors.shape[0])))
            risks.append(float(np.mean(sorted_errors[:keep])))
        axis.plot(coverages, risks, label=spec.title, color=color, linewidth=2.0)
    axis.set_xlabel("Coverage kept (lowest uncertainty first)")
    axis.set_ylabel("Mean squared error on retained pixels")
    axis.set_title("CCRR risk-coverage behavior across paper conditions")
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    output_png = Path(output_png)
    output_svg = Path(output_svg)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)


def save_ccrr_supplementary_gif(
    prediction_path: str | Path,
    output_path: str | Path,
    *,
    sample_index: int,
    frame_indices: list[int] | None = None,
) -> None:
    bundle = load_predictions(prediction_path)
    frame_indices = frame_indices or list(range(bundle["ground_truth"].shape[1]))
    frames: list[Image.Image] = []
    row_keys = ["ground_truth", "baseline_learned", "residual_refinement", "ccrr", "uncertainty"]
    labels = [MODEL_LABELS.get(key, key.replace("_", " ").title()) for key in row_keys]
    for frame_index in frame_indices:
        fig, axes = plt.subplots(1, len(row_keys), figsize=(13.5, 3.0))
        fig.patch.set_facecolor("white")
        for axis, key, label in zip(axes, row_keys, labels):
            image = bundle[key][sample_index, frame_index]
            if key == "uncertainty":
                image = _normalize_map(image)
                cmap = UNCERTAINTY_CMAP
            else:
                image = _safe_image(image)
                cmap = IMAGE_CMAP
            axis.imshow(image, cmap=cmap, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(label, fontsize=10, pad=6)
        fig.suptitle(f"CCRR supplementary sequence, sample {sample_index}, frame {frame_index}", fontsize=12, y=0.98)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(Image.fromarray(rgba[..., :3].copy()))
        plt.close(fig)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=700, loop=0)
