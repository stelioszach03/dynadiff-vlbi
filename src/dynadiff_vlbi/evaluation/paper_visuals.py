"""Publication-facing figure generation from verified experiment outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMAGE_CMAP = "inferno"
UNCERTAINTY_CMAP = "viridis"
CORRECTION_CMAP = "coolwarm"
ERROR_CMAP = "magma"

DISPLAY_LABELS = {
    "ground_truth": "Ground truth",
    "dirty": "Dirty",
    "tikhonov": "Tikhonov",
    "baseline_learned": "Baseline 3D U-Net",
    "baseline_data_consistent": "Baseline + data consistency",
    "residual_refinement": "Residual refinement",
    "ccrr": "CCRR",
    "uncertainty": "Uncertainty",
    "residual_correction": "Residual correction",
    "pre_dc_prediction": "Pre-DC prediction",
    "absolute_error": "Absolute error",
}


@dataclass(frozen=True)
class PredictionCondition:
    """One paper figure condition tied to a saved prediction file."""

    key: str
    title: str
    prediction_path: Path


def load_predictions(path: str | Path) -> dict[str, np.ndarray]:
    """Load a saved `test_predictions.npz` bundle into memory."""

    with np.load(Path(path)) as payload:
        return {key: payload[key] for key in payload.files}


def _safe_image(image: np.ndarray) -> np.ndarray:
    """Clamp reconstruction images for display."""

    return np.clip(image, 0.0, 1.0)


def _normalize_map(image: np.ndarray) -> np.ndarray:
    """Normalize a single map to [0, 1] for display."""

    image = np.asarray(image, dtype=np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_value) / (max_value - min_value)


def _sample_gains(
    bundle: dict[str, np.ndarray],
    reference_key: str = "baseline_learned",
    candidate_key: str = "residual_refinement",
) -> np.ndarray:
    """Per-sample gain of one reconstruction over another."""

    baseline = bundle[reference_key]
    ground_truth = bundle["ground_truth"]
    baseline_mse = ((baseline - ground_truth) ** 2).mean(axis=(1, 2, 3))
    residual_mse = ((bundle[candidate_key] - ground_truth) ** 2).mean(axis=(1, 2, 3))
    return baseline_mse - residual_mse


def select_best_improvement(
    bundle: dict[str, np.ndarray],
    reference_key: str = "baseline_learned",
    candidate_key: str = "residual_refinement",
) -> tuple[int, int]:
    """Select the sample/frame with the strongest reconstruction gain."""

    sample_index = int(np.argmax(_sample_gains(bundle, reference_key=reference_key, candidate_key=candidate_key)))
    baseline = bundle[reference_key][sample_index]
    residual = bundle[candidate_key][sample_index]
    ground_truth = bundle["ground_truth"][sample_index]
    baseline_frame_mse = ((baseline - ground_truth) ** 2).mean(axis=(1, 2))
    residual_frame_mse = ((residual - ground_truth) ** 2).mean(axis=(1, 2))
    frame_index = int(np.argmax(baseline_frame_mse - residual_frame_mse))
    return sample_index, frame_index


def select_representative_improvement(
    bundle: dict[str, np.ndarray],
    reference_key: str = "baseline_learned",
    candidate_key: str = "residual_refinement",
) -> tuple[int, int]:
    """Select a reproducible median-gain example instead of a best-case one."""

    gains = _sample_gains(bundle, reference_key=reference_key, candidate_key=candidate_key)
    ordering = np.argsort(gains)
    sample_index = int(ordering[len(ordering) // 2])
    baseline = bundle[reference_key][sample_index]
    candidate = bundle[candidate_key][sample_index]
    ground_truth = bundle["ground_truth"][sample_index]
    frame_gains = ((baseline - ground_truth) ** 2).mean(axis=(1, 2)) - ((candidate - ground_truth) ** 2).mean(axis=(1, 2))
    frame_index = int(np.argsort(frame_gains)[len(frame_gains) // 2])
    return sample_index, frame_index


def select_hard_improvement(
    bundle: dict[str, np.ndarray],
    reference_key: str = "baseline_learned",
    candidate_key: str = "residual_refinement",
) -> tuple[int, int]:
    """Select a hard example from the high-error subset with the strongest candidate gain."""

    reference = bundle[reference_key]
    ground_truth = bundle["ground_truth"]
    reference_mse = ((reference - ground_truth) ** 2).mean(axis=(1, 2, 3))
    threshold = float(np.quantile(reference_mse, 0.75))
    hard_indices = np.flatnonzero(reference_mse >= threshold)
    if hard_indices.size == 0:
        return select_best_improvement(bundle, reference_key=reference_key, candidate_key=candidate_key)
    gains = _sample_gains(bundle, reference_key=reference_key, candidate_key=candidate_key)
    sample_index = int(hard_indices[np.argmax(gains[hard_indices])])
    frame_gains = ((reference[sample_index] - ground_truth[sample_index]) ** 2).mean(axis=(1, 2)) - (
        (bundle[candidate_key][sample_index] - ground_truth[sample_index]) ** 2
    ).mean(axis=(1, 2))
    frame_index = int(np.argmax(frame_gains))
    return sample_index, frame_index


def select_best_uncertainty_alignment(bundle: dict[str, np.ndarray]) -> tuple[int, int]:
    """Select the sample/frame with the strongest error-uncertainty correlation."""

    residual = bundle["residual_refinement"]
    ground_truth = bundle["ground_truth"]
    uncertainty = bundle["uncertainty"]
    sample_scores: list[float] = []
    for sample_index in range(residual.shape[0]):
        error = np.abs(residual[sample_index] - ground_truth[sample_index]).ravel()
        score = np.corrcoef(error, uncertainty[sample_index].ravel())[0, 1]
        sample_scores.append(float(np.nan_to_num(score, nan=-1.0)))
    best_sample = int(np.argmax(np.asarray(sample_scores)))
    frame_scores: list[float] = []
    for frame_index in range(residual.shape[1]):
        error = np.abs(residual[best_sample, frame_index] - ground_truth[best_sample, frame_index]).ravel()
        score = np.corrcoef(error, uncertainty[best_sample, frame_index].ravel())[0, 1]
        frame_scores.append(float(np.nan_to_num(score, nan=-1.0)))
    best_frame = int(np.argmax(np.asarray(frame_scores)))
    return best_sample, best_frame


def _render_panel(
    arrays: list[np.ndarray],
    labels: list[str],
    cmaps: list[str],
    row_title: str | None = None,
    figure_size: tuple[float, float] = (13.5, 3.2),
) -> np.ndarray:
    """Render one labeled row of panels to an RGB array."""

    fig, axes = plt.subplots(1, len(arrays), figsize=figure_size)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")
    for axis, image, label, cmap in zip(axes, arrays, labels, cmaps):
        axis.imshow(image, cmap=cmap, interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(label, fontsize=11, pad=6)
        for spine in axis.spines.values():
            spine.set_visible(False)
    if row_title:
        fig.text(0.01, 0.5, row_title, va="center", ha="left", rotation=90, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 1.0))
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb


def save_condition_comparison_figure(
    conditions: list[PredictionCondition],
    output_png: str | Path,
    output_svg: str | Path,
    selection_manifest: str | Path,
) -> dict[str, dict[str, Any]]:
    """Save a multi-condition qualitative comparison figure."""

    bundles = {condition.key: load_predictions(condition.prediction_path) for condition in conditions}
    selection_payload: dict[str, dict[str, Any]] = {}
    columns = [
        "ground_truth",
        "dirty",
        "tikhonov",
        "baseline_learned",
        "residual_refinement",
        "uncertainty",
    ]

    fig, axes = plt.subplots(len(conditions), len(columns), figsize=(15.5, 10.8))
    fig.patch.set_facecolor("white")

    for row_index, condition in enumerate(conditions):
        bundle = bundles[condition.key]
        sample_index, frame_index = select_best_improvement(bundle)
        selection_payload[condition.key] = {
            "prediction_path": str(condition.prediction_path),
            "selection_rule": "max residual-refinement gain over baseline by sample MSE, then by frame MSE",
            "sample_index": sample_index,
            "frame_index": frame_index,
        }

        absolute_error = np.abs(bundle["residual_refinement"][sample_index, frame_index] - bundle["ground_truth"][sample_index, frame_index])
        uncertainty = _normalize_map(bundle["uncertainty"][sample_index, frame_index])
        images: dict[str, np.ndarray] = {
            "ground_truth": _safe_image(bundle["ground_truth"][sample_index, frame_index]),
            "dirty": _safe_image(bundle["dirty"][sample_index, frame_index]),
            "tikhonov": _safe_image(bundle["tikhonov"][sample_index, frame_index]),
            "baseline_learned": _safe_image(bundle["baseline_learned"][sample_index, frame_index]),
            "residual_refinement": _safe_image(bundle["residual_refinement"][sample_index, frame_index]),
            "uncertainty": uncertainty,
            "absolute_error": _normalize_map(absolute_error),
        }

        for column_index, key in enumerate(columns):
            axis = axes[row_index, column_index]
            cmap = IMAGE_CMAP if key != "uncertainty" else UNCERTAINTY_CMAP
            axis.imshow(images[key], cmap=cmap, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(DISPLAY_LABELS[key], fontsize=12, pad=8)
            if column_index == 0:
                axis.set_ylabel(
                    f"{condition.title}\n(sample {sample_index}, frame {frame_index})",
                    fontsize=11,
                    rotation=90,
                    labelpad=16,
                    va="center",
                )
            border_color = "#1d4ed8" if key == "residual_refinement" else "#e5e7eb"
            border_width = 2.2 if key == "residual_refinement" else 0.8
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(border_width)
                spine.set_edgecolor(border_color)

    fig.suptitle(
        "Qualitative comparison across the main paper conditions using automatically selected high-gain examples",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_png = Path(output_png)
    output_svg = Path(output_svg)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)

    selection_path = Path(selection_manifest)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")
    return selection_payload


def save_temporal_sequence_figure(
    prediction_path: str | Path,
    output_png: str | Path,
    output_svg: str | Path,
    selection_manifest: str | Path,
    frame_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Save a multi-frame temporal comparison figure for one default32 sample."""

    bundle = load_predictions(prediction_path)
    sample_index, _ = select_best_improvement(bundle)
    frame_indices = frame_indices or [0, 2, 4, 6]
    row_specs = [
        ("ground_truth", IMAGE_CMAP),
        ("dirty", IMAGE_CMAP),
        ("baseline_learned", IMAGE_CMAP),
        ("residual_refinement", IMAGE_CMAP),
        ("uncertainty", UNCERTAINTY_CMAP),
    ]

    fig, axes = plt.subplots(len(row_specs), len(frame_indices), figsize=(12.8, 11.8))
    fig.patch.set_facecolor("white")
    for row_index, (key, cmap) in enumerate(row_specs):
        for column_index, frame_index in enumerate(frame_indices):
            axis = axes[row_index, column_index]
            image = bundle[key][sample_index, frame_index]
            if key == "uncertainty":
                image = _normalize_map(image)
            else:
                image = _safe_image(image)
            axis.imshow(image, cmap=cmap, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(f"Frame {frame_index}", fontsize=11, pad=6)
            if column_index == 0:
                axis.set_ylabel(DISPLAY_LABELS[key], fontsize=11, rotation=90, labelpad=12, va="center")
            border_color = "#1d4ed8" if key == "residual_refinement" else "#e5e7eb"
            border_width = 2.0 if key == "residual_refinement" else 0.8
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(border_width)
                spine.set_edgecolor(border_color)
    fig.suptitle(
        f"Temporal behavior on a strong default32 example (sample {sample_index})",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    output_png = Path(output_png)
    output_svg = Path(output_svg)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "prediction_path": str(prediction_path),
        "sample_index": sample_index,
        "frame_indices": frame_indices,
        "selection_rule": "best residual-refinement sample by sample-level MSE gain over the baseline",
    }
    Path(selection_manifest).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def save_uncertainty_alignment_figure(
    prediction_path: str | Path,
    output_png: str | Path,
    output_svg: str | Path,
    selection_manifest: str | Path,
) -> dict[str, Any]:
    """Save an uncertainty-focused figure for one strongly aligned sample/frame."""

    bundle = load_predictions(prediction_path)
    sample_index, frame_index = select_best_uncertainty_alignment(bundle)
    residual = _safe_image(bundle["residual_refinement"][sample_index, frame_index])
    baseline = _safe_image(bundle["baseline_learned"][sample_index, frame_index])
    ground_truth = _safe_image(bundle["ground_truth"][sample_index, frame_index])
    error = _normalize_map(np.abs(residual - ground_truth))
    uncertainty = _normalize_map(bundle["uncertainty"][sample_index, frame_index])
    correction = _normalize_map(bundle["residual_correction"][sample_index, frame_index])

    images = [ground_truth, baseline, residual, error, uncertainty, correction]
    labels = [
        DISPLAY_LABELS["ground_truth"],
        DISPLAY_LABELS["baseline_learned"],
        DISPLAY_LABELS["residual_refinement"],
        DISPLAY_LABELS["absolute_error"],
        DISPLAY_LABELS["uncertainty"],
        DISPLAY_LABELS["residual_correction"],
    ]
    cmaps = [IMAGE_CMAP, IMAGE_CMAP, IMAGE_CMAP, ERROR_CMAP, UNCERTAINTY_CMAP, CORRECTION_CMAP]

    fig, axes = plt.subplots(1, len(images), figsize=(14.5, 3.8))
    fig.patch.set_facecolor("white")
    for axis, image, label, cmap in zip(axes, images, labels, cmaps):
        axis.imshow(image, cmap=cmap, interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(label, fontsize=11, pad=6)
        border_color = "#1d4ed8" if label == DISPLAY_LABELS["residual_refinement"] else "#e5e7eb"
        border_width = 2.0 if label == DISPLAY_LABELS["residual_refinement"] else 0.8
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(border_width)
            spine.set_edgecolor(border_color)
    fig.suptitle(
        f"Uncertainty and residual-correction behavior on a high-correlation default32 example (sample {sample_index}, frame {frame_index})",
        fontsize=14,
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    output_png = Path(output_png)
    output_svg = Path(output_svg)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "prediction_path": str(prediction_path),
        "sample_index": sample_index,
        "frame_index": frame_index,
        "selection_rule": "maximum error-uncertainty correlation over samples, then over frames",
    }
    Path(selection_manifest).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def save_supplementary_gif(
    prediction_path: str | Path,
    output_path: str | Path,
    row_title: str,
    frame_indices: list[int] | None = None,
    sample_index: int | None = None,
) -> dict[str, Any]:
    """Save a GIF sequence showing the temporal evolution of one selected sample."""

    bundle = load_predictions(prediction_path)
    if sample_index is None:
        sample_index, _ = select_best_improvement(bundle)
    frame_indices = frame_indices or list(range(bundle["ground_truth"].shape[1]))

    frames: list[Image.Image] = []
    labels = [
        DISPLAY_LABELS["ground_truth"],
        DISPLAY_LABELS["dirty"],
        DISPLAY_LABELS["baseline_learned"],
        DISPLAY_LABELS["residual_refinement"],
        DISPLAY_LABELS["uncertainty"],
    ]
    cmaps = [IMAGE_CMAP, IMAGE_CMAP, IMAGE_CMAP, IMAGE_CMAP, UNCERTAINTY_CMAP]

    for frame_index in frame_indices:
        arrays = [
            _safe_image(bundle["ground_truth"][sample_index, frame_index]),
            _safe_image(bundle["dirty"][sample_index, frame_index]),
            _safe_image(bundle["baseline_learned"][sample_index, frame_index]),
            _safe_image(bundle["residual_refinement"][sample_index, frame_index]),
            _normalize_map(bundle["uncertainty"][sample_index, frame_index]),
        ]
        rgb = _render_panel(
            arrays=arrays,
            labels=labels,
            cmaps=cmaps,
            row_title=f"{row_title}\nframe {frame_index}",
            figure_size=(13.6, 3.0),
        )
        frames.append(Image.fromarray(rgb))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=700,
        loop=0,
    )
    return {
        "prediction_path": str(prediction_path),
        "sample_index": sample_index,
        "frame_indices": frame_indices,
        "row_title": row_title,
    }


def _load_mask(dataset_path: str | Path) -> np.ndarray:
    with np.load(Path(dataset_path)) as payload:
        return payload["mask"]


def save_sparse_uv_killer_figure(
    prediction_path: str | Path,
    output_png: str | Path,
    output_svg: str | Path,
    selection_manifest: str | Path,
) -> dict[str, Any]:
    """Save a main-result qualitative figure for the sparse-uv condition."""

    bundle = load_predictions(prediction_path)
    sample_index, frame_index = select_best_improvement(
        bundle,
        reference_key="baseline_data_consistent",
        candidate_key="residual_refinement",
    )

    columns = [
        ("ground_truth", IMAGE_CMAP),
        ("dirty", IMAGE_CMAP),
        ("baseline_learned", IMAGE_CMAP),
        ("baseline_data_consistent", IMAGE_CMAP),
        ("residual_refinement", IMAGE_CMAP),
        ("uncertainty", UNCERTAINTY_CMAP),
    ]
    fig, axes = plt.subplots(2, len(columns), figsize=(15.0, 5.4))
    fig.patch.set_facecolor("white")

    for column_index, (key, cmap) in enumerate(columns):
        image = bundle[key][sample_index, frame_index]
        if key == "uncertainty":
            image = _normalize_map(image)
        else:
            image = _safe_image(image)
        axes[0, column_index].imshow(image, cmap=cmap, interpolation="nearest")
        axes[0, column_index].set_xticks([])
        axes[0, column_index].set_yticks([])
        axes[0, column_index].set_title(DISPLAY_LABELS[key], fontsize=11, pad=6)

        if key in {"uncertainty", "ground_truth"}:
            error_map = np.zeros_like(bundle["ground_truth"][sample_index, frame_index], dtype=np.float32)
        else:
            error_map = np.abs(bundle[key][sample_index, frame_index] - bundle["ground_truth"][sample_index, frame_index])
            error_map = _normalize_map(error_map)
        axes[1, column_index].imshow(error_map, cmap=ERROR_CMAP, interpolation="nearest")
        axes[1, column_index].set_xticks([])
        axes[1, column_index].set_yticks([])
        if column_index == 0:
            axes[0, column_index].set_ylabel("Reconstruction", fontsize=11)
            axes[1, column_index].set_ylabel("Abs. error", fontsize=11)

    fig.suptitle(
        f"Sparse-uv condition: residual refinement improves over the data-consistent baseline on a documented high-gain example (sample {sample_index}, frame {frame_index})",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    output_png = Path(output_png)
    output_svg = Path(output_svg)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "prediction_path": str(prediction_path),
        "sample_index": sample_index,
        "frame_index": frame_index,
        "selection_rule": "maximum residual-refinement gain over baseline_data_consistent by sample MSE, then by frame MSE",
    }
    Path(selection_manifest).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def save_realism_bridge_figure(
    prediction_path: str | Path,
    dataset_path: str | Path,
    output_png: str | Path,
    output_svg: str | Path,
    selection_manifest: str | Path,
) -> dict[str, Any]:
    """Save a realism-bridge figure with station-inspired uv coverage."""

    bundle = load_predictions(prediction_path)
    mask = _load_mask(dataset_path)
    sample_index, frame_index = select_best_improvement(bundle)
    mean_mask = mask[sample_index].mean(axis=0)

    columns = [
        ("uv_coverage", mean_mask, "cividis"),
        ("ground_truth", _safe_image(bundle["ground_truth"][sample_index, frame_index]), IMAGE_CMAP),
        ("dirty", _safe_image(bundle["dirty"][sample_index, frame_index]), IMAGE_CMAP),
        ("baseline_learned", _safe_image(bundle["baseline_learned"][sample_index, frame_index]), IMAGE_CMAP),
        (
            "baseline_data_consistent",
            _safe_image(bundle["baseline_data_consistent"][sample_index, frame_index]),
            IMAGE_CMAP,
        ),
        ("residual_refinement", _safe_image(bundle["residual_refinement"][sample_index, frame_index]), IMAGE_CMAP),
        ("uncertainty", _normalize_map(bundle["uncertainty"][sample_index, frame_index]), UNCERTAINTY_CMAP),
    ]

    fig, axes = plt.subplots(1, len(columns), figsize=(16.5, 3.8))
    fig.patch.set_facecolor("white")
    labels = {
        "uv_coverage": "Mean uv coverage",
        **DISPLAY_LABELS,
    }
    for axis, (key, image, cmap) in zip(axes, columns):
        axis.imshow(image, cmap=cmap, interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(labels[key], fontsize=11, pad=6)
        border_color = "#1d4ed8" if key == "residual_refinement" else "#d1d5db"
        border_width = 2.0 if key == "residual_refinement" else 0.8
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(border_width)
            spine.set_edgecolor(border_color)

    fig.suptitle(
        f"Realism bridge with station-inspired rotating baselines (sample {sample_index}, frame {frame_index})",
        fontsize=14,
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    output_png = Path(output_png)
    output_svg = Path(output_svg)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "prediction_path": str(prediction_path),
        "dataset_path": str(dataset_path),
        "sample_index": sample_index,
        "frame_index": frame_index,
        "selection_rule": "maximum residual-refinement gain over baseline_learned by sample MSE, then by frame MSE",
    }
    Path(selection_manifest).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _risk_coverage_curve(
    target: np.ndarray,
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    min_coverage: float = 0.1,
    num_points: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    errors = ((target - prediction) ** 2).reshape(-1)
    uncertainties = uncertainty.reshape(-1)
    ordering = np.argsort(uncertainties)
    sorted_errors = errors[ordering]
    coverages = np.linspace(min_coverage, 1.0, num_points)
    risks = []
    for coverage in coverages:
        keep = max(1, int(round(coverage * sorted_errors.shape[0])))
        risks.append(float(np.mean(sorted_errors[:keep])))
    return coverages, np.asarray(risks, dtype=np.float32)


def save_uncertainty_risk_coverage_figure(
    conditions: list[PredictionCondition],
    output_png: str | Path,
    output_svg: str | Path,
) -> None:
    """Save a risk-coverage comparison across multiple conditions."""

    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    fig.patch.set_facecolor("white")
    palette = ["#1d4ed8", "#dc2626", "#059669", "#7c3aed", "#b45309"]

    for color, condition in zip(palette, conditions):
        bundle = load_predictions(condition.prediction_path)
        coverages, risks = _risk_coverage_curve(
            target=bundle["ground_truth"],
            prediction=bundle["residual_refinement"],
            uncertainty=bundle["uncertainty"],
        )
        axis.plot(coverages, risks, label=condition.title, color=color, linewidth=2.0)

    axis.set_xlabel("Coverage kept (lowest uncertainty first)")
    axis.set_ylabel("Mean squared error on retained pixels")
    axis.set_title("Residual-refinement risk-coverage behavior across paper conditions")
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    output_png = Path(output_png)
    output_svg = Path(output_svg)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)
