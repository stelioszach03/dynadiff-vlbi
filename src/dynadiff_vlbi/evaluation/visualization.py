"""Visualization helpers for comparing reconstructions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def select_frame_indices(sequence_length: int, num_frames: int) -> list[int]:
    """Pick a small set of evenly spaced frame indices."""

    if num_frames >= sequence_length:
        return list(range(sequence_length))
    frame_indices = np.linspace(0, sequence_length - 1, num_frames, dtype=int)
    return sorted(set(int(index) for index in frame_indices))


def save_reconstruction_panel(
    path: str | Path,
    ground_truth: np.ndarray,
    dirty: np.ndarray,
    learned_mean: np.ndarray,
    uncertainty: np.ndarray,
    frames_to_plot: int,
    dpi: int,
    tikhonov: np.ndarray | None = None,
) -> None:
    """Save a compact figure comparing target, baseline, prediction, and uncertainty."""

    frame_indices = select_frame_indices(ground_truth.shape[0], frames_to_plot)
    rows = [
        ("Ground truth", ground_truth, "inferno"),
        ("Dirty", dirty, "inferno"),
    ]
    if tikhonov is not None:
        rows.append(("Tikhonov", tikhonov, "inferno"))
    rows.extend(
        [
            ("Learned mean", learned_mean, "inferno"),
            ("Abs error", np.abs(learned_mean - ground_truth), "magma"),
            ("Uncertainty", uncertainty, "magma"),
        ]
    )

    fig, axes = plt.subplots(len(rows), len(frame_indices), figsize=(3.0 * len(frame_indices), 2.4 * len(rows)))
    if len(rows) == 1:
        axes = np.asarray([axes])
    if len(frame_indices) == 1:
        axes = axes[:, None]

    for row_index, (row_title, data, cmap) in enumerate(rows):
        for col_index, frame_index in enumerate(frame_indices):
            axis = axes[row_index, col_index]
            vmin, vmax = (0.0, 1.0) if row_title not in {"Abs error", "Uncertainty"} else (0.0, None)
            axis.imshow(data[frame_index], cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(f"Frame {frame_index}")
            if col_index == 0:
                axis.set_ylabel(row_title)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_phase2_comparison_panel(
    path: str | Path,
    ground_truth: np.ndarray,
    dirty: np.ndarray,
    tikhonov: np.ndarray,
    baseline_learned: np.ndarray | None,
    baseline_data_consistent: np.ndarray | None,
    visibility_conditioned: np.ndarray | None,
    uncertainty: np.ndarray | None,
    frames_to_plot: int,
    dpi: int,
    residual_refinement: np.ndarray | None = None,
    ccrr: np.ndarray | None = None,
) -> None:
    """Save a comparison figure including baseline and phase 2 reconstructions."""

    frame_indices = select_frame_indices(ground_truth.shape[0], frames_to_plot)
    rows = [
        ("Ground truth", ground_truth, "inferno"),
        ("Dirty", dirty, "inferno"),
        ("Tikhonov", tikhonov, "inferno"),
    ]
    if baseline_learned is not None:
        rows.append(("Baseline 3D U-Net", baseline_learned, "inferno"))
    if baseline_data_consistent is not None:
        rows.extend(
            [
                ("Baseline + data consistency", baseline_data_consistent, "inferno"),
                ("DC abs error", np.abs(baseline_data_consistent - ground_truth), "magma"),
            ]
        )
    if visibility_conditioned is not None:
        rows.extend(
            [
                ("Visibility-conditioned", visibility_conditioned, "inferno"),
                ("VC abs error", np.abs(visibility_conditioned - ground_truth), "magma"),
            ]
        )
    if residual_refinement is not None:
        rows.extend(
            [
                ("Residual refinement", residual_refinement, "inferno"),
                ("Residual abs error", np.abs(residual_refinement - ground_truth), "magma"),
            ]
        )
    if ccrr is not None:
        rows.extend(
            [
                ("CCRR", ccrr, "inferno"),
                ("CCRR abs error", np.abs(ccrr - ground_truth), "magma"),
            ]
        )
    if uncertainty is not None:
        if ccrr is not None:
            uncertainty_label = "CCRR uncertainty"
        elif residual_refinement is not None:
            uncertainty_label = "Residual uncertainty"
        else:
            uncertainty_label = "VC uncertainty"
        rows.append((uncertainty_label, uncertainty, "magma"))

    fig, axes = plt.subplots(len(rows), len(frame_indices), figsize=(3.1 * len(frame_indices), 2.4 * len(rows)))
    if len(rows) == 1:
        axes = np.asarray([axes])
    if len(frame_indices) == 1:
        axes = axes[:, None]

    for row_index, (row_title, data, cmap) in enumerate(rows):
        for col_index, frame_index in enumerate(frame_indices):
            axis = axes[row_index, col_index]
            error_rows = {
                "DC abs error",
                "VC abs error",
                "VC uncertainty",
                "Residual abs error",
                "Residual uncertainty",
                "CCRR abs error",
                "CCRR uncertainty",
            }
            vmin, vmax = (0.0, 1.0) if row_title not in error_rows else (0.0, None)
            axis.imshow(data[frame_index], cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(f"Frame {frame_index}")
            if col_index == 0:
                axis.set_ylabel(row_title)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
