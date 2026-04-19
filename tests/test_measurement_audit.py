from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dynadiff_vlbi.evaluation.measurement_audit import (
    MeasurementAuditError,
    build_deterministic_heldout_masks,
    compute_pre_post_dc_audit,
    summarize_triangle_support,
    validate_prediction_bundle,
)
from dynadiff_vlbi.physics.fourier_operator import fft2c
from dynadiff_vlbi.physics.sampling import conjugate_index


def test_build_deterministic_heldout_masks_is_repeatable() -> None:
    mask = np.zeros((2, 8, 8), dtype=np.float32)
    frame_uv_indices = np.asarray(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[1, 2], [3, 4], [5, 6]],
        ],
        dtype=np.int64,
    )
    for frame_index in range(mask.shape[0]):
        for row, col in frame_uv_indices[frame_index]:
            mask[frame_index, row, col] = 1.0
            mask[frame_index, conjugate_index(int(row), 8), conjugate_index(int(col), 8)] = 1.0

    heldout_first, enforced_first, metadata_first = build_deterministic_heldout_masks(
        mask=mask,
        frame_uv_indices=frame_uv_indices,
        base_seed=7,
        sample_index=3,
        fraction=0.34,
    )
    heldout_second, enforced_second, metadata_second = build_deterministic_heldout_masks(
        mask=mask,
        frame_uv_indices=frame_uv_indices,
        base_seed=7,
        sample_index=3,
        fraction=0.34,
    )

    np.testing.assert_array_equal(heldout_first, heldout_second)
    np.testing.assert_array_equal(enforced_first, enforced_second)
    assert metadata_first == metadata_second
    np.testing.assert_array_less(heldout_first + enforced_first, mask + 1e-6)


def test_summarize_triangle_support_counts_only_all_heldout_triangles() -> None:
    observed_mask = np.zeros((1, 8, 8), dtype=np.float32)
    heldout_mask = np.zeros_like(observed_mask)
    baseline_pairs = np.asarray([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
    frame_uv_indices = np.asarray([[[1, 2], [2, 3], [3, 4]]], dtype=np.int64)
    for row, col in frame_uv_indices[0]:
        observed_mask[0, row, col] = 1.0

    heldout_mask[0, 1, 2] = 1.0
    heldout_mask[0, 2, 3] = 1.0
    support = summarize_triangle_support(
        observed_mask=observed_mask,
        heldout_mask=heldout_mask,
        baseline_pairs=baseline_pairs,
        frame_uv_indices=frame_uv_indices,
        max_triangles=1,
    )

    assert support["all_heldout"] == 0
    assert support["mixed"] == 1
    assert support["enforced_only"] == 0


def test_validate_prediction_bundle_requires_pre_dc_prediction(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    ground_truth = np.zeros((1, 2, 8, 8), dtype=np.float32)
    np.savez_compressed(
        dataset_dir / "test.npz",
        ground_truth=ground_truth,
        dirty=ground_truth,
        vis_real=np.zeros_like(ground_truth),
        vis_imag=np.zeros_like(ground_truth),
        mask=np.zeros_like(ground_truth),
        ring_radius_px=np.ones((1,), dtype=np.float32),
        hotspot_coords_px=np.zeros((1, 2, 2), dtype=np.float32),
        baseline_pairs=np.zeros((0, 2), dtype=np.int32),
        frame_uv_indices=np.zeros((2, 0, 2), dtype=np.int32),
    )
    prediction_path = tmp_path / "predictions.npz"
    np.savez_compressed(
        prediction_path,
        ground_truth=ground_truth,
        baseline_prediction=ground_truth,
        ccrr=ground_truth,
    )

    with pytest.raises(MeasurementAuditError):
        validate_prediction_bundle(prediction_path=prediction_path, dataset_dir=dataset_dir)


def test_compute_pre_post_dc_audit_extracts_stage_metrics(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    ground_truth = np.zeros((1, 2, 8, 8), dtype=np.float32)
    ground_truth[0, :, 3, 4] = 1.0
    mask = np.zeros_like(ground_truth)
    frame_uv_indices = np.asarray([[[2, 3]], [[2, 3]]], dtype=np.int32)
    baseline_pairs = np.asarray([[0, 1]], dtype=np.int32)
    for frame_index in range(mask.shape[1]):
        row, col = frame_uv_indices[frame_index, 0]
        mask[0, frame_index, row, col] = 1.0
        mask[0, frame_index, conjugate_index(int(row), 8), conjugate_index(int(col), 8)] = 1.0
    measurements = fft2c(ground_truth[0]) * mask[0]
    np.savez_compressed(
        dataset_dir / "test.npz",
        ground_truth=ground_truth,
        dirty=np.zeros_like(ground_truth),
        vis_real=measurements.real[None].astype(np.float32),
        vis_imag=measurements.imag[None].astype(np.float32),
        mask=mask.astype(np.float32),
        ring_radius_px=np.ones((1,), dtype=np.float32),
        hotspot_coords_px=np.zeros((1, 2, 2), dtype=np.float32),
        baseline_pairs=baseline_pairs,
        frame_uv_indices=frame_uv_indices,
    )
    prediction_path = tmp_path / "predictions.npz"
    np.savez_compressed(
        prediction_path,
        ground_truth=ground_truth,
        baseline_prediction=np.zeros_like(ground_truth),
        pre_dc_prediction=(0.5 * ground_truth).astype(np.float32),
        ccrr=ground_truth,
    )

    summary = compute_pre_post_dc_audit(prediction_path=prediction_path, dataset_dir=dataset_dir)

    assert summary["stages"]["baseline_prediction"]["sample_count"] == 1
    assert summary["stages"]["baseline_prediction"]["observed_visibility_rmse_mean"] > summary["stages"]["pre_dc_prediction"]["observed_visibility_rmse_mean"]
    assert summary["stages"]["pre_dc_prediction"]["observed_visibility_rmse_mean"] > summary["stages"]["ccrr_post_dc"]["observed_visibility_rmse_mean"]
