from __future__ import annotations

import numpy as np

from dynadiff_vlbi.data.measurement_holdout import (
    available_baseline_indices,
    baseline_track_order,
    build_structured_holdout_split,
    closure_triangle_support_counts,
)


def _toy_observation() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequence_length = 3
    image_size = 8
    baseline_count = 4
    mask = np.zeros((sequence_length, image_size, image_size), dtype=np.float32)
    frame_uv_indices = np.asarray(
        [
            [[2, 3], [2, 5], [4, 3], [4, 5]],
            [[2, 3], [2, 5], [4, 3], [4, 5]],
            [[2, 3], [2, 5], [4, 3], [4, 5]],
        ],
        dtype=np.int64,
    )
    frame_uv_coords = np.asarray(
        [
            [[-0.7, -0.1], [-0.2, 0.4], [0.2, -0.4], [0.7, 0.1]],
            [[-0.7, -0.1], [-0.2, 0.4], [0.2, -0.4], [0.7, 0.1]],
            [[-0.7, -0.1], [-0.2, 0.4], [0.2, -0.4], [0.7, 0.1]],
        ],
        dtype=np.float32,
    )
    for frame_index in range(sequence_length):
        for row, col in frame_uv_indices[frame_index]:
            mask[frame_index, row, col] = 1.0
    measurements = np.ones((sequence_length, image_size, image_size), dtype=np.complex64)
    baseline_pairs = np.asarray([[0, 1], [1, 2], [0, 2], [0, 3]], dtype=np.int64)
    return measurements, mask, frame_uv_indices, frame_uv_coords, baseline_pairs


def _toy_station_positions() -> np.ndarray:
    return np.asarray(
        [
            [-0.8, -0.1],
            [-0.2, 0.7],
            [0.5, -0.5],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )


def test_baseline_track_order_is_deterministic() -> None:
    _, _, _, frame_uv_coords, _ = _toy_observation()
    first = baseline_track_order(frame_uv_coords)
    second = baseline_track_order(frame_uv_coords.copy())
    assert np.array_equal(first, second)


def test_structured_holdout_is_deterministic_and_disjoint() -> None:
    measurements, mask, frame_uv_indices, frame_uv_coords, _ = _toy_observation()
    split_a = build_structured_holdout_split(
        measurements=measurements,
        observed_mask=mask,
        frame_uv_indices=frame_uv_indices,
        frame_uv_coords=frame_uv_coords,
        base_seed=7,
        sample_index=3,
        support_fraction=0.5,
    )
    split_b = build_structured_holdout_split(
        measurements=measurements,
        observed_mask=mask,
        frame_uv_indices=frame_uv_indices,
        frame_uv_coords=frame_uv_coords,
        base_seed=7,
        sample_index=3,
        support_fraction=0.5,
    )
    assert np.array_equal(split_a.support_mask, split_b.support_mask)
    assert np.array_equal(split_a.target_mask, split_b.target_mask)
    assert np.all((split_a.support_mask * split_a.target_mask) == 0.0)
    assert np.array_equal(split_a.support_mask + split_a.target_mask, mask)


def test_closure_support_counts_report_mixed_and_all_target() -> None:
    _, mask, frame_uv_indices, frame_uv_coords, baseline_pairs = _toy_observation()
    split = build_structured_holdout_split(
        measurements=np.ones_like(mask, dtype=np.complex64),
        observed_mask=mask,
        frame_uv_indices=frame_uv_indices,
        frame_uv_coords=frame_uv_coords,
        base_seed=5,
        sample_index=1,
        support_fraction=0.5,
    )
    counts = closure_triangle_support_counts(
        target_mask=split.target_mask,
        support_mask=split.support_mask,
        baseline_pairs=baseline_pairs,
        frame_uv_indices=frame_uv_indices,
        max_triangles=8,
    )
    assert counts["mixed"] >= 0
    assert counts["all_target"] >= 0
    assert counts["support_only"] >= 0


def test_scan_segment_holdout_is_deterministic() -> None:
    measurements, mask, frame_uv_indices, frame_uv_coords, _ = _toy_observation()
    split_a = build_structured_holdout_split(
        measurements=measurements,
        observed_mask=mask,
        frame_uv_indices=frame_uv_indices,
        frame_uv_coords=frame_uv_coords,
        base_seed=11,
        sample_index=2,
        support_fraction=0.6,
        strategy="scan_segment_blocks",
    )
    split_b = build_structured_holdout_split(
        measurements=measurements,
        observed_mask=mask,
        frame_uv_indices=frame_uv_indices,
        frame_uv_coords=frame_uv_coords,
        base_seed=11,
        sample_index=2,
        support_fraction=0.6,
        strategy="scan_segment_blocks",
    )
    assert np.array_equal(split_a.support_mask, split_b.support_mask)
    assert np.array_equal(split_a.target_mask, split_b.target_mask)
    assert split_a.strategy == "scan_segment_blocks"
    assert split_a.target_unit_count >= 1
    assert np.array_equal(split_a.support_mask + split_a.target_mask, mask)


def test_station_dropout_requires_station_metadata_and_is_deterministic() -> None:
    measurements, mask, frame_uv_indices, frame_uv_coords, baseline_pairs = _toy_observation()
    station_positions = _toy_station_positions()
    split_a = build_structured_holdout_split(
        measurements=measurements,
        observed_mask=mask,
        frame_uv_indices=frame_uv_indices,
        frame_uv_coords=frame_uv_coords,
        baseline_pairs=baseline_pairs,
        station_positions=station_positions,
        base_seed=17,
        sample_index=4,
        support_fraction=0.5,
        strategy="station_dropout",
    )
    split_b = build_structured_holdout_split(
        measurements=measurements,
        observed_mask=mask,
        frame_uv_indices=frame_uv_indices,
        frame_uv_coords=frame_uv_coords,
        baseline_pairs=baseline_pairs,
        station_positions=station_positions,
        base_seed=17,
        sample_index=4,
        support_fraction=0.5,
        strategy="station_dropout",
    )
    assert np.array_equal(split_a.support_mask, split_b.support_mask)
    assert np.array_equal(split_a.target_mask, split_b.target_mask)
    assert split_a.strategy == "station_dropout"
    assert 1 <= split_a.target_unit_count < station_positions.shape[0]
    assert np.array_equal(split_a.support_mask + split_a.target_mask, mask)


def test_missing_baselines_are_ignored_in_target_unit_counts() -> None:
    measurements, mask, frame_uv_indices, frame_uv_coords, baseline_pairs = _toy_observation()
    mask[:, frame_uv_indices[:, 3, 0], frame_uv_indices[:, 3, 1]] = 0.0
    available = available_baseline_indices(
        observed_mask=mask,
        frame_uv_indices=frame_uv_indices,
    )
    split = build_structured_holdout_split(
        measurements=measurements,
        observed_mask=mask,
        frame_uv_indices=frame_uv_indices,
        frame_uv_coords=frame_uv_coords,
        baseline_pairs=baseline_pairs,
        station_positions=_toy_station_positions(),
        base_seed=19,
        sample_index=1,
        support_fraction=0.5,
        strategy="baseline_track_blocks",
    )
    assert available.shape[0] == 3
    assert split.target_unit_count + split.support_unit_count == 3
