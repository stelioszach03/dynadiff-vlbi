"""Structured support/target holdout utilities for earned measurement consistency."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from dynadiff_vlbi.physics.classical_reconstruction import dirty_image_reconstruction
from dynadiff_vlbi.physics.sampling import conjugate_index


@dataclass(frozen=True)
class HoldoutSplit:
    """Support and target partitions of one observed measurement sequence."""

    support_mask: np.ndarray
    target_mask: np.ndarray
    support_measurements: np.ndarray
    target_measurements: np.ndarray
    support_dirty: np.ndarray
    support_fraction: float
    target_unit_count: int
    support_unit_count: int
    strategy: str


HOLDOUT_STRATEGY_LABELS: dict[str, str] = {
    "baseline_track_blocks": "Baseline-track blocks",
    "scan_segment_blocks": "Scan-segment blocks",
    "station_dropout": "Station dropout",
}

HOLDOUT_STRATEGY_DESCRIPTIONS: dict[str, str] = {
    "baseline_track_blocks": (
        "Withhold a deterministic contiguous block of baseline tracks across time. "
        "This stresses generalization across unseen baseline trajectories."
    ),
    "scan_segment_blocks": (
        "Withhold a deterministic contiguous block of scan-like time segments while retaining the DC coefficient. "
        "This stresses temporal extrapolation under missing scan windows."
    ),
    "station_dropout": (
        "Withhold all baselines incident to a deterministic subset of stations across time. "
        "This stresses station-structured missingness rather than pointwise dropout."
    ),
}


def normalized_support_fraction(fraction: float) -> float:
    """Clamp a requested support fraction into a numerically safe open interval."""

    return float(np.clip(fraction, 1e-3, 1.0))


def select_epoch_support_fraction(train_support_fractions: tuple[float, ...], epoch_index: int) -> float:
    """Cycle deterministically through configured support fractions across epochs."""

    if not train_support_fractions:
        return 1.0
    position = max(epoch_index - 1, 0) % len(train_support_fractions)
    return normalized_support_fraction(float(train_support_fractions[position]))


def deterministic_start_index(*, length: int, base_seed: int, sample_index: int, salt: int = 0) -> int:
    """Return one deterministic cyclic start index."""

    if length <= 0:
        return 0
    return (int(base_seed) * 10007 + int(sample_index) * 97 + int(salt) * 313) % length


def baseline_track_order(frame_uv_coords: np.ndarray) -> np.ndarray:
    """Return a stable baseline-track ordering by mean length and mean angle."""

    if frame_uv_coords.size == 0:
        return np.zeros((0,), dtype=np.int64)

    mean_coords = np.mean(frame_uv_coords.astype(np.float64), axis=0)
    lengths = np.linalg.norm(mean_coords, axis=-1)
    angles = np.mod(np.arctan2(mean_coords[:, 1], mean_coords[:, 0]), 2.0 * np.pi)
    baseline_indices = np.arange(mean_coords.shape[0], dtype=np.int64)
    order = np.lexsort((angles, lengths, baseline_indices))
    return order.astype(np.int64)


def available_baseline_indices(
    *,
    observed_mask: np.ndarray,
    frame_uv_indices: np.ndarray,
) -> np.ndarray:
    """Return baseline indices that are observed at least once in the sample."""

    if frame_uv_indices.size == 0:
        return np.zeros((0,), dtype=np.int64)
    available: list[int] = []
    for baseline_index in range(int(frame_uv_indices.shape[1])):
        rows = frame_uv_indices[:, baseline_index, 0].astype(np.int64)
        cols = frame_uv_indices[:, baseline_index, 1].astype(np.int64)
        frame_axis = np.arange(int(frame_uv_indices.shape[0]), dtype=np.int64)
        if np.any(observed_mask[frame_axis, rows, cols] > 0.0):
            available.append(baseline_index)
    return np.asarray(available, dtype=np.int64)


def station_order(station_positions: np.ndarray) -> np.ndarray:
    """Return a stable station ordering by polar angle and radius."""

    if station_positions.size == 0:
        return np.zeros((0,), dtype=np.int64)
    positions = np.asarray(station_positions, dtype=np.float64)
    radii = np.linalg.norm(positions, axis=-1)
    angles = np.mod(np.arctan2(positions[:, 1], positions[:, 0]), 2.0 * np.pi)
    station_indices = np.arange(positions.shape[0], dtype=np.int64)
    order = np.lexsort((radii, angles, station_indices))
    return order.astype(np.int64)


def deterministic_target_baselines(
    *,
    frame_uv_coords: np.ndarray,
    observed_mask: np.ndarray | None = None,
    frame_uv_indices: np.ndarray | None = None,
    support_fraction: float,
    base_seed: int,
    sample_index: int,
) -> np.ndarray:
    """Select a deterministic contiguous block of baseline tracks to hold out."""

    ordered = baseline_track_order(frame_uv_coords)
    if observed_mask is not None and frame_uv_indices is not None:
        available = set(
            available_baseline_indices(
                observed_mask=observed_mask,
                frame_uv_indices=frame_uv_indices,
            ).tolist()
        )
        ordered = np.asarray([index for index in ordered.tolist() if index in available], dtype=np.int64)
    baseline_count = int(ordered.shape[0])
    if baseline_count == 0:
        return np.zeros((0,), dtype=np.int64)

    support_fraction = normalized_support_fraction(support_fraction)
    target_count = max(1, int(round((1.0 - support_fraction) * baseline_count)))
    target_count = min(target_count, baseline_count)
    start = deterministic_start_index(length=baseline_count, base_seed=base_seed, sample_index=sample_index)
    selected = [ordered[(start + offset) % baseline_count] for offset in range(target_count)]
    return np.asarray(selected, dtype=np.int64)


def deterministic_target_frames(
    *,
    observed_mask: np.ndarray,
    support_fraction: float,
    base_seed: int,
    sample_index: int,
) -> np.ndarray:
    """Select a deterministic contiguous block of frames to hold out."""

    sequence_length = int(observed_mask.shape[0])
    if sequence_length == 0:
        return np.zeros((0,), dtype=np.int64)

    support_fraction = normalized_support_fraction(support_fraction)
    center = observed_mask.shape[-1] // 2
    observed_counts = observed_mask.reshape(sequence_length, -1).sum(axis=1).astype(np.float64)
    observed_counts -= observed_mask[:, center, center].astype(np.float64)
    target_total = max(1.0, float((1.0 - support_fraction) * observed_counts.sum()))
    start = deterministic_start_index(length=sequence_length, base_seed=base_seed, sample_index=sample_index, salt=17)

    best_length = 1
    best_gap = float("inf")
    for length in range(1, max(sequence_length, 2)):
        frame_indices = [(start + offset) % sequence_length for offset in range(length)]
        count = float(observed_counts[frame_indices].sum())
        gap = abs(count - target_total)
        if gap < best_gap - 1e-12 or (np.isclose(gap, best_gap) and length < best_length):
            best_gap = gap
            best_length = length

    return np.asarray([(start + offset) % sequence_length for offset in range(best_length)], dtype=np.int64)


def deterministic_target_stations(
    *,
    station_positions: np.ndarray,
    baseline_pairs: np.ndarray,
    available_baseline_indices: np.ndarray | None = None,
    support_fraction: float,
    base_seed: int,
    sample_index: int,
) -> np.ndarray:
    """Select a deterministic subset of stations whose incident baselines are held out."""

    station_positions = np.asarray(station_positions, dtype=np.float32)
    baseline_pairs = np.asarray(baseline_pairs, dtype=np.int64)
    if available_baseline_indices is not None and np.asarray(available_baseline_indices).size > 0:
        baseline_pairs = baseline_pairs[np.asarray(available_baseline_indices, dtype=np.int64)]
    station_count = int(station_positions.shape[0])
    if station_count == 0 or baseline_pairs.size == 0:
        return np.zeros((0,), dtype=np.int64)

    support_fraction = normalized_support_fraction(support_fraction)
    target_fraction = 1.0 - support_fraction
    total_baselines = float(baseline_pairs.shape[0])
    available_station_indices = sorted({int(value) for value in baseline_pairs.reshape(-1).tolist()})
    ordered = np.asarray(
        [index for index in station_order(station_positions).tolist() if index in available_station_indices],
        dtype=np.int64,
    )
    available_station_count = int(ordered.shape[0])
    if available_station_count <= 1:
        return np.zeros((0,), dtype=np.int64)

    best_station_count = 1
    best_gap = float("inf")
    start = deterministic_start_index(
        length=available_station_count,
        base_seed=base_seed,
        sample_index=sample_index,
        salt=31,
    )
    for station_subset_size in range(1, available_station_count):
        target_station_set = {
            int(ordered[(start + offset) % available_station_count]) for offset in range(station_subset_size)
        }
        target_baseline_fraction = float(
            np.mean(
                [
                    1.0 if int(first) in target_station_set or int(second) in target_station_set else 0.0
                    for first, second in baseline_pairs.tolist()
                ]
            )
        )
        gap = abs(target_baseline_fraction - target_fraction)
        if gap < best_gap - 1e-12 or (
            np.isclose(gap, best_gap) and station_subset_size < best_station_count
        ):
            best_gap = gap
            best_station_count = station_subset_size

    selected = [ordered[(start + offset) % available_station_count] for offset in range(best_station_count)]
    return np.asarray(selected, dtype=np.int64)


def _restore_dc_support(
    *,
    observed_mask: np.ndarray,
    support_mask: np.ndarray,
    target_mask: np.ndarray,
) -> None:
    """Keep the DC coefficient in the support set for numerical stability."""

    center = observed_mask.shape[-1] // 2
    if observed_mask.ndim == 3:
        support_mask[:, center, center] = observed_mask[:, center, center]
        target_mask[:, center, center] = 0.0
    else:
        support_mask[center, center] = observed_mask[center, center]
        target_mask[center, center] = 0.0


def _split_from_target_baselines(
    *,
    measurements: np.ndarray,
    observed_mask: np.ndarray,
    frame_uv_indices: np.ndarray,
    target_baseline_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct support/target masks from target baseline indices."""

    image_size = int(observed_mask.shape[-1])
    target_mask = np.zeros_like(observed_mask, dtype=np.float32)
    support_mask = observed_mask.astype(np.float32).copy()
    target_set = set(int(value) for value in np.asarray(target_baseline_indices, dtype=np.int64).tolist())
    for frame_index in range(observed_mask.shape[0]):
        for baseline_index in target_set:
            row, col = frame_uv_indices[frame_index, baseline_index]
            row_int = int(row)
            col_int = int(col)
            if observed_mask[frame_index, row_int, col_int] <= 0.0:
                continue
            target_mask[frame_index, row_int, col_int] = 1.0
            support_mask[frame_index, row_int, col_int] = 0.0
            conj_row = conjugate_index(row_int, image_size)
            conj_col = conjugate_index(col_int, image_size)
            if observed_mask[frame_index, conj_row, conj_col] > 0.0:
                target_mask[frame_index, conj_row, conj_col] = 1.0
                support_mask[frame_index, conj_row, conj_col] = 0.0

    _restore_dc_support(observed_mask=observed_mask, support_mask=support_mask, target_mask=target_mask)
    _ = measurements
    return support_mask.astype(np.float32), target_mask.astype(np.float32)


def build_structured_holdout_split(
    *,
    measurements: np.ndarray,
    observed_mask: np.ndarray,
    frame_uv_indices: np.ndarray,
    frame_uv_coords: np.ndarray,
    baseline_pairs: np.ndarray | None = None,
    station_positions: np.ndarray | None = None,
    base_seed: int,
    sample_index: int,
    support_fraction: float,
    strategy: str = "baseline_track_blocks",
) -> HoldoutSplit:
    """Split observed coefficients into support and target sets with structured holdout."""

    if strategy not in HOLDOUT_STRATEGY_DESCRIPTIONS:
        raise ValueError(f"Unsupported holdout strategy '{strategy}'.")

    support_fraction = normalized_support_fraction(support_fraction)
    if support_fraction >= 0.999:
        support_mask = observed_mask.astype(np.float32)
        target_mask = np.zeros_like(observed_mask, dtype=np.float32)
        return HoldoutSplit(
            support_mask=support_mask,
            target_mask=target_mask,
            support_measurements=(measurements * support_mask).astype(np.complex64),
            target_measurements=np.zeros_like(measurements, dtype=np.complex64),
            support_dirty=dirty_image_reconstruction(measurements=measurements, mask=support_mask),
            support_fraction=support_fraction,
            target_unit_count=0,
            support_unit_count=int(frame_uv_indices.shape[1]),
            strategy=strategy,
        )

    if strategy == "baseline_track_blocks":
        available_baselines = available_baseline_indices(
            observed_mask=observed_mask,
            frame_uv_indices=frame_uv_indices,
        )
        target_baselines = deterministic_target_baselines(
            frame_uv_coords=frame_uv_coords,
            observed_mask=observed_mask,
            frame_uv_indices=frame_uv_indices,
            support_fraction=support_fraction,
            base_seed=base_seed,
            sample_index=sample_index,
        )
        support_mask, target_mask = _split_from_target_baselines(
            measurements=measurements,
            observed_mask=observed_mask,
            frame_uv_indices=frame_uv_indices,
            target_baseline_indices=target_baselines,
        )
        target_unit_count = int(target_baselines.shape[0])
        support_unit_count = max(int(available_baselines.shape[0]) - target_unit_count, 0)
    elif strategy == "scan_segment_blocks":
        target_frames = deterministic_target_frames(
            observed_mask=observed_mask,
            support_fraction=support_fraction,
            base_seed=base_seed,
            sample_index=sample_index,
        )
        target_mask = np.zeros_like(observed_mask, dtype=np.float32)
        support_mask = observed_mask.astype(np.float32).copy()
        target_mask[target_frames] = observed_mask[target_frames]
        support_mask[target_frames] = 0.0
        _restore_dc_support(observed_mask=observed_mask, support_mask=support_mask, target_mask=target_mask)
        target_unit_count = int(target_frames.shape[0])
        support_unit_count = max(int(observed_mask.shape[0]) - target_unit_count, 0)
    else:
        if baseline_pairs is None or station_positions is None:
            raise ValueError("station_dropout requires baseline_pairs and station_positions.")
        available_baselines = available_baseline_indices(
            observed_mask=observed_mask,
            frame_uv_indices=frame_uv_indices,
        )
        target_stations = deterministic_target_stations(
            station_positions=station_positions,
            baseline_pairs=baseline_pairs,
            available_baseline_indices=available_baselines,
            support_fraction=support_fraction,
            base_seed=base_seed,
            sample_index=sample_index,
        )
        target_station_set = set(int(value) for value in target_stations.tolist())
        target_baseline_indices = [
            baseline_index
            for baseline_index in available_baselines.tolist()
            for first, second in [np.asarray(baseline_pairs, dtype=np.int64)[baseline_index]]
            if int(first) in target_station_set or int(second) in target_station_set
        ]
        support_mask, target_mask = _split_from_target_baselines(
            measurements=measurements,
            observed_mask=observed_mask,
            frame_uv_indices=frame_uv_indices,
            target_baseline_indices=np.asarray(target_baseline_indices, dtype=np.int64),
        )
        target_unit_count = len(target_station_set)
        available_station_count = len(
            {
                int(value)
                for baseline_index in available_baselines.tolist()
                for value in np.asarray(baseline_pairs, dtype=np.int64)[baseline_index].tolist()
            }
        )
        support_unit_count = max(int(available_station_count) - target_unit_count, 0)

    support_measurements = (measurements * support_mask).astype(np.complex64)
    target_measurements = (measurements * target_mask).astype(np.complex64)
    support_dirty = dirty_image_reconstruction(measurements=measurements, mask=support_mask)
    return HoldoutSplit(
        support_mask=support_mask.astype(np.float32),
        target_mask=target_mask.astype(np.float32),
        support_measurements=support_measurements,
        target_measurements=target_measurements,
        support_dirty=support_dirty.astype(np.float32),
        support_fraction=support_fraction,
        target_unit_count=target_unit_count,
        support_unit_count=support_unit_count,
        strategy=strategy,
    )


def closure_triangle_support_counts(
    *,
    target_mask: np.ndarray,
    support_mask: np.ndarray,
    baseline_pairs: np.ndarray,
    frame_uv_indices: np.ndarray,
    max_triangles: int,
) -> dict[str, int]:
    """Count closure support for all-target and mixed-target triangles."""

    if baseline_pairs.size == 0 or frame_uv_indices.size == 0:
        return {"all_target": 0, "mixed": 0, "support_only": 0}

    pair_lookup = {
        tuple(pair.tolist()): pair_index for pair_index, pair in enumerate(np.asarray(baseline_pairs, dtype=np.int64))
    }
    station_count = int(np.max(baseline_pairs)) + 1
    triangles = list(combinations(range(station_count), 3))[:max_triangles]
    counts = {"all_target": 0, "mixed": 0, "support_only": 0}
    for frame_index in range(target_mask.shape[0]):
        frame_target = target_mask[frame_index].astype(bool)
        frame_support = support_mask[frame_index].astype(bool)
        frame_indices = frame_uv_indices[frame_index]
        for first, second, third in triangles:
            pair_ab = pair_lookup.get((first, second))
            pair_bc = pair_lookup.get((second, third))
            pair_ac = pair_lookup.get((first, third))
            if pair_ab is None or pair_bc is None or pair_ac is None:
                continue
            row_ab, col_ab = frame_indices[pair_ab]
            row_bc, col_bc = frame_indices[pair_bc]
            row_ac, col_ac = frame_indices[pair_ac]
            target_edges = [
                frame_target[row_ab, col_ab],
                frame_target[row_bc, col_bc],
                frame_target[row_ac, col_ac],
            ]
            support_edges = [
                frame_support[row_ab, col_ab],
                frame_support[row_bc, col_bc],
                frame_support[row_ac, col_ac],
            ]
            if all(target_edges):
                counts["all_target"] += 1
            elif any(target_edges):
                counts["mixed"] += 1
            elif all(support_edges):
                counts["support_only"] += 1
    return counts
