"""Sparse Fourier-domain sampling mask generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynadiff_vlbi.utils.config import SamplingConfig


@dataclass(frozen=True)
class SamplingMetadata:
    """Optional metadata describing the observation geometry."""

    mode: str
    station_positions: np.ndarray
    baseline_pairs: np.ndarray
    frame_uv_indices: np.ndarray
    frame_uv_coords: np.ndarray


def conjugate_index(index: int, size: int) -> int:
    """Return the centered-spectrum conjugate index for an axis."""

    center = size // 2
    return (2 * center - index) % size


def enforce_hermitian_symmetry(mask: np.ndarray) -> np.ndarray:
    """Mirror selected Fourier locations to satisfy real-image symmetry."""

    sym_mask = mask.astype(bool).copy()
    for row, col in np.argwhere(sym_mask):
        sym_row = conjugate_index(int(row), sym_mask.shape[0])
        sym_col = conjugate_index(int(col), sym_mask.shape[1])
        sym_mask[sym_row, sym_col] = True
    return sym_mask


def _empty_sampling_metadata(sequence_length: int) -> SamplingMetadata:
    return SamplingMetadata(
        mode="random_radial",
        station_positions=np.zeros((0, 2), dtype=np.float32),
        baseline_pairs=np.zeros((0, 2), dtype=np.int32),
        frame_uv_indices=np.zeros((sequence_length, 0, 2), dtype=np.int32),
        frame_uv_coords=np.zeros((sequence_length, 0, 2), dtype=np.float32),
    )


def _sampling_weights(image_size: int, radial_exponent: float) -> np.ndarray:
    coords = np.arange(image_size) - (image_size // 2)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)
    radius_norm = radius / max(radius.max(), 1.0)
    weights = np.exp(-radial_exponent * radius_norm)
    return weights.astype(np.float64)


def generate_base_mask(
    image_size: int,
    config: SamplingConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a single sparse uv mask before per-frame missing-coverage dropout."""

    total_points = image_size * image_size
    target_points = max(1, int(round(config.coverage * total_points)))
    weights = _sampling_weights(image_size=image_size, radial_exponent=config.radial_exponent)
    center = image_size // 2
    if not config.include_dc:
        weights[center, center] = 0.0
    flat_weights = weights.reshape(-1)
    if flat_weights.sum() <= 0.0:
        raise ValueError("Sampling weights are all zero; cannot build a mask.")
    probabilities = flat_weights / flat_weights.sum()
    chosen = rng.choice(total_points, size=min(target_points, total_points), replace=False, p=probabilities)
    mask = np.zeros((image_size, image_size), dtype=bool)
    mask.reshape(-1)[chosen] = True
    if config.include_dc:
        mask[center, center] = True
    if config.hermitian_symmetric:
        mask = enforce_hermitian_symmetry(mask)
    return mask


def _rotation_matrix(theta: float) -> np.ndarray:
    return np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )


def _build_station_layout(config: SamplingConfig, rng: np.random.Generator) -> np.ndarray:
    """Construct a compact irregular station layout on the unit disk."""

    if config.station_count < 4:
        raise ValueError("station_count must be at least 4 for station-inspired sampling.")

    angles = np.linspace(0.0, 2.0 * np.pi, config.station_count, endpoint=False, dtype=np.float32)
    angles += rng.uniform(-0.08, 0.08, size=config.station_count).astype(np.float32)
    radii = np.ones(config.station_count, dtype=np.float32) * 0.92
    if config.station_count >= 8:
        radii[::4] = 0.72
    if config.station_count >= 12:
        radii[1::5] = 0.58
    if config.station_jitter > 0.0:
        radii += rng.uniform(
            -config.station_jitter,
            config.station_jitter,
            size=config.station_count,
        ).astype(np.float32)
    radii = np.clip(radii, 0.35, 1.0)
    x_coords = radii * np.cos(angles)
    y_coords = radii * np.sin(angles)
    return np.stack([x_coords, y_coords], axis=1).astype(np.float32)


def _baseline_pairs(station_count: int) -> np.ndarray:
    pairs = []
    for first in range(station_count - 1):
        for second in range(first + 1, station_count):
            pairs.append([first, second])
    return np.asarray(pairs, dtype=np.int32)


def _to_grid_index(coord: np.ndarray, image_size: int) -> np.ndarray:
    coord = np.clip(coord, -1.0, 1.0)
    scaled = 0.5 * (coord + 1.0) * float(image_size - 1)
    return np.rint(scaled).astype(np.int32)


def build_sampling_metadata(
    image_size: int,
    sequence_length: int,
    config: SamplingConfig,
    rng: np.random.Generator,
) -> SamplingMetadata:
    """Build optional station-inspired geometry metadata for one split."""

    if config.mode != "station_tracks":
        return _empty_sampling_metadata(sequence_length=sequence_length)

    station_positions = _build_station_layout(config=config, rng=rng)
    baseline_pairs = _baseline_pairs(station_positions.shape[0])
    max_baseline = 1e-6
    for first, second in baseline_pairs:
        baseline = station_positions[second] - station_positions[first]
        max_baseline = max(max_baseline, float(np.linalg.norm(baseline)))

    total_rotation = np.deg2rad(config.earth_rotation_degrees)
    if sequence_length <= 1:
        frame_angles = np.zeros(1, dtype=np.float32)
    else:
        frame_angles = np.linspace(
            -0.5 * total_rotation,
            0.5 * total_rotation,
            sequence_length,
            dtype=np.float32,
        )

    frame_uv_indices = []
    frame_uv_coords = []
    for frame_angle in frame_angles:
        rotation = _rotation_matrix(float(frame_angle))
        rotated = station_positions @ rotation.T
        baseline_vectors = []
        for first, second in baseline_pairs:
            vector = (rotated[second] - rotated[first]) / max_baseline
            baseline_vectors.append(np.clip(vector, -1.0, 1.0))
        uv_coords = np.asarray(baseline_vectors, dtype=np.float32)
        grid_cols = _to_grid_index(uv_coords[:, 0], image_size=image_size)
        grid_rows = _to_grid_index(uv_coords[:, 1], image_size=image_size)
        frame_uv_indices.append(np.stack([grid_rows, grid_cols], axis=1))
        frame_uv_coords.append(uv_coords)

    return SamplingMetadata(
        mode=config.mode,
        station_positions=station_positions.astype(np.float32),
        baseline_pairs=baseline_pairs.astype(np.int32),
        frame_uv_indices=np.stack(frame_uv_indices).astype(np.int32),
        frame_uv_coords=np.stack(frame_uv_coords).astype(np.float32),
    )


def _frame_mask_from_station_tracks(
    image_size: int,
    frame_uv_indices: np.ndarray,
    config: SamplingConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.zeros((image_size, image_size), dtype=bool)
    baseline_count = frame_uv_indices.shape[0]
    if baseline_count == 0:
        return mask
    keep = np.ones(baseline_count, dtype=bool)
    if config.missing_fraction > 0.0:
        keep &= rng.random(baseline_count) >= config.missing_fraction
    for row, col in frame_uv_indices[keep]:
        mask[int(row), int(col)] = True
    center = image_size // 2
    if config.include_dc:
        mask[center, center] = True
    if config.hermitian_symmetric:
        mask = enforce_hermitian_symmetry(mask)
    return mask


def _apply_scan_gaps(
    temporal_mask: np.ndarray,
    config: SamplingConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply contiguous scan-like gaps across time."""

    if config.scan_gap_probability <= 0.0 or config.scan_gap_length <= 0:
        return temporal_mask

    gapped = temporal_mask.copy()
    sequence_length = gapped.shape[0]
    frame_index = 0
    while frame_index < sequence_length:
        if rng.random() < config.scan_gap_probability:
            gap_end = min(sequence_length, frame_index + config.scan_gap_length)
            gapped[frame_index:gap_end] = 0.0
            frame_index = gap_end
        else:
            frame_index += 1
    return gapped


def generate_temporal_uv_mask(
    image_size: int,
    sequence_length: int,
    config: SamplingConfig,
    rng: np.random.Generator,
    metadata: SamplingMetadata | None = None,
) -> np.ndarray:
    """Generate a time-indexed uv mask with optional missing coverage per frame."""

    if config.mode == "station_tracks":
        metadata = metadata or build_sampling_metadata(
            image_size=image_size,
            sequence_length=sequence_length,
            config=config,
            rng=rng,
        )
        temporal_mask = [
            _frame_mask_from_station_tracks(
                image_size=image_size,
                frame_uv_indices=metadata.frame_uv_indices[frame_index],
                config=config,
                rng=rng,
            ).astype(np.float32)
            for frame_index in range(sequence_length)
        ]
        return _apply_scan_gaps(np.stack(temporal_mask, axis=0), config=config, rng=rng)

    base_mask = generate_base_mask(image_size=image_size, config=config, rng=rng)
    center = image_size // 2
    temporal_mask = []
    for _ in range(sequence_length):
        frame_mask = base_mask.copy()
        if config.missing_fraction > 0.0:
            keep = rng.random((image_size, image_size)) >= config.missing_fraction
            frame_mask &= keep
        if config.include_dc:
            frame_mask[center, center] = True
        if config.hermitian_symmetric:
            frame_mask = enforce_hermitian_symmetry(frame_mask)
        temporal_mask.append(frame_mask.astype(np.float32))
    return _apply_scan_gaps(np.stack(temporal_mask, axis=0), config=config, rng=rng)


def mask_coverage(mask: np.ndarray) -> float:
    """Return the average observed Fourier coverage."""

    return float(mask.mean())
