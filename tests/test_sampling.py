from __future__ import annotations

import numpy as np

from dynadiff_vlbi.physics.sampling import (
    build_sampling_metadata,
    conjugate_index,
    generate_temporal_uv_mask,
    mask_coverage,
)
from dynadiff_vlbi.utils.config import SamplingConfig


def test_temporal_uv_mask_is_reproducible_and_reasonably_sparse() -> None:
    config = SamplingConfig(
        coverage=0.12,
        radial_exponent=1.2,
        missing_fraction=0.10,
        hermitian_symmetric=True,
        include_dc=True,
    )
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(11)
    mask_a = generate_temporal_uv_mask(image_size=32, sequence_length=4, config=config, rng=rng_a)
    mask_b = generate_temporal_uv_mask(image_size=32, sequence_length=4, config=config, rng=rng_b)

    assert mask_a.shape == (4, 32, 32)
    assert np.array_equal(mask_a, mask_b)
    assert 0.05 <= mask_coverage(mask_a) <= 0.30


def test_temporal_uv_mask_preserves_hermitian_symmetry() -> None:
    config = SamplingConfig(
        coverage=0.10,
        radial_exponent=1.0,
        missing_fraction=0.20,
        hermitian_symmetric=True,
        include_dc=True,
    )
    mask = generate_temporal_uv_mask(
        image_size=32,
        sequence_length=3,
        config=config,
        rng=np.random.default_rng(5),
    )
    for frame in mask:
        for row, col in np.argwhere(frame > 0.5):
            assert frame[conjugate_index(int(row), 32), conjugate_index(int(col), 32)] > 0.5


def test_station_track_sampling_builds_geometry_and_mask() -> None:
    config = SamplingConfig(
        coverage=0.08,
        radial_exponent=1.0,
        missing_fraction=0.10,
        hermitian_symmetric=True,
        include_dc=True,
        mode="station_tracks",
        station_count=10,
        earth_rotation_degrees=120.0,
        station_jitter=0.05,
    )
    rng = np.random.default_rng(17)
    metadata = build_sampling_metadata(image_size=32, sequence_length=4, config=config, rng=rng)
    mask = generate_temporal_uv_mask(
        image_size=32,
        sequence_length=4,
        config=config,
        rng=np.random.default_rng(17),
        metadata=metadata,
    )

    assert metadata.station_positions.shape == (10, 2)
    assert metadata.baseline_pairs.shape[0] == 45
    assert metadata.frame_uv_indices.shape == (4, 45, 2)
    assert mask.shape == (4, 32, 32)
    assert mask_coverage(mask) > 0.01


def test_scan_gap_sampling_can_zero_entire_frames() -> None:
    config = SamplingConfig(
        coverage=0.08,
        radial_exponent=1.0,
        missing_fraction=0.0,
        hermitian_symmetric=True,
        include_dc=True,
        mode="station_tracks",
        station_count=8,
        earth_rotation_degrees=90.0,
        station_jitter=0.05,
        scan_gap_probability=1.0,
        scan_gap_length=1,
    )
    metadata = build_sampling_metadata(
        image_size=32,
        sequence_length=4,
        config=config,
        rng=np.random.default_rng(3),
    )
    mask = generate_temporal_uv_mask(
        image_size=32,
        sequence_length=4,
        config=config,
        rng=np.random.default_rng(3),
        metadata=metadata,
    )

    assert np.any(mask.sum(axis=(1, 2)) == 0.0)
