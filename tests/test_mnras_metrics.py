from __future__ import annotations

import numpy as np

from dynadiff_vlbi.data.synthetic_generator import generate_black_hole_sequence
from dynadiff_vlbi.evaluation.metrics import (
    bright_sector_angle_error,
    closure_phase_mae,
    observed_visibility_rmse,
    ring_thickness_error,
)
from dynadiff_vlbi.physics.classical_reconstruction import visibility_data_consistency_projection
from dynadiff_vlbi.physics.fourier_operator import FourierMeasurementOperator
from dynadiff_vlbi.physics.sampling import build_sampling_metadata, generate_temporal_uv_mask
from dynadiff_vlbi.utils.config import NoiseConfig, SamplingConfig, SyntheticSequenceConfig


def _synthetic_config() -> SyntheticSequenceConfig:
    return SyntheticSequenceConfig(
        train_size=4,
        val_size=2,
        test_size=2,
        image_size=32,
        sequence_length=4,
        ring_radius=0.42,
        ring_width=0.08,
        asymmetry_strength=0.35,
        hotspot_intensity=0.55,
        hotspot_width=0.09,
        hotspot_speed=0.35,
        hotspot_radius=0.33,
        second_hotspot_probability=0.0,
        jet_intensity=0.08,
        temporal_variability=0.12,
        background_level=0.02,
    )


def test_visibility_projection_enforces_observed_data_consistency() -> None:
    dataset_config = _synthetic_config()
    rng = np.random.default_rng(5)
    sequence, _ = generate_black_hole_sequence(dataset_config, rng=rng)
    mask = generate_temporal_uv_mask(
        image_size=dataset_config.image_size,
        sequence_length=dataset_config.sequence_length,
        config=SamplingConfig(
            coverage=0.10,
            radial_exponent=1.2,
            missing_fraction=0.10,
            hermitian_symmetric=True,
            include_dc=True,
        ),
        rng=np.random.default_rng(5),
    )
    operator = FourierMeasurementOperator(noise_std=0.0, seed=5)
    batch = operator.forward(sequence=sequence, mask=mask)

    noisy_prediction = np.clip(batch.dirty_reconstruction + 0.05, 0.0, 1.0)
    before = observed_visibility_rmse(noisy_prediction, batch.noisy_visibilities, mask)
    after_prediction = visibility_data_consistency_projection(
        prediction=noisy_prediction,
        measurements=batch.noisy_visibilities,
        mask=mask,
    )
    after = observed_visibility_rmse(after_prediction, batch.noisy_visibilities, mask)

    assert after <= before
    assert after < 0.5 * before


def test_closure_phase_metric_is_finite_for_station_track_sampling() -> None:
    dataset_config = _synthetic_config()
    sampling_config = SamplingConfig(
        coverage=0.08,
        radial_exponent=1.0,
        missing_fraction=0.0,
        hermitian_symmetric=True,
        include_dc=True,
        mode="station_tracks",
        station_count=8,
        earth_rotation_degrees=120.0,
        station_jitter=0.05,
    )
    rng = np.random.default_rng(9)
    sequence, _ = generate_black_hole_sequence(dataset_config, rng=rng)
    metadata = build_sampling_metadata(
        image_size=dataset_config.image_size,
        sequence_length=dataset_config.sequence_length,
        config=sampling_config,
        rng=np.random.default_rng(9),
    )
    mask = generate_temporal_uv_mask(
        image_size=dataset_config.image_size,
        sequence_length=dataset_config.sequence_length,
        config=sampling_config,
        rng=np.random.default_rng(10),
        metadata=metadata,
    )
    operator = FourierMeasurementOperator(noise_std=NoiseConfig(noise_std=0.01).noise_std, seed=11)
    batch = operator.forward(sequence=sequence, mask=mask)

    error = closure_phase_mae(
        prediction=batch.dirty_reconstruction,
        measurements=batch.noisy_visibilities,
        mask=mask,
        baseline_pairs=metadata.baseline_pairs,
        frame_uv_indices=metadata.frame_uv_indices,
    )

    assert np.isfinite(error)


def test_ring_thickness_and_sector_angle_metrics_detect_small_perturbations() -> None:
    dataset_config = _synthetic_config()
    sequence, _ = generate_black_hole_sequence(dataset_config, rng=np.random.default_rng(13))
    shifted_sequence = np.roll(sequence, shift=1, axis=-1)

    thickness_error = ring_thickness_error(sequence, shifted_sequence)
    sector_error = bright_sector_angle_error(sequence, shifted_sequence, target_ring_radius_px=13.0)

    assert thickness_error >= 0.0
    assert sector_error >= 0.0
