"""Synthetic dynamic black-hole-like sequence generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dynadiff_vlbi.data.feature_formatting import build_temporal_uv_grid
from dynadiff_vlbi.data.io import save_npz
from dynadiff_vlbi.physics.fourier_operator import FourierMeasurementOperator
from dynadiff_vlbi.physics.noise import apply_structured_visibility_corruption
from dynadiff_vlbi.physics.sampling import build_sampling_metadata, generate_temporal_uv_mask
from dynadiff_vlbi.utils.config import NoiseConfig, SamplingConfig, SyntheticSequenceConfig


def _normalized_grid(image_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = np.linspace(-1.0, 1.0, image_size, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)
    angle = np.arctan2(yy, xx)
    return xx, yy, radius, angle


def _gaussian_2d(
    xx: np.ndarray,
    yy: np.ndarray,
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
    amplitude: float,
) -> np.ndarray:
    return amplitude * np.exp(
        -0.5 * (((xx - center_x) / sigma_x) ** 2 + ((yy - center_y) / sigma_y) ** 2)
    )


def _pixel_coordinates(image_size: int, x: float, y: float) -> tuple[float, float]:
    scale = 0.5 * (image_size - 1)
    return (x + 1.0) * scale, (y + 1.0) * scale


def generate_black_hole_sequence(
    config: SyntheticSequenceConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Generate one grayscale black-hole-like movie and associated metadata."""

    xx, yy, radius, angle = _normalized_grid(config.image_size)
    base_ring = np.exp(-0.5 * ((radius - config.ring_radius) / max(config.ring_width, 1e-3)) ** 2)
    seq = np.zeros((config.sequence_length, config.image_size, config.image_size), dtype=np.float32)

    asymmetry_phase = rng.uniform(-np.pi, np.pi)
    asymmetry_drift = rng.uniform(-0.15, 0.15)
    hotspot_angle = rng.uniform(-np.pi, np.pi)
    hotspot_direction = rng.choice([-1.0, 1.0])
    hotspot_phase = rng.uniform(-np.pi, np.pi)
    global_phase = rng.uniform(-np.pi, np.pi)
    jet_angle = rng.uniform(-0.6, 0.6)
    add_second_hotspot = rng.random() < config.second_hotspot_probability
    hotspot_coords = np.zeros((config.sequence_length, 2), dtype=np.float32)

    for t in range(config.sequence_length):
        time_fraction = t / max(config.sequence_length - 1, 1)
        global_mod = 1.0 + 0.5 * config.temporal_variability * np.sin(2.0 * np.pi * time_fraction + global_phase)
        frame_phase = asymmetry_phase + asymmetry_drift * t
        asymmetry = 1.0 + config.asymmetry_strength * np.cos(angle - frame_phase)
        frame = global_mod * base_ring * asymmetry

        hotspot_theta = hotspot_angle + hotspot_direction * config.hotspot_speed * t
        hotspot_x = config.hotspot_radius * np.cos(hotspot_theta)
        hotspot_y = config.hotspot_radius * np.sin(hotspot_theta)
        hotspot_amp = config.hotspot_intensity * (
            1.0 + 0.6 * config.temporal_variability * np.sin(2.0 * np.pi * time_fraction + hotspot_phase)
        )
        frame += _gaussian_2d(
            xx=xx,
            yy=yy,
            center_x=hotspot_x,
            center_y=hotspot_y,
            sigma_x=config.hotspot_width,
            sigma_y=config.hotspot_width,
            amplitude=hotspot_amp,
        )
        hotspot_coords[t] = np.asarray(_pixel_coordinates(config.image_size, hotspot_x, hotspot_y), dtype=np.float32)

        if add_second_hotspot:
            second_theta = hotspot_theta + np.pi / 2.5
            second_x = (config.hotspot_radius - 0.06) * np.cos(second_theta)
            second_y = (config.hotspot_radius - 0.06) * np.sin(second_theta)
            frame += _gaussian_2d(
                xx=xx,
                yy=yy,
                center_x=second_x,
                center_y=second_y,
                sigma_x=config.hotspot_width * 1.2,
                sigma_y=config.hotspot_width * 1.2,
                amplitude=0.55 * config.hotspot_intensity,
            )

        if config.jet_intensity > 0.0:
            cos_a = np.cos(jet_angle)
            sin_a = np.sin(jet_angle)
            x_rot = xx * cos_a + yy * sin_a
            y_rot = -xx * sin_a + yy * cos_a
            jet = np.exp(-0.5 * (((x_rot - 0.45) / 0.23) ** 2 + (y_rot / 0.05) ** 2))
            jet *= (x_rot > 0.0).astype(np.float32)
            frame += config.jet_intensity * global_mod * jet

        frame += config.background_level
        frame = np.clip(frame, 0.0, None)
        seq[t] = frame.astype(np.float32)

    seq -= seq.min()
    seq /= max(float(seq.max()), 1e-6)
    ring_radius_px = config.ring_radius * 0.5 * (config.image_size - 1)
    metadata = {
        "ring_radius_px": float(ring_radius_px),
        "hotspot_coords_px": hotspot_coords,
    }
    return seq.astype(np.float32), metadata


def generate_split_dataset(
    num_sequences: int,
    dataset_config: SyntheticSequenceConfig,
    sampling_config: SamplingConfig,
    noise_config: NoiseConfig,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate one synthetic split with measurements and metadata."""

    rng = np.random.default_rng(seed)
    structured_corruptions = (
        sampling_config.mode == "station_tracks"
        and (
            noise_config.baseline_noise_jitter > 0.0
            or noise_config.gain_amplitude_std > 0.0
            or noise_config.gain_phase_std > 0.0
        )
    )
    operator = FourierMeasurementOperator(
        noise_std=0.0 if structured_corruptions else noise_config.noise_std,
        seed=seed + 137,
    )
    noise_rng = np.random.default_rng(seed + 137)
    uv_coords = build_temporal_uv_grid(
        image_size=dataset_config.image_size,
        sequence_length=dataset_config.sequence_length,
    )
    sampling_metadata = build_sampling_metadata(
        image_size=dataset_config.image_size,
        sequence_length=dataset_config.sequence_length,
        config=sampling_config,
        rng=np.random.default_rng(seed + 53),
    )

    ground_truth = []
    dirty = []
    vis_real = []
    vis_imag = []
    masks = []
    ring_radii = []
    hotspot_coords = []

    for _ in range(num_sequences):
        sequence, metadata = generate_black_hole_sequence(config=dataset_config, rng=rng)
        mask = generate_temporal_uv_mask(
            image_size=dataset_config.image_size,
            sequence_length=dataset_config.sequence_length,
            config=sampling_config,
            rng=rng,
            metadata=sampling_metadata,
        )
        measurement_batch = operator.forward(sequence=sequence, mask=mask)
        noisy_visibilities = measurement_batch.noisy_visibilities
        dirty_reconstruction = measurement_batch.dirty_reconstruction
        if structured_corruptions:
            noisy_visibilities = apply_structured_visibility_corruption(
                clean_visibilities=measurement_batch.clean_visibilities,
                mask=mask,
                noise_config=noise_config,
                rng=noise_rng,
                baseline_pairs=sampling_metadata.baseline_pairs,
                frame_uv_indices=sampling_metadata.frame_uv_indices,
            )
            dirty_reconstruction = operator.dirty_reconstruct(noisy_visibilities, mask)
        ground_truth.append(sequence.astype(np.float32))
        dirty.append(dirty_reconstruction.astype(np.float32))
        vis_real.append(noisy_visibilities.real.astype(np.float32))
        vis_imag.append(noisy_visibilities.imag.astype(np.float32))
        masks.append(mask.astype(np.float32))
        ring_radii.append(np.float32(metadata["ring_radius_px"]))
        hotspot_coords.append(np.asarray(metadata["hotspot_coords_px"], dtype=np.float32))

    return {
        "ground_truth": np.stack(ground_truth).astype(np.float32),
        "dirty": np.stack(dirty).astype(np.float32),
        "vis_real": np.stack(vis_real).astype(np.float32),
        "vis_imag": np.stack(vis_imag).astype(np.float32),
        "mask": np.stack(masks).astype(np.float32),
        "ring_radius_px": np.asarray(ring_radii, dtype=np.float32),
        "hotspot_coords_px": np.stack(hotspot_coords).astype(np.float32),
        "uv_coords": uv_coords.astype(np.float32),
        "station_positions": sampling_metadata.station_positions.astype(np.float32),
        "baseline_pairs": sampling_metadata.baseline_pairs.astype(np.int32),
        "frame_uv_indices": sampling_metadata.frame_uv_indices.astype(np.int32),
        "frame_uv_coords": sampling_metadata.frame_uv_coords.astype(np.float32),
    }


def generate_dataset_splits(
    output_dir: str | Path,
    dataset_config: SyntheticSequenceConfig,
    sampling_config: SamplingConfig,
    noise_config: NoiseConfig,
    base_seed: int,
) -> dict[str, Path]:
    """Generate and save train/val/test splits as NPZ files."""

    output_dir = Path(output_dir)
    split_sizes = {
        "train": dataset_config.train_size,
        "val": dataset_config.val_size,
        "test": dataset_config.test_size,
    }
    split_offsets = {"train": 0, "val": 1_000, "test": 2_000}
    saved_paths: dict[str, Path] = {}
    for split_name, split_size in split_sizes.items():
        arrays = generate_split_dataset(
            num_sequences=split_size,
            dataset_config=dataset_config,
            sampling_config=sampling_config,
            noise_config=noise_config,
            seed=base_seed + split_offsets[split_name],
        )
        split_path = output_dir / f"{split_name}.npz"
        save_npz(split_path, arrays)
        saved_paths[split_name] = split_path
    return saved_paths
