"""Noise models for synthetic Fourier-domain measurements."""

from __future__ import annotations

import numpy as np

from dynadiff_vlbi.physics.sampling import conjugate_index
from dynadiff_vlbi.utils.config import NoiseConfig


def add_complex_gaussian_noise(
    measurements: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add circular complex Gaussian noise to Fourier measurements."""

    if noise_std <= 0.0:
        return measurements.astype(np.complex64, copy=True)
    scale = noise_std / np.sqrt(2.0)
    noise = rng.normal(scale=scale, size=measurements.shape) + 1j * rng.normal(
        scale=scale,
        size=measurements.shape,
    )
    return (measurements + noise).astype(np.complex64)


def _complex_gaussian_sample(
    rng: np.random.Generator,
    noise_std: float,
) -> np.complex64:
    if noise_std <= 0.0:
        return np.complex64(0.0 + 0.0j)
    scale = noise_std / np.sqrt(2.0)
    return np.complex64(rng.normal(scale=scale) + 1j * rng.normal(scale=scale))


def apply_structured_visibility_corruption(
    clean_visibilities: np.ndarray,
    mask: np.ndarray,
    noise_config: NoiseConfig,
    rng: np.random.Generator,
    baseline_pairs: np.ndarray | None = None,
    frame_uv_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Apply station-inspired gain and baseline-dependent noise corruptions."""

    sampled = (clean_visibilities * mask).astype(np.complex64, copy=True)
    if (
        baseline_pairs is None
        or frame_uv_indices is None
        or baseline_pairs.size == 0
        or frame_uv_indices.size == 0
    ):
        return add_complex_gaussian_noise(sampled, noise_std=noise_config.noise_std, rng=rng)

    corrupted = np.zeros_like(sampled, dtype=np.complex64)
    filled = np.zeros_like(mask, dtype=bool)
    station_count = int(np.max(baseline_pairs)) + 1
    image_size = sampled.shape[-1]
    center = image_size // 2

    for frame_index in range(sampled.shape[0]):
        amplitudes = np.exp(
            rng.normal(loc=0.0, scale=noise_config.gain_amplitude_std, size=station_count)
        ).astype(np.float32)
        phases = rng.normal(loc=0.0, scale=noise_config.gain_phase_std, size=station_count).astype(np.float32)
        station_gains = amplitudes * np.exp(1j * phases)
        baseline_noise = np.ones(baseline_pairs.shape[0], dtype=np.float32)
        if noise_config.baseline_noise_jitter > 0.0:
            baseline_noise *= np.exp(
                rng.normal(
                    loc=0.0,
                    scale=noise_config.baseline_noise_jitter,
                    size=baseline_pairs.shape[0],
                )
            ).astype(np.float32)

        for pair_index, (first, second) in enumerate(baseline_pairs):
            row, col = frame_uv_indices[frame_index, pair_index]
            row_i = int(row)
            col_i = int(col)
            if mask[frame_index, row_i, col_i] <= 0.0:
                continue

            gain = station_gains[int(second)] * np.conj(station_gains[int(first)])
            value = sampled[frame_index, row_i, col_i] * gain
            local_noise = noise_config.noise_std * float(baseline_noise[pair_index])
            value = np.complex64(value + _complex_gaussian_sample(rng, local_noise))
            corrupted[frame_index, row_i, col_i] = value
            filled[frame_index, row_i, col_i] = True

            sym_row = conjugate_index(row_i, image_size)
            sym_col = conjugate_index(col_i, image_size)
            if mask[frame_index, sym_row, sym_col] > 0.0:
                corrupted[frame_index, sym_row, sym_col] = np.conj(value)
                filled[frame_index, sym_row, sym_col] = True

        if mask[frame_index, center, center] > 0.0 and not filled[frame_index, center, center]:
            corrupted[frame_index, center, center] = np.complex64(
                sampled[frame_index, center, center] + _complex_gaussian_sample(rng, noise_config.noise_std)
            )
            filled[frame_index, center, center] = True

    remaining = mask.astype(bool) & ~filled
    if np.any(remaining):
        generic_noisy = add_complex_gaussian_noise(sampled, noise_std=noise_config.noise_std, rng=rng)
        corrupted[remaining] = generic_noisy[remaining]

    return corrupted.astype(np.complex64)
