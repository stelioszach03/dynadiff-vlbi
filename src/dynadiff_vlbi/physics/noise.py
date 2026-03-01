"""Noise models for synthetic Fourier-domain measurements."""

from __future__ import annotations

import numpy as np


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
