"""Classical sequence reconstruction baselines."""

from __future__ import annotations

import numpy as np

from dynadiff_vlbi.physics.fourier_operator import fft2c, ifft2c


def dirty_image_reconstruction(measurements: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the zero-filled inverse FFT reconstruction."""

    return ifft2c(measurements * mask).real.astype(np.float32)


def tikhonov_iterative_reconstruction(
    measurements: np.ndarray,
    mask: np.ndarray,
    lambda_reg: float = 0.02,
    iterations: int = 12,
    step_size: float = 0.8,
) -> np.ndarray:
    """Refine the dirty image with a small Tikhonov-regularized gradient descent loop."""

    estimate = dirty_image_reconstruction(measurements, mask)
    for _ in range(iterations):
        residual = mask * (fft2c(estimate) - measurements)
        gradient = ifft2c(residual).real + lambda_reg * estimate
        estimate = estimate - step_size * gradient
        estimate = np.clip(estimate, 0.0, 1.0)
    return estimate.astype(np.float32)
