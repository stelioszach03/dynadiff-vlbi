"""Centered FFT-based forward operator for sparse VLBI-like measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynadiff_vlbi.physics.noise import add_complex_gaussian_noise


def fft2c(image: np.ndarray) -> np.ndarray:
    """Centered 2D FFT over the last two axes."""

    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    """Centered 2D inverse FFT over the last two axes."""

    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )


@dataclass
class MeasurementBatch:
    """Container for one synthetic measurement sequence."""

    clean_visibilities: np.ndarray
    noisy_visibilities: np.ndarray
    dirty_reconstruction: np.ndarray
    mask: np.ndarray


class FourierMeasurementOperator:
    """Sparse Fourier measurement model with optional complex Gaussian noise."""

    def __init__(self, noise_std: float = 0.0, seed: int | None = None) -> None:
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

    def apply_noise(self, measurements: np.ndarray) -> np.ndarray:
        """Apply complex Gaussian noise to sampled Fourier coefficients."""

        return add_complex_gaussian_noise(
            measurements=measurements,
            noise_std=self.noise_std,
            rng=self.rng,
        )

    def dirty_reconstruct(self, measurements: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Reconstruct a dirty image sequence from masked Fourier measurements."""

        masked_measurements = measurements * mask
        dirty = ifft2c(masked_measurements).real
        return dirty.astype(np.float32)

    def forward(self, sequence: np.ndarray, mask: np.ndarray) -> MeasurementBatch:
        """Map a real image sequence to sparse noisy Fourier measurements."""

        clean_visibilities = fft2c(sequence).astype(np.complex64)
        sampled = clean_visibilities * mask
        noisy_visibilities = self.apply_noise(sampled)
        dirty_reconstruction = self.dirty_reconstruct(noisy_visibilities, mask)
        return MeasurementBatch(
            clean_visibilities=clean_visibilities,
            noisy_visibilities=noisy_visibilities.astype(np.complex64),
            dirty_reconstruction=dirty_reconstruction,
            mask=mask.astype(np.float32),
        )
