"""CLEAN deconvolution baseline for radio interferometry.

Implements a Hogbom-style CLEAN algorithm adapted for the benchmark's
Fourier-masking measurement model. CLEAN is the standard baseline in
radio astronomy imaging and its inclusion is expected by MNRAS reviewers.

The algorithm iteratively:
  1. Finds the peak in the dirty image (residual map)
  2. Subtracts a fraction (loop gain) of the dirty beam at that location
  3. Records the component in a model image
  4. Convolves the final model with a restoring beam (Gaussian)

This implementation works within the benchmark's deterministic holdout
protocol: it uses only the support-set measurements for imaging.

References:
  - Hogbom (1974), A&AS 15, 417 (original CLEAN)
  - Clark (1980), A&A 89, 377 (Clark CLEAN variant)
  - Cornwell (2008), IEEE JSTSP 2, 793 (multi-scale CLEAN)
"""

from __future__ import annotations

import numpy as np

from dynadiff_vlbi.physics.fourier_operator import fft2c, ifft2c


def _gaussian_beam(image_size: int, fwhm_px: float) -> np.ndarray:
    """Generate a 2D Gaussian restoring beam."""
    sigma = fwhm_px / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    center = (image_size - 1) / 2.0
    yy, xx = np.indices((image_size, image_size), dtype=np.float64)
    beam = np.exp(-0.5 * (((xx - center) / sigma) ** 2 + ((yy - center) / sigma) ** 2))
    beam /= beam.sum()
    return beam.astype(np.float32)


def clean_deconvolve(
    dirty_image: np.ndarray,
    mask: np.ndarray,
    measurements: np.ndarray | None = None,
    loop_gain: float = 0.1,
    n_iterations: int = 500,
    threshold: float = 1e-4,
    beam_fwhm_px: float = 3.0,
) -> np.ndarray:
    """Hogbom CLEAN deconvolution for a single 2D frame.

    Args:
        dirty_image: (H, W) dirty image from inverse FFT
        mask: (H, W) Fourier-domain sampling mask
        measurements: (H, W) complex visibilities (if None, derived from dirty)
        loop_gain: Fraction of peak subtracted per iteration
        n_iterations: Maximum CLEAN iterations
        threshold: Stop when peak residual < threshold * initial peak
        beam_fwhm_px: FWHM of restoring Gaussian beam in pixels

    Returns:
        (H, W) CLEAN-restored image
    """
    H, W = dirty_image.shape

    # Dirty beam (PSF): inverse FFT of the sampling mask
    dirty_beam = np.real(ifft2c(mask.astype(np.complex64)))
    dirty_beam_peak = np.abs(dirty_beam).max()
    if dirty_beam_peak > 1e-8:
        dirty_beam /= dirty_beam_peak

    # Initialize residual
    residual = dirty_image.copy().astype(np.float64)
    model = np.zeros((H, W), dtype=np.float64)

    initial_peak = np.abs(residual).max()
    stop_threshold = threshold * initial_peak

    center_y, center_x = H // 2, W // 2

    for iteration in range(n_iterations):
        # Find peak in residual
        peak_val = np.abs(residual).max()
        if peak_val < stop_threshold:
            break

        peak_idx = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
        peak_y, peak_x = peak_idx

        # Subtract shifted dirty beam
        component = loop_gain * residual[peak_y, peak_x]
        model[peak_y, peak_x] += component

        # Shift dirty beam to peak location
        shift_y = peak_y - center_y
        shift_x = peak_x - center_x
        shifted_beam = np.roll(np.roll(dirty_beam, shift_y, axis=0), shift_x, axis=1)
        residual -= component * shifted_beam

    # Restore: convolve model with Gaussian beam + add residual
    restoring_beam = _gaussian_beam(H, beam_fwhm_px)
    model_ft = fft2c(model.astype(np.float32))
    beam_ft = fft2c(restoring_beam)
    restored = np.real(ifft2c(model_ft * beam_ft)) + residual.astype(np.float32)

    # Ensure non-negative and normalize
    restored = np.clip(restored, 0.0, None)
    peak = restored.max()
    if peak > 1e-8:
        restored /= peak
    return restored.astype(np.float32)


def clean_reconstruct_sequence(
    dirty_sequence: np.ndarray,
    mask_sequence: np.ndarray,
    measurements_sequence: np.ndarray | None = None,
    loop_gain: float = 0.1,
    n_iterations: int = 500,
    threshold: float = 1e-4,
    beam_fwhm_px: float = 3.0,
) -> np.ndarray:
    """Apply CLEAN to each frame in a temporal sequence.

    Args:
        dirty_sequence: (T, H, W) dirty images
        mask_sequence: (T, H, W) Fourier sampling masks
        measurements_sequence: (T, H, W) complex visibilities (optional)
        loop_gain, n_iterations, threshold, beam_fwhm_px: CLEAN parameters

    Returns:
        (T, H, W) CLEAN-restored sequence
    """
    T = dirty_sequence.shape[0]
    restored = np.zeros_like(dirty_sequence)

    for t in range(T):
        meas = measurements_sequence[t] if measurements_sequence is not None else None
        restored[t] = clean_deconvolve(
            dirty_image=dirty_sequence[t],
            mask=mask_sequence[t],
            measurements=meas,
            loop_gain=loop_gain,
            n_iterations=n_iterations,
            threshold=threshold,
            beam_fwhm_px=beam_fwhm_px,
        )

    return restored
