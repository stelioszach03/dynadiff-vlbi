"""Sparse Fourier-domain sampling mask generation."""

from __future__ import annotations

import numpy as np

from dynadiff_vlbi.utils.config import SamplingConfig


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


def generate_temporal_uv_mask(
    image_size: int,
    sequence_length: int,
    config: SamplingConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a time-indexed uv mask with optional missing coverage per frame."""

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
    return np.stack(temporal_mask, axis=0)


def mask_coverage(mask: np.ndarray) -> float:
    """Return the average observed Fourier coverage."""

    return float(mask.mean())
