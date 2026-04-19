"""Forward operators for generalization experiments across inverse problems.

Provides a unified interface for different linear forward operators that
share the same support-target holdout evaluation protocol. This enables
the earned-vs-enforced benchmark to generalize beyond VLBI to:

  1. Accelerated MRI (Fourier undersampling with Cartesian/radial masks)
  2. Sparse-view CT (Radon transform with limited angles)
  3. Compressed sensing (random Gaussian measurements)

Each operator implements:
  - forward(x, mask) -> y: measurement with optional mask
  - adjoint(y, mask) -> x_dirty: pseudo-inverse / dirty reconstruction
  - support_target_split(mask, fraction) -> (support_mask, target_mask)

References:
  - Lustig et al. (2007), Compressed Sensing MRI, MRM
  - Sidky & Pan (2008), Image Reconstruction in Circular CT, PMB
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class MeasurementResult:
    """Unified measurement result across forward operators."""

    measurements: np.ndarray  # complex or real measurements
    dirty_reconstruction: np.ndarray  # adjoint / pseudo-inverse
    mask: np.ndarray  # measurement mask
    support_mask: np.ndarray | None  # support subset
    target_mask: np.ndarray | None  # held-out subset


class ForwardOperator(ABC):
    """Abstract forward operator for linear inverse problems."""

    @abstractmethod
    def forward(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply forward operator with measurement mask."""

    @abstractmethod
    def adjoint(self, measurements: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply adjoint (dirty reconstruction)."""

    @abstractmethod
    def generate_mask(
        self, image_size: int, undersampling_factor: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate a sampling mask."""

    def support_target_split(
        self,
        mask: np.ndarray,
        support_fraction: float,
        rng: np.random.Generator,
        strategy: str = "random",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split measurement mask into support and target subsets."""
        observed = np.where(mask.ravel() > 0)[0]
        n_observed = len(observed)
        n_support = int(n_observed * support_fraction)

        if strategy == "random":
            rng.shuffle(observed)
            support_indices = observed[:n_support]
            target_indices = observed[n_support:]
        elif strategy == "contiguous_blocks":
            # Block-structured holdout (like baseline_track_blocks)
            block_size = max(1, n_observed // 10)
            n_blocks = n_observed // block_size
            n_support_blocks = int(n_blocks * support_fraction)
            block_order = np.arange(n_blocks)
            rng.shuffle(block_order)
            support_blocks = block_order[:n_support_blocks]
            target_blocks = block_order[n_support_blocks:]
            support_indices = np.concatenate(
                [observed[b * block_size:(b + 1) * block_size] for b in support_blocks]
            )
            target_indices = np.concatenate(
                [observed[b * block_size:(b + 1) * block_size] for b in target_blocks]
            )
        else:
            raise ValueError(f"Unknown split strategy: {strategy}")

        support_mask = np.zeros_like(mask.ravel())
        target_mask = np.zeros_like(mask.ravel())
        support_mask[support_indices] = 1.0
        target_mask[target_indices] = 1.0
        return support_mask.reshape(mask.shape), target_mask.reshape(mask.shape)


class MRIForwardOperator(ForwardOperator):
    """Accelerated MRI: 2D Fourier undersampling.

    Implements Cartesian, radial, and variable-density sampling patterns
    commonly used in compressed sensing MRI.
    """

    def __init__(self, noise_std: float = 0.01) -> None:
        self.noise_std = noise_std

    def forward(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))
        return (kspace * mask).astype(np.complex64)

    def adjoint(self, measurements: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(measurements)))).astype(np.float32)

    def generate_mask(
        self,
        image_size: int,
        undersampling_factor: float = 4.0,
        rng: np.random.Generator | None = None,
        pattern: str = "variable_density",
    ) -> np.ndarray:
        """Generate MRI sampling mask.

        Args:
            image_size: Spatial dimension.
            undersampling_factor: R = full / sampled (e.g., 4x acceleration).
            rng: Random number generator.
            pattern: "cartesian", "radial", or "variable_density".
        """
        if rng is None:
            rng = np.random.default_rng()

        target_fraction = 1.0 / undersampling_factor
        mask = np.zeros((image_size, image_size), dtype=np.float32)

        if pattern == "cartesian":
            # Random Cartesian lines
            n_lines = max(1, int(image_size * target_fraction))
            lines = rng.choice(image_size, size=n_lines, replace=False)
            mask[lines, :] = 1.0
            # Always include center lines (ACS)
            center = image_size // 2
            acs_width = max(1, image_size // 16)
            mask[center - acs_width:center + acs_width, :] = 1.0

        elif pattern == "radial":
            # Random radial spokes
            n_spokes = max(1, int(np.pi * image_size * target_fraction / 2))
            center = (image_size - 1) / 2.0
            for _ in range(n_spokes):
                angle = rng.uniform(0, np.pi)
                for r in np.linspace(-center, center, image_size * 2):
                    x = int(round(center + r * np.cos(angle)))
                    y = int(round(center + r * np.sin(angle)))
                    if 0 <= x < image_size and 0 <= y < image_size:
                        mask[y, x] = 1.0

        elif pattern == "variable_density":
            # Variable-density random sampling (polynomial decay)
            center = (image_size - 1) / 2.0
            yy, xx = np.indices((image_size, image_size))
            dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2) / center
            prob = np.clip(1.0 - dist ** 2, 0.05, 1.0)
            prob *= target_fraction / prob.mean()
            prob = np.clip(prob, 0.0, 1.0)
            mask = (rng.random((image_size, image_size)) < prob).astype(np.float32)
            # Ensure center is always sampled
            acs = max(1, image_size // 16)
            c = image_size // 2
            mask[c - acs:c + acs, c - acs:c + acs] = 1.0

        return mask


class CTForwardOperator(ForwardOperator):
    """Sparse-view CT: Radon transform with limited angles.

    Uses a simplified 2D Radon transform (sinogram) for the forward model.
    The sparse-view problem withholds certain projection angles.
    """

    def __init__(self, n_angles: int = 180, noise_std: float = 0.01) -> None:
        self.n_angles = n_angles
        self.noise_std = noise_std

    def _radon(self, image: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Simplified Radon transform via rotation and summation."""
        from scipy.ndimage import rotate
        H, W = image.shape
        sinogram = np.zeros((len(angles), max(H, W)), dtype=np.float32)
        for i, angle in enumerate(angles):
            rotated = rotate(image, -angle, reshape=False, order=1)
            sinogram[i] = rotated.sum(axis=0)[:max(H, W)]
        return sinogram

    def _backproject(self, sinogram: np.ndarray, angles: np.ndarray, image_size: int) -> np.ndarray:
        """Filtered backprojection."""
        from scipy.ndimage import rotate
        recon = np.zeros((image_size, image_size), dtype=np.float64)
        # Ram-Lak filter in Fourier domain
        n_det = sinogram.shape[1]
        freq = np.fft.fftfreq(n_det)
        ram_lak = np.abs(freq)
        for i, angle in enumerate(angles):
            projection = sinogram[i]
            filtered = np.real(np.fft.ifft(np.fft.fft(projection) * ram_lak))
            # Backproject by smearing along angle
            row = np.tile(filtered, (image_size, 1))
            rotated = rotate(row, angle, reshape=False, order=1)
            recon += rotated
        recon *= np.pi / len(angles)
        return np.clip(recon, 0, None).astype(np.float32)

    def forward(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Forward Radon transform at sampled angles."""
        angles = np.linspace(0, 180, self.n_angles, endpoint=False)
        sampled_angles = angles[mask.ravel()[:self.n_angles].astype(bool)]
        return self._radon(image, sampled_angles)

    def adjoint(self, measurements: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Filtered backprojection from sparse measurements."""
        angles = np.linspace(0, 180, self.n_angles, endpoint=False)
        sampled_angles = angles[mask.ravel()[:self.n_angles].astype(bool)]
        image_size = measurements.shape[1] if measurements.ndim > 1 else 64
        return self._backproject(measurements, sampled_angles, image_size)

    def generate_mask(
        self,
        image_size: int,
        undersampling_factor: float = 4.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate angle sampling mask for sparse-view CT."""
        if rng is None:
            rng = np.random.default_rng()

        n_sampled = max(1, int(self.n_angles / undersampling_factor))
        mask = np.zeros(self.n_angles, dtype=np.float32)
        selected = rng.choice(self.n_angles, size=n_sampled, replace=False)
        mask[selected] = 1.0
        return mask


class CompressedSensingOperator(ForwardOperator):
    """Random Gaussian measurement operator for compressed sensing."""

    def __init__(self, n_measurements: int = 256, noise_std: float = 0.01, seed: int = 42) -> None:
        self.n_measurements = n_measurements
        self.noise_std = noise_std
        self._A: np.ndarray | None = None
        self.seed = seed

    def _get_matrix(self, n: int) -> np.ndarray:
        if self._A is None or self._A.shape[1] != n:
            rng = np.random.default_rng(self.seed)
            self._A = rng.standard_normal((self.n_measurements, n)).astype(np.float32) / np.sqrt(self.n_measurements)
        return self._A

    def forward(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        x = image.ravel()
        A = self._get_matrix(len(x))
        y = A @ x
        return y[mask.astype(bool)]

    def adjoint(self, measurements: np.ndarray, mask: np.ndarray) -> np.ndarray:
        A = self._get_matrix(measurements.shape[0] if mask is None else mask.sum().astype(int))
        return (A.T @ measurements).reshape(int(np.sqrt(A.shape[1])), -1).astype(np.float32)

    def generate_mask(
        self,
        image_size: int,
        undersampling_factor: float = 2.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        n_active = max(1, int(self.n_measurements / undersampling_factor))
        mask = np.zeros(self.n_measurements, dtype=np.float32)
        if rng is None:
            rng = np.random.default_rng()
        selected = rng.choice(self.n_measurements, size=n_active, replace=False)
        mask[selected] = 1.0
        return mask
