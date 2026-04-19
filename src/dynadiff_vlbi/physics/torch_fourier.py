"""Torch-centered Fourier operators for differentiable learning components."""

from __future__ import annotations

import torch


def fft2c_torch(image: torch.Tensor) -> torch.Tensor:
    """Centered 2D FFT over the last two axes."""

    return torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(image, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        ),
        dim=(-2, -1),
    )


def ifft2c_torch(kspace: torch.Tensor) -> torch.Tensor:
    """Centered 2D inverse FFT over the last two axes."""

    return torch.fft.fftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(kspace, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        ),
        dim=(-2, -1),
    )
