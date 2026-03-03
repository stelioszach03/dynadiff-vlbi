"""Feature formatting helpers for visibility-conditioned models."""

from __future__ import annotations

import numpy as np


def build_temporal_uv_grid(image_size: int, sequence_length: int) -> np.ndarray:
    """Construct a normalized uv-coordinate grid repeated across time."""

    freqs = np.fft.fftshift(np.fft.fftfreq(image_size)).astype(np.float32)
    freqs = freqs / max(float(np.max(np.abs(freqs))), 1e-6)
    vv, uu = np.meshgrid(freqs, freqs, indexing="ij")
    uv_grid = np.stack([uu, vv], axis=0).astype(np.float32)
    return np.repeat(uv_grid[:, None, :, :], sequence_length, axis=1).astype(np.float32)


def format_dirty_input(dirty: np.ndarray) -> np.ndarray:
    """Convert a dirty image sequence into a channel-first tensor."""

    return dirty[None].astype(np.float32)


def format_visibility_tensor(
    vis_real: np.ndarray,
    vis_imag: np.ndarray,
    mask: np.ndarray,
    representation: str = "real_imag",
    include_mask_channel: bool = True,
    include_uv_coords: bool = False,
    uv_coords: np.ndarray | None = None,
) -> np.ndarray:
    """Build a stable channel-first visibility tensor for learning."""

    if representation != "real_imag":
        raise ValueError(f"Unsupported visibility representation '{representation}'.")

    observed_real = vis_real * mask
    observed_imag = vis_imag * mask
    scale = max(float(np.max(np.sqrt(observed_real**2 + observed_imag**2))), 1e-6)
    channels = [
        (observed_real / scale).astype(np.float32),
        (observed_imag / scale).astype(np.float32),
    ]
    if include_mask_channel:
        channels.append(mask.astype(np.float32))
    if include_uv_coords:
        if uv_coords is None:
            uv_coords = build_temporal_uv_grid(image_size=mask.shape[-1], sequence_length=mask.shape[0])
        channels.extend([uv_coords[0].astype(np.float32), uv_coords[1].astype(np.float32)])
    return np.stack(channels, axis=0).astype(np.float32)
