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


def build_observation_metadata_channels(
    mask: np.ndarray,
    frame_uv_coords: np.ndarray | None = None,
    frame_uv_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Build lightweight station/baseline-aware metadata maps for learning."""

    sequence_length, image_size, _ = mask.shape
    time_coords = np.linspace(-1.0, 1.0, sequence_length, dtype=np.float32)[:, None, None]
    time_map = np.broadcast_to(time_coords, (sequence_length, image_size, image_size)).astype(np.float32)

    baseline_length_map = np.zeros_like(mask, dtype=np.float32)
    baseline_angle_map = np.zeros_like(mask, dtype=np.float32)
    baseline_id_map = np.zeros_like(mask, dtype=np.float32)

    if frame_uv_coords is not None and frame_uv_indices is not None and frame_uv_coords.size > 0:
        max_length = max(float(np.linalg.norm(frame_uv_coords, axis=-1).max()), 1e-6)
        baseline_count = max(int(frame_uv_indices.shape[1]), 1)
        for frame_index in range(frame_uv_indices.shape[0]):
            for baseline_index, ((row, col), (u_coord, v_coord)) in enumerate(
                zip(frame_uv_indices[frame_index], frame_uv_coords[frame_index])
            ):
                row_int = int(row)
                col_int = int(col)
                length = float(np.sqrt(u_coord**2 + v_coord**2) / max_length)
                angle = float(np.arctan2(v_coord, u_coord) / np.pi)
                baseline_length_map[frame_index, row_int, col_int] = length
                baseline_angle_map[frame_index, row_int, col_int] = angle
                baseline_id_map[frame_index, row_int, col_int] = (
                    2.0 * baseline_index / max(baseline_count - 1, 1) - 1.0
                )

    return np.stack(
        [
            time_map.astype(np.float32),
            baseline_length_map.astype(np.float32),
            baseline_angle_map.astype(np.float32),
            baseline_id_map.astype(np.float32),
        ],
        axis=0,
    )


def format_visibility_tensor(
    vis_real: np.ndarray,
    vis_imag: np.ndarray,
    mask: np.ndarray,
    representation: str = "real_imag",
    include_mask_channel: bool = True,
    include_uv_coords: bool = False,
    uv_coords: np.ndarray | None = None,
    include_observation_metadata: bool = False,
    frame_uv_coords: np.ndarray | None = None,
    frame_uv_indices: np.ndarray | None = None,
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
    if include_observation_metadata:
        metadata_channels = build_observation_metadata_channels(
            mask=mask,
            frame_uv_coords=frame_uv_coords,
            frame_uv_indices=frame_uv_indices,
        )
        channels.extend([channel.astype(np.float32) for channel in metadata_channels])
    return np.stack(channels, axis=0).astype(np.float32)
