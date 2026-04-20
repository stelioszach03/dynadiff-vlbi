"""Adapter: ``VisibilityConditionedDataset`` -> oracle training batches.

Loads benchmark-real samples (the same ones ``Phase2Trainer`` and the
EMC evaluator consume), applies a randomly-chosen deterministic holdout
split per (sample, frame), builds the partial-2D-DFT measurement
operator at the observed row/col positions, computes the Woodbury
posterior-variance teacher, and stacks the result into a padded oracle
batch with an attention-safe ``support_padding_mask``.

By construction, this trains the oracle on the *exact* data distribution
(UV coverage, noise characteristics, image statistics) that the benchmark
evaluates on -- no distribution-shift gap between oracle training and
benchmark partition-time use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping

import numpy as np
import torch

from dynadiff_vlbi.data.measurement_holdout import build_structured_holdout_split
from dynadiff_vlbi.oracle.teacher import compute_importance_teacher


DETERMINISTIC_STRATEGIES: tuple[str, ...] = (
    "baseline_track_blocks",
    "scan_segment_blocks",
    "station_dropout",
)


@dataclass
class OracleDataAdapterConfig:
    """Hyperparameters for the oracle data adapter."""

    support_fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    signal_var: float = 1.0
    noise_var: float = 0.01
    # Max support / target positions to retain per frame. Observations
    # above the cap are truncated to keep teacher computation tractable
    # at 128x128 scale; below the cap we pad and mask.
    max_support: int = 256
    max_target: int = 256
    # Frames per sample are independent oracle training examples.
    use_all_frames: bool = True


def _build_partial_dft_rows(
    positions: np.ndarray,  # [M, 2] of (row, col) in the HxW grid
    image_size: int,
) -> np.ndarray:
    """Rows of the 2D DFT matrix at the given (row, col) positions.

    Returns a complex64 array of shape [M, HxW] where
        A[k, m*W + n] = exp(-2πi (row_k * m + col_k * n) / H) / sqrt(H*W)

    With H == W == image_size.
    """
    if positions.shape[0] == 0:
        return np.zeros((0, image_size * image_size), dtype=np.complex64)
    rows = positions[:, 0].astype(np.float64)
    cols = positions[:, 1].astype(np.float64)
    m_grid, n_grid = np.meshgrid(
        np.arange(image_size, dtype=np.float64),
        np.arange(image_size, dtype=np.float64),
        indexing="ij",
    )  # [H, W]
    phase = -2.0 * math.pi * (
        np.outer(rows, m_grid.ravel()) / image_size
        + np.outer(cols, n_grid.ravel()) / image_size
    )
    scale = 1.0 / math.sqrt(image_size * image_size)
    return (np.exp(1j * phase) * scale).astype(np.complex64)


def _frame_oracle_example(
    *,
    measurements: np.ndarray,          # [H, W] complex64
    observed_mask: np.ndarray,         # [H, W] float32
    support_mask: np.ndarray,          # [H, W] float32 (from holdout split)
    target_mask: np.ndarray,           # [H, W] float32
    image_size: int,
    frame_index: int,
    sequence_length: int,
    signal_var: float,
    noise_var: float,
) -> tuple[dict[str, np.ndarray] | None, int, int]:
    """Produce one oracle example from a single frame."""

    sup_positions = np.argwhere(support_mask > 0.0).astype(np.int64)  # [m_sup, 2]
    tgt_positions = np.argwhere(target_mask > 0.0).astype(np.int64)   # [m_tgt, 2]
    if sup_positions.shape[0] == 0 or tgt_positions.shape[0] == 0:
        return None, 0, 0

    A_sup = _build_partial_dft_rows(sup_positions, image_size)
    A_tgt = _build_partial_dft_rows(tgt_positions, image_size)

    sup_vals = measurements[sup_positions[:, 0], sup_positions[:, 1]]
    support_vis = np.stack([sup_vals.real, sup_vals.imag, np.abs(sup_vals)], axis=-1).astype(np.float32)

    # Normalise UV tokens to [-1, 1] using the image grid extent.
    def _uv_norm(pos: np.ndarray) -> np.ndarray:
        u = pos[:, 0].astype(np.float32) / max(image_size - 1, 1) * 2.0 - 1.0
        v = pos[:, 1].astype(np.float32) / max(image_size - 1, 1) * 2.0 - 1.0
        t = np.full((pos.shape[0],), (
            (frame_index / max(sequence_length - 1, 1)) * 2.0 - 1.0
            if sequence_length > 1
            else 0.0
        ), dtype=np.float32)
        return np.stack([u, v, t], axis=-1)

    support_uv = _uv_norm(sup_positions)
    target_uv = _uv_norm(tgt_positions)

    teacher = (
        compute_importance_teacher(
            torch.from_numpy(A_tgt),
            torch.from_numpy(A_sup),
            signal_var=signal_var,
            noise_var=noise_var,
        )
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return {
        "support_visibilities": support_vis,
        "support_uv": support_uv,
        "target_uv": target_uv,
        "teacher_importance": teacher,
    }, sup_positions.shape[0], tgt_positions.shape[0]


def _pad_stack(
    arrays: list[np.ndarray],
    max_len: int,
    feature_dim: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad per-example arrays to a common max length and build a bool mask.

    Returns ``(padded, pad_mask)`` with shapes
        padded:   [B, max_len, feature_dim]  or [B, max_len] if feature_dim==0
        pad_mask: [B, max_len]  (True where padded)
    """
    B = len(arrays)
    if feature_dim == 0:
        padded = np.zeros((B, max_len), dtype=dtype)
    else:
        padded = np.zeros((B, max_len, feature_dim), dtype=dtype)
    pad_mask = np.ones((B, max_len), dtype=bool)
    for i, arr in enumerate(arrays):
        n = min(arr.shape[0], max_len)
        if feature_dim == 0:
            padded[i, :n] = arr[:n]
        else:
            padded[i, :n] = arr[:n]
        pad_mask[i, :n] = False
    return padded, pad_mask


class VisibilityDatasetOracleAdapter:
    """Wrap a :class:`VisibilityConditionedDataset` DataLoader as an
    oracle-training epoch iterator.

    Each emitted batch is a dict with the keys expected by
    :func:`distill_oracle_step`:
        support_visibilities, support_uv, target_uv, teacher_importance,
        support_padding_mask, support_fraction.

    Batch assembly: pulls a batch of B samples from the dataloader, picks
    one (strategy, alpha) pair per batch, applies the holdout split per
    sample x frame, drops frames without both support and target, pads
    support/target token lists to a common length, stacks, returns.
    """

    def __init__(
        self,
        dataloader: Iterable[Mapping[str, torch.Tensor]],
        image_size: int,
        sequence_length: int,
        base_seed: int,
        device: torch.device,
        rng: np.random.Generator,
        adapter_config: OracleDataAdapterConfig | None = None,
        strategies: tuple[str, ...] = DETERMINISTIC_STRATEGIES,
    ) -> None:
        self.dataloader = dataloader
        self.image_size = int(image_size)
        self.sequence_length = int(sequence_length)
        self.base_seed = int(base_seed)
        self.device = device
        self.rng = rng
        self.cfg = adapter_config or OracleDataAdapterConfig()
        self.strategies = tuple(strategies)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for loader_batch in self.dataloader:
            strategy = str(self.rng.choice(self.strategies))
            alpha = float(self.rng.choice(self.cfg.support_fractions))

            vis_real = loader_batch["vis_real"] if "vis_real" in loader_batch else None
            vis_imag = loader_batch["vis_imag"] if "vis_imag" in loader_batch else None
            mask = loader_batch["mask"] if "mask" in loader_batch else None
            frame_uv_indices = loader_batch["frame_uv_indices"]
            frame_uv_coords = loader_batch["frame_uv_coords"]
            baseline_pairs = loader_batch.get("baseline_pairs")
            station_positions = loader_batch.get("station_positions")

            # Tensors come from DataLoader; detach to numpy for the
            # existing holdout machinery (pure-numpy API).
            def _np(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
                return x

            vis_real_np = _np(vis_real)
            vis_imag_np = _np(vis_imag)
            mask_np = _np(mask)
            frame_uv_indices_np = _np(frame_uv_indices)
            frame_uv_coords_np = _np(frame_uv_coords)
            baseline_pairs_np = _np(baseline_pairs)
            station_positions_np = _np(station_positions)

            B = int(vis_real_np.shape[0])
            sv_list, su_list, tu_list, ti_list = [], [], [], []
            max_sup, max_tgt = 0, 0

            for b in range(B):
                measurements_b = (vis_real_np[b] + 1j * vis_imag_np[b]).astype(np.complex64)
                observed_b = mask_np[b].astype(np.float32)
                split_kwargs = dict(
                    measurements=measurements_b,
                    observed_mask=observed_b,
                    frame_uv_indices=frame_uv_indices_np[b].astype(np.int64),
                    frame_uv_coords=frame_uv_coords_np[b].astype(np.float32),
                    base_seed=self.base_seed,
                    sample_index=b,
                    support_fraction=alpha,
                    strategy=strategy,
                )
                if strategy == "station_dropout":
                    if baseline_pairs_np is None or station_positions_np is None:
                        # Fall back silently to another strategy for this sample.
                        split_kwargs["strategy"] = "baseline_track_blocks"
                    else:
                        split_kwargs["baseline_pairs"] = baseline_pairs_np[b].astype(np.int64)
                        split_kwargs["station_positions"] = station_positions_np[b].astype(np.float32)
                split = build_structured_holdout_split(**split_kwargs)

                seq_len = split.support_mask.shape[0]
                frames = range(seq_len) if self.cfg.use_all_frames else (seq_len // 2,)
                for frame_index in frames:
                    example, ns, nt = _frame_oracle_example(
                        measurements=measurements_b[frame_index],
                        observed_mask=observed_b[frame_index],
                        support_mask=split.support_mask[frame_index],
                        target_mask=split.target_mask[frame_index],
                        image_size=self.image_size,
                        frame_index=frame_index,
                        sequence_length=seq_len,
                        signal_var=self.cfg.signal_var,
                        noise_var=self.cfg.noise_var,
                    )
                    if example is None:
                        continue
                    sv_list.append(example["support_visibilities"])
                    su_list.append(example["support_uv"])
                    tu_list.append(example["target_uv"])
                    ti_list.append(example["teacher_importance"])
                    max_sup = max(max_sup, ns)
                    max_tgt = max(max_tgt, nt)

            if not sv_list:
                continue

            max_sup = min(max_sup, self.cfg.max_support)
            max_tgt = min(max_tgt, self.cfg.max_target)

            sup_vis_padded, sup_pad_mask = _pad_stack(sv_list, max_sup, 3, np.float32)
            sup_uv_padded, _ = _pad_stack(su_list, max_sup, 3, np.float32)
            tgt_uv_padded, _ = _pad_stack(tu_list, max_tgt, 3, np.float32)
            teacher_padded, _ = _pad_stack(ti_list, max_tgt, 0, np.float32)

            yield {
                "support_visibilities": torch.from_numpy(sup_vis_padded).to(self.device),
                "support_uv": torch.from_numpy(sup_uv_padded).to(self.device),
                "target_uv": torch.from_numpy(tgt_uv_padded).to(self.device),
                "teacher_importance": torch.from_numpy(teacher_padded).to(self.device),
                "support_padding_mask": torch.from_numpy(sup_pad_mask).to(self.device),
                "support_fraction": torch.tensor(alpha, device=self.device),
            }
