"""Loss functions for reconstruction training."""

from __future__ import annotations

from itertools import combinations

import torch
import torch.nn.functional as F

from dynadiff_vlbi.physics.torch_fourier import fft2c_torch


def temporal_difference_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize mismatch in frame-to-frame dynamics."""

    if prediction.shape[2] <= 1:
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    pred_diff = prediction[:, :, 1:] - prediction[:, :, :-1]
    target_diff = target[:, :, 1:] - target[:, :, :-1]
    return F.mse_loss(pred_diff, target_diff)


def gaussian_nll_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    log_variance: torch.Tensor,
) -> torch.Tensor:
    """Compute a simple per-pixel Gaussian negative log-likelihood."""

    precision = torch.exp(-log_variance)
    squared_error = (prediction - target) ** 2
    return 0.5 * torch.mean(precision * squared_error + log_variance)


def visibility_consistency_loss(
    prediction: torch.Tensor,
    measurements: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize mismatch with observed visibilities at sampled coefficients."""

    predicted_vis = fft2c_torch(prediction.squeeze(1))
    residual = (predicted_vis - measurements) * mask
    denom = mask.sum().clamp_min(1.0)
    return (torch.abs(residual) ** 2).sum() / denom


def _normalized_bispectrum(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return value / torch.abs(value).clamp_min(eps)


def closure_consistency_loss(
    prediction: torch.Tensor,
    measurements: torch.Tensor,
    mask: torch.Tensor,
    baseline_pairs: torch.Tensor | None,
    frame_uv_indices: torch.Tensor | None,
    max_triangles: int = 24,
) -> torch.Tensor:
    """Compute a phase-safe closure loss using normalized complex bispectra."""

    if baseline_pairs is None or frame_uv_indices is None:
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)

    predicted_vis = fft2c_torch(prediction.squeeze(1))
    if baseline_pairs.ndim == 3:
        baseline_pairs_shared = baseline_pairs[0]
    else:
        baseline_pairs_shared = baseline_pairs

    pair_lookup = {
        tuple(pair): pair_index
        for pair_index, pair in enumerate(baseline_pairs_shared.detach().cpu().tolist())
    }
    if not pair_lookup:
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)

    station_count = int(baseline_pairs_shared.max().item()) + 1
    triangles = list(combinations(range(station_count), 3))[:max_triangles]
    if not triangles:
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)

    losses: list[torch.Tensor] = []
    batch_size = prediction.shape[0]
    for batch_index in range(batch_size):
        batch_frame_uv = frame_uv_indices[batch_index] if frame_uv_indices.ndim == 4 else frame_uv_indices
        for frame_index in range(prediction.shape[2]):
            frame_mask = mask[batch_index, frame_index]
            frame_indices = batch_frame_uv[frame_index]
            for first, second, third in triangles:
                pair_ab = pair_lookup.get((first, second))
                pair_bc = pair_lookup.get((second, third))
                pair_ac = pair_lookup.get((first, third))
                if pair_ab is None or pair_bc is None or pair_ac is None:
                    continue

                row_ab, col_ab = frame_indices[pair_ab].tolist()
                row_bc, col_bc = frame_indices[pair_bc].tolist()
                row_ac, col_ac = frame_indices[pair_ac].tolist()
                if (
                    frame_mask[row_ab, col_ab] <= 0.0
                    or frame_mask[row_bc, col_bc] <= 0.0
                    or frame_mask[row_ac, col_ac] <= 0.0
                ):
                    continue

                pred_ab = predicted_vis[batch_index, frame_index, row_ab, col_ab]
                pred_bc = predicted_vis[batch_index, frame_index, row_bc, col_bc]
                pred_ac = predicted_vis[batch_index, frame_index, row_ac, col_ac]
                meas_ab = measurements[batch_index, frame_index, row_ab, col_ab]
                meas_bc = measurements[batch_index, frame_index, row_bc, col_bc]
                meas_ac = measurements[batch_index, frame_index, row_ac, col_ac]

                bis_pred = pred_ab * pred_bc * torch.conj(pred_ac)
                bis_meas = meas_ab * meas_bc * torch.conj(meas_ac)
                losses.append(torch.abs(_normalized_bispectrum(bis_pred) - _normalized_bispectrum(bis_meas)) ** 2)

    if not losses:
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    return torch.stack(losses).mean().real


def compute_loss_dict(
    prediction: torch.Tensor,
    target: torch.Tensor,
    temporal_loss_weight: float,
) -> dict[str, torch.Tensor]:
    """Return total, reconstruction, and temporal losses."""

    reconstruction = F.mse_loss(prediction, target)
    temporal = temporal_difference_loss(prediction, target)
    total = reconstruction + temporal_loss_weight * temporal
    return {
        "total": total,
        "reconstruction": reconstruction,
        "temporal": temporal,
    }


def compute_phase2_loss_dict(
    prediction: torch.Tensor,
    target: torch.Tensor,
    temporal_loss_weight: float,
    log_variance: torch.Tensor | None = None,
    heteroscedastic_loss_weight: float = 0.0,
    consistency_prediction: torch.Tensor | None = None,
    measurements: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    visibility_loss_weight: float = 0.0,
    baseline_pairs: torch.Tensor | None = None,
    frame_uv_indices: torch.Tensor | None = None,
    closure_loss_weight: float = 0.0,
    closure_max_triangles: int = 24,
) -> dict[str, torch.Tensor]:
    """Return total, reconstruction, temporal, and optional heteroscedastic losses."""

    reconstruction = F.mse_loss(prediction, target)
    temporal = temporal_difference_loss(prediction, target)
    heteroscedastic = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    visibility = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    closure = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    consistency_source = consistency_prediction if consistency_prediction is not None else prediction
    total = reconstruction + temporal_loss_weight * temporal
    if log_variance is not None and heteroscedastic_loss_weight > 0.0:
        heteroscedastic = gaussian_nll_loss(prediction, target, log_variance)
        total = total + heteroscedastic_loss_weight * heteroscedastic
    if measurements is not None and mask is not None and visibility_loss_weight > 0.0:
        visibility = visibility_consistency_loss(
            prediction=consistency_source,
            measurements=measurements,
            mask=mask,
        )
        total = total + visibility_loss_weight * visibility
    if (
        measurements is not None
        and mask is not None
        and baseline_pairs is not None
        and frame_uv_indices is not None
        and closure_loss_weight > 0.0
    ):
        closure = closure_consistency_loss(
            prediction=consistency_source,
            measurements=measurements,
            mask=mask,
            baseline_pairs=baseline_pairs,
            frame_uv_indices=frame_uv_indices,
            max_triangles=closure_max_triangles,
        )
        total = total + closure_loss_weight * closure
    return {
        "total": total,
        "reconstruction": reconstruction,
        "temporal": temporal,
        "heteroscedastic": heteroscedastic,
        "visibility": visibility,
        "closure": closure,
    }


def compute_emc_loss_dict(
    prediction: torch.Tensor,
    target: torch.Tensor,
    temporal_loss_weight: float,
    measurements: torch.Tensor,
    support_mask: torch.Tensor,
    target_mask: torch.Tensor,
    log_variance: torch.Tensor | None = None,
    heteroscedastic_loss_weight: float = 0.0,
    consistency_prediction: torch.Tensor | None = None,
    support_visibility_loss_weight: float = 0.0,
    target_visibility_loss_weight: float = 0.0,
    baseline_pairs: torch.Tensor | None = None,
    frame_uv_indices: torch.Tensor | None = None,
    target_closure_loss_weight: float = 0.0,
    closure_max_triangles: int = 24,
) -> dict[str, torch.Tensor]:
    """Return EMC losses that separate support-enforced and target-earned consistency."""

    reconstruction = F.mse_loss(prediction, target)
    temporal = temporal_difference_loss(prediction, target)
    heteroscedastic = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    support_visibility = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    target_visibility = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    target_closure = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    consistency_source = consistency_prediction if consistency_prediction is not None else prediction
    total = reconstruction + temporal_loss_weight * temporal

    if log_variance is not None and heteroscedastic_loss_weight > 0.0:
        heteroscedastic = gaussian_nll_loss(prediction, target, log_variance)
        total = total + heteroscedastic_loss_weight * heteroscedastic
    if support_visibility_loss_weight > 0.0:
        support_visibility = visibility_consistency_loss(
            prediction=consistency_source,
            measurements=measurements,
            mask=support_mask,
        )
        total = total + support_visibility_loss_weight * support_visibility
    if target_visibility_loss_weight > 0.0:
        target_visibility = visibility_consistency_loss(
            prediction=prediction,
            measurements=measurements,
            mask=target_mask,
        )
        total = total + target_visibility_loss_weight * target_visibility
    if (
        target_closure_loss_weight > 0.0
        and baseline_pairs is not None
        and frame_uv_indices is not None
    ):
        target_closure = closure_consistency_loss(
            prediction=prediction,
            measurements=measurements,
            mask=target_mask,
            baseline_pairs=baseline_pairs,
            frame_uv_indices=frame_uv_indices,
            max_triangles=closure_max_triangles,
        )
        total = total + target_closure_loss_weight * target_closure
    return {
        "total": total,
        "reconstruction": reconstruction,
        "temporal": temporal,
        "heteroscedastic": heteroscedastic,
        "support_visibility": support_visibility,
        "target_visibility": target_visibility,
        "target_closure": target_closure,
    }
