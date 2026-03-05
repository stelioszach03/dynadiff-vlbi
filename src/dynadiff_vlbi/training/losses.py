"""Loss functions for reconstruction training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
) -> dict[str, torch.Tensor]:
    """Return total, reconstruction, temporal, and optional heteroscedastic losses."""

    reconstruction = F.mse_loss(prediction, target)
    temporal = temporal_difference_loss(prediction, target)
    heteroscedastic = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    total = reconstruction + temporal_loss_weight * temporal
    if log_variance is not None and heteroscedastic_loss_weight > 0.0:
        heteroscedastic = gaussian_nll_loss(prediction, target, log_variance)
        total = total + heteroscedastic_loss_weight * heteroscedastic
    return {
        "total": total,
        "reconstruction": reconstruction,
        "temporal": temporal,
        "heteroscedastic": heteroscedastic,
    }
