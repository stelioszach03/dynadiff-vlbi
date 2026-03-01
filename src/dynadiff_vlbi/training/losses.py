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
