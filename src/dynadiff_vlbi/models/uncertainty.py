"""Uncertainty estimation helpers based on Monte Carlo dropout."""

from __future__ import annotations

import torch
from torch import nn


def _enable_dropout(module: nn.Module) -> None:
    if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        module.train()


def mc_dropout_predict(
    model: nn.Module,
    inputs: torch.Tensor,
    n_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run repeated stochastic forward passes and return mean and std predictions."""

    was_training = model.training
    model.eval()
    model.apply(_enable_dropout)
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            samples.append(model(inputs))
    sample_stack = torch.stack(samples, dim=0)
    predictive_mean = sample_stack.mean(dim=0)
    predictive_std = sample_stack.std(dim=0, unbiased=False)
    model.train(was_training)
    return predictive_mean, predictive_std, sample_stack
