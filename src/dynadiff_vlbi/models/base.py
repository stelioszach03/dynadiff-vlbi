"""Base model abstractions for learned reconstructions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseReconstructionModel(nn.Module, ABC):
    """Abstract base class for learned reconstruction models."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a corrupted input sequence to a reconstructed sequence."""

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce Monte Carlo dropout predictions through the model."""

        from dynadiff_vlbi.models.uncertainty import mc_dropout_predict

        return mc_dropout_predict(model=self, inputs=x, n_samples=n_samples)
