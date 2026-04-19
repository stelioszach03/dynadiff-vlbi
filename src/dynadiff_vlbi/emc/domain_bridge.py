"""Domain-bridge helpers for released public EHT measurement products."""

from __future__ import annotations

import numpy as np


class EHTDomainBridge:
    """Normalize released support measurements into a synthetic-friendly amplitude range."""

    def __init__(self, eps: float = 1.0e-8) -> None:
        self.eps = float(eps)

    def normalize_support(
        self,
        support_vis: np.ndarray,
        support_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        amplitudes = np.abs(np.asarray(support_vis)[np.asarray(support_mask) > 0])
        if amplitudes.size == 0:
            scale = 1.0
        else:
            scale = float(np.median(amplitudes))
        scale = max(scale, self.eps)
        return np.asarray(support_vis, dtype=np.complex64) / np.float32(scale), scale

    def normalize_dirty(self, dirty_recon: np.ndarray, scale: float) -> np.ndarray:
        return np.asarray(dirty_recon, dtype=np.float32) / np.float32(max(float(scale), self.eps))

    def denormalize_prediction(self, pred: np.ndarray, scale: float) -> np.ndarray:
        # The Fourier transform is linear, so image-domain proxies inherit the visibility scale.
        return np.asarray(pred, dtype=np.float32) * np.float32(max(float(scale), self.eps))

    def denormalize_visibilities(self, vis: np.ndarray, scale: float) -> np.ndarray:
        return np.asarray(vis) * np.float32(max(float(scale), self.eps))
