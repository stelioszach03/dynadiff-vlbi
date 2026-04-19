"""Earned Measurement Consistency model path built on top of CCRR."""

from __future__ import annotations

from dynadiff_vlbi.models.ccrr import CCRROutput, ClosureConsistentResidualRefinementModel
from dynadiff_vlbi.utils.config import ModelConfig


class EarnedMeasurementConsistencyModel(ClosureConsistentResidualRefinementModel):
    """CCRR architecture trained and evaluated with structured held-out measurements."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)


__all__ = ["CCRROutput", "EarnedMeasurementConsistencyModel"]
