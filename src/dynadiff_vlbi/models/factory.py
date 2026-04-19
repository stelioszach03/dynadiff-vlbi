"""Factories for model construction."""

from __future__ import annotations

from dynadiff_vlbi.models.ccrr import ClosureConsistentResidualRefinementModel
from dynadiff_vlbi.models.emc_ccrr import EarnedMeasurementConsistencyModel
from dynadiff_vlbi.models.residual_visibility_refinement import ResidualVisibilityRefinementModel
from dynadiff_vlbi.models.temporal_unet import TemporalUNet3D
from dynadiff_vlbi.models.visibility_conditioned import VisibilityConditionedReconstructor
from dynadiff_vlbi.utils.config import ModelConfig


def build_model(model_config: ModelConfig):
    """Instantiate a model from the config."""

    if model_config.model_type == "baseline":
        return TemporalUNet3D(model_config)
    if model_config.model_type == "visibility_conditioned":
        return VisibilityConditionedReconstructor(model_config)
    if model_config.model_type == "residual_visibility_refinement":
        return ResidualVisibilityRefinementModel(model_config)
    if model_config.model_type == "ccrr":
        return ClosureConsistentResidualRefinementModel(model_config)
    if model_config.model_type == "emc_ccrr":
        return EarnedMeasurementConsistencyModel(model_config)
    raise ValueError(f"Unsupported model_type '{model_config.model_type}'.")
