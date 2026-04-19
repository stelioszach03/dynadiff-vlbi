from __future__ import annotations

from pathlib import Path

from dynadiff_vlbi.models.factory import build_model
from dynadiff_vlbi.utils.config import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]


def test_phase2_config_loading_preserves_visibility_model_settings() -> None:
    config = load_experiment_config(
        base_path=ROOT / "configs/phase2_visibility_default32.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="smoke",
        default_base_path=ROOT / "configs/base.yaml",
    )
    assert config.model.model_type == "visibility_conditioned"
    assert config.model.include_dirty_input is True
    assert config.model.include_uv_coords is True
    assert config.model.uncertainty_head is True
    assert config.training.heteroscedastic_loss_weight > 0.0


def test_tuned_phase2_config_loading_preserves_warmup_and_weight() -> None:
    config = load_experiment_config(
        base_path=ROOT / "configs/phase2_visibility_tuned005_default32.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="default32",
        default_base_path=ROOT / "configs/base.yaml",
    )
    assert config.training.heteroscedastic_loss_weight == 0.05
    assert config.training.reconstruction_warmup_epochs == 2


def test_residual_refinement_config_loading_preserves_backbone_and_refinement_settings() -> None:
    config = load_experiment_config(
        base_path=ROOT / "configs/phase2_residual_refine_default32.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="smoke",
        default_base_path=ROOT / "configs/base.yaml",
    )
    assert config.model.model_type == "residual_visibility_refinement"
    assert config.model.freeze_backbone is True
    assert config.model.include_dirty_input is True
    assert config.model.refinement_channels == 8
    assert config.training.heteroscedastic_loss_weight == 0.10
    assert config.training.reconstruction_warmup_epochs == 2


def test_residual_refinement_exp64_config_builds_cleanly() -> None:
    config = load_experiment_config(
        base_path=ROOT / "configs/phase2_residual_refine_exp64.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="exp64",
        default_base_path=ROOT / "configs/base.yaml",
    )
    assert config.model.model_type == "residual_visibility_refinement"
    assert config.model.refinement_channels == 16
    build_model(config.model)


def test_mnras_bridge_config_loading_preserves_station_track_sampling() -> None:
    config = load_experiment_config(
        base_path=ROOT / "configs/mnras_realism_bridge_default32.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="smoke",
        default_base_path=ROOT / "configs/base.yaml",
    )
    assert config.sampling.mode == "station_tracks"
    assert config.sampling.station_count == 14
    assert config.sampling.earth_rotation_degrees == 130.0
    assert config.model.model_type == "residual_visibility_refinement"


def test_ccrr_default32_config_builds_with_metadata_and_data_consistency() -> None:
    config = load_experiment_config(
        base_path=ROOT / "configs/ccrr_default32.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="smoke",
        default_base_path=ROOT / "configs/base.yaml",
    )
    assert config.model.model_type == "ccrr"
    assert config.model.include_observation_metadata is True
    assert config.model.dc_enabled is True
    assert config.training.closure_loss_weight > 0.0
    build_model(config.model)


def test_ccrr_bridge2_config_preserves_structured_corruption_settings() -> None:
    config = load_experiment_config(
        base_path=ROOT / "configs/ccrr_realism_bridge2_default32.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="smoke",
        default_base_path=ROOT / "configs/base.yaml",
    )
    assert config.model.model_type == "ccrr"
    assert config.sampling.mode == "station_tracks"
    assert config.sampling.scan_gap_probability > 0.0
    assert config.noise.baseline_noise_jitter > 0.0
    assert config.noise.gain_phase_std > 0.0


def test_ccrr_noise_high_and_sparse_uv_configs_preserve_named_hard_conditions() -> None:
    noise_high = load_experiment_config(
        base_path=ROOT / "configs/ccrr_noise_high_default32.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="default32",
        default_base_path=ROOT / "configs/base.yaml",
    )
    sparse_uv = load_experiment_config(
        base_path=ROOT / "configs/ccrr_sparse_uv_default32.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="default32",
        default_base_path=ROOT / "configs/base.yaml",
    )
    assert noise_high.noise.noise_std == 0.06
    assert sparse_uv.sampling.coverage == 0.06
    assert sparse_uv.sampling.missing_fraction == 0.20


def test_emc_config_loading_preserves_structured_holdout_settings() -> None:
    config = load_experiment_config(
        base_path=ROOT / "configs/emc_default32.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="smoke",
        default_base_path=ROOT / "configs/base.yaml",
    )
    assert config.model.model_type == "emc_ccrr"
    assert config.holdout.enabled is True
    assert config.holdout.strategy == "baseline_track_blocks"
    assert config.holdout.train_support_fractions == (0.8, 0.6, 0.4, 0.2)
    assert config.training.target_visibility_loss_weight == 0.06
    assert config.training.target_closure_loss_weight == 0.0
    build_model(config.model)


def test_emc_real_data_and_ablation_configs_build_cleanly() -> None:
    real_data = load_experiment_config(
        base_path=ROOT / "configs/emc_real_m87_public_validation.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="smoke",
        default_base_path=ROOT / "configs/base.yaml",
    )
    no_dc = load_experiment_config(
        base_path=ROOT / "configs/emc_ablation_no_dc_default32.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="smoke",
        default_base_path=ROOT / "configs/base.yaml",
    )
    no_metadata = load_experiment_config(
        base_path=ROOT / "configs/emc_ablation_no_metadata_default32.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="smoke",
        default_base_path=ROOT / "configs/base.yaml",
    )

    assert real_data.holdout.min_eval_closure_triangles == 12
    assert no_dc.model.dc_enabled is False
    assert no_metadata.model.include_observation_metadata is False
    build_model(real_data.model)
    build_model(no_dc.model)
