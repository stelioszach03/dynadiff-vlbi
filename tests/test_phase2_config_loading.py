from __future__ import annotations

from pathlib import Path

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
