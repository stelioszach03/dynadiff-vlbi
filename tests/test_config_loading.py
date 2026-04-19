from __future__ import annotations

from pathlib import Path

from dynadiff_vlbi.utils.config import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]


def test_custom_config_without_preset_uses_custom_values_without_preset_override() -> None:
    config = load_experiment_config(
        base_path=ROOT / "configs/paper_noise_high.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset=None,
        default_base_path=ROOT / "configs/base.yaml",
    )
    assert config.noise.noise_std == 0.06
    assert config.preset_name == "custom"


def test_custom_config_overrides_preset_for_noise_and_sampling_values() -> None:
    noise_config = load_experiment_config(
        base_path=ROOT / "configs/paper_noise_high.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="default32",
        default_base_path=ROOT / "configs/base.yaml",
    )
    sparse_config = load_experiment_config(
        base_path=ROOT / "configs/paper_sparse_uv.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="default32",
        default_base_path=ROOT / "configs/base.yaml",
    )

    assert noise_config.noise.noise_std == 0.06
    assert sparse_config.sampling.coverage == 0.06
    assert sparse_config.sampling.missing_fraction == 0.20


def test_default_base_with_smoke_preset_keeps_smoke_sizes() -> None:
    config = load_experiment_config(
        base_path=ROOT / "configs/base.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="smoke",
        default_base_path=ROOT / "configs/base.yaml",
    )
    assert config.dataset.train_size == 16
    assert config.dataset.val_size == 4
    assert config.dataset.test_size == 4
