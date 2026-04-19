from __future__ import annotations

from pathlib import Path

from dynadiff_vlbi.data.synthetic_generator import generate_dataset_splits
from dynadiff_vlbi.data.visibility_dataset import VisibilityConditionedDataset
from dynadiff_vlbi.utils.config import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]


def test_visibility_dataset_exposes_formatted_inputs(tmp_path: Path) -> None:
    config = load_experiment_config(
        base_path=ROOT / "configs/phase2_visibility_default32.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="smoke",
        default_base_path=ROOT / "configs/base.yaml",
    )
    data_dir = tmp_path / "phase2_smoke_data"
    generate_dataset_splits(
        output_dir=data_dir,
        dataset_config=config.dataset,
        sampling_config=config.sampling,
        noise_config=config.noise,
        base_seed=config.project.seed,
    )

    dataset = VisibilityConditionedDataset(data_dir / "train.npz", model_config=config.model)
    sample = dataset[0]

    assert sample["visibility_input"].shape == (5, config.dataset.sequence_length, 32, 32)
    assert sample["dirty_input"].shape == (1, config.dataset.sequence_length, 32, 32)
    assert sample["uv_coords"].shape == (2, config.dataset.sequence_length, 32, 32)
    assert sample["target"].shape == (config.dataset.sequence_length, 32, 32)
    assert int(sample["sample_index"].item()) == 0


def test_residual_refinement_dataset_uses_visibility_conditioned_inputs(tmp_path: Path) -> None:
    config = load_experiment_config(
        base_path=ROOT / "configs/phase2_residual_refine_default32.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="smoke",
        default_base_path=ROOT / "configs/base.yaml",
    )
    data_dir = tmp_path / "phase22_smoke_data"
    generate_dataset_splits(
        output_dir=data_dir,
        dataset_config=config.dataset,
        sampling_config=config.sampling,
        noise_config=config.noise,
        base_seed=config.project.seed,
    )

    dataset = VisibilityConditionedDataset(data_dir / "train.npz", model_config=config.model)
    sample = dataset[0]

    assert sample["visibility_input"].shape[0] == 5
    assert sample["dirty_input"].shape == (1, config.dataset.sequence_length, 32, 32)
    assert sample["target"].shape == (config.dataset.sequence_length, 32, 32)


def test_station_bridge_visibility_dataset_keeps_geometry_metadata(tmp_path: Path) -> None:
    config = load_experiment_config(
        base_path=ROOT / "configs/mnras_realism_bridge_default32.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="smoke",
        default_base_path=ROOT / "configs/base.yaml",
    )
    data_dir = tmp_path / "mnras_visibility_smoke_data"
    generate_dataset_splits(
        output_dir=data_dir,
        dataset_config=config.dataset,
        sampling_config=config.sampling,
        noise_config=config.noise,
        base_seed=config.project.seed,
    )

    dataset = VisibilityConditionedDataset(data_dir / "train.npz", model_config=config.model)
    sample = dataset[0]

    assert sample["visibility_input"].shape[0] == 5
    assert sample["station_positions"].shape[1] == 2
    assert sample["baseline_pairs"].shape[1] == 2
    assert sample["frame_uv_indices"].shape[0] == config.dataset.sequence_length


def test_ccrr_dataset_exposes_observation_metadata_channels(tmp_path: Path) -> None:
    config = load_experiment_config(
        base_path=ROOT / "configs/ccrr_default32.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="smoke",
        default_base_path=ROOT / "configs/base.yaml",
    )
    data_dir = tmp_path / "ccrr_smoke_data"
    generate_dataset_splits(
        output_dir=data_dir,
        dataset_config=config.dataset,
        sampling_config=config.sampling,
        noise_config=config.noise,
        base_seed=config.project.seed,
    )

    dataset = VisibilityConditionedDataset(data_dir / "train.npz", model_config=config.model)
    sample = dataset[0]

    assert sample["visibility_input"].shape[0] == 9
    assert sample["baseline_pairs"].shape[1] == 2
    assert sample["frame_uv_indices"].shape[0] == config.dataset.sequence_length
