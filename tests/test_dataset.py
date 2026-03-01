from __future__ import annotations

from pathlib import Path

from dynadiff_vlbi.data.dataset import DynamicVLBIDataset
from dynadiff_vlbi.data.io import load_npz
from dynadiff_vlbi.data.synthetic_generator import generate_dataset_splits
from dynadiff_vlbi.utils.config import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]


def test_generated_dataset_splits_and_loader_match_smoke_preset(tmp_path: Path) -> None:
    config = load_experiment_config(
        base_path=ROOT / "configs/base.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="smoke",
    )
    data_dir = tmp_path / "smoke_data"
    generate_dataset_splits(
        output_dir=data_dir,
        dataset_config=config.dataset,
        sampling_config=config.sampling,
        noise_config=config.noise,
        base_seed=config.project.seed,
    )

    train_arrays = load_npz(data_dir / "train.npz")
    val_arrays = load_npz(data_dir / "val.npz")
    test_arrays = load_npz(data_dir / "test.npz")

    assert train_arrays["ground_truth"].shape[0] == config.dataset.train_size
    assert val_arrays["ground_truth"].shape[0] == config.dataset.val_size
    assert test_arrays["ground_truth"].shape[0] == config.dataset.test_size
    assert train_arrays["ground_truth"].shape[1:] == (
        config.dataset.sequence_length,
        config.dataset.image_size,
        config.dataset.image_size,
    )
    assert train_arrays["mask"].shape == train_arrays["ground_truth"].shape
    assert train_arrays["hotspot_coords_px"].shape == (config.dataset.train_size, config.dataset.sequence_length, 2)

    dataset = DynamicVLBIDataset(data_dir / "train.npz")
    sample = dataset[0]
    assert sample["input"].shape == (
        config.dataset.sequence_length,
        config.dataset.image_size,
        config.dataset.image_size,
    )
    assert sample["target"].shape == sample["input"].shape
    assert sample["hotspot_coords_px"].shape == (config.dataset.sequence_length, 2)
