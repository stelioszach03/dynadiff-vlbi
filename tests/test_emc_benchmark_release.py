from __future__ import annotations

from pathlib import Path

import numpy as np

from dynadiff_vlbi.evaluation.benchmark_release import export_split_manifests
from dynadiff_vlbi.utils.config import DEFAULT_BASE_CONFIG_PATH, load_experiment_config


ROOT = Path(__file__).resolve().parents[1]


def _write_toy_dataset(output_dir: Path) -> None:
    sequence_length = 3
    image_size = 8
    sample_count = 2
    ground_truth = np.zeros((sample_count, sequence_length, image_size, image_size), dtype=np.float32)
    vis_real = np.ones_like(ground_truth, dtype=np.float32)
    vis_imag = np.zeros_like(ground_truth, dtype=np.float32)
    mask = np.zeros_like(ground_truth, dtype=np.float32)
    frame_uv_indices = np.asarray(
        [
            [[2, 3], [2, 5], [4, 3], [4, 5]],
            [[2, 3], [2, 5], [4, 3], [4, 5]],
            [[2, 3], [2, 5], [4, 3], [4, 5]],
        ],
        dtype=np.int64,
    )
    frame_uv_coords = np.asarray(
        [
            [[-0.7, -0.1], [-0.2, 0.4], [0.2, -0.4], [0.7, 0.1]],
            [[-0.7, -0.1], [-0.2, 0.4], [0.2, -0.4], [0.7, 0.1]],
            [[-0.7, -0.1], [-0.2, 0.4], [0.2, -0.4], [0.7, 0.1]],
        ],
        dtype=np.float32,
    )
    for sample_index in range(sample_count):
        for frame_index in range(sequence_length):
            for row, col in frame_uv_indices[frame_index]:
                mask[sample_index, frame_index, row, col] = 1.0
    np.savez_compressed(
        output_dir / "test.npz",
        ground_truth=ground_truth,
        vis_real=vis_real,
        vis_imag=vis_imag,
        mask=mask,
        station_positions=np.asarray(
            [
                [-0.8, -0.1],
                [-0.2, 0.7],
                [0.5, -0.5],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        ),
        baseline_pairs=np.asarray([[0, 1], [1, 2], [0, 2], [0, 3]], dtype=np.int32),
        frame_uv_indices=frame_uv_indices.astype(np.int32),
        frame_uv_coords=frame_uv_coords.astype(np.float32),
    )


def test_export_split_manifests_is_deterministic(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    _write_toy_dataset(dataset_dir)
    config = load_experiment_config(
        base_path=ROOT / "configs/emc_benchmark_station_dropout_default32.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="default32",
        default_base_path=ROOT / DEFAULT_BASE_CONFIG_PATH,
    )
    config.holdout.eval_support_fractions = (0.6,)

    manifest_a = export_split_manifests(
        config=config,
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "manifest_a",
    )
    manifest_b = export_split_manifests(
        config=config,
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "manifest_b",
    )

    split_a = np.load(tmp_path / "manifest_a" / "support_60_split_manifest.npz")
    split_b = np.load(tmp_path / "manifest_b" / "support_60_split_manifest.npz")
    assert manifest_a["strategy"] == "station_dropout"
    assert manifest_b["strategy_label"] == "Station dropout"
    assert np.array_equal(split_a["support_mask"], split_b["support_mask"])
    assert np.array_equal(split_a["target_mask"], split_b["target_mask"])
    assert np.all(split_a["target_unit_count"] >= 1)
