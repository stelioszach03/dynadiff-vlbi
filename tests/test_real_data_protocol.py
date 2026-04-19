from __future__ import annotations

from pathlib import Path

import numpy as np

from dynadiff_vlbi.data.feature_formatting import build_temporal_uv_grid
from dynadiff_vlbi.data.io import save_npz
from dynadiff_vlbi.evaluation.emc_protocol import ComparatorSpec, load_comparators
from dynadiff_vlbi.evaluation.real_data_protocol import evaluate_real_data_condition
from dynadiff_vlbi.physics.classical_reconstruction import dirty_image_reconstruction
from dynadiff_vlbi.utils.config import load_experiment_config
from dynadiff_vlbi.utils.device import get_device


ROOT = Path(__file__).resolve().parents[1]


def test_real_data_protocol_runs_without_ground_truth(tmp_path: Path) -> None:
    sequence_length = 3
    image_size = 8
    sample_count = 1
    frame_uv_indices = np.asarray(
        [
            [
                [[2, 2], [2, 5], [5, 3]],
                [[2, 2], [2, 5], [5, 3]],
                [[2, 2], [2, 5], [5, 3]],
            ]
        ],
        dtype=np.int32,
    )
    frame_uv_coords = np.asarray(
        [
            [
                [[-0.5, -0.5], [0.3, -0.4], [0.1, 0.6]],
                [[-0.5, -0.5], [0.3, -0.4], [0.1, 0.6]],
                [[-0.5, -0.5], [0.3, -0.4], [0.1, 0.6]],
            ]
        ],
        dtype=np.float32,
    )
    mask = np.zeros((sample_count, sequence_length, image_size, image_size), dtype=np.float32)
    vis_real = np.zeros_like(mask)
    vis_imag = np.zeros_like(mask)
    for frame_index in range(sequence_length):
        for baseline_index, (row, col) in enumerate(frame_uv_indices[0, frame_index]):
            mask[0, frame_index, row, col] = 1.0
            vis_real[0, frame_index, row, col] = 0.4 + 0.1 * baseline_index
            vis_imag[0, frame_index, row, col] = 0.05 * (frame_index + 1)
    dirty = dirty_image_reconstruction(
        measurements=(vis_real[0] + 1j * vis_imag[0]).astype(np.complex64),
        mask=mask[0],
    )[None]
    save_npz(
        tmp_path / "dataset" / "test.npz",
        {
            "sample_id": np.asarray(["toy_real_000"], dtype="<U32"),
            "day_of_year": np.asarray([95], dtype=np.int32),
            "band": np.asarray(["hi"], dtype="<U4"),
            "release_code": np.asarray(["TEST-REAL"], dtype="<U16"),
            "target_name": np.asarray(["Toy"], dtype="<U16"),
            "campaign_year": np.asarray([2017], dtype=np.int32),
            "pipeline_name": np.asarray(["Toy pipeline"], dtype="<U24"),
            "vis_real": vis_real.astype(np.float32),
            "vis_imag": vis_imag.astype(np.float32),
            "vis_sigma": np.where(mask > 0.0, 0.1, 0.0).astype(np.float32),
            "vis_weight": np.where(mask > 0.0, 100.0, 0.0).astype(np.float32),
            "mask": mask.astype(np.float32),
            "dirty": dirty.astype(np.float32),
            "uv_coords": build_temporal_uv_grid(image_size=image_size, sequence_length=sequence_length).astype(
                np.float32
            ),
            "baseline_pairs": np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int32),
            "frame_uv_indices": frame_uv_indices.astype(np.int32),
            "frame_uv_coords": frame_uv_coords.astype(np.float32),
            "station_positions": np.asarray([[-0.6, -0.2], [0.1, 0.5], [0.7, -0.4]], dtype=np.float32),
        },
    )

    config = load_experiment_config(
        base_path=ROOT / "configs/emc_real_m87_public_validation.yaml",
        train_path=ROOT / "configs/train.yaml",
        eval_path=ROOT / "configs/eval.yaml",
        preset="smoke",
        default_base_path=ROOT / "configs/base.yaml",
    )
    comparators = load_comparators(
        [
            ComparatorSpec("dirty", "Dirty", "dirty"),
            ComparatorSpec("tikhonov", "Tikhonov", "tikhonov"),
        ],
        device=get_device(),
    )
    summary = evaluate_real_data_condition(
        config=config,
        dataset_dir=tmp_path / "dataset",
        comparators=comparators,
        output_dir=tmp_path / "outputs",
        support_fractions=(0.6,),
    )

    assert summary["sample_count"] == 1
    assert "60" in summary["support_fractions"]
    emc_free_metrics = summary["support_fractions"]["60"]["models"]["dirty"]
    assert "heldout_weighted_chi2" in emc_free_metrics
    assert "observed_reduced_chi2" in emc_free_metrics
    assert (tmp_path / "outputs" / "logs" / "real_data_protocol_summary.json").exists()
    assert (tmp_path / "outputs" / "logs" / "per_sample_support_60.csv").exists()
    assert (tmp_path / "outputs" / "predictions" / "support_60.npz").exists()
