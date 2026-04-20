"""Generate the backcompat reference fixture for measurement_holdout.

Run this ONCE on a "known-good" HEAD to freeze the deterministic output
of ``build_structured_holdout_split``. The test
``tests/test_benchmark_backcompat.py`` loads the produced ``.npz`` and
asserts that subsequent refactors preserve bit-exact deterministic
behaviour for the 3 original strategies.

Usage:
    python tests/fixtures/generate_holdout_backcompat_reference.py

Output:
    tests/fixtures/holdout_backcompat_reference.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.data.measurement_holdout import build_structured_holdout_split


def _toy_observation():
    """Synthetic minibatch matching the pattern used in test_emc_holdout.py."""
    sequence_length = 3
    image_size = 8
    mask = np.zeros((sequence_length, image_size, image_size), dtype=np.float32)
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
    for frame_index in range(sequence_length):
        for row, col in frame_uv_indices[frame_index]:
            mask[frame_index, row, col] = 1.0
    measurements = np.ones((sequence_length, image_size, image_size), dtype=np.complex64)
    baseline_pairs = np.asarray([[0, 1], [1, 2], [0, 2], [0, 3]], dtype=np.int64)
    station_positions = np.asarray(
        [[-0.8, -0.1], [-0.2, 0.7], [0.5, -0.5], [0.8, 0.2]], dtype=np.float32
    )
    return measurements, mask, frame_uv_indices, frame_uv_coords, baseline_pairs, station_positions


STRATEGIES = ("baseline_track_blocks", "scan_segment_blocks", "station_dropout")
SUPPORT_FRACTIONS = (0.2, 0.5, 0.8)
BASE_SEED = 7
SAMPLE_INDEX = 3


def main() -> None:
    measurements, mask, frame_uv_indices, frame_uv_coords, baseline_pairs, station_positions = (
        _toy_observation()
    )
    out: dict[str, np.ndarray] = {}
    for strategy in STRATEGIES:
        for alpha in SUPPORT_FRACTIONS:
            kwargs = dict(
                measurements=measurements,
                observed_mask=mask,
                frame_uv_indices=frame_uv_indices,
                frame_uv_coords=frame_uv_coords,
                base_seed=BASE_SEED,
                sample_index=SAMPLE_INDEX,
                support_fraction=alpha,
                strategy=strategy,
            )
            if strategy == "station_dropout":
                kwargs["baseline_pairs"] = baseline_pairs
                kwargs["station_positions"] = station_positions
            split = build_structured_holdout_split(**kwargs)
            tag = f"{strategy}__{alpha:.2f}"
            out[f"{tag}__support_mask"] = split.support_mask
            out[f"{tag}__target_mask"] = split.target_mask
            out[f"{tag}__support_unit_count"] = np.asarray(split.support_unit_count)
            out[f"{tag}__target_unit_count"] = np.asarray(split.target_unit_count)

    fixture_path = ROOT / "tests" / "fixtures" / "holdout_backcompat_reference.npz"
    np.savez_compressed(fixture_path, **out)
    print(f"wrote {fixture_path} with {len(out)} arrays across "
          f"{len(STRATEGIES)} strategies x {len(SUPPORT_FRACTIONS)} support fractions")


if __name__ == "__main__":
    main()
