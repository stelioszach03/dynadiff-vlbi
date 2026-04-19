from __future__ import annotations

import numpy as np
import pytest

from dynadiff_vlbi.evaluation.ehtim_bridge import (
    centered_frequency_axis,
    predict_ehtim_bridge_sequence,
    representative_support_pairs,
)


def test_centered_frequency_axis_matches_fft_grid() -> None:
    axis = centered_frequency_axis(8)
    np.testing.assert_allclose(axis, np.asarray([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))


def test_representative_support_pairs_uses_first_available_pair() -> None:
    support_mask = np.zeros((2, 8, 8), dtype=np.float32)
    support_mask[0, 3, 5] = 1.0
    support_mask[1, 2, 1] = 1.0
    frame_uv_indices = np.zeros((2, 3, 2), dtype=np.int64)
    frame_uv_indices[0, 0] = np.asarray([3, 5])
    frame_uv_indices[0, 1] = np.asarray([3, 5])
    frame_uv_indices[1, 2] = np.asarray([2, 1])
    baseline_pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)

    mapping = representative_support_pairs(
        support_mask=support_mask,
        frame_uv_indices=frame_uv_indices,
        baseline_pairs=baseline_pairs,
    )

    assert mapping[(0, 3, 5)] == (0, 1)
    assert mapping[(1, 2, 1)] == (1, 2)


def test_predict_ehtim_bridge_sequence_returns_finite_sequence() -> None:
    pytest.importorskip("ehtim")

    measurements = np.zeros((2, 8, 8), dtype=np.complex64)
    measurements[0, 4, 4] = 1.0 + 0.0j
    measurements[1, 4, 4] = 0.8 + 0.0j
    support_mask = np.zeros((2, 8, 8), dtype=np.float32)
    support_mask[:, 4, 4] = 1.0
    sigma = np.ones((2, 8, 8), dtype=np.float32) * 0.05
    frame_uv_indices = np.zeros((2, 1, 2), dtype=np.int64)
    frame_uv_indices[:, 0] = np.asarray([4, 4])
    baseline_pairs = np.asarray([[0, 1]], dtype=np.int64)
    station_labels = np.asarray(["AA", "AP"])
    station_positions = np.asarray([[-0.92, -0.35], [-0.84, -0.22]], dtype=np.float32)

    prediction = predict_ehtim_bridge_sequence(
        measurements=measurements,
        support_mask=support_mask,
        sigma=sigma,
        frame_uv_indices=frame_uv_indices,
        baseline_pairs=baseline_pairs,
        station_labels=station_labels,
        station_positions=station_positions,
        rf_hz=230.0e9,
        source_name="toy",
        mjd=58000.0,
    )

    assert prediction.shape == (2, 8, 8)
    assert np.isfinite(prediction).all()
    assert float(prediction.sum()) > 0.0
