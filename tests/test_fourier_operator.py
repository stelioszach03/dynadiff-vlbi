from __future__ import annotations

import numpy as np

from dynadiff_vlbi.physics.fourier_operator import FourierMeasurementOperator
from dynadiff_vlbi.physics.sampling import generate_temporal_uv_mask
from dynadiff_vlbi.utils.config import SamplingConfig


def test_full_mask_zero_noise_recovers_input_sequence() -> None:
    rng = np.random.default_rng(3)
    sequence = rng.random((4, 32, 32), dtype=np.float32)
    full_mask = np.ones_like(sequence, dtype=np.float32)
    operator = FourierMeasurementOperator(noise_std=0.0, seed=17)
    measurement_batch = operator.forward(sequence=sequence, mask=full_mask)

    np.testing.assert_allclose(measurement_batch.dirty_reconstruction, sequence, atol=1e-5)


def test_operator_shapes_and_noise_reproducibility() -> None:
    rng = np.random.default_rng(9)
    sequence = rng.random((4, 32, 32), dtype=np.float32)
    sampling_config = SamplingConfig(
        coverage=0.12,
        radial_exponent=1.2,
        missing_fraction=0.10,
        hermitian_symmetric=True,
        include_dc=True,
    )
    mask = generate_temporal_uv_mask(image_size=32, sequence_length=4, config=sampling_config, rng=rng)
    operator_a = FourierMeasurementOperator(noise_std=0.03, seed=101)
    operator_b = FourierMeasurementOperator(noise_std=0.03, seed=101)
    batch_a = operator_a.forward(sequence=sequence, mask=mask)
    batch_b = operator_b.forward(sequence=sequence, mask=mask)

    assert batch_a.clean_visibilities.shape == (4, 32, 32)
    assert batch_a.noisy_visibilities.shape == (4, 32, 32)
    assert batch_a.dirty_reconstruction.shape == (4, 32, 32)
    np.testing.assert_allclose(batch_a.noisy_visibilities, batch_b.noisy_visibilities)
