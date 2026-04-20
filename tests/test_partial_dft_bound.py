"""Tests for Theorem 1' — partial-DFT earned-vs-enforced gap verification.

See theory/partial_dft_bound.tex for the theorem statement and
theory/numerical_verification_partial_dft.py for the verification logic.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest

from theory.numerical_verification_partial_dft import (
    build_partial_dft,
    coherence,
    run_verification,
    sweep_and_verify,
)
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_partial_dft_has_unit_coherence():
    """A partial DFT built from the standard DFT basis has coherence mu = 1."""
    rng = np.random.default_rng(0)
    A = build_partial_dft(n=64, m=32, rng=rng)
    mu = coherence(A)
    assert mu == pytest.approx(1.0, abs=1e-10), f"expected mu=1, got {mu}"


def test_build_partial_dft_is_deterministic_given_seed():
    """Same seed -> same partial-DFT matrix."""
    A1 = build_partial_dft(n=32, m=16, rng=np.random.default_rng(42))
    A2 = build_partial_dft(n=32, m=16, rng=np.random.default_rng(42))
    np.testing.assert_allclose(A1, A2)


def test_partial_dft_rows_have_unit_l2_norm():
    """Each row of the partial DFT has ||row||_2 = 1."""
    rng = np.random.default_rng(7)
    A = build_partial_dft(n=64, m=20, rng=rng)
    row_norms = np.linalg.norm(A, axis=1)
    np.testing.assert_allclose(row_norms, np.ones(20), atol=1e-10)


def test_analytical_matches_monte_carlo_at_reference_alpha():
    """Theorem 1' verification: analytical (Sigma_post) vs Monte Carlo agree
    within 5% at alpha = 0.5 (the reference support fraction)."""
    summary = run_verification(
        n=128,
        m=64,
        k_sparsity=6,
        signal_std=1.0,
        noise_std=0.1,
        alphas=[0.2, 0.5, 0.8],
        n_monte_carlo=2000,
        reference_alpha=0.5,
        seed=1234,
    )
    rel_err = summary["relative_error_at_ref_alpha"]
    assert rel_err < 0.05, (
        f"analytical vs Monte Carlo disagree by {rel_err:.2%} at alpha=0.5 "
        f"(threshold: 5%)"
    )


def test_verification_is_reproducible():
    """Same seed produces byte-identical summary across calls."""
    kwargs = dict(
        n=64,
        m=32,
        k_sparsity=4,
        signal_std=1.0,
        noise_std=0.1,
        alphas=[0.3, 0.5, 0.7],
        n_monte_carlo=500,
        reference_alpha=0.5,
        seed=99,
    )
    s1 = run_verification(**kwargs)
    s2 = run_verification(**kwargs)
    assert s1 == s2, "verification is not deterministic under fixed seed"


def test_bound_shape_decreases_monotonically_in_alpha():
    """Theorem 1' bound scales like (1-alpha)/alpha and should decrease
    monotonically as alpha increases."""
    results, _ = sweep_and_verify(
        n=64,
        m=32,
        k_sparsity=4,
        alphas=[0.2, 0.4, 0.5, 0.6, 0.8],
        n_monte_carlo=500,
        reference_alpha=0.5,
        seed=11,
    )
    bounds = [r.theorem1prime_bound for r in results]
    for i in range(1, len(bounds)):
        assert bounds[i] <= bounds[i - 1] + 1e-9, (
            f"bound should be non-increasing in alpha; got {bounds}"
        )


def test_full_verification_runs_in_under_60_seconds_and_writes_artefacts(tmp_path: Path):
    """End-to-end smoke test: full verification completes quickly and
    writes both the figure and the JSON."""
    fig_path = tmp_path / "fig.pdf"
    json_path = tmp_path / "summary.json"

    start = time.perf_counter()
    summary = run_verification(
        n=128,
        m=64,
        k_sparsity=6,
        signal_std=1.0,
        noise_std=0.1,
        alphas=[0.2, 0.3, 0.5, 0.7, 0.9],
        n_monte_carlo=1500,
        reference_alpha=0.5,
        seed=2024,
        output_figure=fig_path,
        output_json=json_path,
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 60.0, f"verification took {elapsed:.1f}s (limit: 60s)"
    assert fig_path.exists(), "figure not written"
    assert json_path.exists(), "JSON not written"

    with open(json_path) as f:
        loaded = json.load(f)
    assert loaded["coherence"] == pytest.approx(1.0, abs=1e-9)
    assert len(loaded["sweep"]) == 5
    # At reference alpha the fitted C1 forces exact agreement with analytical.
    ref_row = next(r for r in loaded["sweep"] if math.isclose(r["alpha"], 0.5))
    assert math.isclose(ref_row["analytical_gap"], ref_row["theorem1prime_bound"], rel_tol=1e-6)
