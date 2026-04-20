"""Monte-Carlo verification of Theorem 1' (partial-DFT gap bound).

Companion to theory/partial_dft_bound.tex. Mirrors the structure of
theory/consistency_bounds.py (Theorem 1 for the isotropic-Gaussian
measurement case) but replaces the i.i.d. Gaussian matrix A with a
partial-DFT operator, which is what actually governs VLBI imaging.

Theorem 1' (reproduced here for quick reference):
  For a partial-DFT operator A with coherence mu, a k-sparse signal x,
  and a support-target partition with support fraction alpha >= alpha_min
  (the recovery regime), the earned-vs-enforced gap satisfies
      Delta(alpha) <= C1 * (1-alpha) * mu^2 * k * log(n) / (alpha * m) * ||x||^2 + sigma_n^2
  with an absolute constant C1.

This script:
  1. Computes the EXACT analytical gap via the posterior covariance
     Sigma_post = (sigma_x^{-2} I + sigma_n^{-2} A_S^* A_S)^{-1}.
  2. Estimates the gap empirically via Monte-Carlo over signals drawn
     from a k-sparse prior.
  3. Sweeps alpha in {0.2, 0.3, ..., 0.95} and compares the shape of
     Delta(alpha) to the Theorem 1' bound (up to the absolute constant
     C1, which is fit once at a reference alpha).
  4. Saves a plot and a JSON table, and prints the percentage agreement
     between theory and Monte Carlo.

The script is designed to run end-to-end in under 60 seconds on a CPU.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PartialDFTGapResult:
    """Gap-bound result for a single support fraction with partial DFT."""

    support_fraction: float
    n_signal: int
    m_measurements: int
    k_sparsity: int
    mu_coherence: float
    analytical_gap: float  # From Sigma_post
    empirical_gap: float  # From Monte Carlo
    theorem1prime_bound: float  # C1 * (1-alpha) * mu^2 k log n / (alpha m) * ||x||^2
    relative_error: float  # |analytical - empirical| / max(|analytical|, eps)


# ---------------------------------------------------------------------------
# Partial-DFT operator construction
# ---------------------------------------------------------------------------


def build_partial_dft(
    n: int,
    m: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build an m-by-n partial DFT matrix.

    Rows are the m rows of the unitary n-point DFT indexed by a uniformly
    random subset of size m of {0, 1, ..., n-1}. This is the classical
    Candes-Tao / Rudelson-Vershynin sensing model.

    Args:
        n: Signal dimension.
        m: Number of measurements (must satisfy m <= n).
        rng: NumPy random generator.

    Returns:
        Complex-valued (m, n) array. Rows have l2-norm 1 each.
    """
    if m > n:
        raise ValueError(f"m={m} cannot exceed n={n} for a partial DFT")
    idx = rng.choice(n, size=m, replace=False)
    freqs = np.arange(n)
    angles = -2 * np.pi * np.outer(idx, freqs) / n
    A = np.exp(1j * angles) / math.sqrt(n)
    return A


def coherence(A: np.ndarray) -> float:
    """Coherence mu(A) = sqrt(n) * max |A_{i,j}|."""
    n = A.shape[1]
    return float(math.sqrt(n) * np.max(np.abs(A)))


# ---------------------------------------------------------------------------
# Posterior & gap computation
# ---------------------------------------------------------------------------


def analytical_gap(
    A: np.ndarray,
    support_fraction: float,
    signal_std: float,
    noise_std: float,
    signal_norm_sq: float,
) -> tuple[float, float, float]:
    """Compute the exact earned-vs-enforced gap via the posterior covariance.

    Returns:
        (enforced_error, earned_error, gap)
    """
    m, n = A.shape
    m_sup = max(1, int(round(m * support_fraction)))
    m_tgt = m - m_sup
    if m_tgt <= 0:
        raise ValueError("support_fraction too large; m_tgt must be positive")

    A_sup = A[:m_sup]
    A_tgt = A[m_sup:]

    # Posterior covariance given support (real-valued on the signal side).
    # For complex A and real x, x | y_sup ~ N(mu_post, Sigma_post) with
    #   Sigma_post^{-1} = sigma_x^{-2} I + sigma_n^{-2} * Re(A_sup^H A_sup).
    gram_sup = (A_sup.conj().T @ A_sup).real
    precision = np.eye(n) / signal_std**2 + gram_sup / noise_std**2
    Sigma_post = np.linalg.inv(precision)

    # Earned error = (1/|T|) [ tr(A_T Sigma_post A_T^H) + sigma_n^2 |T| ]
    target_pred_cov = (A_tgt.conj() @ Sigma_post @ A_tgt.T).real
    earned = float(np.trace(target_pred_cov)) / m_tgt + noise_std**2

    enforced = noise_std**2

    # Scale by ||x||^2 factor so numbers are comparable across signal
    # distributions. The posterior variance is computed assuming
    # sigma_x^2 per coordinate, i.e. expected ||x||^2 = n * sigma_x^2.
    # We rescale the variance component so the reported gap corresponds
    # to a signal of norm signal_norm_sq.
    scale = signal_norm_sq / (n * signal_std**2)
    variance_component = earned - enforced
    earned_rescaled = enforced + scale * variance_component
    gap = earned_rescaled - enforced

    return enforced, earned_rescaled, gap


def empirical_gap_monte_carlo(
    A: np.ndarray,
    support_fraction: float,
    signal_std: float,
    noise_std: float,
    k_sparsity: int,
    n_monte_carlo: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Estimate the gap empirically over draws of (x, epsilon).

    The signal x is drawn as k-sparse with Gaussian nonzeros (sigma_std).
    The estimator is the exact MMSE (posterior mean) given y_sup.
    """
    m, n = A.shape
    m_sup = max(1, int(round(m * support_fraction)))
    m_tgt = m - m_sup
    A_sup = A[:m_sup]
    A_tgt = A[m_sup:]

    gram_sup = (A_sup.conj().T @ A_sup).real
    precision = np.eye(n) / signal_std**2 + gram_sup / noise_std**2
    Sigma_post = np.linalg.inv(precision)

    enforced_errs, earned_errs = [], []
    for _ in range(n_monte_carlo):
        x = np.zeros(n)
        support_idx = rng.choice(n, size=k_sparsity, replace=False)
        x[support_idx] = rng.standard_normal(k_sparsity) * signal_std * math.sqrt(n / k_sparsity)

        eps_sup = (
            rng.standard_normal(m_sup) + 1j * rng.standard_normal(m_sup)
        ) * noise_std / math.sqrt(2)
        eps_tgt = (
            rng.standard_normal(m_tgt) + 1j * rng.standard_normal(m_tgt)
        ) * noise_std / math.sqrt(2)

        y_sup = A_sup @ x + eps_sup
        y_tgt = A_tgt @ x + eps_tgt

        # MMSE estimator (posterior mean). For complex A, real x, the
        # posterior mean is sigma_n^{-2} Sigma_post Re(A_sup^H y_sup).
        x_hat = (Sigma_post @ (A_sup.conj().T @ y_sup).real) / noise_std**2

        enforced_errs.append(float(np.mean(np.abs(A_sup @ x_hat - y_sup) ** 2)))
        earned_errs.append(float(np.mean(np.abs(A_tgt @ x_hat - y_tgt) ** 2)))

    return float(np.mean(enforced_errs)), float(np.mean(earned_errs)), float(
        np.mean(earned_errs) - np.mean(enforced_errs)
    )


def theorem1prime_bound(
    alpha: float,
    mu: float,
    k: int,
    m: int,
    n: int,
    signal_norm_sq: float,
    C1: float,
) -> float:
    """RHS of Theorem 1' (without the sigma_n^2 noise term)."""
    if alpha <= 0 or alpha >= 1:
        return float("inf")
    return C1 * (1 - alpha) * mu**2 * k * math.log(n) / (alpha * m) * signal_norm_sq


# ---------------------------------------------------------------------------
# Sweep + verification
# ---------------------------------------------------------------------------


def sweep_and_verify(
    n: int = 256,
    m: int = 128,
    k_sparsity: int = 8,
    signal_std: float = 1.0,
    noise_std: float = 0.1,
    alphas: list[float] | None = None,
    n_monte_carlo: int = 2000,
    reference_alpha: float = 0.5,
    seed: int = 1729,
) -> tuple[list[PartialDFTGapResult], float]:
    """Sweep alpha and verify Theorem 1'.

    Returns:
        (results_list, fitted_C1)
    """
    if alphas is None:
        alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    rng = np.random.default_rng(seed)
    A = build_partial_dft(n, m, rng)
    mu = coherence(A)
    signal_norm_sq = float(n * signal_std**2)  # E[||x||^2] for Gaussian prior

    # Pass 1: compute analytical and empirical gaps at each alpha.
    analytical = {}
    empirical = {}
    for alpha in alphas:
        _, _, g_ana = analytical_gap(
            A, alpha, signal_std, noise_std, signal_norm_sq
        )
        _, _, g_mc = empirical_gap_monte_carlo(
            A, alpha, signal_std, noise_std, k_sparsity, n_monte_carlo, rng
        )
        analytical[alpha] = g_ana
        empirical[alpha] = g_mc

    # Fit C1 so that Theorem 1' bound equals the analytical gap at reference_alpha.
    if reference_alpha not in analytical:
        raise ValueError(f"reference_alpha={reference_alpha} not in sweep")
    ref_gap = analytical[reference_alpha]
    ref_bound_no_C1 = theorem1prime_bound(
        reference_alpha, mu, k_sparsity, m, n, signal_norm_sq, C1=1.0
    )
    C1_fit = ref_gap / ref_bound_no_C1 if ref_bound_no_C1 > 0 else 1.0

    # Build results.
    results = []
    for alpha in alphas:
        bound = theorem1prime_bound(alpha, mu, k_sparsity, m, n, signal_norm_sq, C1_fit)
        rel_err = abs(analytical[alpha] - empirical[alpha]) / max(abs(analytical[alpha]), 1e-12)
        results.append(
            PartialDFTGapResult(
                support_fraction=alpha,
                n_signal=n,
                m_measurements=m,
                k_sparsity=k_sparsity,
                mu_coherence=mu,
                analytical_gap=float(analytical[alpha]),
                empirical_gap=float(empirical[alpha]),
                theorem1prime_bound=float(bound),
                relative_error=float(rel_err),
            )
        )
    return results, float(C1_fit)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_one_panel(ax, results: list[PartialDFTGapResult], C1_fit: float, label: str) -> None:
    alphas = [r.support_fraction for r in results]
    ana = [r.analytical_gap for r in results]
    emp = [r.empirical_gap for r in results]
    bound = [r.theorem1prime_bound for r in results]
    ax.plot(alphas, ana, "o-", label=r"Analytical ($\Sigma_{\rm post}$)", linewidth=2)
    ax.plot(alphas, emp, "s--", label="Monte Carlo", linewidth=1.5)
    ax.plot(alphas, bound, "k:", label=f"Theorem 1' bound ($C_1$={C1_fit:.3f})", linewidth=1.5)
    ax.set_xlabel(r"Support fraction $\alpha$")
    ax.set_ylabel(r"Gap $\Delta(\alpha)$")
    ax.set_title(
        f"{label}: n={results[0].n_signal}, m={results[0].m_measurements}, "
        f"k={results[0].k_sparsity}, $\\mu$={results[0].mu_coherence:.3f}"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_results(
    results: list[PartialDFTGapResult],
    C1_fit: float,
    output_path: Path,
) -> None:
    """Plot analytical, empirical, and Theorem 1' bound vs alpha (single panel)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    _plot_one_panel(ax, results, C1_fit, label="Theorem 1'")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_multi_panel(
    panels: list[dict],
    output_path: Path,
) -> None:
    """Plot one panel per (n, m, k) configuration, side-by-side.

    Each entry of `panels` is a dict with keys: label, results (list of
    PartialDFTGapResult), C1_fit (float).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]
    for ax, panel in zip(axes, panels):
        _plot_one_panel(ax, panel["results"], panel["C1_fit"], panel["label"])
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point used by the test + CLI
# ---------------------------------------------------------------------------


def run_verification(
    output_figure: Path | None = None,
    output_json: Path | None = None,
    **sweep_kwargs,
) -> dict:
    """Run the full verification, return summary dict, and optionally save artefacts."""
    results, C1_fit = sweep_and_verify(**sweep_kwargs)

    summary = {
        "fitted_C1": C1_fit,
        "coherence": results[0].mu_coherence,
        "n_signal": results[0].n_signal,
        "m_measurements": results[0].m_measurements,
        "k_sparsity": results[0].k_sparsity,
        "sweep": [
            {
                "alpha": r.support_fraction,
                "analytical_gap": r.analytical_gap,
                "empirical_gap": r.empirical_gap,
                "theorem1prime_bound": r.theorem1prime_bound,
                "relative_error": r.relative_error,
            }
            for r in results
        ],
        "max_relative_error": max(r.relative_error for r in results),
        "relative_error_at_ref_alpha": next(
            r.relative_error for r in results if abs(r.support_fraction - 0.5) < 1e-9
        ),
    }

    if output_figure is not None:
        plot_results(results, C1_fit, output_figure)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)

    return summary


def _print_panel(label: str, summary: dict) -> None:
    print("=" * 64)
    print(f"{label}: n={summary['n_signal']}, m={summary['m_measurements']}, "
          f"k={summary['k_sparsity']}, mu={summary['coherence']:.4f}")
    print(f"Fitted C1 (at alpha=0.5): {summary['fitted_C1']:.4f}")
    print(f"{'alpha':>6} {'analytical':>12} {'empirical':>12} {'bound':>12} {'rel_err':>10}")
    for row in summary["sweep"]:
        print(
            f"{row['alpha']:>6.2f} "
            f"{row['analytical_gap']:>12.6f} "
            f"{row['empirical_gap']:>12.6f} "
            f"{row['theorem1prime_bound']:>12.6f} "
            f"{row['relative_error']:>10.4%}"
        )
    rel_at_02 = next(
        (r["relative_error"] for r in summary["sweep"] if abs(r["alpha"] - 0.2) < 1e-9),
        float("nan"),
    )
    print(f"Max rel error : {summary['max_relative_error']:.4%}")
    print(f"Rel err @ 0.2 : {rel_at_02:.4%}")
    print(f"Rel err @ 0.5 : {summary['relative_error_at_ref_alpha']:.4%}")
    print()


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    fig_path = repo_root / "figures_out" / "partial_dft_numerical_verification.pdf"
    json_path = repo_root / "theory" / "partial_dft_verification_results.json"

    # Panel A: original small-scale sanity check (n=256).
    # Panel B: EHT-per-frame regime (n=1024); checks whether the Theorem 1'
    # bound tightens at larger n as the asymptotic analysis predicts.
    panel_configs = [
        dict(
            label="Panel A — small-scale (n=256)",
            n=256,
            m=128,
            k_sparsity=8,
            signal_std=1.0,
            noise_std=0.1,
            alphas=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            n_monte_carlo=2000,
            reference_alpha=0.5,
            seed=1729,
        ),
        dict(
            label="Panel B — EHT-per-frame regime (n=1024)",
            n=1024,
            m=512,
            k_sparsity=32,
            signal_std=1.0,
            noise_std=0.1,
            alphas=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            n_monte_carlo=2000,
            reference_alpha=0.5,
            seed=1729,
        ),
    ]

    import time as _time

    panels_out = []
    all_summaries = {}
    t_total_start = _time.perf_counter()
    for cfg in panel_configs:
        label = cfg.pop("label")
        t0 = _time.perf_counter()
        results, C1_fit = sweep_and_verify(**cfg)
        t1 = _time.perf_counter()
        summary = {
            "fitted_C1": C1_fit,
            "coherence": results[0].mu_coherence,
            "n_signal": results[0].n_signal,
            "m_measurements": results[0].m_measurements,
            "k_sparsity": results[0].k_sparsity,
            "sweep": [
                {
                    "alpha": r.support_fraction,
                    "analytical_gap": r.analytical_gap,
                    "empirical_gap": r.empirical_gap,
                    "theorem1prime_bound": r.theorem1prime_bound,
                    "relative_error": r.relative_error,
                }
                for r in results
            ],
            "max_relative_error": max(r.relative_error for r in results),
            "relative_error_at_ref_alpha": next(
                r.relative_error for r in results if abs(r.support_fraction - 0.5) < 1e-9
            ),
            "runtime_seconds": t1 - t0,
        }
        panels_out.append({"label": label, "results": results, "C1_fit": C1_fit})
        all_summaries[label] = summary
        _print_panel(label, summary)

    plot_multi_panel(panels_out, fig_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    t_total = _time.perf_counter() - t_total_start

    print("=" * 64)
    print("Summary")
    print("=" * 64)
    for label, s in all_summaries.items():
        rel_02 = next(
            (r["relative_error"] for r in s["sweep"] if abs(r["alpha"] - 0.2) < 1e-9),
            float("nan"),
        )
        print(
            f"{label}\n"
            f"  max rel err : {s['max_relative_error']:.4%}\n"
            f"  rel err @0.2: {rel_02:.4%}\n"
            f"  rel err @0.5: {s['relative_error_at_ref_alpha']:.4%}\n"
            f"  runtime     : {s['runtime_seconds']:.2f} s"
        )
    print(f"\nTotal runtime: {t_total:.2f} s")
    print(f"Figure saved to: {fig_path}")
    print(f"JSON   saved to: {json_path}")
