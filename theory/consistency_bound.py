#!/usr/bin/env python3
"""Numerical verification of a stylized earned-versus-enforced consistency gap."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "theory" / "figures"
DEFAULT_RESULTS_PATH = ROOT / "theory" / "consistency_bound_results.json"


@dataclass(frozen=True)
class ConsistencyBoundPoint:
    alpha: float
    support_dim: int
    target_dim: int
    enforced_rmse: float
    earned_rmse: float
    empirical_gap: float
    conditional_rmse_proxy: float
    mutual_information_nats: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--measurement-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--test-samples", type=int, default=2048)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--alphas", default="0.2,0.4,0.6,0.8")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    return parser.parse_args()


def _alphas(raw: str) -> list[float]:
    values = [float(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]
    if not values:
        raise ValueError("At least one alpha value is required.")
    return values


def _make_measurement_operator(measurement_dim: int, latent_dim: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.normal(size=(measurement_dim, latent_dim))
    smooth = np.linspace(0.6, 1.4, measurement_dim, dtype=np.float64)[:, None]
    operator = base * smooth
    operator /= np.linalg.norm(operator, axis=0, keepdims=True) + 1.0e-8
    return operator.astype(np.float64)


def _sample_measurements(
    *,
    operator: np.ndarray,
    sample_count: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    latent_dim = int(operator.shape[1])
    # Correlated latent amplitudes yield a non-trivial support->target dependency structure.
    latent = rng.normal(size=(sample_count, latent_dim))
    latent[:, 1:] += 0.45 * latent[:, :-1]
    measurements = latent @ operator.T
    measurements += noise_std * rng.normal(size=measurements.shape)
    return measurements.astype(np.float64)


def _ridge_predict(
    *,
    train_support: np.ndarray,
    train_target: np.ndarray,
    test_support: np.ndarray,
    ridge: float,
) -> np.ndarray:
    support_mean = train_support.mean(axis=0, keepdims=True)
    target_mean = train_target.mean(axis=0, keepdims=True)
    centered_support = train_support - support_mean
    centered_target = train_target - target_mean
    gram = centered_support.T @ centered_support
    weights = np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0], dtype=np.float64),
        centered_support.T @ centered_target,
    )
    return (test_support - support_mean) @ weights + target_mean


def _enforced_predict(
    *,
    operator: np.ndarray,
    support_indices: np.ndarray,
    target_indices: np.ndarray,
    support_measurements: np.ndarray,
    ridge: float,
    alpha: float,
    prior_mean: np.ndarray,
) -> np.ndarray:
    support_operator = operator[support_indices]
    target_operator = operator[target_indices]
    gram = support_operator.T @ support_operator
    projector = np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0], dtype=np.float64),
        support_operator.T,
    )
    latent_estimate = support_measurements @ projector.T
    support_consistent = latent_estimate @ target_operator.T
    blend = float(np.clip(alpha, 0.0, 1.0))
    return blend * support_consistent + (1.0 - blend) * prior_mean


def _conditional_covariance(joint: np.ndarray, support_indices: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    sigma = np.cov(joint, rowvar=False)
    sigma_ss = sigma[np.ix_(support_indices, support_indices)]
    sigma_tt = sigma[np.ix_(target_indices, target_indices)]
    sigma_ts = sigma[np.ix_(target_indices, support_indices)]
    sigma_st = sigma[np.ix_(support_indices, target_indices)]
    sigma_ss_inv = np.linalg.pinv(sigma_ss, hermitian=True)
    conditional = sigma_tt - sigma_ts @ sigma_ss_inv @ sigma_st
    conditional = 0.5 * (conditional + conditional.T)
    return conditional


def _mutual_information(joint: np.ndarray, support_indices: np.ndarray, target_indices: np.ndarray) -> float:
    sigma = np.cov(joint, rowvar=False)
    sigma_tt = sigma[np.ix_(target_indices, target_indices)]
    sigma_cond = _conditional_covariance(joint, support_indices, target_indices)
    eig_t = np.clip(np.linalg.eigvalsh(sigma_tt), 1.0e-8, None)
    eig_c = np.clip(np.linalg.eigvalsh(sigma_cond), 1.0e-8, None)
    return float(0.5 * (np.log(eig_t).sum() - np.log(eig_c).sum()))


def simulate_consistency_bound(
    *,
    alphas: list[float],
    measurement_dim: int,
    latent_dim: int,
    train_samples: int,
    test_samples: int,
    noise_std: float,
    ridge: float,
    seed: int,
) -> list[ConsistencyBoundPoint]:
    rng = np.random.default_rng(seed)
    operator = _make_measurement_operator(measurement_dim, latent_dim, rng)
    train_measurements = _sample_measurements(
        operator=operator,
        sample_count=train_samples,
        noise_std=noise_std,
        rng=rng,
    )
    test_measurements = _sample_measurements(
        operator=operator,
        sample_count=test_samples,
        noise_std=noise_std,
        rng=rng,
    )
    measurement_order = rng.permutation(measurement_dim)

    points: list[ConsistencyBoundPoint] = []
    for alpha in alphas:
        support_dim = max(1, min(measurement_dim - 1, int(round(alpha * measurement_dim))))
        support_indices = np.sort(measurement_order[:support_dim])
        target_indices = np.sort(measurement_order[support_dim:])

        train_support = train_measurements[:, support_indices]
        train_target = train_measurements[:, target_indices]
        test_support = test_measurements[:, support_indices]
        test_target = test_measurements[:, target_indices]

        earned_prediction = _ridge_predict(
            train_support=train_support,
            train_target=train_target,
            test_support=test_support,
            ridge=ridge,
        )
        enforced_prediction = _enforced_predict(
            operator=operator,
            support_indices=support_indices,
            target_indices=target_indices,
            support_measurements=test_support,
            ridge=ridge,
            alpha=float(alpha),
            prior_mean=train_target.mean(axis=0, keepdims=True),
        )

        earned_rmse = float(np.sqrt(np.mean((earned_prediction - test_target) ** 2)))
        enforced_rmse = float(np.sqrt(np.mean((enforced_prediction - test_target) ** 2)))

        conditional_cov = _conditional_covariance(train_measurements, support_indices, target_indices)
        conditional_rmse_proxy = float(
            np.sqrt(max(float(np.trace(conditional_cov) / conditional_cov.shape[0]), 0.0))
        )
        mutual_information_nats = _mutual_information(train_measurements, support_indices, target_indices)

        points.append(
            ConsistencyBoundPoint(
                alpha=float(alpha),
                support_dim=int(support_dim),
                target_dim=int(target_indices.shape[0]),
                enforced_rmse=enforced_rmse,
                earned_rmse=earned_rmse,
                empirical_gap=float(enforced_rmse - earned_rmse),
                conditional_rmse_proxy=conditional_rmse_proxy,
                mutual_information_nats=mutual_information_nats,
            )
        )

    return points


def save_consistency_bound_figure(
    *,
    points: list[ConsistencyBoundPoint],
    output_pdf: Path,
    output_png: Path,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    alphas = np.asarray([point.alpha for point in points], dtype=np.float64)
    empirical_gap = np.asarray([point.empirical_gap for point in points], dtype=np.float64)
    conditional_proxy = np.asarray([point.conditional_rmse_proxy for point in points], dtype=np.float64)
    enforced_rmse = np.asarray([point.enforced_rmse for point in points], dtype=np.float64)
    earned_rmse = np.asarray([point.earned_rmse for point in points], dtype=np.float64)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), constrained_layout=True)

    axes[0].plot(alphas, empirical_gap, color="#c62828", marker="o", linewidth=2.2, label="Empirical gap")
    axes[0].plot(
        alphas,
        conditional_proxy,
        color="#455a64",
        marker="s",
        linestyle="--",
        linewidth=2.0,
        label="Conditional-RMSE proxy",
    )
    axes[0].set_xlabel(r"Support fraction $\alpha$")
    axes[0].set_ylabel("Held-out RMSE scale")
    axes[0].set_title("Support-conditioned gap proxy")
    axes[0].legend(frameon=False)

    axes[1].plot(alphas, enforced_rmse, color="#1e88e5", marker="o", linewidth=2.2, label="Enforced")
    axes[1].plot(alphas, earned_rmse, color="#2e7d32", marker="o", linewidth=2.2, label="Earned")
    axes[1].set_xlabel(r"Support fraction $\alpha$")
    axes[1].set_ylabel("Expected held-out RMSE")
    axes[1].set_title("Stylized Gaussian-source verification")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.set_xlim(min(alphas) - 0.02, max(alphas) + 0.02)

    fig.suptitle(
        "Stylized enforcement-versus-earning gap under a correlated Gaussian source",
        fontsize=11,
    )
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_results(points: list[ConsistencyBoundPoint], results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "points": [asdict(point) for point in points],
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    results_path = Path(args.results_path)
    points = simulate_consistency_bound(
        alphas=_alphas(args.alphas),
        measurement_dim=args.measurement_dim,
        latent_dim=args.latent_dim,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        noise_std=args.noise_std,
        ridge=args.ridge,
        seed=args.seed,
    )
    save_consistency_bound_figure(
        points=points,
        output_pdf=output_dir / "consistency_bound.pdf",
        output_png=output_dir / "consistency_bound.png",
    )
    save_results(points, results_path)
    print(
        json.dumps(
            {
                "figure_pdf": str((output_dir / "consistency_bound.pdf").resolve()),
                "figure_png": str((output_dir / "consistency_bound.png").resolve()),
                "results_path": str(results_path.resolve()),
                "points": [asdict(point) for point in points],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
