from __future__ import annotations

from dynadiff_vlbi.evaluation.paper_artifacts import (
    aggregate_seed_repeats,
    build_condition_verdict_rows,
    compute_verdict,
)


def _summary(
    *,
    mse: float,
    psnr: float,
    ssim: float,
    temporal: float,
    ring: float,
    hotspot: float,
) -> dict[str, float]:
    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "temporal_consistency": temporal,
        "ring_radius_error": ring,
        "hotspot_localization_error": hotspot,
    }


def test_compute_verdict_respects_metric_direction() -> None:
    assert compute_verdict(1.0, 0.5, "lower") == "win"
    assert compute_verdict(1.0, 1.0, "lower") == "tie"
    assert compute_verdict(1.0, 2.0, "lower") == "loss"
    assert compute_verdict(1.0, 2.0, "higher") == "win"


def test_seed_repeat_aggregation_tracks_residual_verdict_counts() -> None:
    loaded = {
        "seed7": {
            "dirty": _summary(mse=4.0, psnr=10.0, ssim=0.1, temporal=0.5, ring=4.0, hotspot=7.0),
            "tikhonov": _summary(mse=3.0, psnr=12.0, ssim=0.2, temporal=0.4, ring=3.0, hotspot=6.5),
            "baseline_learned": _summary(mse=1.0, psnr=20.0, ssim=0.8, temporal=0.20, ring=1.0, hotspot=5.5),
            "visibility_conditioned": _summary(mse=1.2, psnr=19.0, ssim=0.7, temporal=0.18, ring=1.1, hotspot=5.3),
            "residual_refinement": _summary(mse=0.9, psnr=21.0, ssim=0.81, temporal=0.19, ring=1.0, hotspot=5.6),
            "uncertainty": {"empirical_95_coverage": 0.99, "error_uncertainty_correlation": 0.65},
        },
        "seed19": {
            "dirty": _summary(mse=4.1, psnr=9.8, ssim=0.11, temporal=0.51, ring=4.2, hotspot=7.1),
            "tikhonov": _summary(mse=3.1, psnr=11.8, ssim=0.21, temporal=0.41, ring=3.1, hotspot=6.6),
            "baseline_learned": _summary(mse=1.1, psnr=19.5, ssim=0.79, temporal=0.22, ring=1.0, hotspot=5.4),
            "visibility_conditioned": _summary(mse=1.3, psnr=18.7, ssim=0.71, temporal=0.19, ring=1.2, hotspot=5.2),
            "residual_refinement": _summary(mse=1.0, psnr=20.2, ssim=0.83, temporal=0.21, ring=1.0, hotspot=5.5),
            "uncertainty": {"empirical_95_coverage": 1.0, "error_uncertainty_correlation": 0.66},
        },
        "seed31": {
            "dirty": _summary(mse=4.2, psnr=9.7, ssim=0.12, temporal=0.49, ring=4.1, hotspot=7.2),
            "tikhonov": _summary(mse=3.2, psnr=11.9, ssim=0.22, temporal=0.40, ring=3.2, hotspot=6.7),
            "baseline_learned": _summary(mse=0.8, psnr=21.5, ssim=0.84, temporal=0.18, ring=0.9, hotspot=5.3),
            "visibility_conditioned": _summary(mse=1.1, psnr=19.9, ssim=0.75, temporal=0.17, ring=1.0, hotspot=5.1),
            "residual_refinement": _summary(mse=0.7, psnr=22.0, ssim=0.85, temporal=0.16, ring=0.9, hotspot=5.2),
            "uncertainty": {"empirical_95_coverage": 1.0, "error_uncertainty_correlation": 0.68},
        },
    }
    rows, verdict_counts = aggregate_seed_repeats(
        loaded=loaded,
        seed_keys=["seed7", "seed19", "seed31"],
        model_order=[
            "dirty",
            "tikhonov",
            "baseline_learned",
            "visibility_conditioned",
            "residual_refinement",
        ],
    )

    residual_row = next(row for row in rows if row["model"] == "residual_refinement")
    assert residual_row["mse_mean"] < next(row for row in rows if row["model"] == "baseline_learned")["mse_mean"]
    assert verdict_counts["mse"] == {"win": 3, "tie": 0, "loss": 0}
    assert verdict_counts["hotspot_localization_error"] == {"win": 1, "tie": 0, "loss": 2}


def test_build_condition_verdict_rows_marks_wins_and_losses() -> None:
    loaded = {
        "noise_high": {
            "baseline_learned": _summary(mse=1.0, psnr=20.0, ssim=0.8, temporal=0.20, ring=1.0, hotspot=5.0),
            "residual_refinement": _summary(mse=0.9, psnr=20.5, ssim=0.79, temporal=0.19, ring=1.0, hotspot=5.2),
        }
    }
    rows = build_condition_verdict_rows(loaded=loaded, condition_keys=["noise_high"])
    verdicts = {row["metric"]: row["verdict_vs_baseline"] for row in rows}
    assert verdicts["mse"] == "win"
    assert verdicts["ssim"] == "loss"
    assert verdicts["ring_radius_error"] == "tie"
