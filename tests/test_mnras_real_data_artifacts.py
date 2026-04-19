from __future__ import annotations

from dynadiff_vlbi.evaluation.mnras_real_data_artifacts import build_ablation_rows, build_real_data_rows


def test_build_real_data_rows_picks_best_models() -> None:
    summary = {
        "support_fractions": {
            "80": {
                "support_fraction": 0.8,
                "mean_support_coefficients": 10.0,
                "mean_target_coefficients": 2.0,
                "mean_all_target_triangles": 1.0,
                "mean_mixed_triangles": 5.0,
                "models": {
                    "dirty": {
                        "heldout_visibility_rmse": 0.4,
                        "observed_visibility_rmse": 0.3,
                        "support_visibility_rmse": 0.1,
                        "heldout_closure_phase_mae": 1.0,
                    },
                    "tikhonov": {
                        "heldout_visibility_rmse": 0.35,
                        "observed_visibility_rmse": 0.2,
                        "support_visibility_rmse": 0.05,
                        "heldout_closure_phase_mae": 0.9,
                    },
                    "baseline_learned": {
                        "heldout_visibility_rmse": 0.32,
                        "observed_visibility_rmse": 0.4,
                        "support_visibility_rmse": 0.4,
                        "heldout_closure_phase_mae": 0.3,
                    },
                    "visibility_conditioned": {
                        "heldout_visibility_rmse": 0.33,
                        "observed_visibility_rmse": 0.45,
                        "support_visibility_rmse": 0.45,
                        "heldout_closure_phase_mae": 0.25,
                    },
                    "residual_refinement": {
                        "heldout_visibility_rmse": 0.34,
                        "observed_visibility_rmse": 0.41,
                        "support_visibility_rmse": 0.39,
                        "heldout_closure_phase_mae": 0.32,
                    },
                    "ccrr": {
                        "heldout_visibility_rmse": 0.31,
                        "observed_visibility_rmse": 0.28,
                        "support_visibility_rmse": 0.18,
                        "heldout_closure_phase_mae": 0.31,
                    },
                    "emc": {
                        "heldout_visibility_rmse": 0.36,
                        "observed_visibility_rmse": 0.27,
                        "support_visibility_rmse": 0.17,
                        "heldout_closure_phase_mae": 0.29,
                    },
                },
            }
        }
    }
    rows = build_real_data_rows(summary)
    assert len(rows) == 1
    row = rows[0]
    assert row["best_heldout_model"] == "ccrr"
    assert row["best_observed_model"] == "tikhonov"


def test_build_ablation_rows_uses_emc_as_delta_reference() -> None:
    summary = {
        "support_fractions": {
            "80": {
                "models": {
                    "ccrr": {
                        "heldout_visibility_rmse": 0.12,
                        "heldout_closure_phase_mae": 1.5,
                        "support_visibility_rmse": 0.02,
                        "ssim": 0.7,
                        "temporal_consistency": 0.002,
                    },
                    "emc": {
                        "heldout_visibility_rmse": 0.10,
                        "heldout_closure_phase_mae": 1.4,
                        "support_visibility_rmse": 0.02,
                        "ssim": 0.75,
                        "temporal_consistency": 0.0021,
                    },
                    "emc_no_dc": {
                        "heldout_visibility_rmse": 0.11,
                        "heldout_closure_phase_mae": 1.45,
                        "support_visibility_rmse": 0.08,
                        "ssim": 0.85,
                        "temporal_consistency": 0.0015,
                    },
                    "emc_with_closure": {
                        "heldout_visibility_rmse": 0.09,
                        "heldout_closure_phase_mae": 1.35,
                        "support_visibility_rmse": 0.02,
                        "ssim": 0.77,
                        "temporal_consistency": 0.0022,
                    },
                    "emc_no_metadata": {
                        "heldout_visibility_rmse": 0.105,
                        "heldout_closure_phase_mae": 1.42,
                        "support_visibility_rmse": 0.02,
                        "ssim": 0.74,
                        "temporal_consistency": 0.0023,
                    },
                    "emc_no_uncertainty": {
                        "heldout_visibility_rmse": 0.115,
                        "heldout_closure_phase_mae": 1.43,
                        "support_visibility_rmse": 0.02,
                        "ssim": 0.73,
                        "temporal_consistency": 0.0020,
                    },
                }
            }
        }
    }
    rows = build_ablation_rows(summary)
    delta_by_model = {row["model"]: row["delta_vs_emc_heldout_visibility_rmse"] for row in rows}
    assert delta_by_model["emc"] == 0.0
    assert delta_by_model["ccrr"] > 0.0
    assert delta_by_model["emc_with_closure"] < 0.0
