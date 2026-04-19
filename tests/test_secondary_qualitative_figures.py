from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from dynadiff_vlbi.evaluation.emc_artifacts import save_emc_secondary_qualitative_figure
from dynadiff_vlbi.evaluation.public_eht_suite_artifacts import (
    PublicTrackSpec,
    save_public_qualitative_figure,
    select_public_qualitative_example,
)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "model", "heldout_visibility_rmse"])
        writer.writeheader()
        writer.writerows(rows)


def _write_synthetic_bundle(path: Path) -> None:
    ground_truth = np.zeros((5, 3, 4, 4), dtype=np.float32)
    ground_truth[:, :, 1:3, 1:3] = 1.0
    dirty = np.clip(ground_truth - 0.25, -1.0, 1.0)
    baseline = np.clip(ground_truth - 0.10, 0.0, 1.0)
    residual = np.clip(ground_truth - 0.05, 0.0, 1.0)
    ccrr = np.clip(ground_truth - 0.08, 0.0, 1.0)
    emc = np.clip(ground_truth - 0.06, 0.0, 1.0)
    target_mask = np.zeros((5, 3, 4, 4), dtype=np.float32)
    target_mask[:, :, 0, 0] = 1.0

    # Make sample 2 the representative positive-gain case.
    ccrr[2, 0] = np.clip(ground_truth[2, 0] - 0.12, 0.0, 1.0)
    ccrr[2, 1] = np.clip(ground_truth[2, 1] - 0.15, 0.0, 1.0)
    ccrr[2, 2] = np.clip(ground_truth[2, 2] - 0.09, 0.0, 1.0)
    emc[2, 0] = np.clip(ground_truth[2, 0] - 0.10, 0.0, 1.0)
    emc[2, 1] = np.clip(ground_truth[2, 1] - 0.04, 0.0, 1.0)
    emc[2, 2] = np.clip(ground_truth[2, 2] - 0.08, 0.0, 1.0)

    np.savez(
        path,
        ground_truth=ground_truth,
        dirty=dirty,
        baseline_learned=baseline,
        residual_refinement=residual,
        ccrr=ccrr,
        emc=emc,
        target_mask=target_mask,
    )


def test_save_emc_secondary_qualitative_figure_writes_deterministic_selection(tmp_path: Path) -> None:
    prediction_path = tmp_path / "synthetic_predictions.npz"
    _write_synthetic_bundle(prediction_path)
    csv_path = tmp_path / "per_sample_support_40.csv"
    _write_rows(
        csv_path,
        [
            {"sample_index": 0, "model": "ccrr", "heldout_visibility_rmse": 0.10},
            {"sample_index": 0, "model": "emc", "heldout_visibility_rmse": 0.11},
            {"sample_index": 1, "model": "ccrr", "heldout_visibility_rmse": 0.20},
            {"sample_index": 1, "model": "emc", "heldout_visibility_rmse": 0.25},
            {"sample_index": 2, "model": "ccrr", "heldout_visibility_rmse": 0.30},
            {"sample_index": 2, "model": "emc", "heldout_visibility_rmse": 0.20},
            {"sample_index": 3, "model": "ccrr", "heldout_visibility_rmse": 0.40},
            {"sample_index": 3, "model": "emc", "heldout_visibility_rmse": 0.30},
            {"sample_index": 4, "model": "ccrr", "heldout_visibility_rmse": 0.50},
            {"sample_index": 4, "model": "emc", "heldout_visibility_rmse": 0.55},
        ],
    )

    manifest_path = tmp_path / "synthetic.selection.json"
    selection = save_emc_secondary_qualitative_figure(
        prediction_path=prediction_path,
        per_sample_csv=csv_path,
        output_png=tmp_path / "synthetic.png",
        output_svg=tmp_path / "synthetic.svg",
        selection_manifest=manifest_path,
        support_fraction_tag="40",
        condition_title="Synthetic test condition",
    )

    assert selection["sample_index"] == 3
    assert selection["frame_index"] == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["figure_layout"][-1] == "Held-out target mask"
    assert (tmp_path / "synthetic.png").exists()
    assert (tmp_path / "synthetic.svg").exists()


def _write_public_track(
    root: Path,
    *,
    target: str,
    campaign_year: str,
    release_code: str,
    gains: list[tuple[int, float]],
    occupancies: list[float],
) -> PublicTrackSpec:
    output_dir = root / f"{target.lower().replace(' ', '_')}_{campaign_year}_{release_code.lower()}"
    logs_dir = output_dir / "logs"
    predictions_dir = output_dir / "predictions"
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "target": target,
        "campaign_year": campaign_year,
        "release_code": release_code,
        "support_fractions": {"60": {"models": {}}},
    }
    summary_path = output_dir / "logs" / "real_data_protocol_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    rows: list[dict[str, object]] = []
    sample_count = len(gains)
    target_mask = np.zeros((sample_count, 3, 4, 4), dtype=np.float32)
    support_mask = np.zeros((sample_count, 3, 4, 4), dtype=np.float32)
    images = {}
    for key, scale in [
        ("dirty", 0.05),
        ("baseline_learned", 0.10),
        ("residual_refinement", 0.15),
        ("emc", 0.20),
        ("ehtim_bridge", 0.25),
        ("tikhonov", 0.30),
    ]:
        images[key] = np.full((sample_count, 3, 4, 4), scale, dtype=np.float32)
    for sample_index, (_, gain) in enumerate(gains):
        rows.extend(
            [
                {"sample_index": sample_index, "model": "residual_refinement", "heldout_visibility_rmse": 0.40},
                {"sample_index": sample_index, "model": "emc", "heldout_visibility_rmse": 0.40 - gain},
            ]
        )
        target_mask[sample_index, 0, 0, 0] = occupancies[0]
        target_mask[sample_index, 1, : int(occupancies[1]), 0] = 1.0
        target_mask[sample_index, 2, : int(occupancies[2]), 0] = 1.0
        support_mask[sample_index, :, 1, 1] = 1.0

    _write_rows(logs_dir / "per_sample_support_60.csv", rows)
    np.savez(
        predictions_dir / "support_60.npz",
        support_mask=support_mask,
        target_mask=target_mask,
        **images,
    )
    per_sample_paths = [logs_dir / f"per_sample_support_{tag}.csv" for tag in ("80", "60", "40", "20")]
    return PublicTrackSpec(
        family="baseline_track_blocks",
        output_dir=output_dir,
        summary_path=summary_path,
        per_sample_paths=per_sample_paths,
    )


def test_select_public_qualitative_example_uses_release_and_sample_medians(tmp_path: Path) -> None:
    m87_2018 = _write_public_track(
        tmp_path,
        target="M87",
        campaign_year="2018",
        release_code="2024-D01-01",
        gains=[(0, 0.020), (1, -0.010), (2, 0.050), (3, 0.005)],
        occupancies=[1.0, 3.0, 2.0],
    )
    m87_2017 = _write_public_track(
        tmp_path,
        target="M87",
        campaign_year="2017",
        release_code="2019-D01-01",
        gains=[(0, 0.090), (1, 0.110), (2, 0.080)],
        occupancies=[1.0, 2.0, 3.0],
    )

    selection = select_public_qualitative_example([m87_2017, m87_2018], support_fraction_tag="60")

    assert selection["target"] == "M87"
    assert selection["campaign_year"] == "2018"
    assert selection["sample_index"] == 0
    assert selection["frame_index"] == 2
    assert selection["release_median_gap_emc_minus_residual"] == pytest.approx(-0.0125)
    assert selection["pooled_public_median_gap_emc_minus_residual"] == pytest.approx(-0.05)


def test_save_public_qualitative_figure_reads_bundle_and_writes_expected_labels(tmp_path: Path) -> None:
    track = _write_public_track(
        tmp_path,
        target="M87",
        campaign_year="2018",
        release_code="2024-D01-01",
        gains=[(0, 0.020), (1, -0.010), (2, 0.050), (3, 0.005)],
        occupancies=[1.0, 3.0, 2.0],
    )

    manifest_path = tmp_path / "public.selection.json"
    selection = save_public_qualitative_figure(
        track_specs=[track],
        output_png=tmp_path / "public.png",
        output_svg=tmp_path / "public.svg",
        selection_manifest=manifest_path,
        support_fraction_tag="60",
    )

    assert selection["track_label"] == "M87 2018 (2024-D01-01)"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["figure_layout"] == [
        "Support-only dirty",
        "Baseline 3D U-Net",
        "Residual refinement",
        "EMC",
        "eht-imaging bridge",
        "Tikhonov",
        "Support / held-out mask",
    ]
    assert "non-negative" not in manifest["selection_rule"]
    assert (tmp_path / "public.png").exists()
    assert (tmp_path / "public.svg").exists()
