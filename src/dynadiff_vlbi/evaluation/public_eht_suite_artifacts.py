"""Artifact generation for the multi-release public-EHT EMC validation suite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from dynadiff_vlbi.evaluation.ccrr_artifacts import paired_bootstrap_stats
from dynadiff_vlbi.evaluation.paper_artifacts import format_value, save_json, write_csv, write_markdown_table
from dynadiff_vlbi.evaluation.paper_visuals import IMAGE_CMAP, load_predictions


MODEL_LABELS: dict[str, str] = {
    "dirty": "Dirty",
    "tikhonov": "Tikhonov",
    "ehtim_bridge": "eht-imaging bridge",
    "baseline_learned": "Baseline 3D U-Net",
    "visibility_conditioned": "Standalone Visibility",
    "residual_refinement": "Residual Refinement",
    "ccrr": "CCRR",
    "emc": "EMC",
}

MODEL_COLORS: dict[str, str] = {
    "dirty": "#6b7280",
    "tikhonov": "#b45309",
    "ehtim_bridge": "#059669",
    "baseline_learned": "#111827",
    "residual_refinement": "#2563eb",
    "ccrr": "#7c3aed",
    "emc": "#dc2626",
}

PLOT_MODEL_ORDER = [
    "dirty",
    "tikhonov",
    "ehtim_bridge",
    "baseline_learned",
    "residual_refinement",
    "ccrr",
    "emc",
]

METRIC_DIRECTIONS = {
    "heldout_visibility_rmse": "lower",
    "observed_visibility_rmse": "lower",
    "heldout_reduced_chi2": "lower",
    "observed_reduced_chi2": "lower",
}


@dataclass(frozen=True)
class PublicTrackSpec:
    """One public-EHT track summary produced by the suite runner."""

    family: str
    output_dir: Path
    summary_path: Path
    per_sample_paths: list[Path]

    def per_sample_path(self, support_fraction_tag: str) -> Path:
        return self.output_dir / "logs" / f"per_sample_support_{support_fraction_tag}.csv"

    def prediction_path(self, support_fraction_tag: str) -> Path:
        return self.output_dir / "predictions" / f"support_{support_fraction_tag}.npz"


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    import csv

    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: Any) -> float:
    if value in {None, "", "nan", "NaN"}:
        return float("nan")
    return float(value)


def _fmt(value: float) -> str:
    return "n/a" if math.isnan(float(value)) else format_value(float(value))


def _descending_support_tags(summary: dict[str, Any]) -> list[str]:
    return sorted(summary["support_fractions"].keys(), key=lambda item: int(item), reverse=True)


def _model_metrics(summary: dict[str, Any], support_tag: str, model_key: str) -> dict[str, float]:
    return {
        key: _safe_float(value)
        for key, value in summary["support_fractions"][support_tag]["models"][model_key].items()
    }


def _track_label(summary: dict[str, Any]) -> str:
    return f"{summary['target']} {summary['campaign_year']} ({summary['release_code']})"


def _track_priority_key(summary: dict[str, Any]) -> tuple[str, str]:
    return str(summary["target"]), str(summary["campaign_year"])


def _mean_support_metric(summary: dict[str, Any], model_key: str, metric_key: str) -> float:
    values = [
        _model_metrics(summary, support_tag, model_key).get(metric_key, float("nan"))
        for support_tag in _descending_support_tags(summary)
    ]
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if not np.all(np.isnan(array)) else float("nan")


def _group_sample_metrics(rows: list[dict[str, str]], metric_key: str) -> dict[int, dict[str, float]]:
    grouped: dict[int, dict[str, float]] = {}
    for row in rows:
        sample_index = int(row.get("sample_index", row.get("sample_id", "0")))
        metric_value = _safe_float(row.get(metric_key))
        if math.isnan(metric_value):
            continue
        grouped.setdefault(sample_index, {})[str(row["model"])] = metric_value
    return grouped


def select_public_qualitative_example(
    track_specs: list[PublicTrackSpec],
    support_fraction_tag: str = "60",
) -> dict[str, Any]:
    """Select one deterministic public-EHT qualitative example."""

    preferred_order = [
        ("M87", "2018"),
        ("M87", "2017"),
        ("Centaurus A", "2017"),
        ("3C279", "2017"),
    ]
    spec_by_priority: dict[tuple[str, str], PublicTrackSpec] = {}
    summary_by_priority: dict[tuple[str, str], dict[str, Any]] = {}
    for spec in track_specs:
        summary = load_json(spec.summary_path)
        priority_key = _track_priority_key(summary)
        spec_by_priority[priority_key] = spec
        summary_by_priority[priority_key] = summary

    release_records: list[dict[str, Any]] = []
    pooled_sample_gaps: list[float] = []
    for priority_rank, priority_key in enumerate(preferred_order):
        spec = spec_by_priority.get(priority_key)
        summary = summary_by_priority.get(priority_key)
        if spec is None or summary is None:
            continue
        rows = load_csv_rows(spec.per_sample_path(support_fraction_tag))
        grouped = _group_sample_metrics(rows, "heldout_visibility_rmse")
        sample_gaps = sorted(
            [
                (sample_index, model_map["emc"] - model_map["residual_refinement"])
                for sample_index, model_map in grouped.items()
                if "residual_refinement" in model_map and "emc" in model_map
            ],
            key=lambda item: item[0],
        )
        if not sample_gaps:
            continue
        pooled_sample_gaps.extend(gap for _, gap in sample_gaps)
        release_gap_values = [gap for _, gap in sample_gaps]
        release_records.append(
            {
                "priority_rank": priority_rank,
                "priority_key": priority_key,
                "spec": spec,
                "summary": summary,
                "sample_gaps": sample_gaps,
                "release_median_gap": float(np.median(np.asarray(release_gap_values, dtype=np.float64))),
            }
        )

    if not release_records:
        raise ValueError("No eligible public-EHT qualitative tracks were found for support 60%.")

    pooled_public_median_gap = float(np.median(np.asarray(pooled_sample_gaps, dtype=np.float64)))
    for record in release_records:
        record["release_gap_distance_to_pooled"] = abs(record["release_median_gap"] - pooled_public_median_gap)
    chosen_record = min(
        release_records,
        key=lambda record: (record["release_gap_distance_to_pooled"], record["priority_rank"]),
    )
    chosen_spec = chosen_record["spec"]
    chosen_summary = chosen_record["summary"]
    release_median_gap = float(chosen_record["release_median_gap"])
    sample_index, sample_gap = min(
        chosen_record["sample_gaps"],
        key=lambda item: (abs(item[1] - release_median_gap), item[0]),
    )
    selection_rule = (
        "release median EMC-minus-residual held-out-visibility gap closest to the pooled public median at 60% "
        "support, then sample closest to that release median, then frame with median target-mask occupancy"
    )
    bundle = load_predictions(chosen_spec.prediction_path(support_fraction_tag))
    target_mask = np.asarray(bundle["target_mask"][sample_index], dtype=np.float32)
    occupancy = target_mask.reshape(target_mask.shape[0], -1).sum(axis=1)
    frame_order = np.argsort(occupancy, kind="stable")
    frame_index = int(frame_order[len(frame_order) // 2])

    return {
        "track_label": _track_label(chosen_summary),
        "target": chosen_summary["target"],
        "campaign_year": chosen_summary["campaign_year"],
        "release_code": chosen_summary["release_code"],
        "family": chosen_spec.family,
        "support_fraction_tag": support_fraction_tag,
        "sample_index": int(sample_index),
        "frame_index": frame_index,
        "sample_gap_emc_minus_residual": float(sample_gap),
        "release_median_gap_emc_minus_residual": release_median_gap,
        "pooled_public_median_gap_emc_minus_residual": pooled_public_median_gap,
        "release_gap_distance_to_pooled": float(chosen_record["release_gap_distance_to_pooled"]),
        "sample_gap_distance_to_release_median": float(abs(sample_gap - release_median_gap)),
        "frame_target_occupancy": float(occupancy[frame_index]),
        "tie_break_order": [f"{target} {year}" for target, year in preferred_order],
        "tie_break_decision": f"selected {_track_label(chosen_summary)} by minimum release-median distance to pooled public median; ties broken by the documented release order",
        "selection_rule": selection_rule,
        "prediction_path": str(chosen_spec.prediction_path(support_fraction_tag)),
        "per_sample_csv": str(chosen_spec.per_sample_path(support_fraction_tag)),
    }


def build_public_matrix_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the primary public-EHT benchmark matrix for the baseline-track family."""

    rows: list[dict[str, Any]] = []
    for summary in summaries:
        track_label = _track_label(summary)
        for support_tag in _descending_support_tags(summary):
            support_payload = summary["support_fractions"][support_tag]
            models = support_payload["models"]
            best_heldout_key = min(models, key=lambda key: _safe_float(models[key]["heldout_visibility_rmse"]))
            best_observed_key = min(models, key=lambda key: _safe_float(models[key]["observed_visibility_rmse"]))
            rows.append(
                {
                    "track_label": track_label,
                    "release_code": summary["release_code"],
                    "target": summary["target"],
                    "campaign_year": summary["campaign_year"],
                    "support_fraction_tag": support_tag,
                    "support_fraction": float(support_payload["support_fraction"]),
                    "sample_count": int(summary["sample_count"]),
                    "mean_target_coefficients": float(support_payload["mean_target_coefficients"]),
                    "mean_all_target_triangles": float(support_payload["mean_all_target_triangles"]),
                    "best_heldout_model": best_heldout_key,
                    "best_heldout_model_label": MODEL_LABELS[best_heldout_key],
                    "best_heldout_visibility_rmse": _safe_float(models[best_heldout_key]["heldout_visibility_rmse"]),
                    "best_observed_model": best_observed_key,
                    "best_observed_model_label": MODEL_LABELS[best_observed_key],
                    "best_observed_visibility_rmse": _safe_float(models[best_observed_key]["observed_visibility_rmse"]),
                    "emc_heldout_visibility_rmse": _safe_float(models["emc"]["heldout_visibility_rmse"]),
                    "ccrr_heldout_visibility_rmse": _safe_float(models["ccrr"]["heldout_visibility_rmse"]),
                    "baseline_heldout_visibility_rmse": _safe_float(models["baseline_learned"]["heldout_visibility_rmse"]),
                    "residual_heldout_visibility_rmse": _safe_float(models["residual_refinement"]["heldout_visibility_rmse"]),
                    "tikhonov_heldout_visibility_rmse": _safe_float(models["tikhonov"]["heldout_visibility_rmse"]),
                    "ehtim_bridge_heldout_visibility_rmse": _safe_float(models["ehtim_bridge"]["heldout_visibility_rmse"]),
                    "dirty_heldout_visibility_rmse": _safe_float(models["dirty"]["heldout_visibility_rmse"]),
                    "emc_observed_visibility_rmse": _safe_float(models["emc"]["observed_visibility_rmse"]),
                    "emc_heldout_reduced_chi2": _safe_float(models["emc"]["heldout_reduced_chi2"]),
                    "days_present": ",".join(str(value) for value in support_payload.get("days_present", [])),
                    "bands_present": ",".join(str(value) for value in support_payload.get("bands_present", [])),
                }
            )
    return rows


def write_public_matrix_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    markdown_rows = [
        [
            str(row["track_label"]),
            str(row["support_fraction_tag"]),
            str(row["sample_count"]),
            _fmt(float(row["mean_target_coefficients"])),
            str(row["best_heldout_model_label"]),
            _fmt(float(row["best_heldout_visibility_rmse"])),
            _fmt(float(row["emc_heldout_visibility_rmse"])),
            _fmt(float(row["ccrr_heldout_visibility_rmse"])),
            _fmt(float(row["baseline_heldout_visibility_rmse"])),
            _fmt(float(row["ehtim_bridge_heldout_visibility_rmse"])),
            _fmt(float(row["tikhonov_heldout_visibility_rmse"])),
            str(row["best_observed_model_label"]),
        ]
        for row in rows
    ]
    write_markdown_table(
        path,
        headers=[
            "Track",
            "Support (%)",
            "Cases",
            "Held-out coeffs",
            "Best held-out model",
            "Best held-out VisRMSE",
            "EMC",
            "CCRR",
            "Baseline",
            "eht-imaging",
            "Tikhonov",
            "Best observed model",
        ],
        rows=markdown_rows,
        title="Public EHT Observation-Domain Benchmark Matrix",
        notes=[
            "The matrix uses the baseline-track support-target family across all official public EHT releases included in this study.",
            "Real-data evaluation is observation-domain only: there is no image-domain ground truth, and the table does not support morphology claims on real observations.",
        ],
    )


def build_sensitivity_rows(
    *,
    baseline_summaries: list[dict[str, Any]],
    station_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize real-data protocol sensitivity beyond one structured family."""

    station_by_track = {_track_label(summary): summary for summary in station_summaries}
    rows: list[dict[str, Any]] = []
    for baseline_summary in baseline_summaries:
        track_label = _track_label(baseline_summary)
        if track_label not in station_by_track:
            continue
        station_summary = station_by_track[track_label]
        for family_key, summary in [
            ("baseline_track_blocks", baseline_summary),
            ("station_dropout", station_summary),
        ]:
            best_mean_key = min(
                MODEL_LABELS,
                key=lambda key: _mean_support_metric(summary, key, "heldout_visibility_rmse")
                if key in summary["support_fractions"]["80"]["models"]
                else float("inf"),
            )
            rows.append(
                {
                    "track_label": track_label,
                    "family": family_key,
                    "family_label": "Baseline-track blocks" if family_key == "baseline_track_blocks" else "Station dropout",
                    "mean_emc_heldout_visibility_rmse": _mean_support_metric(summary, "emc", "heldout_visibility_rmse"),
                    "mean_ccrr_heldout_visibility_rmse": _mean_support_metric(summary, "ccrr", "heldout_visibility_rmse"),
                    "mean_baseline_heldout_visibility_rmse": _mean_support_metric(summary, "baseline_learned", "heldout_visibility_rmse"),
                    "mean_ehtim_bridge_heldout_visibility_rmse": _mean_support_metric(summary, "ehtim_bridge", "heldout_visibility_rmse"),
                    "mean_tikhonov_heldout_visibility_rmse": _mean_support_metric(summary, "tikhonov", "heldout_visibility_rmse"),
                    "best_mean_model": best_mean_key,
                    "best_mean_model_label": MODEL_LABELS[best_mean_key],
                    "best_mean_heldout_visibility_rmse": _mean_support_metric(summary, best_mean_key, "heldout_visibility_rmse"),
                }
            )
    return rows


def write_sensitivity_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    markdown_rows = [
        [
            str(row["track_label"]),
            str(row["family_label"]),
            _fmt(float(row["mean_emc_heldout_visibility_rmse"])),
            _fmt(float(row["mean_ccrr_heldout_visibility_rmse"])),
            _fmt(float(row["mean_baseline_heldout_visibility_rmse"])),
            _fmt(float(row["mean_ehtim_bridge_heldout_visibility_rmse"])),
            _fmt(float(row["mean_tikhonov_heldout_visibility_rmse"])),
            str(row["best_mean_model_label"]),
            _fmt(float(row["best_mean_heldout_visibility_rmse"])),
        ]
        for row in rows
    ]
    write_markdown_table(
        path,
        headers=[
            "Track",
            "Family",
            "Mean EMC",
            "Mean CCRR",
            "Mean baseline",
            "Mean eht-imaging",
            "Mean Tikhonov",
            "Best mean model",
            "Best mean VisRMSE",
        ],
        rows=markdown_rows,
        title="Public EHT Protocol-Sensitivity Summary",
        notes=[
            "Means are taken across the 80/60/40/20 support sweep within each track and family.",
            "The table is intended as a split-design robustness check rather than a second benchmark claim.",
        ],
    )


def build_release_robustness_rows(
    *,
    baseline_summaries: list[dict[str, Any]],
    station_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize public-release robustness across support fractions and families."""

    station_by_track = {_track_label(summary): summary for summary in station_summaries}
    rows: list[dict[str, Any]] = []
    for baseline_summary in baseline_summaries:
        track_label = _track_label(baseline_summary)
        baseline_support_tags = _descending_support_tags(baseline_summary)
        emc_baseline_values = np.asarray(
            [_model_metrics(baseline_summary, tag, "emc")["heldout_visibility_rmse"] for tag in baseline_support_tags],
            dtype=np.float64,
        )
        ehtim_baseline_values = np.asarray(
            [_model_metrics(baseline_summary, tag, "ehtim_bridge")["heldout_visibility_rmse"] for tag in baseline_support_tags],
            dtype=np.float64,
        )
        baseline_best_mean_key = min(
            MODEL_LABELS,
            key=lambda key: _mean_support_metric(baseline_summary, key, "heldout_visibility_rmse")
            if key in baseline_summary["support_fractions"]["80"]["models"]
            else float("inf"),
        )
        row = {
            "track_label": track_label,
            "sample_count": int(baseline_summary["sample_count"]),
            "support_fractions": "/".join(baseline_support_tags),
            "days_present": ",".join(str(value) for value in baseline_summary["support_fractions"]["80"].get("days_present", [])),
            "bands_present": ",".join(str(value) for value in baseline_summary["support_fractions"]["80"].get("bands_present", [])),
            "mean_emc_baseline_track": float(np.nanmean(emc_baseline_values)),
            "std_emc_baseline_track": float(np.nanstd(emc_baseline_values)),
            "mean_ehtim_baseline_track": float(np.nanmean(ehtim_baseline_values)),
            "best_baseline_track_model": baseline_best_mean_key,
            "best_baseline_track_model_label": MODEL_LABELS[baseline_best_mean_key],
            "best_baseline_track_mean": _mean_support_metric(
                baseline_summary,
                baseline_best_mean_key,
                "heldout_visibility_rmse",
            ),
        }
        station_summary = station_by_track.get(track_label)
        if station_summary is not None:
            station_support_tags = _descending_support_tags(station_summary)
            emc_station_values = np.asarray(
                [_model_metrics(station_summary, tag, "emc")["heldout_visibility_rmse"] for tag in station_support_tags],
                dtype=np.float64,
            )
            station_best_mean_key = min(
                MODEL_LABELS,
                key=lambda key: _mean_support_metric(station_summary, key, "heldout_visibility_rmse")
                if key in station_summary["support_fractions"]["80"]["models"]
                else float("inf"),
            )
            row.update(
                {
                    "mean_emc_station_dropout": float(np.nanmean(emc_station_values)),
                    "std_emc_station_dropout": float(np.nanstd(emc_station_values)),
                    "emc_family_gap": float(np.nanmean(emc_station_values) - np.nanmean(emc_baseline_values)),
                    "best_station_dropout_model": station_best_mean_key,
                    "best_station_dropout_model_label": MODEL_LABELS[station_best_mean_key],
                    "best_station_dropout_mean": _mean_support_metric(
                        station_summary,
                        station_best_mean_key,
                        "heldout_visibility_rmse",
                    ),
                }
            )
        rows.append(row)
    return rows


def write_release_robustness_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    markdown_rows = [
        [
            str(row["track_label"]),
            str(row["sample_count"]),
            str(row["days_present"]),
            str(row["bands_present"]),
            _fmt(float(row["mean_emc_baseline_track"])),
            _fmt(float(row["std_emc_baseline_track"])),
            _fmt(float(row.get("mean_emc_station_dropout", float("nan")))),
            _fmt(float(row.get("emc_family_gap", float("nan")))),
            _fmt(float(row["mean_ehtim_baseline_track"])),
            str(row["best_baseline_track_model_label"]),
            str(row.get("best_station_dropout_model_label", "n/a")),
        ]
        for row in rows
    ]
    write_markdown_table(
        path,
        headers=[
            "Track",
            "Samples",
            "Days",
            "Bands",
            "Mean EMC (baseline-track)",
            "Std EMC (baseline-track)",
            "Mean EMC (station-dropout)",
            "EMC family gap",
            "Mean eht-imaging",
            "Best baseline-track model",
            "Best station-dropout model",
        ],
        rows=markdown_rows,
        title="Public EHT Release Robustness Summary",
        notes=[
            "The EMC family gap is station-dropout mean minus baseline-track mean, so positive values indicate worse EMC held-out recovery under station-structured missingness.",
            "This table is observation-domain only and is intended to summarize public-release robustness rather than to support morphology claims.",
        ],
    )


def _collect_paired_values(
    *,
    per_sample_paths: list[Path],
    candidate_key: str,
    reference_key: str,
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_values: list[float] = []
    reference_values: list[float] = []
    for csv_path in per_sample_paths:
        rows = load_csv_rows(csv_path)
        grouped: dict[tuple[str, str], dict[str, float]] = {}
        for row in rows:
            sample_id = str(row.get("sample_id", row.get("sample_index", "")))
            support_tag = str(row.get("support_fraction_tag", ""))
            metric_value = _safe_float(row.get(metric_key))
            if math.isnan(metric_value):
                continue
            grouped.setdefault((support_tag, sample_id), {})[str(row["model"])] = metric_value
        for model_map in grouped.values():
            if candidate_key not in model_map or reference_key not in model_map:
                continue
            candidate_values.append(model_map[candidate_key])
            reference_values.append(model_map[reference_key])
    return np.asarray(candidate_values, dtype=np.float64), np.asarray(reference_values, dtype=np.float64)


def build_public_stats_rows(
    *,
    baseline_track_paths: list[Path],
    station_dropout_paths: list[Path],
    synthetic_paths: list[Path],
) -> list[dict[str, Any]]:
    """Build paired-bootstrap robustness rows for the public suite and transfer-gap analysis."""

    specs = [
        ("Synthetic benchmark breadth", synthetic_paths, "emc", "ccrr", "heldout_visibility_rmse"),
        ("Synthetic benchmark breadth", synthetic_paths, "emc", "residual_refinement", "heldout_visibility_rmse"),
        ("Synthetic benchmark breadth", synthetic_paths, "emc", "baseline_learned", "heldout_visibility_rmse"),
        ("Public EHT baseline-track suite", baseline_track_paths, "emc", "ccrr", "heldout_visibility_rmse"),
        ("Public EHT baseline-track suite", baseline_track_paths, "emc", "residual_refinement", "heldout_visibility_rmse"),
        ("Public EHT baseline-track suite", baseline_track_paths, "emc", "baseline_learned", "heldout_visibility_rmse"),
        ("Public EHT baseline-track suite", baseline_track_paths, "emc", "ehtim_bridge", "heldout_visibility_rmse"),
        ("Public EHT baseline-track suite", baseline_track_paths, "emc", "tikhonov", "heldout_visibility_rmse"),
        ("Public EHT baseline-track suite", baseline_track_paths, "emc", "tikhonov", "heldout_reduced_chi2"),
        ("Public EHT station-dropout suite", station_dropout_paths, "emc", "ccrr", "heldout_visibility_rmse"),
        ("Public EHT station-dropout suite", station_dropout_paths, "emc", "baseline_learned", "heldout_visibility_rmse"),
        ("Public EHT station-dropout suite", station_dropout_paths, "emc", "ehtim_bridge", "heldout_visibility_rmse"),
        ("Public EHT station-dropout suite", station_dropout_paths, "emc", "tikhonov", "heldout_visibility_rmse"),
    ]
    rows: list[dict[str, Any]] = []
    for cohort, paths, candidate_key, reference_key, metric_key in specs:
        candidate, reference = _collect_paired_values(
            per_sample_paths=paths,
            candidate_key=candidate_key,
            reference_key=reference_key,
            metric_key=metric_key,
        )
        if candidate.size == 0:
            continue
        stats = paired_bootstrap_stats(candidate, reference, direction=METRIC_DIRECTIONS[metric_key])
        rows.append(
            {
                "cohort": cohort,
                "comparison": f"{MODEL_LABELS[candidate_key]} vs {MODEL_LABELS[reference_key]}",
                "metric": metric_key,
                "n_pairs": int(candidate.size),
                "candidate_mean": float(np.mean(candidate)),
                "reference_mean": float(np.mean(reference)),
                **stats,
            }
        )
    return rows


def write_public_stats_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    markdown_rows = [
        [
            str(row["cohort"]),
            str(row["comparison"]),
            str(row["metric"]),
            str(row["n_pairs"]),
            _fmt(float(row["candidate_mean"])),
            _fmt(float(row["reference_mean"])),
            _fmt(float(row["mean_delta"])),
            f"[{_fmt(float(row['ci_low']))}, {_fmt(float(row['ci_high']))}]",
            _fmt(float(row["win_rate"])),
            _fmt(float(row["p_value"])),
        ]
        for row in rows
    ]
    write_markdown_table(
        path,
        headers=[
            "Cohort",
            "Comparison",
            "Metric",
            "Pairs",
            "Candidate mean",
            "Reference mean",
            "Mean delta",
            "95% CI",
            "Win rate",
            "p-value",
        ],
        rows=markdown_rows,
        title="Synthetic-to-Public Robustness Summary",
        notes=[
            "Mean deltas are direction-aware: positive means the candidate is better under the metric direction.",
            "The public rows pool all sample-support cases within the named public-EHT cohort.",
        ],
    )


def build_day_band_rows(per_sample_paths: list[Path]) -> list[dict[str, Any]]:
    """Summarize public real-data behaviour by day and band from the same per-sample outputs."""

    rows: list[dict[str, Any]] = []
    for csv_path in per_sample_paths:
        grouped: dict[tuple[str, str, str, str], dict[str, list[float]]] = {}
        for row in load_csv_rows(csv_path):
            key = (
                str(row["target"]),
                str(row.get("campaign_year", "")),
                str(row.get("support_fraction_tag", "")),
                str(row.get("day_of_year", "")),
                str(row.get("band", "")),
            )
            grouped.setdefault(key, {}).setdefault(str(row["model"]), []).append(
                _safe_float(row["heldout_visibility_rmse"])
            )
        for (target, campaign_year, support_tag, day, band), model_values in grouped.items():
            if "emc" not in model_values:
                continue
            best_model = min(
                model_values,
                key=lambda key: float(np.nanmean(np.asarray(model_values[key], dtype=np.float64))),
            )
            rows.append(
                {
                    "target": target,
                    "campaign_year": campaign_year,
                    "support_fraction_tag": support_tag,
                    "day_of_year": day,
                    "band": band,
                    "emc_mean_heldout_visibility_rmse": float(np.nanmean(model_values["emc"])),
                    "best_model": best_model,
                    "best_model_label": MODEL_LABELS.get(best_model, best_model),
                    "best_mean_heldout_visibility_rmse": float(
                        np.nanmean(np.asarray(model_values[best_model], dtype=np.float64))
                    ),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["target"],
            row["campaign_year"],
            int(row["support_fraction_tag"]),
            row["day_of_year"],
            row["band"],
        ),
    )


def write_day_band_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    markdown_rows = [
        [
            f"{row['target']} {row['campaign_year']}",
            str(row["support_fraction_tag"]),
            str(row["day_of_year"]),
            str(row["band"]),
            _fmt(float(row["emc_mean_heldout_visibility_rmse"])),
            str(row["best_model_label"]),
            _fmt(float(row["best_mean_heldout_visibility_rmse"])),
        ]
        for row in rows
    ]
    write_markdown_table(
        path,
        headers=["Track", "Support (%)", "Day", "Band", "Mean EMC", "Best model", "Best mean VisRMSE"],
        rows=markdown_rows,
        title="Public EHT Day/Band Stratified Summary",
        notes=[
            "This table reuses the same public benchmark outputs and is intended to expose within-track heterogeneity rather than to create a second ranking.",
        ],
    )


def save_public_qualitative_figure(
    *,
    track_specs: list[PublicTrackSpec],
    output_png: str | Path,
    output_svg: str | Path,
    selection_manifest: str | Path,
    support_fraction_tag: str = "60",
) -> dict[str, Any]:
    """Save one observation-domain public-EHT qualitative panel."""

    selection = select_public_qualitative_example(track_specs, support_fraction_tag=support_fraction_tag)
    bundle = load_predictions(selection["prediction_path"])
    sample_index = int(selection["sample_index"])
    frame_index = int(selection["frame_index"])

    panel_specs = [
        ("dirty", "Support-only dirty"),
        ("baseline_learned", "Baseline 3D U-Net"),
        ("residual_refinement", "Residual refinement"),
        ("emc", "EMC"),
        ("ehtim_bridge", "eht-imaging bridge"),
        ("tikhonov", "Tikhonov"),
    ]
    display_images = [
        np.asarray(bundle[key][sample_index, frame_index], dtype=np.float32)
        for key, _ in panel_specs
    ]
    stacked = np.stack(display_images, axis=0)
    vmin = float(np.min(stacked))
    vmax = float(np.max(stacked))
    if math.isclose(vmin, vmax, rel_tol=0.0, abs_tol=1e-12):
        vmax = vmin + 1e-6

    support_mask = np.asarray(bundle["support_mask"][sample_index, frame_index], dtype=np.float32)
    target_mask = np.asarray(bundle["target_mask"][sample_index, frame_index], dtype=np.float32)
    mask_rgb = np.ones((*support_mask.shape, 3), dtype=np.float32)
    mask_rgb *= np.asarray([0.96, 0.96, 0.96], dtype=np.float32)
    mask_rgb[support_mask > 0.0] = np.asarray([0.29, 0.56, 0.89], dtype=np.float32)
    mask_rgb[target_mask > 0.0] = np.asarray([0.93, 0.46, 0.16], dtype=np.float32)

    fig, axes = plt.subplots(1, len(panel_specs) + 1, figsize=(16.0, 3.6))
    fig.patch.set_facecolor("white")
    for axis, image, (_, label) in zip(axes[:-1], display_images, panel_specs):
        axis.imshow(image, cmap=IMAGE_CMAP, interpolation="nearest", vmin=vmin, vmax=vmax)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(label, fontsize=11, pad=6)
        highlight = label == "EMC"
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.8 if highlight else 0.8)
            spine.set_edgecolor("#dc2626" if highlight else "#d1d5db")

    mask_axis = axes[-1]
    mask_axis.imshow(mask_rgb, interpolation="nearest")
    mask_axis.set_xticks([])
    mask_axis.set_yticks([])
    mask_axis.set_title("Support / held-out mask", fontsize=11, pad=6)
    for spine in mask_axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#d1d5db")

    fig.suptitle(
        f"{selection['track_label']}, support {support_fraction_tag}%: representative public-EHT qualitative example "
        f"(sample {sample_index}, frame {frame_index})",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(Path(output_svg), bbox_inches="tight")
    plt.close(fig)

    payload = {
        **selection,
        "figure_layout": [label for _, label in panel_specs] + ["Support / held-out mask"],
    }
    Path(selection_manifest).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def save_public_suite_figure(
    *,
    summaries: list[dict[str, Any]],
    output_png: str | Path,
    output_svg: str | Path,
) -> None:
    """Save the multi-panel public-EHT support-fraction figure."""

    panel_count = len(summaries)
    cols = 2
    rows = int(math.ceil(panel_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12.4, 4.6 * rows), sharex=False, sharey=False)
    axes_array = np.asarray(axes).reshape(-1)
    fig.patch.set_facecolor("white")
    x_values = [80, 60, 40, 20]

    for axis, summary in zip(axes_array, summaries):
        for model_key in PLOT_MODEL_ORDER:
            heldout_values = [_model_metrics(summary, str(tag), model_key)["heldout_visibility_rmse"] for tag in x_values]
            axis.plot(
                x_values,
                heldout_values,
                marker="o",
                linewidth=2.0,
                color=MODEL_COLORS[model_key],
                label=MODEL_LABELS[model_key],
            )
        axis.set_title(_track_label(summary), fontsize=12)
        axis.set_xlabel("Support fraction (%)", fontsize=10)
        axis.set_ylabel("Held-out visibility RMSE", fontsize=10)
        axis.set_xticks(x_values)
        axis.grid(alpha=0.22)

    for axis in axes_array[panel_count:]:
        axis.axis("off")

    legend_handles = [
        Line2D([0], [0], color=MODEL_COLORS[key], lw=2.0, marker="o", label=MODEL_LABELS[key])
        for key in PLOT_MODEL_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle("Public EHT held-out measurement recovery across releases and targets", fontsize=14, y=1.02)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(Path(output_svg), bbox_inches="tight")
    plt.close(fig)


def save_transfer_gap_figure(
    *,
    stats_rows: list[dict[str, Any]],
    output_png: str | Path,
    output_svg: str | Path,
) -> None:
    """Save a compact synthetic-versus-public transfer-gap figure."""

    comparisons = [
        "EMC vs Baseline 3D U-Net",
        "EMC vs Residual Refinement",
        "EMC vs CCRR",
    ]
    synthetic = {
        row["comparison"]: row
        for row in stats_rows
        if row["cohort"] == "Synthetic benchmark breadth" and row["metric"] == "heldout_visibility_rmse"
    }
    public = {
        row["comparison"]: row
        for row in stats_rows
        if row["cohort"] == "Public EHT baseline-track suite" and row["metric"] == "heldout_visibility_rmse"
    }

    x = np.arange(len(comparisons), dtype=np.float64)
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    fig.patch.set_facecolor("white")
    synthetic_values = [synthetic.get(comp, {}).get("mean_delta", float("nan")) for comp in comparisons]
    public_values = [public.get(comp, {}).get("mean_delta", float("nan")) for comp in comparisons]
    ax.bar(x - width / 2.0, synthetic_values, width=width, color="#dc2626", label="Synthetic breadth")
    ax.bar(x + width / 2.0, public_values, width=width, color="#2563eb", label="Public EHT baseline-track")
    ax.axhline(0.0, color="#111827", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["vs Baseline", "vs Residual", "vs CCRR"])
    ax.set_ylabel("Direction-aware mean delta\n(positive means EMC is better)", fontsize=10)
    ax.set_title("Synthetic-to-public transfer gap on held-out visibility RMSE", fontsize=13)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(Path(output_svg), bbox_inches="tight")
    plt.close(fig)


def build_claim_to_evidence_rows(
    *,
    public_matrix_path: Path,
    sensitivity_path: Path,
    stats_path: Path,
    day_band_path: Path,
) -> list[dict[str, str]]:
    return [
        {
            "claim": "The public-EHT benchmark question remains meaningful across multiple official releases and targets, even when image-domain ground truth is unavailable.",
            "evidence": str(public_matrix_path.resolve()),
        },
        {
            "claim": "Synthetic winners do not transfer directly to public EHT measurement products, and the transfer gap is strongest on the main held-out visibility metric.",
            "evidence": str(stats_path.resolve()),
        },
        {
            "claim": "The earned-consistency conclusions are not tied to one structured missingness family, although robustness on public data is partial rather than universal.",
            "evidence": str(sensitivity_path.resolve()),
        },
        {
            "claim": "Day/band stratification reveals substantial within-track heterogeneity, which is one reason public observation-domain benchmarking remains necessary.",
            "evidence": str(day_band_path.resolve()),
        },
    ]
