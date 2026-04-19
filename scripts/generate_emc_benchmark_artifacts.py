#!/usr/bin/env python3
"""Generate benchmark-first EMC paper artifacts from protocol outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import csv

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.evaluation.emc_artifacts import (  # noqa: E402
    save_emc_qualitative_figure,
    save_emc_secondary_qualitative_figure,
)
from dynadiff_vlbi.evaluation.emc_benchmark_artifacts import (  # noqa: E402
    BenchmarkProtocolSpec,
    benchmark_long_rows,
    build_family_matrix_rows,
    build_realism_rows,
    save_benchmark_manifest,
    save_family_support_figure,
    save_realism_support_figure,
    write_csv,
    write_family_matrix_table,
    write_leaderboard_template,
    write_realism_table,
)
from dynadiff_vlbi.evaluation.ccrr_artifacts import paired_bootstrap_stats  # noqa: E402
from dynadiff_vlbi.evaluation.paper_artifacts import load_json, relativize_payload_paths, repo_relative_path, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default="outputs/emc_benchmark_artifacts")
    parser.add_argument("--bootstrap-root", default="benchmark/bootstrap_64")
    parser.add_argument("--paper-root", default="paper")
    parser.add_argument("--release-root", default="outputs/emc_benchmark_release")
    parser.add_argument("--uq-root", default="outputs/emc_conformal_uq")
    parser.add_argument("--dps-artifact-root", default="outputs/dps_benchmark_artifacts")
    return parser.parse_args()


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_optional_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return _load_csv_rows(path)


def _merge_synthetic_uq_rows(
    rows: list[dict[str, object]],
    uq_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    uq_index = {
        (str(row["family"]), str(row["support_fraction_tag"])): row
        for row in uq_rows
    }
    for row in rows:
        uq_row = uq_index.get((str(row["condition"]), str(row["support_fraction_tag"])))
        if uq_row is None:
            continue
        row["emc_coverage_90"] = float(uq_row["emc_coverage_90"])
        row["emc_miw"] = float(uq_row["emc_miw"])
    return rows


def _merge_synthetic_dps_rows(
    rows: list[dict[str, object]],
    dps_summary_path: Path,
) -> list[dict[str, object]]:
    if not dps_summary_path.exists():
        return rows
    dps_summary = load_json(dps_summary_path)
    support_payloads = dps_summary.get("support_fractions", {})
    for row in rows:
        if str(row["condition"]) != "baseline_tracks":
            continue
        support_tag = str(row["support_fraction_tag"])
        models = support_payloads.get(support_tag, {}).get("models", {})
        dps_metrics = models.get("dps")
        if not dps_metrics:
            continue
        row["dps_heldout_visibility_rmse"] = float(dps_metrics["heldout_visibility_rmse"])
    return rows


def _write_table01_tex(path: Path, rows: list[dict[str, object]]) -> None:
    ordered_rows = sorted(rows, key=lambda row: (str(row["family"]), int(str(row["support_fraction_tag"]))))
    lines: list[str] = []
    for row in ordered_rows:
        dps_value = float(row.get("dps_heldout_visibility_rmse", float("nan")))
        coverage_value = float(row.get("emc_coverage_90", float("nan")))
        miw_value = float(row.get("emc_miw", float("nan")))
        dps_text = "n/a" if np.isnan(dps_value) else f"{dps_value:.6f}"
        coverage_text = "n/a" if np.isnan(coverage_value) else f"{coverage_value:.3f}"
        miw_text = "n/a" if np.isnan(miw_value) else f"{miw_value:.6f}"
        lines.append(
            f"{row['family']} & {row['support_fraction_tag']} & "
            f"{float(row['emc_heldout_visibility_rmse']):.6f} & "
            f"{float(row['baseline_heldout_visibility_rmse']):.6f} & "
            f"{float(row['residual_heldout_visibility_rmse']):.6f} & "
            f"{float(row['ccrr_heldout_visibility_rmse']):.6f} & "
            f"{dps_text} & {coverage_text} & {miw_text} \\\\"
        )
    body = "\n".join(lines)
    tex = (
        "\\begin{table*}\n"
        "\\centering\n"
        "\\caption{Default64 synthetic benchmark matrix across the three structured holdout families and four support fractions. "
        "Lower held-out visibility RMSE is better. DPS is rerun only on the central baseline-track family in this add-on cycle, while EMC conformal UQ is reported as 90 per cent empirical coverage and mean interval width (MIW).}\n"
        "\\label{tab:benchmark-matrix}\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{llccccccc}\n"
        "\\toprule\n"
        "Holdout family & Support (\\%) & EMC & Baseline & Residual & CCRR & DPS & EMC 90\\% cov. & EMC MIW \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )
    path.write_text(tex, encoding="utf-8")


def _collect_paired_values(
    *,
    per_sample_paths: list[Path],
    candidate_key: str,
    reference_key: str,
    metric_key: str,
) -> tuple[list[float], list[float]]:
    candidate_values: list[float] = []
    reference_values: list[float] = []
    for csv_path in per_sample_paths:
        grouped: dict[tuple[str, str], dict[str, float]] = {}
        for row in _load_csv_rows(csv_path):
            sample_id = str(row.get("sample_id", row.get("sample_index", "")))
            support_tag = str(row.get("support_fraction_tag", ""))
            grouped.setdefault((support_tag, sample_id), {})[str(row["model"])] = float(row[metric_key])
        for model_map in grouped.values():
            if candidate_key not in model_map or reference_key not in model_map:
                continue
            candidate_values.append(model_map[candidate_key])
            reference_values.append(model_map[reference_key])
    return candidate_values, reference_values


def _benchmark_specs(output_root: Path) -> tuple[list[BenchmarkProtocolSpec], BenchmarkProtocolSpec]:
    family_specs = [
        BenchmarkProtocolSpec(
            key="baseline_tracks",
            title="Default64 baseline-track holdout",
            family_label="Baseline-track blocks",
            family_description="Deterministic contiguous baseline-track blocks across time.",
            protocol_dir=(output_root / "emc_benchmark_baseline_tracks_protocol").resolve(),
            condition_group="benchmark_family",
        ),
        BenchmarkProtocolSpec(
            key="scan_segments",
            title="Default64 scan-segment holdout",
            family_label="Scan-segment blocks",
            family_description="Deterministic contiguous scan-like temporal windows are withheld.",
            protocol_dir=(output_root / "emc_benchmark_scan_segments_protocol").resolve(),
            condition_group="benchmark_family",
        ),
        BenchmarkProtocolSpec(
            key="station_dropout",
            title="Default64 station-dropout holdout",
            family_label="Station dropout",
            family_description="Deterministic subsets of stations and all incident baselines are withheld together.",
            protocol_dir=(output_root / "emc_benchmark_station_dropout_protocol").resolve(),
            condition_group="benchmark_family",
        ),
    ]
    realism_spec = BenchmarkProtocolSpec(
        key="challenge_inspired_realism",
        title="Challenge-inspired realism track",
        family_label="Challenge-inspired realism",
        family_description=(
            "Public-style station-track sampling with scan gaps, gain corruption, and baseline-dependent noise."
        ),
        protocol_dir=(output_root / "emc_benchmark_challenge_inspired_realism_protocol").resolve(),
        condition_group="challenge_inspired_realism",
    )
    return family_specs, realism_spec


def main() -> int:
    args = parse_args()
    artifact_root = (ROOT / args.artifact_root).resolve()
    bootstrap_root = (ROOT / args.bootstrap_root).resolve()
    paper_root = (ROOT / args.paper_root).resolve()
    release_root = (ROOT / args.release_root).resolve()
    uq_root = (ROOT / args.uq_root).resolve()
    dps_artifact_root = (ROOT / args.dps_artifact_root).resolve()
    output_root = ROOT / "outputs"
    tables_dir = artifact_root / "tables"
    bootstrap_tables_dir = bootstrap_root / "synthetic"
    summaries_dir = artifact_root / "summaries"
    figures_dir = paper_root / "figures"
    paper_tables_dir = paper_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_tables_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    dps_artifact_root.mkdir(parents=True, exist_ok=True)

    family_specs, realism_spec = _benchmark_specs(output_root)
    all_specs = [*family_specs, realism_spec]

    long_rows = benchmark_long_rows(all_specs)
    write_csv(tables_dir / "emc_benchmark_long.csv", long_rows)

    family_matrix_rows = build_family_matrix_rows(family_specs)
    family_matrix_rows = _merge_synthetic_dps_rows(
        family_matrix_rows,
        output_root / "emc_benchmark_baseline_tracks_default64_protocol" / "logs" / "emc_protocol_summary.json",
    )
    family_matrix_rows = _merge_synthetic_uq_rows(
        family_matrix_rows,
        _load_optional_csv(uq_root / "tables" / "synthetic_emc_conformal_uq.csv"),
    )
    write_csv(tables_dir / "emc_benchmark_matrix.csv", family_matrix_rows)
    save_json(tables_dir / "emc_benchmark_matrix.json", {"rows": family_matrix_rows})
    write_family_matrix_table(tables_dir / "emc_benchmark_matrix.md", family_matrix_rows)
    write_csv(paper_tables_dir / "table01_default64_benchmark_matrix.csv", family_matrix_rows)
    _write_table01_tex(paper_tables_dir / "table01_default64_benchmark_matrix.tex", family_matrix_rows)
    write_csv(dps_artifact_root / "synthetic_dps_table.csv", family_matrix_rows)

    synthetic_bootstrap_rows = []
    pooled_family_paths = [
        spec.per_sample_path(support_tag)
        for spec in family_specs
        for support_tag in ("80", "60", "40", "20")
    ]
    for candidate_key, reference_key, comparison in [
        ("emc", "ccrr", "EMC vs CCRR"),
        ("emc", "residual_refinement", "EMC vs Residual Refinement"),
        ("emc", "baseline_learned", "EMC vs Baseline 3D U-Net"),
    ]:
        candidate_values, reference_values = _collect_paired_values(
            per_sample_paths=pooled_family_paths,
            candidate_key=candidate_key,
            reference_key=reference_key,
            metric_key="heldout_visibility_rmse",
        )
        candidate_array = np.asarray(candidate_values, dtype=np.float64)
        reference_array = np.asarray(reference_values, dtype=np.float64)
        stats = paired_bootstrap_stats(candidate_array, reference_array, direction="lower")
        synthetic_bootstrap_rows.append(
            {
                "cohort": "Synthetic benchmark breadth",
                "comparison": comparison,
                "metric": "heldout_visibility_rmse",
                "n_pairs": int(candidate_array.size),
                "candidate_mean": float(np.mean(candidate_array)),
                "reference_mean": float(np.mean(reference_array)),
                **stats,
            }
        )
    write_csv(tables_dir / "emc_benchmark_bootstrap.csv", synthetic_bootstrap_rows)
    save_json(tables_dir / "emc_benchmark_bootstrap.json", {"rows": synthetic_bootstrap_rows})
    from dynadiff_vlbi.evaluation.public_eht_suite_artifacts import write_public_stats_table  # noqa: E402

    write_public_stats_table(tables_dir / "emc_benchmark_bootstrap.md", synthetic_bootstrap_rows)
    write_csv(bootstrap_tables_dir / "emc_benchmark_bootstrap.csv", synthetic_bootstrap_rows)
    save_json(bootstrap_tables_dir / "emc_benchmark_bootstrap.json", {"rows": synthetic_bootstrap_rows})
    write_public_stats_table(bootstrap_tables_dir / "emc_benchmark_bootstrap.md", synthetic_bootstrap_rows)

    realism_rows = build_realism_rows(realism_spec)
    write_csv(tables_dir / "emc_challenge_inspired_realism.csv", realism_rows)
    save_json(tables_dir / "emc_challenge_inspired_realism.json", {"rows": realism_rows})
    write_realism_table(tables_dir / "emc_challenge_inspired_realism.md", realism_rows)

    write_leaderboard_template(artifact_root / "leaderboard_template.csv")

    save_family_support_figure(
        specs=family_specs,
        output_png=figures_dir / "fig02_emc_benchmark_support_curve.png",
        output_svg=figures_dir / "fig02_emc_benchmark_support_curve.svg",
    )
    save_realism_support_figure(
        spec=realism_spec,
        output_png=figures_dir / "fig03_emc_challenge_inspired_realism.png",
        output_svg=figures_dir / "fig03_emc_challenge_inspired_realism.svg",
    )
    representative_selection = save_emc_qualitative_figure(
        prediction_path=family_specs[0].prediction_path("40"),
        per_sample_csv=family_specs[0].per_sample_path("40"),
        output_png=figures_dir / "fig04_emc_benchmark_representative.png",
        output_svg=figures_dir / "fig04_emc_benchmark_representative.svg",
        selection_manifest=figures_dir / "fig04_emc_benchmark_representative.selection.json",
        support_fraction_tag="40",
        condition_title="Benchmark default64 baseline-track family",
        selection_mode="representative",
    )
    synthetic_secondary_selection = save_emc_secondary_qualitative_figure(
        prediction_path=family_specs[0].prediction_path("40"),
        per_sample_csv=family_specs[0].per_sample_path("40"),
        output_png=figures_dir / "fig08_emc_synthetic_qualitative.png",
        output_svg=figures_dir / "fig08_emc_synthetic_qualitative.svg",
        selection_manifest=figures_dir / "fig08_emc_synthetic_qualitative.selection.json",
        support_fraction_tag="40",
        condition_title="Benchmark default64 baseline-track family",
    )
    hard_selection = save_emc_qualitative_figure(
        prediction_path=realism_spec.prediction_path("20"),
        per_sample_csv=realism_spec.per_sample_path("20"),
        output_png=figures_dir / "fig05_emc_realism_hard_example.png",
        output_svg=figures_dir / "fig05_emc_realism_hard_example.svg",
        selection_manifest=figures_dir / "fig05_emc_realism_hard_example.selection.json",
        support_fraction_tag="20",
        condition_title="Challenge-inspired realism",
        selection_mode="hard",
    )

    benchmark_manifest = {
        "release_root": str(release_root),
        "artifact_root": str(artifact_root),
        "bootstrap_root": str(bootstrap_root),
        "paper_root": str(paper_root),
        "protocols": {
            spec.key: {
                "title": spec.title,
                "family_label": spec.family_label,
                "family_description": spec.family_description,
                "summary_path": repo_relative_path(spec.summary_path),
            }
            for spec in all_specs
        },
        "tables": {
            "benchmark_matrix": repo_relative_path(tables_dir / "emc_benchmark_matrix.md"),
            "bootstrap": repo_relative_path(tables_dir / "emc_benchmark_bootstrap.md"),
            "realism_track": repo_relative_path(tables_dir / "emc_challenge_inspired_realism.md"),
            "leaderboard_template": repo_relative_path(artifact_root / "leaderboard_template.csv"),
            "paper_table01_csv": repo_relative_path(paper_tables_dir / "table01_default64_benchmark_matrix.csv"),
            "paper_table01_tex": repo_relative_path(paper_tables_dir / "table01_default64_benchmark_matrix.tex"),
        },
        "figures": {
            "benchmark_support_curve": repo_relative_path(figures_dir / "fig02_emc_benchmark_support_curve.png"),
            "challenge_inspired_realism": repo_relative_path(figures_dir / "fig03_emc_challenge_inspired_realism.png"),
            "representative_example": representative_selection,
            "synthetic_secondary_example": synthetic_secondary_selection,
            "hard_realism_example": hard_selection,
        },
    }
    benchmark_manifest = relativize_payload_paths(benchmark_manifest)
    save_benchmark_manifest(summaries_dir / "emc_benchmark_artifact_manifest.json", benchmark_manifest)
    visual_asset_manifest_path = paper_root / "visual_asset_manifest.json"
    if visual_asset_manifest_path.exists():
        visual_asset_manifest = load_json(visual_asset_manifest_path)
    else:
        visual_asset_manifest = {}
    visual_asset_manifest["emc_benchmark_figures"] = {
        "fig02_emc_benchmark_support_curve": {
            "sources": [repo_relative_path(spec.summary_path) for spec in family_specs],
        },
        "fig03_emc_challenge_inspired_realism": {
            "source": repo_relative_path(realism_spec.summary_path),
        },
        "fig04_emc_benchmark_representative": representative_selection,
        "fig08_emc_synthetic_qualitative": synthetic_secondary_selection,
        "fig05_emc_realism_hard_example": hard_selection,
    }
    visual_asset_manifest = relativize_payload_paths(visual_asset_manifest)
    save_json(visual_asset_manifest_path, visual_asset_manifest)

    print(
        {
            "artifact_root": str(artifact_root),
            "paper_root": str(paper_root),
            "release_root": str(release_root),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
