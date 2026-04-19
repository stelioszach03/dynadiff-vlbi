#!/usr/bin/env python3
"""Generate paper-facing artifacts for the multi-release public-EHT validation suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.evaluation.paper_artifacts import relativize_payload_paths, repo_relative_path, save_json, write_csv  # noqa: E402
from dynadiff_vlbi.evaluation.public_eht_suite_artifacts import (  # noqa: E402
    PublicTrackSpec,
    build_claim_to_evidence_rows,
    build_day_band_rows,
    build_public_matrix_rows,
    build_release_gap_rows,
    build_release_robustness_rows,
    build_public_stats_rows,
    build_sensitivity_rows,
    load_json,
    save_public_qualitative_figure,
    save_public_suite_figure,
    save_transfer_gap_figure,
    write_day_band_table,
    write_public_matrix_table,
    write_release_gap_table,
    write_release_robustness_table,
    write_public_stats_table,
    write_sensitivity_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", default="outputs/public_eht_suite")
    parser.add_argument("--artifact-root", default="outputs/public_eht_suite_artifacts")
    parser.add_argument("--bootstrap-root", default="benchmark/bootstrap_64")
    parser.add_argument("--paper-root", default="paper")
    parser.add_argument("--uq-root", default="outputs/emc_conformal_uq")
    parser.add_argument("--dps-artifact-root", default="outputs/dps_benchmark_artifacts")
    return parser.parse_args()


def _synthetic_per_sample_paths(output_root: Path) -> list[Path]:
    protocol_dirs = [
        output_root / "emc_benchmark_baseline_tracks_protocol",
        output_root / "emc_benchmark_scan_segments_protocol",
        output_root / "emc_benchmark_station_dropout_protocol",
    ]
    return [
        protocol_dir / "logs" / f"per_sample_support_{support_tag}.csv"
        for protocol_dir in protocol_dirs
        for support_tag in ("80", "60", "40", "20")
    ]


def _track_specs_from_manifest(suite_manifest: dict[str, object], family: str) -> list[PublicTrackSpec]:
    runs = [
        run
        for run in suite_manifest["runs"]  # type: ignore[index]
        if isinstance(run, dict) and run.get("family") == family
    ]
    specs: list[PublicTrackSpec] = []
    for run in runs:
        output_dir = Path(str(run["output_dir"]))
        per_sample_paths = [output_dir / "logs" / f"per_sample_support_{tag}.csv" for tag in ("80", "60", "40", "20")]
        specs.append(
            PublicTrackSpec(
                family=family,
                output_dir=output_dir,
                summary_path=Path(str(run["summary_path"])),
                per_sample_paths=per_sample_paths,
            )
        )
    return specs


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _merge_public_uq_rows(
    public_matrix_rows: list[dict[str, object]],
    robustness_rows: list[dict[str, object]],
    uq_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_track_support: dict[tuple[str, str], dict[str, float]] = {}
    per_track_model: dict[tuple[str, str], list[float]] = {}
    for row in uq_rows:
        track_key = (str(row["track_label"]), str(row["support_fraction_tag"]))
        per_track_support.setdefault(track_key, {})
        per_track_support[track_key][str(row["model"])] = float(row["mean_interval_width"])
        per_track_model.setdefault((str(row["track_label"]), str(row["model"])), []).append(
            float(row["mean_interval_width"])
        )
    for row in public_matrix_rows:
        support_key = (str(row["track_label"]), str(row["support_fraction_tag"]))
        support_payload = per_track_support.get(support_key, {})
        row["emc_miw"] = support_payload.get("emc", float("nan"))
        row["emc_tto_miw"] = support_payload.get("emc_tto", float("nan"))
    for row in robustness_rows:
        track_label = str(row["track_label"])
        emc_values = per_track_model.get((track_label, "emc"), [])
        emc_tto_values = per_track_model.get((track_label, "emc_tto"), [])
        row["mean_emc_miw"] = float(sum(emc_values) / len(emc_values)) if emc_values else float("nan")
        row["mean_emc_tto_miw"] = float(sum(emc_tto_values) / len(emc_tto_values)) if emc_tto_values else float("nan")
    return public_matrix_rows, robustness_rows


def _write_table02_tex(path: Path, rows: list[dict[str, object]]) -> None:
    ordered_rows = sorted(rows, key=lambda row: str(row["track_label"]))
    lines: list[str] = []
    for row in ordered_rows:
        dps_text = "n/a" if np.isnan(float(row.get("mean_dps_baseline_track", float("nan")))) else f"{float(row['mean_dps_baseline_track']):.6f}"
        emc_miw_text = "n/a" if np.isnan(float(row.get("mean_emc_miw", float("nan")))) else f"{float(row['mean_emc_miw']):.6f}"
        emc_tto_miw_text = "n/a" if np.isnan(float(row.get("mean_emc_tto_miw", float("nan")))) else f"{float(row['mean_emc_tto_miw']):.6f}"
        lines.append(
            f"{row['track_label']} & {int(row['sample_count'])} & "
            f"{float(row['mean_emc_baseline_track']):.6f} & "
            f"{float(row['mean_emc_tto_baseline_track']):.6f} & "
            f"{dps_text} & "
            f"{float(row['emc_tto_uplift_baseline_track']):+.6f} & "
            f"{emc_miw_text} & {emc_tto_miw_text} & "
            f"{row['best_baseline_track_model_label']} & "
            f"{float(row['best_baseline_track_mean']):.6f} \\\\"
        )
    tex = (
        "\\begin{table*}\n"
        "\\centering\n"
        "\\caption{Release-level public-EHT baseline-track summary with TTO, DPS, and conformal MIW. Positive TTO uplift means lower held-out visibility RMSE than plain EMC. Public UQ reports MIW only because image-domain ground truth is unavailable.}\n"
        "\\label{tab:public-release-means}\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lccccccccc}\n"
        "\\toprule\n"
        "Track & Samples & EMC mean & EMC-TTO mean & DPS mean & TTO uplift & EMC MIW & EMC-TTO MIW & Best baseline-track model & Best mean \\\\\n"
        "\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )
    path.write_text(tex, encoding="utf-8")


def main() -> int:
    args = parse_args()
    suite_root = (ROOT / args.suite_root).resolve()
    artifact_root = (ROOT / args.artifact_root).resolve()
    bootstrap_root = (ROOT / args.bootstrap_root).resolve()
    paper_root = (ROOT / args.paper_root).resolve()
    uq_root = (ROOT / args.uq_root).resolve()
    dps_artifact_root = (ROOT / args.dps_artifact_root).resolve()
    tables_dir = artifact_root / "tables"
    bootstrap_tables_dir = bootstrap_root / "public_eht"
    summaries_dir = artifact_root / "summaries"
    figures_dir = paper_root / "figures"
    paper_tables_dir = paper_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_tables_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    dps_artifact_root.mkdir(parents=True, exist_ok=True)

    suite_manifest = json.loads((suite_root / "suite_manifest.json").read_text(encoding="utf-8"))
    baseline_specs = _track_specs_from_manifest(suite_manifest, "baseline_track_blocks")
    station_specs = _track_specs_from_manifest(suite_manifest, "station_dropout")

    baseline_summaries = [load_json(spec.summary_path) for spec in baseline_specs]
    station_summaries = [load_json(spec.summary_path) for spec in station_specs]

    public_matrix_rows = build_public_matrix_rows(baseline_summaries)
    sensitivity_rows = build_sensitivity_rows(
        baseline_summaries=baseline_summaries,
        station_summaries=station_summaries,
    )
    robustness_rows = build_release_robustness_rows(
        baseline_summaries=baseline_summaries,
        station_summaries=station_summaries,
    )
    public_matrix_rows, robustness_rows = _merge_public_uq_rows(
        public_matrix_rows,
        robustness_rows,
        _load_csv_rows(uq_root / "tables" / "public_emc_conformal_uq.csv"),
    )
    write_csv(tables_dir / "emc_public_eht_matrix.csv", public_matrix_rows)
    save_json(tables_dir / "emc_public_eht_matrix.json", {"rows": public_matrix_rows})
    write_public_matrix_table(tables_dir / "emc_public_eht_matrix.md", public_matrix_rows)
    write_csv(tables_dir / "emc_public_eht_sensitivity.csv", sensitivity_rows)
    save_json(tables_dir / "emc_public_eht_sensitivity.json", {"rows": sensitivity_rows})
    write_sensitivity_table(tables_dir / "emc_public_eht_sensitivity.md", sensitivity_rows)
    write_csv(tables_dir / "emc_public_eht_release_robustness.csv", robustness_rows)
    save_json(tables_dir / "emc_public_eht_release_robustness.json", {"rows": robustness_rows})
    write_release_robustness_table(tables_dir / "emc_public_eht_release_robustness.md", robustness_rows)
    write_csv(paper_tables_dir / "table02_public_eht_release_means.csv", robustness_rows)
    write_csv(paper_tables_dir / "table02_public_eht_release_robustness.csv", robustness_rows)
    _write_table02_tex(paper_tables_dir / "table02_public_eht_release_means.tex", robustness_rows)
    write_csv(dps_artifact_root / "public_dps_release_summary.csv", robustness_rows)

    stats_rows = build_public_stats_rows(
        baseline_track_paths=[path for spec in baseline_specs for path in spec.per_sample_paths],
        station_dropout_paths=[path for spec in station_specs for path in spec.per_sample_paths],
        synthetic_paths=_synthetic_per_sample_paths((ROOT / "outputs").resolve()),
    )
    write_csv(tables_dir / "emc_public_eht_stats.csv", stats_rows)
    save_json(tables_dir / "emc_public_eht_stats.json", {"rows": stats_rows})
    write_public_stats_table(tables_dir / "emc_public_eht_stats.md", stats_rows)
    write_csv(bootstrap_tables_dir / "emc_public_eht_stats.csv", stats_rows)
    save_json(bootstrap_tables_dir / "emc_public_eht_stats.json", {"rows": stats_rows})
    write_public_stats_table(bootstrap_tables_dir / "emc_public_eht_stats.md", stats_rows)

    day_band_rows = build_day_band_rows([path for spec in baseline_specs for path in spec.per_sample_paths])
    write_csv(tables_dir / "emc_public_eht_day_band.csv", day_band_rows)
    save_json(tables_dir / "emc_public_eht_day_band.json", {"rows": day_band_rows})
    write_day_band_table(tables_dir / "emc_public_eht_day_band.md", day_band_rows)

    release_gap_rows = build_release_gap_rows(baseline_specs)
    write_csv(tables_dir / "emc_public_eht_release_gaps.csv", release_gap_rows)
    save_json(tables_dir / "emc_public_eht_release_gaps.json", {"rows": release_gap_rows})
    write_release_gap_table(tables_dir / "emc_public_eht_release_gaps.md", release_gap_rows)
    write_csv(bootstrap_tables_dir / "emc_public_eht_release_gaps.csv", release_gap_rows)
    save_json(bootstrap_tables_dir / "emc_public_eht_release_gaps.json", {"rows": release_gap_rows})
    write_release_gap_table(bootstrap_tables_dir / "emc_public_eht_release_gaps.md", release_gap_rows)

    save_public_suite_figure(
        summaries=baseline_summaries,
        output_png=figures_dir / "fig06_emc_public_eht_suite.png",
        output_svg=figures_dir / "fig06_emc_public_eht_suite.svg",
    )
    save_transfer_gap_figure(
        stats_rows=stats_rows,
        release_gap_rows=release_gap_rows,
        output_png=figures_dir / "fig07_emc_public_transfer_gap.png",
        output_svg=figures_dir / "fig07_emc_public_transfer_gap.svg",
    )
    public_qualitative_selection = save_public_qualitative_figure(
        track_specs=baseline_specs,
        output_png=figures_dir / "fig09_public_eht_qualitative.png",
        output_svg=figures_dir / "fig09_public_eht_qualitative.svg",
        selection_manifest=figures_dir / "fig09_public_eht_qualitative.selection.json",
        support_fraction_tag="60",
    )

    claim_rows = build_claim_to_evidence_rows(
        public_matrix_path=tables_dir / "emc_public_eht_matrix.md",
        sensitivity_path=tables_dir / "emc_public_eht_sensitivity.md",
        stats_path=tables_dir / "emc_public_eht_stats.md",
        day_band_path=tables_dir / "emc_public_eht_day_band.md",
        release_gap_path=tables_dir / "emc_public_eht_release_gaps.md",
    )
    save_json(summaries_dir / "public_eht_claim_to_evidence.json", {"rows": claim_rows})
    (summaries_dir / "public_eht_claim_to_evidence.md").write_text(
        "# Public EHT Claim-to-Evidence Map\n\n"
        + "\n".join(f"- {row['claim']} -> `{row['evidence']}`" for row in claim_rows)
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "suite_root": repo_relative_path(suite_root),
        "artifact_root": repo_relative_path(artifact_root),
        "bootstrap_root": repo_relative_path(bootstrap_root),
        "paper_root": repo_relative_path(paper_root),
        "tables": {
            "public_matrix": repo_relative_path(tables_dir / "emc_public_eht_matrix.md"),
            "protocol_sensitivity": repo_relative_path(tables_dir / "emc_public_eht_sensitivity.md"),
            "release_robustness": repo_relative_path(tables_dir / "emc_public_eht_release_robustness.md"),
            "stats": repo_relative_path(tables_dir / "emc_public_eht_stats.md"),
            "day_band": repo_relative_path(tables_dir / "emc_public_eht_day_band.md"),
            "release_gaps": repo_relative_path(tables_dir / "emc_public_eht_release_gaps.md"),
            "bootstrap_stats": repo_relative_path(bootstrap_tables_dir / "emc_public_eht_stats.md"),
            "bootstrap_release_gaps": repo_relative_path(bootstrap_tables_dir / "emc_public_eht_release_gaps.md"),
        },
        "figures": {
            "public_suite": repo_relative_path(figures_dir / "fig06_emc_public_eht_suite.png"),
            "transfer_gap": repo_relative_path(figures_dir / "fig07_emc_public_transfer_gap.png"),
            "public_qualitative": public_qualitative_selection,
        },
    }
    manifest = relativize_payload_paths(manifest)
    visual_asset_manifest_path = paper_root / "visual_asset_manifest.json"
    if visual_asset_manifest_path.exists():
        visual_asset_manifest = load_json(visual_asset_manifest_path)
    else:
        visual_asset_manifest = {}
    visual_asset_manifest["public_eht_suite_figures"] = {
        "fig06_emc_public_eht_suite": {
            "sources": [repo_relative_path(spec.summary_path) for spec in baseline_specs],
        },
        "fig07_emc_public_transfer_gap": {
            "source": repo_relative_path(tables_dir / "emc_public_eht_stats.md"),
            "release_gap_source": repo_relative_path(tables_dir / "emc_public_eht_release_gaps.md"),
        },
        "fig09_public_eht_qualitative": public_qualitative_selection,
    }
    visual_asset_manifest = relativize_payload_paths(visual_asset_manifest)
    save_json(visual_asset_manifest_path, visual_asset_manifest)
    save_json(summaries_dir / "public_eht_artifact_manifest.json", manifest)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
