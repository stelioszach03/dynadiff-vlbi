#!/usr/bin/env python3
"""Generate paper-facing artifacts for the multi-release public-EHT validation suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.evaluation.paper_artifacts import save_json, write_csv  # noqa: E402
from dynadiff_vlbi.evaluation.public_eht_suite_artifacts import (  # noqa: E402
    PublicTrackSpec,
    build_claim_to_evidence_rows,
    build_day_band_rows,
    build_public_matrix_rows,
    build_release_robustness_rows,
    build_public_stats_rows,
    build_sensitivity_rows,
    load_json,
    save_public_qualitative_figure,
    save_public_suite_figure,
    save_transfer_gap_figure,
    write_day_band_table,
    write_public_matrix_table,
    write_release_robustness_table,
    write_public_stats_table,
    write_sensitivity_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", default="outputs/public_eht_suite")
    parser.add_argument("--artifact-root", default="outputs/public_eht_suite_artifacts")
    parser.add_argument("--paper-root", default="paper")
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


def main() -> int:
    args = parse_args()
    suite_root = (ROOT / args.suite_root).resolve()
    artifact_root = (ROOT / args.artifact_root).resolve()
    paper_root = (ROOT / args.paper_root).resolve()
    tables_dir = artifact_root / "tables"
    summaries_dir = artifact_root / "summaries"
    figures_dir = paper_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    suite_manifest = json.loads((suite_root / "suite_manifest.json").read_text(encoding="utf-8"))
    baseline_specs = _track_specs_from_manifest(suite_manifest, "baseline_track_blocks")
    station_specs = _track_specs_from_manifest(suite_manifest, "station_dropout")

    baseline_summaries = [load_json(spec.summary_path) for spec in baseline_specs]
    station_summaries = [load_json(spec.summary_path) for spec in station_specs]

    public_matrix_rows = build_public_matrix_rows(baseline_summaries)
    write_csv(tables_dir / "emc_public_eht_matrix.csv", public_matrix_rows)
    save_json(tables_dir / "emc_public_eht_matrix.json", {"rows": public_matrix_rows})
    write_public_matrix_table(tables_dir / "emc_public_eht_matrix.md", public_matrix_rows)

    sensitivity_rows = build_sensitivity_rows(
        baseline_summaries=baseline_summaries,
        station_summaries=station_summaries,
    )
    write_csv(tables_dir / "emc_public_eht_sensitivity.csv", sensitivity_rows)
    save_json(tables_dir / "emc_public_eht_sensitivity.json", {"rows": sensitivity_rows})
    write_sensitivity_table(tables_dir / "emc_public_eht_sensitivity.md", sensitivity_rows)

    robustness_rows = build_release_robustness_rows(
        baseline_summaries=baseline_summaries,
        station_summaries=station_summaries,
    )
    write_csv(tables_dir / "emc_public_eht_release_robustness.csv", robustness_rows)
    save_json(tables_dir / "emc_public_eht_release_robustness.json", {"rows": robustness_rows})
    write_release_robustness_table(tables_dir / "emc_public_eht_release_robustness.md", robustness_rows)

    stats_rows = build_public_stats_rows(
        baseline_track_paths=[path for spec in baseline_specs for path in spec.per_sample_paths],
        station_dropout_paths=[path for spec in station_specs for path in spec.per_sample_paths],
        synthetic_paths=_synthetic_per_sample_paths((ROOT / "outputs").resolve()),
    )
    write_csv(tables_dir / "emc_public_eht_stats.csv", stats_rows)
    save_json(tables_dir / "emc_public_eht_stats.json", {"rows": stats_rows})
    write_public_stats_table(tables_dir / "emc_public_eht_stats.md", stats_rows)

    day_band_rows = build_day_band_rows([path for spec in baseline_specs for path in spec.per_sample_paths])
    write_csv(tables_dir / "emc_public_eht_day_band.csv", day_band_rows)
    save_json(tables_dir / "emc_public_eht_day_band.json", {"rows": day_band_rows})
    write_day_band_table(tables_dir / "emc_public_eht_day_band.md", day_band_rows)

    save_public_suite_figure(
        summaries=baseline_summaries,
        output_png=figures_dir / "fig06_emc_public_eht_suite.png",
        output_svg=figures_dir / "fig06_emc_public_eht_suite.svg",
    )
    save_transfer_gap_figure(
        stats_rows=stats_rows,
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
    )
    save_json(summaries_dir / "public_eht_claim_to_evidence.json", {"rows": claim_rows})
    (summaries_dir / "public_eht_claim_to_evidence.md").write_text(
        "# Public EHT Claim-to-Evidence Map\n\n"
        + "\n".join(f"- {row['claim']} -> `{row['evidence']}`" for row in claim_rows)
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "suite_root": str(suite_root),
        "artifact_root": str(artifact_root),
        "paper_root": str(paper_root),
        "tables": {
            "public_matrix": str((tables_dir / "emc_public_eht_matrix.md").resolve()),
            "protocol_sensitivity": str((tables_dir / "emc_public_eht_sensitivity.md").resolve()),
            "release_robustness": str((tables_dir / "emc_public_eht_release_robustness.md").resolve()),
            "stats": str((tables_dir / "emc_public_eht_stats.md").resolve()),
            "day_band": str((tables_dir / "emc_public_eht_day_band.md").resolve()),
        },
        "figures": {
            "public_suite": str((figures_dir / "fig06_emc_public_eht_suite.png").resolve()),
            "transfer_gap": str((figures_dir / "fig07_emc_public_transfer_gap.png").resolve()),
            "public_qualitative": public_qualitative_selection,
        },
    }
    visual_asset_manifest_path = paper_root / "visual_asset_manifest.json"
    if visual_asset_manifest_path.exists():
        visual_asset_manifest = load_json(visual_asset_manifest_path)
    else:
        visual_asset_manifest = {}
    visual_asset_manifest["public_eht_suite_figures"] = {
        "fig06_emc_public_eht_suite": {
            "sources": [str(spec.summary_path) for spec in baseline_specs],
        },
        "fig07_emc_public_transfer_gap": {
            "source": str((tables_dir / "emc_public_eht_stats.md").resolve()),
        },
        "fig09_public_eht_qualitative": public_qualitative_selection,
    }
    save_json(visual_asset_manifest_path, visual_asset_manifest)
    save_json(summaries_dir / "public_eht_artifact_manifest.json", manifest)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
