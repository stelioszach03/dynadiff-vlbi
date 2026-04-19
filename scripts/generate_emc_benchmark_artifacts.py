#!/usr/bin/env python3
"""Generate benchmark-first EMC paper artifacts from protocol outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from dynadiff_vlbi.evaluation.paper_artifacts import load_json, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default="outputs/emc_benchmark_artifacts")
    parser.add_argument("--paper-root", default="paper")
    parser.add_argument("--release-root", default="outputs/emc_benchmark_release")
    return parser.parse_args()


def _benchmark_specs(output_root: Path) -> tuple[list[BenchmarkProtocolSpec], BenchmarkProtocolSpec]:
    family_specs = [
        BenchmarkProtocolSpec(
            key="baseline_tracks",
            title="Default32 baseline-track holdout",
            family_label="Baseline-track blocks",
            family_description="Deterministic contiguous baseline-track blocks across time.",
            protocol_dir=(output_root / "emc_benchmark_baseline_tracks_protocol").resolve(),
            condition_group="benchmark_family",
        ),
        BenchmarkProtocolSpec(
            key="scan_segments",
            title="Default32 scan-segment holdout",
            family_label="Scan-segment blocks",
            family_description="Deterministic contiguous scan-like temporal windows are withheld.",
            protocol_dir=(output_root / "emc_benchmark_scan_segments_protocol").resolve(),
            condition_group="benchmark_family",
        ),
        BenchmarkProtocolSpec(
            key="station_dropout",
            title="Default32 station-dropout holdout",
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
    paper_root = (ROOT / args.paper_root).resolve()
    release_root = (ROOT / args.release_root).resolve()
    output_root = ROOT / "outputs"
    tables_dir = artifact_root / "tables"
    summaries_dir = artifact_root / "summaries"
    figures_dir = paper_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    family_specs, realism_spec = _benchmark_specs(output_root)
    all_specs = [*family_specs, realism_spec]

    long_rows = benchmark_long_rows(all_specs)
    write_csv(tables_dir / "emc_benchmark_long.csv", long_rows)

    family_matrix_rows = build_family_matrix_rows(family_specs)
    write_csv(tables_dir / "emc_benchmark_matrix.csv", family_matrix_rows)
    save_json(tables_dir / "emc_benchmark_matrix.json", {"rows": family_matrix_rows})
    write_family_matrix_table(tables_dir / "emc_benchmark_matrix.md", family_matrix_rows)

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
        condition_title="Benchmark default32 baseline-track family",
        selection_mode="representative",
    )
    synthetic_secondary_selection = save_emc_secondary_qualitative_figure(
        prediction_path=family_specs[0].prediction_path("40"),
        per_sample_csv=family_specs[0].per_sample_path("40"),
        output_png=figures_dir / "fig08_emc_synthetic_qualitative.png",
        output_svg=figures_dir / "fig08_emc_synthetic_qualitative.svg",
        selection_manifest=figures_dir / "fig08_emc_synthetic_qualitative.selection.json",
        support_fraction_tag="40",
        condition_title="Benchmark default32 baseline-track family",
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
        "paper_root": str(paper_root),
        "protocols": {
            spec.key: {
                "title": spec.title,
                "family_label": spec.family_label,
                "family_description": spec.family_description,
                "summary_path": str(spec.summary_path),
            }
            for spec in all_specs
        },
        "tables": {
            "benchmark_matrix": str((tables_dir / "emc_benchmark_matrix.md").resolve()),
            "realism_track": str((tables_dir / "emc_challenge_inspired_realism.md").resolve()),
            "leaderboard_template": str((artifact_root / "leaderboard_template.csv").resolve()),
        },
        "figures": {
            "benchmark_support_curve": str((figures_dir / "fig02_emc_benchmark_support_curve.png").resolve()),
            "challenge_inspired_realism": str((figures_dir / "fig03_emc_challenge_inspired_realism.png").resolve()),
            "representative_example": representative_selection,
            "synthetic_secondary_example": synthetic_secondary_selection,
            "hard_realism_example": hard_selection,
        },
    }
    save_benchmark_manifest(summaries_dir / "emc_benchmark_artifact_manifest.json", benchmark_manifest)
    visual_asset_manifest_path = paper_root / "visual_asset_manifest.json"
    if visual_asset_manifest_path.exists():
        visual_asset_manifest = load_json(visual_asset_manifest_path)
    else:
        visual_asset_manifest = {}
    visual_asset_manifest["emc_benchmark_figures"] = {
        "fig02_emc_benchmark_support_curve": {
            "sources": [str(spec.summary_path) for spec in family_specs],
        },
        "fig03_emc_challenge_inspired_realism": {
            "source": str(realism_spec.summary_path),
        },
        "fig04_emc_benchmark_representative": representative_selection,
        "fig08_emc_synthetic_qualitative": synthetic_secondary_selection,
        "fig05_emc_realism_hard_example": hard_selection,
    }
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
