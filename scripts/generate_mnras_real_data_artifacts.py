#!/usr/bin/env python3
"""Generate the final MNRAS-strengthening EMC artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.evaluation.mnras_real_data_artifacts import (  # noqa: E402
    SummarySpec,
    build_ablation_rows,
    build_claim_to_evidence_rows,
    build_real_data_rows,
    build_statistical_rows,
    build_tradeoff_rows,
    load_json as load_artifact_json,
    save_real_data_support_figure,
    save_tradeoff_figure,
    write_ablation_table,
    write_real_data_table,
    write_statistical_table,
    write_tradeoff_table,
)
from dynadiff_vlbi.evaluation.paper_artifacts import load_json as load_paper_json, save_json, write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default="outputs/mnras_real_data_artifacts")
    parser.add_argument("--paper-root", default="paper")
    return parser.parse_args()


def _benchmark_per_sample_paths(output_root: Path) -> list[Path]:
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


def _real_data_per_sample_paths(output_root: Path) -> list[Path]:
    protocol_dir = output_root / "emc_real_m87_public_validation"
    return [protocol_dir / "logs" / f"per_sample_support_{support_tag}.csv" for support_tag in ("80", "60", "40", "20")]


def main() -> int:
    args = parse_args()
    artifact_root = (ROOT / args.artifact_root).resolve()
    paper_root = (ROOT / args.paper_root).resolve()
    output_root = (ROOT / "outputs").resolve()
    tables_dir = artifact_root / "tables"
    summaries_dir = artifact_root / "summaries"
    figures_dir = paper_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    real_data_summary_path = output_root / "emc_real_m87_public_validation" / "logs" / "real_data_protocol_summary.json"
    ablation_summary_path = output_root / "emc_ablation_protocol" / "logs" / "emc_protocol_summary.json"
    benchmark_matrix_path = output_root / "emc_benchmark_artifacts" / "tables" / "emc_benchmark_matrix.md"
    default_benchmark_spec = SummarySpec(
        key="baseline_tracks",
        title="Default32 baseline-track benchmark",
        summary_path=output_root / "emc_benchmark_baseline_tracks_protocol" / "logs" / "emc_protocol_summary.json",
    )
    realism_spec = SummarySpec(
        key="challenge_inspired_realism",
        title="Challenge-inspired realism",
        summary_path=output_root / "emc_benchmark_challenge_inspired_realism_protocol" / "logs" / "emc_protocol_summary.json",
    )

    real_data_summary = load_artifact_json(real_data_summary_path)
    real_data_rows = build_real_data_rows(real_data_summary)
    write_csv(tables_dir / "emc_public_m87_validation.csv", real_data_rows)
    save_json(tables_dir / "emc_public_m87_validation.json", {"rows": real_data_rows})
    write_real_data_table(tables_dir / "emc_public_m87_validation.md", real_data_rows)

    ablation_summary = load_artifact_json(ablation_summary_path)
    ablation_rows = build_ablation_rows(ablation_summary)
    write_csv(tables_dir / "emc_component_ablations.csv", ablation_rows)
    save_json(tables_dir / "emc_component_ablations.json", {"rows": ablation_rows})
    write_ablation_table(tables_dir / "emc_component_ablations.md", ablation_rows)

    statistical_rows = build_statistical_rows(
        benchmark_per_sample_paths=_benchmark_per_sample_paths(output_root),
        real_data_per_sample_paths=_real_data_per_sample_paths(output_root),
    )
    write_csv(tables_dir / "emc_statistical_robustness.csv", statistical_rows)
    save_json(tables_dir / "emc_statistical_robustness.json", {"rows": statistical_rows})
    write_statistical_table(tables_dir / "emc_statistical_robustness.md", statistical_rows)

    tradeoff_rows = build_tradeoff_rows([default_benchmark_spec, realism_spec])
    write_csv(tables_dir / "emc_tradeoff_summary.csv", tradeoff_rows)
    save_json(tables_dir / "emc_tradeoff_summary.json", {"rows": tradeoff_rows})
    write_tradeoff_table(tables_dir / "emc_tradeoff_summary.md", tradeoff_rows)

    save_real_data_support_figure(
        summary=real_data_summary,
        output_png=figures_dir / "fig06_emc_public_m87_validation.png",
        output_svg=figures_dir / "fig06_emc_public_m87_validation.svg",
    )
    save_tradeoff_figure(
        specs=[default_benchmark_spec, realism_spec],
        output_png=figures_dir / "fig07_emc_tradeoff_curve.png",
        output_svg=figures_dir / "fig07_emc_tradeoff_curve.svg",
    )

    visual_asset_manifest_path = paper_root / "visual_asset_manifest.json"
    if visual_asset_manifest_path.exists():
        visual_asset_manifest = load_paper_json(visual_asset_manifest_path)
    else:
        visual_asset_manifest = {}
    visual_asset_manifest["mnras_strengthening_figures"] = {
        "fig06_emc_public_m87_validation": {
            "source": str(real_data_summary_path.resolve()),
        },
        "fig07_emc_tradeoff_curve": {
            "sources": [
                str(default_benchmark_spec.summary_path.resolve()),
                str(realism_spec.summary_path.resolve()),
            ],
        },
    }
    save_json(visual_asset_manifest_path, visual_asset_manifest)

    claim_rows = build_claim_to_evidence_rows(
        benchmark_matrix_path=benchmark_matrix_path,
        real_data_table_path=tables_dir / "emc_public_m87_validation.md",
        bootstrap_table_path=tables_dir / "emc_statistical_robustness.md",
        ablation_table_path=tables_dir / "emc_component_ablations.md",
        tradeoff_table_path=tables_dir / "emc_tradeoff_summary.md",
    )
    save_json(summaries_dir / "mnras_claim_to_evidence.json", {"rows": claim_rows})
    (summaries_dir / "mnras_claim_to_evidence.md").write_text(
        "# MNRAS Claim-to-Evidence Map\n\n"
        + "\n".join(f"- {row['claim']} -> `{row['evidence']}`" for row in claim_rows)
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "artifact_root": str(artifact_root),
        "paper_root": str(paper_root),
        "tables": {
            "public_m87_validation": str((tables_dir / "emc_public_m87_validation.md").resolve()),
            "component_ablations": str((tables_dir / "emc_component_ablations.md").resolve()),
            "statistical_robustness": str((tables_dir / "emc_statistical_robustness.md").resolve()),
            "tradeoff_summary": str((tables_dir / "emc_tradeoff_summary.md").resolve()),
        },
        "figures": {
            "public_m87_validation": str((figures_dir / "fig06_emc_public_m87_validation.png").resolve()),
            "tradeoff_curve": str((figures_dir / "fig07_emc_tradeoff_curve.png").resolve()),
        },
    }
    save_json(summaries_dir / "mnras_real_data_artifact_manifest.json", manifest)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
