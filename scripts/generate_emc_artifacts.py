#!/usr/bin/env python3
"""Generate paper-facing EMC tables, figures, and summary files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.evaluation.emc_artifacts import (  # noqa: E402
    FULL_MODEL_ORDER,
    LEARNED_MODEL_ORDER,
    MODEL_LABELS,
    ProtocolSpec,
    build_verdict_rows,
    flatten_protocol_rows,
    load_json,
    protocol_claim_summary,
    save_emc_qualitative_figure,
    save_emc_schematic,
    save_json,
    save_support_fraction_figure,
    write_csv,
    write_markdown_table,
)
from dynadiff_vlbi.evaluation.paper_artifacts import format_value  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default="outputs/emc_paper_artifacts")
    parser.add_argument("--paper-root", default="paper")
    return parser.parse_args()


def _fmt(value: float) -> str:
    import math

    return "n/a" if math.isnan(float(value)) else format_value(float(value))


def _table_rows(spec: ProtocolSpec) -> list[list[str]]:
    summary = load_json(spec.summary_path)
    rows: list[list[str]] = []
    for support_tag in sorted(summary["support_fractions"].keys(), key=lambda item: int(item)):
        for model_key in FULL_MODEL_ORDER:
            metrics = summary["support_fractions"][support_tag]["models"][model_key]
            rows.append(
                [
                    support_tag,
                    MODEL_LABELS[model_key],
                    _fmt(float(metrics["heldout_visibility_rmse"])),
                    _fmt(float(metrics["heldout_closure_phase_mae"])),
                    _fmt(float(metrics["mse"])),
                    _fmt(float(metrics["ssim"])),
                    _fmt(float(metrics["temporal_consistency"])),
                ]
            )
    return rows


def _verdict_table_rows(rows: list[dict[str, object]]) -> list[list[str]]:
    ordered: list[list[str]] = []
    metric_order = {
        "heldout_visibility_rmse": 0,
        "heldout_closure_phase_mae": 1,
        "mse": 2,
        "ssim": 3,
    }
    rows = sorted(
        rows,
        key=lambda row: (
            row["condition_title"],
            int(str(row["support_fraction_tag"])),
            row["comparison"],
            metric_order[str(row["metric"])],
        ),
    )
    for row in rows:
        ordered.append(
            [
                str(row["condition_title"]),
                str(row["support_fraction_tag"]),
                str(row["comparison"]),
                str(row["metric"]),
                _fmt(float(row["emc_value"])),
                _fmt(float(row["other_value"])),
                str(row["verdict"]),
            ]
        )
    return ordered


def main() -> int:
    args = parse_args()
    artifact_root = (ROOT / args.artifact_root).resolve()
    paper_root = (ROOT / args.paper_root).resolve()
    tables_dir = artifact_root / "tables"
    summaries_dir = artifact_root / "summaries"
    figures_dir = paper_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    default32 = ProtocolSpec(
        key="default32",
        title="Default32 structured-holdout protocol",
        protocol_dir=(ROOT / "outputs" / "emc_default32_protocol").resolve(),
    )
    sparse_uv = ProtocolSpec(
        key="sparse_uv",
        title="Sparse-uv structured-holdout protocol",
        protocol_dir=(ROOT / "outputs" / "emc_sparse_uv_protocol").resolve(),
    )
    specs = [default32, sparse_uv]

    protocol_rows = []
    for spec in specs:
        protocol_rows.extend(flatten_protocol_rows(spec))
    write_csv(tables_dir / "emc_protocol_long.csv", protocol_rows)

    for spec in specs:
        write_markdown_table(
            tables_dir / f"emc_{spec.key}_support_sweep.md",
            headers=["Support (%)", "Model", "Held-out VisRMSE", "Held-out Closure", "MSE", "SSIM", "Temporal"],
            rows=_table_rows(spec),
            title=f"{spec.title} Support-Fraction Sweep",
            notes=[
                "Support-set coefficients are provided to the model and the DC layer; target hold-out coefficients are not.",
                "Held-out closure is reported only when all-target triangle support is sufficient. The 80% rows are therefore marked n/a.",
            ],
        )

    verdict_rows = build_verdict_rows(specs)
    write_csv(tables_dir / "emc_verdicts.csv", verdict_rows)
    write_markdown_table(
        tables_dir / "emc_verdicts.md",
        headers=["Condition", "Support (%)", "Comparison", "Metric", "EMC", "Other", "Verdict"],
        rows=_verdict_table_rows(verdict_rows),
        title="EMC Verdicts Versus Learned Comparators",
        notes=[
            "Wins and losses are metric-direction aware.",
            "This table focuses on attribution-relevant comparisons against the baseline 3D U-Net, residual refinement, and CCRR.",
        ],
    )

    claim_summary = protocol_claim_summary(specs)
    save_json(summaries_dir / "emc_claim_summary.json", claim_summary)

    save_emc_schematic(
        png_path=figures_dir / "fig01_emc_schematic.png",
        svg_path=figures_dir / "fig01_emc_schematic.svg",
    )
    save_support_fraction_figure(
        specs=specs,
        output_png=figures_dir / "fig02_emc_support_fraction_curve.png",
        output_svg=figures_dir / "fig02_emc_support_fraction_curve.svg",
    )
    default_selection = save_emc_qualitative_figure(
        prediction_path=default32.prediction_path("40"),
        per_sample_csv=default32.per_sample_path("40"),
        output_png=figures_dir / "fig03_emc_representative_default32.png",
        output_svg=figures_dir / "fig03_emc_representative_default32.svg",
        selection_manifest=figures_dir / "fig03_emc_representative_default32.selection.json",
        support_fraction_tag="40",
        condition_title="Default32",
        selection_mode="representative",
    )
    sparse_selection = save_emc_qualitative_figure(
        prediction_path=sparse_uv.prediction_path("20"),
        per_sample_csv=sparse_uv.per_sample_path("20"),
        output_png=figures_dir / "fig04_emc_sparse_uv_hard.png",
        output_svg=figures_dir / "fig04_emc_sparse_uv_hard.svg",
        selection_manifest=figures_dir / "fig04_emc_sparse_uv_hard.selection.json",
        support_fraction_tag="20",
        condition_title="Sparse-uv",
        selection_mode="hard",
    )

    save_json(
        paper_root / "visual_asset_manifest.json",
        {
            "emc_figures": {
                "fig01_emc_schematic": {
                    "source": "scripts/generate_emc_artifacts.py",
                },
                "fig02_emc_support_fraction_curve": {
                    "sources": [str(spec.summary_path) for spec in specs],
                },
                "fig03_emc_representative_default32": default_selection,
                "fig04_emc_sparse_uv_hard": sparse_selection,
            }
        },
    )

    print(
        {
            "artifact_root": str(artifact_root),
            "paper_root": str(paper_root),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
