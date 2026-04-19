#!/usr/bin/env python3
"""Generate publication-facing figures and supplementary media from saved outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.evaluation.paper_artifacts import save_methods_figure
from dynadiff_vlbi.evaluation.paper_visuals import (
    PredictionCondition,
    save_condition_comparison_figure,
    save_supplementary_gif,
    save_temporal_sequence_figure,
    save_uncertainty_alignment_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--paper-root", default="paper")
    return parser.parse_args()


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"None of the expected paths exist: {joined}")


def main() -> int:
    args = parse_args()
    output_root = (ROOT / args.output_root).resolve()
    paper_root = (ROOT / args.paper_root).resolve()
    figures_dir = paper_root / "figures"
    media_dir = paper_root / "supplementary_media"
    figures_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    default32_predictions = output_root / "compare_default32_residual_refine_fair8/predictions/test_predictions.npz"
    noise_high_predictions = output_root / "paper_noise_high_residual_refine/predictions/test_predictions.npz"
    sparse_uv_predictions = output_root / "paper_sparse_uv_residual_refine/predictions/test_predictions.npz"
    exp64_predictions = _first_existing(
        output_root / "paper_exp64_residual_refine_clean/predictions/test_predictions.npz",
        output_root / "paper_exp64_residual_refine/predictions/test_predictions.npz",
    )

    condition_specs = [
        PredictionCondition("default32", "Default32", default32_predictions),
        PredictionCondition("noise_high", "High noise", noise_high_predictions),
        PredictionCondition("sparse_uv", "Sparse uv", sparse_uv_predictions),
        PredictionCondition("exp64", "Exp64", exp64_predictions),
    ]

    manifest: dict[str, object] = {}

    save_methods_figure(
        png_path=figures_dir / "fig01_residual_refinement_schematic.png",
        svg_path=figures_dir / "fig01_residual_refinement_schematic.svg",
    )
    manifest["fig01_residual_refinement_schematic"] = {
        "source": "schematic redrawn from the locked residual-refinement architecture",
        "png": str(figures_dir / "fig01_residual_refinement_schematic.png"),
        "svg": str(figures_dir / "fig01_residual_refinement_schematic.svg"),
    }

    manifest["fig02_condition_comparison"] = save_condition_comparison_figure(
        conditions=condition_specs,
        output_png=figures_dir / "fig02_condition_comparison.png",
        output_svg=figures_dir / "fig02_condition_comparison.svg",
        selection_manifest=figures_dir / "fig02_condition_comparison.selection.json",
    )

    manifest["fig03_temporal_sequence"] = save_temporal_sequence_figure(
        prediction_path=default32_predictions,
        output_png=figures_dir / "fig03_temporal_sequence_default32.png",
        output_svg=figures_dir / "fig03_temporal_sequence_default32.svg",
        selection_manifest=figures_dir / "fig03_temporal_sequence_default32.selection.json",
        frame_indices=[0, 2, 4, 6],
    )

    manifest["fig04_uncertainty_alignment"] = save_uncertainty_alignment_figure(
        prediction_path=default32_predictions,
        output_png=figures_dir / "fig04_uncertainty_alignment.png",
        output_svg=figures_dir / "fig04_uncertainty_alignment.svg",
        selection_manifest=figures_dir / "fig04_uncertainty_alignment.selection.json",
    )

    manifest["supplementary_default32_gif"] = save_supplementary_gif(
        prediction_path=default32_predictions,
        output_path=media_dir / "supp_default32_sequence.gif",
        row_title="Default32",
    )
    manifest["supplementary_sparse_uv_gif"] = save_supplementary_gif(
        prediction_path=sparse_uv_predictions,
        output_path=media_dir / "supp_sparse_uv_sequence.gif",
        row_title="Sparse uv",
    )

    (paper_root / "visual_asset_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
