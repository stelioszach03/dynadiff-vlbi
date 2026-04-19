#!/usr/bin/env python3
"""Run the CCRR reviewer-risk measurement audit on archived default32 seed repeats."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.evaluation.measurement_audit import (  # noqa: E402
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_MAX_TRIANGLES,
    DEFAULT_MIN_TOTAL_HELDOUT_TRIANGLES,
    DEFAULT_MIN_VALID_CLOSURE_SAMPLES,
    MeasurementAuditSpec,
    generate_measurement_audit_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--data-root", default="data/generated")
    parser.add_argument("--artifact-root", default="outputs/ccrr_paper_artifacts")
    parser.add_argument("--paper-root", default="paper")
    parser.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)
    parser.add_argument("--max-triangles", type=int, default=DEFAULT_MAX_TRIANGLES)
    parser.add_argument("--min-total-heldout-triangles", type=int, default=DEFAULT_MIN_TOTAL_HELDOUT_TRIANGLES)
    parser.add_argument("--min-valid-closure-samples", type=int, default=DEFAULT_MIN_VALID_CLOSURE_SAMPLES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = (ROOT / args.output_root).resolve()
    data_root = (ROOT / args.data_root).resolve()
    artifact_root = (ROOT / args.artifact_root).resolve()
    paper_root = (ROOT / args.paper_root).resolve()

    seed_specs = [
        MeasurementAuditSpec("seed7", "Default32 seed 7", 7, "ccrr_seed7_main", data_root / "ccrr_default32_seed7_shared", output_root),
        MeasurementAuditSpec("seed19", "Default32 seed 19", 19, "ccrr_seed19_main", data_root / "ccrr_default32_seed19_shared", output_root),
        MeasurementAuditSpec("seed31", "Default32 seed 31", 31, "ccrr_seed31_main", data_root / "ccrr_default32_seed31_shared", output_root),
        MeasurementAuditSpec("seed43", "Default32 seed 43", 43, "ccrr_seed43_main", data_root / "ccrr_default32_seed43_shared", output_root),
        MeasurementAuditSpec("seed59", "Default32 seed 59", 59, "ccrr_seed59_main", data_root / "ccrr_default32_seed59_shared", output_root),
    ]
    no_dc_spec = MeasurementAuditSpec(
        "no_dc",
        "No DC layer",
        7,
        "ccrr_ablation_no_dc",
        data_root / "ccrr_default32_seed7_shared",
        output_root,
    )
    no_closure_spec = MeasurementAuditSpec(
        "no_closure",
        "No closure loss",
        7,
        "ccrr_ablation_no_closure",
        data_root / "ccrr_default32_seed7_shared",
        output_root,
    )

    summary = generate_measurement_audit_artifacts(
        seed_specs=seed_specs,
        no_dc_spec=no_dc_spec,
        no_closure_spec=no_closure_spec,
        artifact_root=artifact_root,
        paper_root=paper_root,
        holdout_fraction=args.holdout_fraction,
        max_triangles=args.max_triangles,
        min_total_heldout_triangles=args.min_total_heldout_triangles,
        min_valid_closure_samples=args.min_valid_closure_samples,
    )
    print(json.dumps(summary["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
