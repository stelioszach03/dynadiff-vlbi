#!/usr/bin/env python3
"""Train and evaluate the bounded DPS add-on comparator."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--artifact-root", default="outputs/dps_benchmark_artifacts")
    parser.add_argument("--run-name", default="dps_default64_baseline_tracks")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print(f"[run] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    output_root = (ROOT / args.output_root).resolve()
    artifact_root = (ROOT / args.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    baseline_manifest = json.loads(
        (output_root / "emc_benchmark_release" / "results_manifests" / "baseline_tracks.json").read_text(encoding="utf-8")
    )
    dps_checkpoint = output_root / args.run_name / "checkpoints" / "best.pt"
    if not args.skip_training and not dps_checkpoint.exists():
        _run(
            [
                args.python,
                "scripts/train_dps_baseline.py",
                "--data-dir",
                baseline_manifest["dataset_dir"],
                "--output-root",
                args.output_root,
                "--run-name",
                args.run_name,
                "--timesteps",
                str(args.timesteps),
                "--ddim-steps",
                str(args.ddim_steps),
                "--epochs",
                str(args.epochs),
            ]
        )

    _run(
        [
            args.python,
            "scripts/run_emc_benchmark.py",
            "--target",
            "baseline_tracks",
            "--python",
            args.python,
            "--output-root",
            args.output_root,
            "--data-root",
            "data/generated",
            "--dps-checkpoint",
            str(dps_checkpoint.relative_to(ROOT)),
        ]
    )
    _run(
        [
            args.python,
            "scripts/run_emc_public_eht_suite.py",
            "--families",
            "baseline_track_blocks",
            "--output-root",
            "outputs/public_eht_suite",
            "--dps-checkpoint",
            str(dps_checkpoint.relative_to(ROOT)),
        ]
    )

    (artifact_root / "dps_addon_manifest.json").write_text(
        json.dumps(
            {
                "dps_checkpoint": str(dps_checkpoint.relative_to(ROOT)),
                "timesteps": int(args.timesteps),
                "ddim_steps": int(args.ddim_steps),
                "epochs": int(args.epochs),
                "synthetic_condition": "baseline_tracks",
                "public_family": "baseline_track_blocks",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact_root": str(artifact_root),
                "dps_checkpoint": str(dps_checkpoint),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
