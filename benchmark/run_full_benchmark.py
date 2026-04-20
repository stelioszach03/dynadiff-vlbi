#!/usr/bin/env python3
"""Thin wrapper around ``scripts/run_emc_benchmark.py``.

Exposes the friendlier argument names used in the thesis-extension plan
(``--config``, ``--oracle-ckpt``, ``--output``) without duplicating any
benchmark logic; forwards to the canonical runner in ``scripts/``.

Usage examples:

  # Deterministic partition, full benchmark, output to runs/synthetic_det:
  python benchmark/run_full_benchmark.py \\
      --config configs/thesis_extension/synthetic_deterministic.yaml \\
      --output runs/synthetic_det

  # Adaptive partition using a trained oracle:
  python benchmark/run_full_benchmark.py \\
      --config configs/thesis_extension/synthetic_adaptive.yaml \\
      --oracle-ckpt checkpoints/oracle/v1/best.ckpt \\
      --output runs/synthetic_adap
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "scripts" / "run_emc_benchmark.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        default=None,
        help="Experiment config YAML (thesis_extension template).",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output root directory (forwarded as --output-root).",
    )
    parser.add_argument(
        "--data-root",
        default="data/generated",
    )
    parser.add_argument(
        "--target",
        default="all",
        choices=[
            "all",
            "families",
            "baseline_tracks",
            "scan_segments",
            "station_dropout",
            "realism",
            "challenge_inspired_realism",
        ],
    )
    parser.add_argument(
        "--partition-mode",
        default=None,
        choices=["deterministic", "adaptive"],
        help="Overrides holdout.partition_mode from config if supplied.",
    )
    parser.add_argument(
        "--oracle-ckpt",
        default=None,
        help="Overrides holdout.oracle_checkpoint from config if supplied.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cmd: list[str] = [args.python, str(CANONICAL), "--target", args.target]
    cmd.extend(["--output-root", str(args.output)])
    cmd.extend(["--data-root", args.data_root])
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.partition_mode is not None:
        cmd.extend(["--partition-mode", args.partition_mode])
    if args.oracle_ckpt is not None:
        cmd.extend(["--oracle-ckpt", args.oracle_ckpt])

    env = dict(os.environ)
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        if not config_path.exists():
            sys.stderr.write(f"[run_full_benchmark] config not found: {config_path}\n")
            return 2
        # Surface the selected config to downstream scripts via an env var.
        env["DYNADIFF_BENCHMARK_CONFIG"] = str(config_path)

    print(f"[run_full_benchmark] forwarding to {CANONICAL}: {' '.join(cmd[1:])}", flush=True)
    return subprocess.call(cmd, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
