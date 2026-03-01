#!/usr/bin/env python3
"""Run a fast end-to-end smoke test using the configured CLI scripts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="smoke", choices=["smoke", "default32", "exp64"])
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"demo_{args.preset}"
    python_executable = sys.executable

    run_command([python_executable, "scripts/generate_toy_dataset.py", "--preset", args.preset])
    run_command(
        [
            python_executable,
            "scripts/train_baseline.py",
            "--preset",
            args.preset,
            "--run-name",
            run_name,
        ]
    )
    run_command(
        [
            python_executable,
            "scripts/evaluate_model.py",
            "--preset",
            args.preset,
            "--run-name",
            run_name,
        ]
    )
    print(f"Smoke demo finished. Outputs are under outputs/{run_name}")


if __name__ == "__main__":
    main()
