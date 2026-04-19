#!/usr/bin/env python3
"""Prepare one official public EHT calibrated-data release for EMC real-data validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.data.eht_public_data import (  # noqa: E402
    ensure_public_eht_repo,
    get_public_eht_release_spec,
    prepare_public_eht_validation_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-code",
        default="2019-D01-01",
        help="Official public EHT release code to prepare.",
    )
    parser.add_argument(
        "--repo-cache-dir",
        default=None,
        help="Local directory where the official public EHT release is cloned or copied.",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Optional existing local copy of the public EHT release to copy instead of cloning.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Evaluation-only dataset output directory.",
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    release_spec = get_public_eht_release_spec(args.release_code)
    repo_cache_dir = (
        ROOT / args.repo_cache_dir
        if args.repo_cache_dir is not None
        else ROOT / "data" / "external" / f"eht-public-{release_spec.release_code}"
    )
    output_dir = (
        ROOT / args.output_dir
        if args.output_dir is not None
        else ROOT / "data" / "real" / f"eht_public_{release_spec.target_id.lower()}_{release_spec.campaign_year}_{release_spec.release_code.lower()}_emc"
    )
    repo_dir = ensure_public_eht_repo(
        destination=repo_cache_dir,
        source_dir=(ROOT / args.source_dir) if args.source_dir is not None else None,
        release_code=release_spec.release_code,
    )
    manifest = prepare_public_eht_validation_dataset(
        source_root=repo_dir,
        output_dir=output_dir,
        release_code=release_spec.release_code,
        image_size=args.image_size,
        sequence_length=args.sequence_length,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
