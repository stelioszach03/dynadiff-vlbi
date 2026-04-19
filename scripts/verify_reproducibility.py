#!/usr/bin/env python3
"""Verify that benchmark/public/seed outputs carry deterministic manifests and stable split exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.evaluation.benchmark_release import export_split_manifests  # noqa: E402
from dynadiff_vlbi.utils.config import _build_config  # noqa: E402
from dynadiff_vlbi.utils.logging_utils import save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", default="outputs/emc_benchmark_release")
    parser.add_argument("--public-root", default="outputs/public_eht_suite")
    parser.add_argument("--seed-root", default="outputs/emc_seed_robustness")
    parser.add_argument("--dps-root", default="outputs/dps_benchmark_artifacts")
    parser.add_argument("--uq-root", default="outputs/emc_conformal_uq")
    parser.add_argument("--workshop-root", default="paper/workshop")
    parser.add_argument("--tmp-root", default="tmp/reproducibility_check")
    parser.add_argument("--report-path", default="benchmark/reproducibility_check.json")
    return parser.parse_args()


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _load_config_from_manifest(path: Path):
    payload = _load_json(path)
    return _build_config(payload, preset_name=str(payload.get("preset_name", "manifest")))


def _compare_split_npz(reference_path: Path, candidate_path: Path) -> dict[str, bool]:
    with np.load(reference_path) as reference, np.load(candidate_path) as candidate:
        checks = {
            "sample_index": np.array_equal(reference["sample_index"], candidate["sample_index"]),
            "support_mask": np.array_equal(reference["support_mask"], candidate["support_mask"]),
            "target_mask": np.array_equal(reference["target_mask"], candidate["target_mask"]),
            "target_unit_count": np.array_equal(reference["target_unit_count"], candidate["target_unit_count"]),
            "support_unit_count": np.array_equal(reference["support_unit_count"], candidate["support_unit_count"]),
        }
    return checks


def _recompute_and_compare(
    *,
    config_manifest_path: Path,
    dataset_dir: Path,
    reference_split_manifest_path: Path,
    tmp_output_dir: Path,
    support_fraction_tag: str = "60",
) -> dict[str, Any]:
    config = _load_config_from_manifest(config_manifest_path)
    tmp_output_dir.mkdir(parents=True, exist_ok=True)
    recomputed = export_split_manifests(
        config=config,
        dataset_dir=dataset_dir,
        output_dir=tmp_output_dir,
    )
    comparisons = _compare_split_npz(
        _resolve_repo_path(_load_json(reference_split_manifest_path)["support_fractions"][support_fraction_tag]["manifest_npz"]),
        tmp_output_dir / f"support_{support_fraction_tag}_split_manifest.npz",
    )
    return {
        "recomputed_split_manifest": str((tmp_output_dir / "split_manifest.json").relative_to(ROOT)),
        "support_fraction_tag": support_fraction_tag,
        "comparisons": comparisons,
        "all_equal": all(comparisons.values()),
        "recomputed_strategy": recomputed["strategy"],
    }


def verify_benchmark(benchmark_root: Path, tmp_root: Path) -> dict[str, Any]:
    manifest_path = benchmark_root / "benchmark_output_manifest.json"
    manifest = _load_json(manifest_path)
    condition_checks: dict[str, Any] = {}
    for key, payload in manifest["conditions"].items():
        split_manifest_path = _resolve_repo_path(payload["split_manifest_path"])
        config_manifest_path = _resolve_repo_path(payload["config_manifest_path"])
        results_manifest_path = _resolve_repo_path(payload["results_manifest_path"])
        summary_path = _resolve_repo_path(payload["summary_path"])
        metrics_csv = _resolve_repo_path(payload["expected_protocol_files"]["metrics_csv"])
        prediction_paths = {
            tag: _resolve_repo_path(path)
            for tag, path in payload["expected_protocol_files"]["support_fraction_predictions"].items()
        }
        checkpoint_paths = {name: _resolve_repo_path(path) for name, path in payload.get("checkpoints", {}).items()}
        for path in [split_manifest_path, config_manifest_path, results_manifest_path, summary_path, metrics_csv, *prediction_paths.values()]:
            _assert_exists(path)
        checkpoint_status = {name: path.exists() for name, path in checkpoint_paths.items()}
        condition_checks[key] = {
            "split_manifest_exists": True,
            "config_manifest_exists": True,
            "results_manifest_exists": True,
            "summary_exists": True,
            "metrics_csv_exists": True,
            "prediction_files_exist": {tag: True for tag in prediction_paths},
            "checkpoints_exist": checkpoint_status,
        }

    first_key = sorted(manifest["conditions"].keys())[0]
    first_payload = manifest["conditions"][first_key]
    recompute = _recompute_and_compare(
        config_manifest_path=_resolve_repo_path(first_payload["config_manifest_path"]),
        dataset_dir=_resolve_repo_path(first_payload["dataset_dir"]),
        reference_split_manifest_path=_resolve_repo_path(first_payload["split_manifest_path"]),
        tmp_output_dir=tmp_root / "benchmark" / first_key,
    )
    return {
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "conditions": condition_checks,
        "recomputed_subset": {first_key: recompute},
    }


def verify_public(public_root: Path, tmp_root: Path) -> dict[str, Any]:
    suite_manifest_path = public_root / "suite_manifest.json"
    suite_manifest = _load_json(suite_manifest_path)
    run_checks: list[dict[str, Any]] = []
    for run in suite_manifest["runs"]:
        split_manifest_path = _resolve_repo_path(run["split_manifest_path"])
        results_manifest_path = _resolve_repo_path(run["results_manifest_path"])
        summary_path = _resolve_repo_path(run["summary_path"])
        output_dir = _resolve_repo_path(run["output_dir"])
        _assert_exists(split_manifest_path)
        _assert_exists(results_manifest_path)
        _assert_exists(summary_path)
        _assert_exists(output_dir / "logs" / "support_fraction_metrics.csv")
        run_checks.append(
            {
                "family": run["family"],
                "release_code": run["release_code"],
                "split_manifest_exists": True,
                "results_manifest_exists": True,
                "summary_exists": True,
            }
        )

    first_run = suite_manifest["runs"][0]
    results_manifest = _load_json(_resolve_repo_path(first_run["results_manifest_path"]))
    recompute = _recompute_and_compare(
        config_manifest_path=_resolve_repo_path(results_manifest["config_manifest_path"]),
        dataset_dir=_resolve_repo_path(results_manifest["dataset_dir"]),
        reference_split_manifest_path=_resolve_repo_path(results_manifest["split_manifest_path"]),
        tmp_output_dir=tmp_root / "public" / f"{first_run['family']}_{first_run['release_code']}",
    )
    return {
        "manifest_path": str(suite_manifest_path.relative_to(ROOT)),
        "runs": run_checks,
        "recomputed_subset": {
            f"{first_run['family']}::{first_run['release_code']}": recompute,
        },
    }


def verify_seed_robustness(seed_root: Path) -> dict[str, Any]:
    manifest_path = seed_root / "seed_robustness_manifest.json"
    manifest = _load_json(manifest_path)
    run_checks: list[dict[str, Any]] = []
    for run in manifest["runs"]:
        dataset_dir = _resolve_repo_path(run["dataset_dir"])
        config_path = _resolve_repo_path(run.get("config_manifest_path", run.get("config_path")))
        protocol_dir = _resolve_repo_path(run["protocol_dir"])
        protocol_summary = _resolve_repo_path(run["protocol_summary"])

        for path in (
            dataset_dir,
            config_path,
            protocol_dir,
            protocol_summary,
            protocol_dir / "logs" / "support_fraction_metrics.csv",
            protocol_dir / "logs" / "per_sample_support_80.csv",
            protocol_dir / "logs" / "per_sample_support_60.csv",
            protocol_dir / "logs" / "per_sample_support_40.csv",
            protocol_dir / "logs" / "per_sample_support_20.csv",
        ):
            _assert_exists(path)
        checkpoint_status = {
            "baseline": _resolve_repo_path(run["baseline_checkpoint"]).exists(),
            "residual": _resolve_repo_path(run["residual_checkpoint"]).exists(),
            "ccrr": _resolve_repo_path(run["ccrr_checkpoint"]).exists(),
            "emc": _resolve_repo_path(run["emc_checkpoint"]).exists(),
        }
        run_checks.append(
            {
                "seed": int(run["seed"]),
                "config_exists": True,
                "dataset_dir_exists": True,
                "protocol_dir_exists": True,
                "protocol_summary_exists": True,
                "checkpoints_exist": checkpoint_status,
                "protocol_metric_files_exist": True,
            }
        )
    return {
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "seeds": [int(value) for value in manifest["seeds"]],
        "runs": run_checks,
    }


def verify_optional_outputs(dps_root: Path, uq_root: Path, workshop_root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {}
    dps_manifest = dps_root / "dps_addon_manifest.json"
    dps_synth = dps_root / "synthetic_dps_table.csv"
    dps_public = dps_root / "public_dps_release_summary.csv"
    if dps_manifest.exists() or dps_synth.exists() or dps_public.exists():
        report["dps"] = {
            "manifest_path": str(dps_manifest.relative_to(ROOT)) if dps_manifest.exists() else None,
            "synthetic_table_exists": dps_synth.exists(),
            "public_table_exists": dps_public.exists(),
        }
    if (uq_root / "conformal_uq_manifest.json").exists():
        report["conformal_uq"] = {
            "manifest_path": str((uq_root / "conformal_uq_manifest.json").relative_to(ROOT)),
            "synthetic_summary_exists": (uq_root / "tables" / "synthetic_emc_conformal_uq.csv").exists(),
            "public_summary_exists": (uq_root / "tables" / "public_emc_conformal_uq.csv").exists(),
        }
    if (workshop_root / "ml4ps_abstract.pdf").exists():
        report["workshop"] = {
            "pdf_path": str((workshop_root / "ml4ps_abstract.pdf").relative_to(ROOT)),
            "style_exists": (workshop_root / "neurips_2025.sty").exists(),
            "tex_exists": (workshop_root / "ml4ps_abstract.tex").exists(),
        }
    return report


def main() -> int:
    args = parse_args()
    benchmark_root = (ROOT / args.benchmark_root).resolve()
    public_root = (ROOT / args.public_root).resolve()
    seed_root = (ROOT / args.seed_root).resolve()
    dps_root = (ROOT / args.dps_root).resolve()
    uq_root = (ROOT / args.uq_root).resolve()
    workshop_root = (ROOT / args.workshop_root).resolve()
    tmp_root = (ROOT / args.tmp_root).resolve()
    report_path = (ROOT / args.report_path).resolve()

    report = {
        "benchmark": verify_benchmark(benchmark_root, tmp_root),
        "public_eht": verify_public(public_root, tmp_root),
        "seed_robustness": verify_seed_robustness(seed_root),
        "optional_outputs": verify_optional_outputs(dps_root, uq_root, workshop_root),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(report_path, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
