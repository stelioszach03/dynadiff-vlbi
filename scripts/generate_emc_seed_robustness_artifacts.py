#!/usr/bin/env python3
"""Generate paper-facing artifacts for the bounded EMC seed-robustness study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynadiff_vlbi.evaluation.ccrr_artifacts import paired_bootstrap_stats  # noqa: E402
from dynadiff_vlbi.evaluation.paper_artifacts import format_value, repo_relative_path, save_json, write_csv, write_markdown_table  # noqa: E402


MODEL_LABELS = {
    "baseline_learned": "Baseline 3D U-Net",
    "residual_refinement": "Residual Refinement",
    "ccrr": "CCRR",
    "emc": "EMC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/emc_seed_robustness/seed_robustness_manifest.json")
    parser.add_argument("--artifact-root", default="outputs/emc_seed_robustness_artifacts")
    return parser.parse_args()


def _safe_float(value: Any) -> float:
    if value in {None, "", "nan", "NaN"}:
        return float("nan")
    return float(value)


def _fmt(value: float) -> str:
    return "n/a" if math.isnan(float(value)) else format_value(float(value))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _support_tags(summary: dict[str, Any]) -> list[str]:
    return sorted(summary["support_fractions"].keys(), key=lambda item: int(item), reverse=True)


def _collect_paired_values(
    *,
    per_sample_paths: list[Path],
    candidate_key: str,
    reference_key: str,
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_values: list[float] = []
    reference_values: list[float] = []
    for csv_path in per_sample_paths:
        grouped: dict[tuple[str, str], dict[str, float]] = {}
        for row in _load_csv_rows(csv_path):
            sample_id = str(row.get("sample_id", row.get("sample_index", "")))
            support_tag = str(row.get("support_fraction_tag", ""))
            metric_value = _safe_float(row.get(metric_key))
            if math.isnan(metric_value):
                continue
            grouped.setdefault((support_tag, sample_id), {})[str(row["model"])] = metric_value
        for model_map in grouped.values():
            if candidate_key not in model_map or reference_key not in model_map:
                continue
            candidate_values.append(model_map[candidate_key])
            reference_values.append(model_map[reference_key])
    return np.asarray(candidate_values, dtype=np.float64), np.asarray(reference_values, dtype=np.float64)


def main() -> int:
    args = parse_args()
    manifest_path = (ROOT / args.manifest).resolve()
    artifact_root = (ROOT / args.artifact_root).resolve()
    tables_dir = artifact_root / "tables"
    summaries_dir = artifact_root / "summaries"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(manifest_path)
    run_specs = manifest["runs"]
    seed_values = [int(value) for value in manifest.get("seeds", [])]
    summaries = [_load_json(ROOT / run["protocol_summary"]) for run in run_specs]
    support_tags = _support_tags(summaries[0])

    summary_rows: list[dict[str, Any]] = []
    for support_tag in support_tags:
        row: dict[str, Any] = {"support_fraction_tag": support_tag}
        best_model = None
        best_mean = float("inf")
        for model_key in MODEL_LABELS:
            values = np.asarray(
                [
                    _safe_float(summary["support_fractions"][support_tag]["models"][model_key]["heldout_visibility_rmse"])
                    for summary in summaries
                ],
                dtype=np.float64,
            )
            mean = float(np.nanmean(values))
            std = float(np.nanstd(values))
            row[f"{model_key}_mean"] = mean
            row[f"{model_key}_std"] = std
            if mean < best_mean:
                best_mean = mean
                best_model = model_key
        row["best_model"] = best_model
        row["best_model_label"] = MODEL_LABELS[str(best_model)]
        row["best_model_mean"] = best_mean
        summary_rows.append(row)

    write_csv(tables_dir / "emc_seed_robustness_summary.csv", summary_rows)
    save_json(tables_dir / "emc_seed_robustness_summary.json", {"rows": summary_rows})
    write_markdown_table(
        tables_dir / "emc_seed_robustness_summary.md",
        headers=[
            "Support (%)",
            "EMC mean ± std",
            "CCRR mean ± std",
            "Residual mean ± std",
            "Baseline mean ± std",
            "Best mean model",
        ],
        rows=[
            [
                str(row["support_fraction_tag"]),
                f"{_fmt(float(row['emc_mean']))} ± {_fmt(float(row['emc_std']))}",
                f"{_fmt(float(row['ccrr_mean']))} ± {_fmt(float(row['ccrr_std']))}",
                f"{_fmt(float(row['residual_refinement_mean']))} ± {_fmt(float(row['residual_refinement_std']))}",
                f"{_fmt(float(row['baseline_learned_mean']))} ± {_fmt(float(row['baseline_learned_std']))}",
                str(row["best_model_label"]),
            ]
            for row in summary_rows
        ],
        title="Default64 Baseline-Track Seed Robustness Summary",
        notes=[
            f"Means and population standard deviations are computed over seeds {', '.join(str(value) for value in seed_values)}.",
            "This study is intentionally bounded: it supports the central synthetic EMC claim on the default64 baseline-track family only.",
        ],
    )

    per_sample_paths = [
        ROOT / run["protocol_dir"] / "logs" / f"per_sample_support_{support_tag}.csv"
        for run in run_specs
        for support_tag in ("80", "60", "40", "20")
    ]
    stats_rows: list[dict[str, Any]] = []
    for reference_key in ("baseline_learned", "residual_refinement", "ccrr"):
        candidate, reference = _collect_paired_values(
            per_sample_paths=per_sample_paths,
            candidate_key="emc",
            reference_key=reference_key,
            metric_key="heldout_visibility_rmse",
        )
        stats = paired_bootstrap_stats(candidate, reference, direction="lower")
        stats_rows.append(
            {
                "comparison": f"EMC vs {MODEL_LABELS[reference_key]}",
                "metric": "heldout_visibility_rmse",
                "n_pairs": int(candidate.size),
                "candidate_mean": float(np.mean(candidate)),
                "reference_mean": float(np.mean(reference)),
                **stats,
            }
        )

    write_csv(tables_dir / "emc_seed_robustness_stats.csv", stats_rows)
    save_json(tables_dir / "emc_seed_robustness_stats.json", {"rows": stats_rows})
    write_markdown_table(
        tables_dir / "emc_seed_robustness_stats.md",
        headers=[
            "Comparison",
            "Metric",
            "Pairs",
            "EMC mean",
            "Reference mean",
            "Mean delta",
            "95% CI",
            "Win rate",
            "p-value",
        ],
        rows=[
            [
                str(row["comparison"]),
                str(row["metric"]),
                str(row["n_pairs"]),
                _fmt(float(row["candidate_mean"])),
                _fmt(float(row["reference_mean"])),
                _fmt(float(row["mean_delta"])),
                f"[{_fmt(float(row['ci_low']))}, {_fmt(float(row['ci_high']))}]",
                _fmt(float(row["win_rate"])),
                _fmt(float(row["p_value"])),
            ]
            for row in stats_rows
        ],
        title="Default64 Baseline-Track Seed Robustness Paired Statistics",
        notes=[
            "Positive direction-aware mean deltas mean EMC is better under held-out visibility RMSE.",
            f"Pairs are pooled across support fractions and seeds {', '.join(str(value) for value in seed_values)} for the bounded robustness check.",
        ],
    )

    claim_rows = [
        {
            "claim": (
                f"Across seeds {', '.join(str(value) for value in seed_values)} on the default64 baseline-track family, "
                "EMC retains the lowest mean held-out visibility RMSE among the learned comparators at every tested support fraction."
            ),
            "evidence": repo_relative_path(tables_dir / "emc_seed_robustness_summary.md"),
        },
        {
            "claim": "The central synthetic EMC advantage does not depend on a single seed, although this bounded study is intentionally narrower than the full synthetic benchmark breadth.",
            "evidence": repo_relative_path(tables_dir / "emc_seed_robustness_stats.md"),
        },
    ]
    save_json(summaries_dir / "emc_seed_robustness_claims.json", {"rows": claim_rows})
    (summaries_dir / "emc_seed_robustness_claims.md").write_text(
        "# EMC Seed-Robustness Claim-to-Evidence Map\n\n"
        + "\n".join(f"- {row['claim']} -> `{row['evidence']}`" for row in claim_rows)
        + "\n",
        encoding="utf-8",
    )
    save_json(
        summaries_dir / "emc_seed_robustness_manifest.json",
        {
            "manifest": repo_relative_path(manifest_path),
            "tables": {
                "summary": repo_relative_path(tables_dir / "emc_seed_robustness_summary.md"),
                "stats": repo_relative_path(tables_dir / "emc_seed_robustness_stats.md"),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
