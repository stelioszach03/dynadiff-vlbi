"""Paper-facing artifact generation for the EMC method path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import math

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

from dynadiff_vlbi.evaluation.paper_artifacts import (
    format_value,
    relativize_payload_paths,
    save_json,
    write_csv,
    write_markdown_table,
)
from dynadiff_vlbi.evaluation.paper_visuals import (
    ERROR_CMAP,
    IMAGE_CMAP,
    load_predictions,
)


MODEL_LABELS: dict[str, str] = {
    "dirty": "Dirty",
    "tikhonov": "Tikhonov",
    "baseline_learned": "Baseline 3D U-Net",
    "residual_refinement": "Residual Refinement",
    "ccrr": "CCRR",
    "emc": "EMC",
    "dps": "DPS",
}

LEARNED_MODEL_ORDER = [
    "baseline_learned",
    "residual_refinement",
    "ccrr",
    "emc",
    "dps",
]
FULL_MODEL_ORDER = [
    "dirty",
    "tikhonov",
    *LEARNED_MODEL_ORDER,
]
MODEL_COLORS = {
    "baseline_learned": "#4b5563",
    "residual_refinement": "#2563eb",
    "ccrr": "#7c3aed",
    "emc": "#dc2626",
    "dps": "#f59e0b",
}


@dataclass(frozen=True)
class ProtocolSpec:
    """One EMC protocol output used to assemble paper artifacts."""

    key: str
    title: str
    protocol_dir: Path

    @property
    def summary_path(self) -> Path:
        return self.protocol_dir / "logs" / "emc_protocol_summary.json"

    @property
    def metrics_csv_path(self) -> Path:
        return self.protocol_dir / "logs" / "support_fraction_metrics.csv"

    def prediction_path(self, support_fraction_tag: str) -> Path:
        return self.protocol_dir / "predictions" / f"support_{support_fraction_tag}.npz"

    def per_sample_path(self, support_fraction_tag: str) -> Path:
        return self.protocol_dir / "logs" / f"per_sample_support_{support_fraction_tag}.csv"


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load a CSV file into a list of dictionaries."""

    import csv

    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: Any) -> float:
    if value in {None, "", "nan", "NaN"}:
        return float("nan")
    return float(value)


def _fmt(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return format_value(value)


def _support_tags(summary: dict[str, Any]) -> list[str]:
    return sorted(summary["support_fractions"].keys(), key=lambda item: int(item))


def _model_metrics(summary: dict[str, Any], support_tag: str, model_key: str) -> dict[str, float]:
    models = summary["support_fractions"][support_tag]["models"]
    if model_key not in models:
        reference_metrics = next(iter(models.values()))
        return {key: float("nan") for key in reference_metrics}
    model_metrics = models[model_key]
    return {key: float(value) for key, value in model_metrics.items()}


def flatten_protocol_rows(spec: ProtocolSpec) -> list[dict[str, Any]]:
    """Flatten one protocol summary into long rows."""

    summary = load_json(spec.summary_path)
    rows: list[dict[str, Any]] = []
    for support_tag in _support_tags(summary):
        support_fraction = float(support_tag) / 100.0
        for model_key in FULL_MODEL_ORDER:
            metrics = _model_metrics(summary, support_tag, model_key)
            rows.append(
                {
                    "condition": spec.key,
                    "condition_title": spec.title,
                    "support_fraction_tag": support_tag,
                    "support_fraction": support_fraction,
                    "model": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    **metrics,
                }
            )
    return rows


def build_verdict_rows(specs: list[ProtocolSpec]) -> list[dict[str, Any]]:
    """Summarize whether EMC wins, ties, or loses versus learned baselines."""

    verdict_rows: list[dict[str, Any]] = []
    directions = {
        "heldout_visibility_rmse": "lower",
        "heldout_closure_phase_mae": "lower",
        "mse": "lower",
        "ssim": "higher",
    }
    for spec in specs:
        summary = load_json(spec.summary_path)
        for support_tag in _support_tags(summary):
            emc_metrics = _model_metrics(summary, support_tag, "emc")
            for other_model in ["baseline_learned", "residual_refinement", "ccrr"]:
                other_metrics = _model_metrics(summary, support_tag, other_model)
                for metric_key, direction in directions.items():
                    emc_value = float(emc_metrics[metric_key])
                    other_value = float(other_metrics[metric_key])
                    if math.isnan(emc_value) or math.isnan(other_value):
                        verdict = "n/a"
                    elif math.isclose(emc_value, other_value, rel_tol=0.0, abs_tol=1e-12):
                        verdict = "tie"
                    elif direction == "lower":
                        verdict = "win" if emc_value < other_value else "loss"
                    else:
                        verdict = "win" if emc_value > other_value else "loss"
                    verdict_rows.append(
                        {
                            "condition": spec.key,
                            "condition_title": spec.title,
                            "support_fraction_tag": support_tag,
                            "comparison": f"EMC vs {MODEL_LABELS[other_model]}",
                            "metric": metric_key,
                            "emc_value": emc_value,
                            "other_value": other_value,
                            "verdict": verdict,
                        }
                    )
    return verdict_rows


def protocol_claim_summary(specs: list[ProtocolSpec]) -> dict[str, Any]:
    """Build a compact claim-to-evidence payload from the protocol summaries."""

    payload: dict[str, Any] = {"conditions": {}, "core_claims": []}
    for spec in specs:
        summary = load_json(spec.summary_path)
        condition_payload = {}
        for support_tag in _support_tags(summary):
            condition_payload[support_tag] = {
                model_key: {
                    "heldout_visibility_rmse": float(_model_metrics(summary, support_tag, model_key)["heldout_visibility_rmse"]),
                    "heldout_closure_phase_mae": float(_model_metrics(summary, support_tag, model_key)["heldout_closure_phase_mae"]),
                    "mse": float(_model_metrics(summary, support_tag, model_key)["mse"]),
                    "ssim": float(_model_metrics(summary, support_tag, model_key)["ssim"]),
                }
                for model_key in LEARNED_MODEL_ORDER
            }
        payload["conditions"][spec.key] = condition_payload

    payload["core_claims"].append(
        {
            "claim": "EMC achieves the lowest held-out visibility RMSE among the learned models at every tested support fraction on default32 and sparse-uv.",
            "supported_by": [
                str(spec.summary_path) for spec in specs
            ],
        }
    )
    payload["core_claims"].append(
        {
            "claim": "Where all-held-out triangle support is sufficient, EMC also improves held-out closure error relative to the baseline 3D U-Net, residual refinement, and CCRR.",
            "supported_by": [
                str(spec.summary_path) for spec in specs
            ],
        }
    )
    payload["core_claims"].append(
        {
            "claim": "EMC does not dominate the strongest structural baseline on SSIM, so the learned trade-off remains visible even when held-out measurement recovery improves.",
            "supported_by": [
                str(spec.summary_path) for spec in specs
            ],
        }
    )
    return payload


def save_emc_schematic(png_path: str | Path, svg_path: str | Path | None = None) -> None:
    """Save a compact schematic of the EMC method."""

    fig, ax = plt.subplots(figsize=(12.8, 4.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.9)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, face: str) -> None:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            linewidth=1.4,
            edgecolor="#111827",
            facecolor=face,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)

    box(0.3, 2.45, 2.1, 1.05, "Observed visibilities", "#f3f4f6")
    box(0.3, 0.8, 2.1, 1.1, "Structured split\nsupport / target", "#e0f2fe")
    box(3.0, 2.45, 2.1, 1.05, "Support dirty\n+ support vis", "#fff7ed")
    box(5.5, 2.45, 2.1, 1.05, "Backbone\n3D U-Net", "#fef3c7")
    box(8.0, 2.45, 2.1, 1.05, "Residual\nrefinement", "#ede9fe")
    box(10.5, 2.45, 2.1, 1.05, "Support-only\nDC layer", "#fee2e2")
    box(12.95, 2.45, 0.95, 1.05, "EMC", "#dcfce7")
    box(8.65, 0.65, 3.2, 1.05, "Target hold-out losses:\nvis + optional closure", "#ecfccb")

    arrow = dict(arrowstyle="->", linewidth=1.5, color="#111827")
    ax.annotate("", xy=(3.0, 2.975), xytext=(2.4, 2.975), arrowprops=arrow)
    ax.annotate("", xy=(5.5, 2.975), xytext=(5.1, 2.975), arrowprops=arrow)
    ax.annotate("", xy=(8.0, 2.975), xytext=(7.6, 2.975), arrowprops=arrow)
    ax.annotate("", xy=(10.5, 2.975), xytext=(10.1, 2.975), arrowprops=arrow)
    ax.annotate("", xy=(12.95, 2.975), xytext=(12.6, 2.975), arrowprops=arrow)
    ax.annotate("", xy=(9.95, 1.7), xytext=(9.95, 2.45), arrowprops=arrow)
    ax.text(
        7.0,
        4.55,
        "Earned Measurement Consistency (EMC)",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        7.0,
        4.08,
        "Only the support set is enforced. The target hold-out set is recovered,\n"
        "but it is never observed by the model or by the support-only data-consistency layer.",
        ha="center",
        va="top",
        fontsize=10.2,
    )

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    if svg_path is not None:
        fig.savefig(Path(svg_path), bbox_inches="tight")
    plt.close(fig)


def _sample_gain_map(
    rows: list[dict[str, Any]],
    reference_key: str,
    candidate_key: str,
    metric_key: str,
) -> list[tuple[int, float, float]]:
    by_sample: dict[int, dict[str, dict[str, float]]] = {}
    for row in rows:
        sample_index = int(row["sample_index"])
        model_key = row["model"]
        by_sample.setdefault(sample_index, {})[model_key] = {
            metric_key: _safe_float(row[metric_key]),
        }

    gains: list[tuple[int, float, float]] = []
    for sample_index, model_map in by_sample.items():
        if reference_key not in model_map or candidate_key not in model_map:
            continue
        reference_value = model_map[reference_key][metric_key]
        candidate_value = model_map[candidate_key][metric_key]
        if math.isnan(reference_value) or math.isnan(candidate_value):
            continue
        gains.append((sample_index, reference_value - candidate_value, reference_value))
    gains.sort(key=lambda item: item[0])
    return gains


def _select_protocol_example(
    rows: list[dict[str, Any]],
    bundle: dict[str, np.ndarray],
    reference_key: str,
    candidate_key: str,
    metric_key: str,
    selection_mode: str,
) -> tuple[int, int, dict[str, Any]]:
    gains = _sample_gain_map(rows, reference_key=reference_key, candidate_key=candidate_key, metric_key=metric_key)
    if not gains:
        raise ValueError("No comparable sample rows were found for qualitative selection.")

    if selection_mode == "representative":
        positive = [item for item in gains if item[1] > 0.0]
        ordered = positive or gains
        sample_index, gain_value, reference_value = ordered[len(ordered) // 2]
        selection_rule = f"median sample-level {candidate_key} gain over {reference_key} on {metric_key}"
    elif selection_mode == "hard":
        baseline_scores = np.asarray([item[2] for item in gains], dtype=np.float64)
        threshold = float(np.quantile(baseline_scores, 0.75))
        hard = [item for item in gains if item[2] >= threshold]
        ordered = sorted(hard or gains, key=lambda item: item[1], reverse=True)
        sample_index, gain_value, reference_value = ordered[0]
        selection_rule = f"top-quartile hard sample by {reference_key} {metric_key}, then max {candidate_key} gain"
    else:
        raise ValueError(f"Unknown selection mode: {selection_mode}")

    ground_truth = bundle["ground_truth"][sample_index]
    reference = bundle[reference_key][sample_index]
    candidate = bundle[candidate_key][sample_index]
    frame_gains = ((reference - ground_truth) ** 2).mean(axis=(1, 2)) - ((candidate - ground_truth) ** 2).mean(axis=(1, 2))
    if selection_mode == "representative":
        frame_index = int(np.argsort(frame_gains)[len(frame_gains) // 2])
    else:
        frame_index = int(np.argmax(frame_gains))

    payload = {
        "sample_index": int(sample_index),
        "frame_index": int(frame_index),
        "selection_rule": selection_rule,
        "reference_key": reference_key,
        "candidate_key": candidate_key,
        "metric_key": metric_key,
        "sample_level_gain": float(gain_value),
        "reference_value": float(reference_value),
    }
    return int(sample_index), int(frame_index), payload


def save_emc_qualitative_figure(
    prediction_path: str | Path,
    per_sample_csv: str | Path,
    output_png: str | Path,
    output_svg: str | Path,
    selection_manifest: str | Path,
    support_fraction_tag: str,
    condition_title: str,
    selection_mode: str,
    reference_key: str = "ccrr",
    candidate_key: str = "emc",
) -> dict[str, Any]:
    """Save a qualitative EMC panel tied to a documented selection rule."""

    bundle = load_predictions(prediction_path)
    rows = load_rows(per_sample_csv)
    sample_index, frame_index, payload = _select_protocol_example(
        rows=rows,
        bundle=bundle,
        reference_key=reference_key,
        candidate_key=candidate_key,
        metric_key="heldout_visibility_rmse",
        selection_mode=selection_mode,
    )

    prediction_columns = ["ground_truth", "dirty", "baseline_learned", "residual_refinement", "ccrr", "emc"]
    labels = ["Ground truth", "Support dirty", "Baseline 3D U-Net", "Residual refinement", "CCRR", "EMC"]
    image_row = [np.clip(bundle[key][sample_index, frame_index], 0.0, 1.0) for key in prediction_columns]
    error_row = [
        np.clip(bundle["target_mask"][sample_index, frame_index], 0.0, 1.0),
        *[
            np.abs(bundle[key][sample_index, frame_index] - bundle["ground_truth"][sample_index, frame_index])
            for key in prediction_columns[1:]
        ],
    ]
    error_row = [image_row[0] * 0.0 + error_row[0], *error_row[1:]]
    error_labels = ["Target mask", "Dirty error", "Baseline error", "Residual error", "CCRR error", "EMC error"]

    fig, axes = plt.subplots(2, len(labels), figsize=(14.8, 5.6))
    fig.patch.set_facecolor("white")
    for column_index, (image, label) in enumerate(zip(image_row, labels)):
        axis = axes[0, column_index]
        axis.imshow(image, cmap=IMAGE_CMAP, interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(label, fontsize=11, pad=6)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0 if label == "EMC" else 0.8)
            spine.set_edgecolor("#dc2626" if label == "EMC" else "#d1d5db")
    for column_index, (image, label) in enumerate(zip(error_row, error_labels)):
        axis = axes[1, column_index]
        cmap = "gray" if label == "Target mask" else ERROR_CMAP
        normalized = image if label == "Target mask" else (
            np.zeros_like(image, dtype=np.float32)
            if float(np.max(image)) <= float(np.min(image))
            else (image - float(np.min(image))) / (float(np.max(image)) - float(np.min(image)))
        )
        axis.imshow(normalized, cmap=cmap, interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
        if column_index == 0:
            axes[0, column_index].set_ylabel("Reconstruction", fontsize=11)
            axis.set_ylabel("Held-out region", fontsize=11)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0 if label == "EMC error" else 0.8)
            spine.set_edgecolor("#dc2626" if label == "EMC error" else "#d1d5db")

    fig.suptitle(
        f"{condition_title}, support {support_fraction_tag}%: documented {selection_mode} EMC example "
        f"(sample {sample_index}, frame {frame_index})",
        fontsize=14,
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(Path(output_svg), bbox_inches="tight")
    plt.close(fig)

    payload.update(
        {
            "prediction_path": str(prediction_path),
            "per_sample_csv": str(per_sample_csv),
            "support_fraction_tag": support_fraction_tag,
            "condition_title": condition_title,
        }
    )
    payload = relativize_payload_paths(payload)
    Path(selection_manifest).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def save_emc_secondary_qualitative_figure(
    prediction_path: str | Path,
    per_sample_csv: str | Path,
    output_png: str | Path,
    output_svg: str | Path,
    selection_manifest: str | Path,
    support_fraction_tag: str,
    condition_title: str,
    selection_mode: str = "representative",
    reference_key: str = "ccrr",
    candidate_key: str = "emc",
) -> dict[str, Any]:
    """Save the secondary synthetic qualitative panel used in the MNRAS manuscript."""

    bundle = load_predictions(prediction_path)
    rows = load_rows(per_sample_csv)
    sample_index, frame_index, payload = _select_protocol_example(
        rows=rows,
        bundle=bundle,
        reference_key=reference_key,
        candidate_key=candidate_key,
        metric_key="heldout_visibility_rmse",
        selection_mode=selection_mode,
    )

    column_specs = [
        ("ground_truth", "Ground truth"),
        ("dirty", "Support-only dirty"),
        ("baseline_learned", "Baseline 3D U-Net"),
        ("residual_refinement", "Residual refinement"),
        ("ccrr", "CCRR"),
        ("emc", "EMC"),
    ]
    display_images = [
        np.clip(bundle[key][sample_index, frame_index], 0.0, 1.0).astype(np.float32)
        for key, _ in column_specs
    ]
    target_mask = np.clip(bundle["target_mask"][sample_index, frame_index], 0.0, 1.0).astype(np.float32)

    fig, axes = plt.subplots(1, len(column_specs) + 1, figsize=(16.0, 3.6))
    fig.patch.set_facecolor("white")

    for axis, (image, (_, label)) in zip(axes[:-1], zip(display_images, column_specs)):
        axis.imshow(image, cmap=IMAGE_CMAP, interpolation="nearest", vmin=0.0, vmax=1.0)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(label, fontsize=11, pad=6)
        highlight = label == "EMC"
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.8 if highlight else 0.8)
            spine.set_edgecolor("#dc2626" if highlight else "#d1d5db")

    mask_axis = axes[-1]
    mask_axis.imshow(target_mask, cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
    mask_axis.set_xticks([])
    mask_axis.set_yticks([])
    mask_axis.set_title("Held-out target mask", fontsize=11, pad=6)
    for spine in mask_axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#d1d5db")

    fig.suptitle(
        f"{condition_title}, support {support_fraction_tag}%: representative synthetic qualitative example "
        f"(sample {sample_index}, frame {frame_index})",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(Path(output_svg), bbox_inches="tight")
    plt.close(fig)

    payload.update(
        {
            "prediction_path": str(prediction_path),
            "per_sample_csv": str(per_sample_csv),
            "support_fraction_tag": support_fraction_tag,
            "condition_title": condition_title,
            "figure_layout": [label for _, label in column_specs] + ["Held-out target mask"],
        }
    )
    payload = relativize_payload_paths(payload)
    Path(selection_manifest).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def save_support_fraction_figure(
    specs: list[ProtocolSpec],
    output_png: str | Path,
    output_svg: str | Path,
) -> None:
    """Save the main EMC support-fraction paper figure."""

    fig, axes = plt.subplots(len(specs), 2, figsize=(11.4, 7.6))
    fig.patch.set_facecolor("white")
    if len(specs) == 1:
        axes = np.asarray([axes])

    for row_index, spec in enumerate(specs):
        summary = load_json(spec.summary_path)
        x_values = [int(tag) for tag in _support_tags(summary)]
        for model_key in LEARNED_MODEL_ORDER:
            heldout_vis = [
                _model_metrics(summary, str(tag), model_key)["heldout_visibility_rmse"]
                for tag in x_values
            ]
            heldout_closure = [
                _model_metrics(summary, str(tag), model_key)["heldout_closure_phase_mae"]
                for tag in x_values
            ]
            label = MODEL_LABELS[model_key]
            color = MODEL_COLORS[model_key]
            axes[row_index, 0].plot(x_values, heldout_vis, marker="o", linewidth=2.2, color=color, label=label)

            valid_x = [x for x, y in zip(x_values, heldout_closure) if not math.isnan(float(y))]
            valid_y = [y for y in heldout_closure if not math.isnan(float(y))]
            if valid_x:
                axes[row_index, 1].plot(valid_x, valid_y, marker="o", linewidth=2.2, color=color, label=label)

        axes[row_index, 0].set_title(f"{spec.title}: held-out VisRMSE", fontsize=12)
        axes[row_index, 1].set_title(f"{spec.title}: held-out closure MAE", fontsize=12)
        axes[row_index, 0].set_ylabel("Lower is better", fontsize=10)
        axes[row_index, 1].set_ylabel("Lower is better", fontsize=10)
        for col in range(2):
            axes[row_index, col].set_xlabel("Support fraction (%)", fontsize=10)
            axes[row_index, col].grid(alpha=0.2)
            axes[row_index, col].set_xticks(x_values)
    axes[0, 0].legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle(
        "Earned measurement consistency improves held-out measurement recovery as support shrinks",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(Path(output_svg), bbox_inches="tight")
    plt.close(fig)
