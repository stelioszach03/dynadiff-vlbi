"""Session 1 analysis: parse all det/adaptive protocol summaries, compute
per-condition diagnostics, emit figures + LaTeX table.

Inputs (from prior pipeline stages):
    outputs/emc_benchmark_{family}_protocol_{deterministic,adaptive}/logs/emc_protocol_summary.json
    outputs/public_eht_suite_{deterministic,adaptive}/{family}/{release_slug}/logs/real_data_protocol_summary.json
    runs/oracle_v1/history.jsonl  (for recall-vs-epoch)

Outputs:
    figures_out/theory_empirics_alignment.pdf
    figures_out/transfer_gap_closure.pdf
    paper/tables/adaptive_partition_results.tex
    /tmp/session1_summary.json  (machine-readable dump)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures_out"
TBL_DIR = ROOT / "paper" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

METRIC_KEY = "heldout_visibility_rmse"   # lower = better; key EMC target
SUPPORT_FRACTIONS = (20, 40, 60, 80)      # percent
SYNTHETIC_FAMILIES = ("baseline_tracks", "scan_segments", "station_dropout")
EHT_FAMILIES = ("baseline_track_blocks", "station_dropout")
EHT_RELEASES = (
    ("2019-D01-01", "M87", 2017, "m87_2017_2019-d01-01"),
    ("2024-D01-01", "M87", 2018, "m87_2018_2024-d01-01"),
    ("2020-D01-01", "3C279", 2017, "3c279_2017_2020-d01-01"),
    ("2021-D03-01", "CenA", 2017, "cena_2017_2021-d03-01"),
)

# Default32 dimensions (for μ²k/αm):
IMAGE_SIZE = 32
N_SIGNAL = IMAGE_SIZE * IMAGE_SIZE   # = 1024
K_EFFECTIVE = 32                      # ≈ sqrt(n); reasonable effective sparsity for ring sources
MU_SQUARED = 1.0                      # partial-DFT canonical coherence squared


# ---------------------------------------------------------------------------
# Synthetic benchmark extraction
# ---------------------------------------------------------------------------


def _load_synth(mode: str) -> dict:
    """Parse all (family, support_fraction, model) points for deterministic/adaptive."""
    results = {}
    for family in SYNTHETIC_FAMILIES:
        path = ROOT / "outputs" / f"emc_benchmark_{family}_protocol_{mode}" / "logs" / "emc_protocol_summary.json"
        if not path.exists():
            print(f"MISSING {path}")
            continue
        summary = json.loads(path.read_text())
        for sf_key, sf_block in summary["support_fractions"].items():
            sf = int(sf_key)
            for model_key, m in sf_block["models"].items():
                if METRIC_KEY not in m:
                    continue
                val = m[METRIC_KEY]
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    continue
                results[(family, sf, model_key)] = float(val)
    return results


def _load_eht(mode: str) -> dict:
    results = {}
    for family in EHT_FAMILIES:
        for release_code, target, year, slug in EHT_RELEASES:
            path = (
                ROOT / "outputs" / f"public_eht_suite_{mode}" / family / slug
                / "logs" / "real_data_protocol_summary.json"
            )
            if not path.exists():
                print(f"MISSING {path}")
                continue
            summary = json.loads(path.read_text())
            # Same shape as synthetic: support_fractions.{level}.models.{name}.metric
            for sf_key, sf_block in summary.get("support_fractions", {}).items():
                sf = int(sf_key)
                for model_key, m in sf_block.get("models", {}).items():
                    if METRIC_KEY not in m:
                        continue
                    val = m[METRIC_KEY]
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        continue
                    results[(release_code, target, year, family, sf, model_key)] = float(val)
    return results


# ---------------------------------------------------------------------------
# Per-condition diagnostic quantities
# ---------------------------------------------------------------------------


def _condition_diagnostics(sf: int, mean_m: float | None = None) -> dict:
    alpha = sf / 100.0
    if mean_m is None:
        # default fallback (rough): for default32, typically ~60 observed per frame
        mean_m = 60.0
    alpha_m = max(alpha * mean_m, 1.0)
    mu2_k_over_alpha_m = MU_SQUARED * K_EFFECTIVE / alpha_m
    # Effective posterior rank ≈ min(m_sup, n) / n
    eff_rank = min(alpha_m, N_SIGNAL) / N_SIGNAL
    return {
        "alpha": alpha,
        "alpha_m": alpha_m,
        "mu2_k_over_alpha_m": mu2_k_over_alpha_m,
        "effective_posterior_rank_ratio": eff_rank,
    }


def _mean_support_count(family: str, sf: int, mode: str) -> float | None:
    path = ROOT / "outputs" / f"emc_benchmark_{family}_protocol_{mode}" / "logs" / "emc_protocol_summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    sf_block = summary["support_fractions"].get(str(sf))
    if sf_block is None:
        return None
    # mean support + target counts = total observed per sample (baselines or frames or stations)
    sup = sf_block.get("mean_support_unit_count")
    tgt = sf_block.get("mean_target_unit_count")
    if sup is None or tgt is None:
        return None
    return float(sup) + float(tgt)


# ---------------------------------------------------------------------------
# Build the per-condition rows
# ---------------------------------------------------------------------------


det = _load_synth("deterministic")
ada = _load_synth("adaptive")
det_eht = _load_eht("deterministic")
ada_eht = _load_eht("adaptive")

synth_rows = []
for family in SYNTHETIC_FAMILIES:
    for sf in SUPPORT_FRACTIONS:
        # Compare EMC (key metric) vs best baseline under the same condition.
        emc_det = det.get((family, sf, "emc"))
        bl_det = det.get((family, sf, "baseline_learned"))
        emc_ada = ada.get((family, sf, "emc"))
        bl_ada = ada.get((family, sf, "baseline_learned"))
        mean_m = _mean_support_count(family, sf, "deterministic")
        diag = _condition_diagnostics(sf, mean_m)
        row = {
            "family": family,
            "support_fraction": sf,
            "mean_observations": mean_m,
            **diag,
            "emc_det": emc_det,
            "baseline_det": bl_det,
            "emc_ada": emc_ada,
            "baseline_ada": bl_ada,
            "gap_det": (bl_det - emc_det) if (emc_det is not None and bl_det is not None) else None,
            "gap_ada": (bl_ada - emc_ada) if (emc_ada is not None and bl_ada is not None) else None,
        }
        row["gap_improvement"] = (
            (row["gap_ada"] - row["gap_det"]) if (row["gap_det"] is not None and row["gap_ada"] is not None) else None
        )
        synth_rows.append(row)

eht_rows = []
for release_code, target, year, slug in EHT_RELEASES:
    for family in EHT_FAMILIES:
        for sf in SUPPORT_FRACTIONS:
            emc_det = det_eht.get((release_code, target, year, family, sf, "emc"))
            bl_det = det_eht.get((release_code, target, year, family, sf, "baseline_learned"))
            emc_ada = ada_eht.get((release_code, target, year, family, sf, "emc"))
            bl_ada = ada_eht.get((release_code, target, year, family, sf, "baseline_learned"))
            diag = _condition_diagnostics(sf, mean_m=None)
            row = {
                "release": release_code,
                "target": target,
                "year": year,
                "family": family,
                "support_fraction": sf,
                **diag,
                "emc_det": emc_det,
                "baseline_det": bl_det,
                "emc_ada": emc_ada,
                "baseline_ada": bl_ada,
                "gap_det": (bl_det - emc_det) if (emc_det is not None and bl_det is not None) else None,
                "gap_ada": (bl_ada - emc_ada) if (emc_ada is not None and bl_ada is not None) else None,
            }
            row["gap_improvement"] = (
                (row["gap_ada"] - row["gap_det"]) if (row["gap_det"] is not None and row["gap_ada"] is not None) else None
            )
            eht_rows.append(row)


# ---------------------------------------------------------------------------
# Quick text report
# ---------------------------------------------------------------------------


def _fmt(x, d=4):
    return "   nan" if x is None else f"{x:>7.{d}f}"


print("=" * 110)
print("SYNTHETIC BENCHMARK (12 conditions): deterministic vs adaptive")
print("=" * 110)
print(f"{'family':<18} {'sf':>4} {'mean_m':>7} {'μ²k/αm':>8} {'emc_det':>10} {'bl_det':>10} {'gap_det':>10} {'emc_ada':>10} {'bl_ada':>10} {'gap_ada':>10} {'Δgap':>10}")
for r in synth_rows:
    print(f"{r['family']:<18} {r['support_fraction']:>4} "
          f"{_fmt(r['mean_observations'], 1)} "
          f"{_fmt(r['mu2_k_over_alpha_m'], 2)} "
          f"{_fmt(r['emc_det'])} {_fmt(r['baseline_det'])} {_fmt(r['gap_det'])} "
          f"{_fmt(r['emc_ada'])} {_fmt(r['baseline_ada'])} {_fmt(r['gap_ada'])} "
          f"{_fmt(r['gap_improvement'])}")

print()
print("=" * 110)
print("PUBLIC EHT (4 releases x 2 families x 4 sf = 32 conditions): deterministic vs adaptive")
print("=" * 110)
print(f"{'release':<15} {'target':<8} {'family':<22} {'sf':>4} "
      f"{'emc_det':>10} {'bl_det':>10} {'gap_det':>10} {'emc_ada':>10} {'bl_ada':>10} {'gap_ada':>10} {'Δgap':>10}")
for r in eht_rows:
    print(f"{r['release']:<15} {r['target']:<8} {r['family']:<22} {r['support_fraction']:>4} "
          f"{_fmt(r['emc_det'])} {_fmt(r['baseline_det'])} {_fmt(r['gap_det'])} "
          f"{_fmt(r['emc_ada'])} {_fmt(r['baseline_ada'])} {_fmt(r['gap_ada'])} "
          f"{_fmt(r['gap_improvement'])}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


# 1) theory_empirics_alignment.pdf
#    X = μ²k/αm ; Y = gap improvement from adaptive.
pts_x = []; pts_y = []; pts_label = []
for r in synth_rows + eht_rows:
    if r.get("gap_improvement") is None or r.get("mu2_k_over_alpha_m") is None:
        continue
    pts_x.append(r["mu2_k_over_alpha_m"])
    pts_y.append(r["gap_improvement"])
    pts_label.append("synthetic" if "release" not in r else f"EHT_{r['target']}")

fig, ax = plt.subplots(figsize=(7.2, 5.0))
xs = np.asarray(pts_x); ys = np.asarray(pts_y)
# Color by synth vs EHT
is_eht = np.asarray([l != "synthetic" for l in pts_label])
ax.scatter(xs[~is_eht], ys[~is_eht], s=55, marker="o", edgecolor="k", facecolor="#3b82f6",
           label=f"Synthetic (n={int((~is_eht).sum())})")
ax.scatter(xs[is_eht], ys[is_eht], s=55, marker="s", edgecolor="k", facecolor="#ef4444",
           label=f"Public EHT (n={int(is_eht.sum())})")
ax.axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.6)
ax.set_xscale("log")
ax.set_xlabel(r"Theorem 2 condition number  $\mu^2 k / (\alpha m)$")
ax.set_ylabel(r"Gap improvement from adaptive  $\Delta_{\mathrm{adap}} - \Delta_{\mathrm{det}}$")
ax.set_title(
    "Theory - empirics alignment\n"
    r"Theorem 2 predicts adaptive helps only when $\mu^2 k/(\alpha m) \ll 1$."
    "\nAt default32, all conditions sit above 1 => no improvement expected."
)
ax.legend(loc="best", fontsize=9)
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIG_DIR / "theory_empirics_alignment.pdf", dpi=150)
plt.close(fig)
print(f"\nwrote {FIG_DIR / 'theory_empirics_alignment.pdf'}")

# 2) transfer_gap_closure.pdf
# Show per-condition bar chart: deterministic gap vs adaptive gap (EHT releases + synthetic families).
fig, axes = plt.subplots(1, 2, figsize=(14, 5.3))

# --- left: synthetic ---
ax = axes[0]
keys = [(r["family"], r["support_fraction"]) for r in synth_rows]
gap_det_vals = [(r["gap_det"] if r["gap_det"] is not None else 0.0) for r in synth_rows]
gap_ada_vals = [(r["gap_ada"] if r["gap_ada"] is not None else 0.0) for r in synth_rows]
x = np.arange(len(keys))
w = 0.40
ax.bar(x - w / 2, gap_det_vals, width=w, color="#3b82f6", label="deterministic")
ax.bar(x + w / 2, gap_ada_vals, width=w, color="#ef4444", alpha=0.85, label="adaptive (oracle)")
ax.axhline(0.0, color="gray", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{k[0][:5]}\nα={k[1]}" for k in keys], fontsize=8, rotation=0)
ax.set_ylabel(r"EMC gap: baseline_learned $-$ EMC  ($\uparrow$ = EMC better)")
ax.set_title("Synthetic benchmark (12 conditions, default32)")
ax.legend(fontsize=9, loc="best")
ax.grid(True, axis="y", alpha=0.3)

# --- right: EHT ---
ax = axes[1]
keys_e = [(r["target"], r["year"], r["family"][:9], r["support_fraction"]) for r in eht_rows]
gap_det_vals_e = [(r["gap_det"] if r["gap_det"] is not None else 0.0) for r in eht_rows]
gap_ada_vals_e = [(r["gap_ada"] if r["gap_ada"] is not None else 0.0) for r in eht_rows]
x = np.arange(len(keys_e))
ax.bar(x - w / 2, gap_det_vals_e, width=w, color="#3b82f6", label="deterministic")
ax.bar(x + w / 2, gap_ada_vals_e, width=w, color="#ef4444", alpha=0.85, label="adaptive (oracle)")
ax.axhline(0.0, color="gray", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{k[0]}\n{k[1]}\nα={k[3]}" for k in keys_e], fontsize=7, rotation=0)
ax.set_ylabel(r"EMC gap: baseline_learned $-$ EMC")
ax.set_title("Public EHT (4 releases × 2 families × 4 sf = 32 conditions)")
ax.legend(fontsize=9, loc="best")
ax.grid(True, axis="y", alpha=0.3)

fig.suptitle(
    "Adaptive vs deterministic partition under the earned-consistency protocol "
    "(default32 regime; Theorem 2 oracle condition not met)",
    fontsize=11, y=1.02,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "transfer_gap_closure.pdf", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {FIG_DIR / 'transfer_gap_closure.pdf'}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------


def _fmt_cell(x, d=3):
    return "" if x is None else f"{x:.{d}f}"

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Session 1 (default32) adaptive- vs deterministic-partition results. "
             r"Columns: EMC - baseline gap on held-out visibility RMSE (positive = EMC better). "
             r"$\mu^2 k / (\alpha m)$ is the Theorem 2 condition number; "
             r"Theorem 2 predicts adaptive helps only when this quantity is $\ll 1$.}")
lines.append(r"\label{tab:adaptive_partition_results}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{llrrrr}")
lines.append(r"\hline")
lines.append(r"Condition & Family & $\alpha$ & $\mu^2 k/(\alpha m)$ & $\Delta_{\text{det}}$ & $\Delta_{\text{adap}}$ \\")
lines.append(r"\hline")
lines.append(r"\multicolumn{6}{l}{\textit{Synthetic (default32, 3 families $\times$ 4 $\alpha$)}} \\")
for r in synth_rows:
    lines.append(
        f"  & {r['family'].replace('_', ' ')} & "
        f"{r['support_fraction']/100:.2f} & "
        f"{_fmt_cell(r['mu2_k_over_alpha_m'], 2)} & "
        f"{_fmt_cell(r['gap_det'], 4)} & "
        f"{_fmt_cell(r['gap_ada'], 4)} \\\\"
    )
lines.append(r"\hline")
lines.append(r"\multicolumn{6}{l}{\textit{Public EHT (4 releases $\times$ 2 families $\times$ 4 $\alpha$)}} \\")
for r in eht_rows:
    lines.append(
        f"{r['target']} {r['year']} & "
        f"{r['family'].replace('_', ' ')} & "
        f"{r['support_fraction']/100:.2f} & "
        f"{_fmt_cell(r['mu2_k_over_alpha_m'], 2)} & "
        f"{_fmt_cell(r['gap_det'], 4)} & "
        f"{_fmt_cell(r['gap_ada'], 4)} \\\\"
    )
lines.append(r"\hline")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
tex_path = TBL_DIR / "adaptive_partition_results.tex"
tex_path.write_text("\n".join(lines))
print(f"wrote {tex_path}")


# ---------------------------------------------------------------------------
# JSON dump
# ---------------------------------------------------------------------------


json_out = Path("/tmp/session1_summary.json")
json_out.write_text(json.dumps({
    "synthetic_rows": synth_rows,
    "eht_rows": eht_rows,
}, indent=2))
print(f"wrote {json_out}")

# Headline numbers
def _mean(xs): xs = [x for x in xs if x is not None]; return float(np.mean(xs)) if xs else float("nan")
print()
print("HEADLINE DELTAS (gap_adaptive - gap_deterministic, positive = adaptive helped)")
print(f"  synthetic: mean = {_mean([r['gap_improvement'] for r in synth_rows]):+.4f}  "
      f"n_improved = {sum(1 for r in synth_rows if r.get('gap_improvement') and r['gap_improvement'] > 0)}/"
      f"{sum(1 for r in synth_rows if r.get('gap_improvement') is not None)}")
print(f"  public EHT: mean = {_mean([r['gap_improvement'] for r in eht_rows]):+.4f}  "
      f"n_improved = {sum(1 for r in eht_rows if r.get('gap_improvement') and r['gap_improvement'] > 0)}/"
      f"{sum(1 for r in eht_rows if r.get('gap_improvement') is not None)}")
