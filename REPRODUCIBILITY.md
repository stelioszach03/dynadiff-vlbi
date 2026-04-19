# Reproducibility Map

This document maps the final `manuscript_v2` claims to exact commands and artifact locations.

## Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

## Table 1 and Figure 2: Default64 synthetic breadth

Claim:

- The released default64 breadth matrix favors EMC across all 12 family-support cells on held-out visibility RMSE, while the bounded add-on cycle reports DPS on the rerun baseline-track family only and EMC conformal UQ as empirical coverage plus mean interval width.

Commands:

```bash
python3.11 scripts/run_emc_benchmark.py --target all --skip-existing
python3.11 scripts/train_dps_baseline.py --data-dir data/generated/ccrr_default64_seed7_shared --output-root outputs --run-name dps_default64_baseline_tracks
python3.11 scripts/run_emc_benchmark.py --target baseline_tracks --python python3.11 --output-root outputs --data-root data/generated --dps-checkpoint outputs/dps_default64_baseline_tracks/checkpoints/best.pt
python3.11 scripts/run_emc_conformal_uq.py
python3.11 scripts/generate_emc_benchmark_artifacts.py --uq-root outputs/emc_conformal_uq --dps-artifact-root outputs/dps_benchmark_artifacts
```

Artifacts:

- `outputs/emc_benchmark_artifacts/tables/emc_benchmark_matrix.csv`
- `outputs/emc_benchmark_artifacts/tables/emc_benchmark_bootstrap.csv`
- `outputs/dps_benchmark_artifacts/synthetic_dps_table.csv`
- `outputs/emc_conformal_uq/tables/synthetic_emc_conformal_uq.csv`
- `paper/tables/table01_default64_benchmark_matrix.tex`
- `paper/figures/fig02_emc_benchmark_support_curve.png`

## Figure 3: challenge-inspired realism

Claim:

- EMC remains strongest on held-out visibility RMSE across the challenge-inspired realism sweep, while residual refinement regains the strongest SSIM at higher support fractions.

Commands:

```bash
python3.11 scripts/run_emc_benchmark.py --target challenge_inspired_realism --skip-existing
python3.11 scripts/generate_emc_benchmark_artifacts.py
```

Artifacts:

- `outputs/emc_benchmark_artifacts/tables/emc_challenge_inspired_realism.csv`
- `paper/figures/fig03_emc_challenge_inspired_realism.png`

## Tables 2--4 and Figures 4--5: public EHT suite

Claim:

- The public-EHT suite is release-aware and mixed: TTO materially helps some releases, harms others, and does not justify a broad real-data dominance claim. The add-on cycle extends the baseline-track release table with DPS and reports MIW only for public conformal UQ.

Commands:

```bash
python3.11 scripts/run_emc_public_eht_suite.py --skip-existing
python3.11 scripts/run_emc_public_eht_suite.py --families baseline_track_blocks --output-root outputs/public_eht_suite --dps-checkpoint outputs/dps_default64_baseline_tracks/checkpoints/best.pt
python3.11 scripts/generate_public_eht_suite_artifacts.py --uq-root outputs/emc_conformal_uq --dps-artifact-root outputs/dps_benchmark_artifacts
```

Artifacts:

- `outputs/public_eht_suite/suite_manifest.json`
- `outputs/public_eht_suite_artifacts/tables/emc_public_eht_matrix.csv`
- `outputs/public_eht_suite_artifacts/tables/emc_public_eht_release_robustness.csv`
- `outputs/public_eht_suite_artifacts/tables/emc_public_eht_release_gaps.csv`
- `outputs/public_eht_suite_artifacts/tables/emc_public_eht_stats.csv`
- `outputs/dps_benchmark_artifacts/public_dps_release_summary.csv`
- `outputs/emc_conformal_uq/tables/public_emc_conformal_uq.csv`
- `paper/tables/table02_public_eht_release_means.tex`
- `paper/tables/table03_public_family_robustness.tex`
- `paper/tables/table04_public_bootstrap.tex`
- `paper/figures/fig06_emc_public_eht_suite.png`
- `paper/figures/fig07_emc_public_transfer_gap.png`

## Table 5: bounded five-seed robustness

Claim:

- The default64 breadth matrix does not by itself establish a universally seed-stable learned-method ordering. On the core default64 baseline-track family over seeds `7, 19, 31, 42, 137`, residual refinement is the strongest mean learned comparator at every support fraction.

Commands:

```bash
python3.11 scripts/run_emc_seed_robustness.py --skip-existing --seeds 7,19,31,42,137
python3.11 scripts/generate_emc_seed_robustness_artifacts.py
```

Artifacts:

- `outputs/emc_seed_robustness/seed_robustness_manifest.json`
- `outputs/emc_seed_robustness_artifacts/tables/emc_seed_robustness_summary.csv`
- `outputs/emc_seed_robustness_artifacts/tables/emc_seed_robustness_stats.csv`
- `paper/tables/table05_seed_robustness.tex`

## Figure 6: theoretical motivation

Claim:

- The stylized enforcement-versus-earning gap decreases as the support fraction increases, consistent with a support-conditioned uncertainty interpretation.

Commands:

```bash
python3.11 theory/consistency_bound.py
```

Artifacts:

- `theory/PROOF_SKETCH.md`
- `theory/consistency_bound_results.json`
- `paper/figures/fig06_consistency_bound.png`
- `paper/figures/fig06_consistency_bound.pdf`

## Reproducibility gate

Command:

```bash
python3.11 scripts/verify_reproducibility.py
```

Artifacts:

- `benchmark/reproducibility_check.json`

## Final manuscript PDF

Command:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode manuscript_v2.tex
```

Artifacts:

- `paper/manuscript_v2.pdf`
- `paper/manuscript.pdf`

## Workshop abstract PDF

Command:

```bash
cd paper/workshop
latexmk -pdf -interaction=nonstopmode ml4ps_abstract.tex
```

Artifacts:

- `paper/workshop/neurips_2025.sty`
- `paper/workshop/ml4ps_abstract.tex`
- `paper/workshop/ml4ps_abstract.pdf`

- `paper/manuscript_v2.tex`
- `paper/manuscript_v2.pdf`

## Non-promoted bounded transfer study

Command:

```bash
python3.11 scripts/run_emc_public_eht_suite.py --families baseline_track_blocks --output-root outputs/public_eht_transfer_variant --emc-checkpoint outputs/emc_benchmark_challenge_inspired_realism_main_noclosure/checkpoints/best.pt --skip-existing
```

Interpretation:

- This one-shot transfer-study variant remains in the audit trail.
- It is not promoted into the paper because it worsened both pooled public baseline-track EMC and the matched synthetic baseline-track EMC check.

## Public releases used

- `2019-D01-01` — `M87 2017`
- `2024-D01-01` — `M87 2018`
- `2020-D01-01` — `3C279 2017`
- `2021-D03-01` — `Centaurus A 2017`
