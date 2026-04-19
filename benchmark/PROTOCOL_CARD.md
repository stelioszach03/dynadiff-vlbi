# Protocol Card

## Deterministic components

- dataset split: fixed generated split per condition
- project seed: `7`
- support fractions: `80%`, `60%`, `40%`, `20%`
- comparator suite: dirty, Tikhonov, baseline 3D U-Net, residual refinement, CCRR, EMC
- holdout families: deterministic and config-driven

## Holdout families

### `baseline_track_blocks`

Observed coefficients are grouped by baseline track and ordered deterministically by average baseline length and angle. A contiguous block of tracks is withheld across time.

### `scan_segment_blocks`

Contiguous temporal scan-like windows are withheld deterministically. This stresses time-localized missingness rather than baseline identity.

### `station_dropout`

Deterministic subsets of stations are withheld together with all incident baselines. This stresses station-structured missingness.

### `challenge_inspired_realism`

This track reuses the earned-consistency protocol but applies it in a public-style corruption regime with:

- station-track sampling
- scan gaps
- per-station gain corruption
- baseline-dependent noise heterogeneity

It is challenge-inspired, not the private ngEHT Challenge #2 dataset.

## Fairness rules

- all learned methods are evaluated on the same support-target partitions
- dirty and Tikhonov reconstructions are rebuilt from the same support-only set
- held-out metrics are always reported separately from support-set metrics
- held-out closure is reported only when all-target triangle support is sufficient

## Required reporting

- support fraction
- holdout family
- held-out visibility RMSE
- held-out closure-phase MAE when defined
- support visibility RMSE
- MSE
- SSIM
- temporal consistency
- config manifest path
- split manifest path

## Minimal rerun surface

One-command benchmark reproduction:

```bash
python3.11 scripts/run_emc_benchmark.py --target all --skip-existing && \
python3.11 scripts/generate_emc_benchmark_artifacts.py
```

Deterministic split export only:

```bash
python3.11 scripts/export_emc_split_manifests.py \
  --base-config configs/emc_benchmark_baseline_tracks_default32.yaml \
  --data-dir data/generated/ccrr_default32_seed7_shared \
  --output-dir outputs/emc_benchmark_release/split_manifests/baseline_tracks
```
