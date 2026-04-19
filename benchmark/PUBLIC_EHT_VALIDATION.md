# Public EHT Observation-Domain Validation

This document records the public-EHT validation path used by the MNRAS manuscript.

The public suite is complementary to the fixed synthetic EMC benchmark. It keeps the same support-target logic, but applies it to official public Event Horizon Telescope calibrated-data releases where no image-domain ground truth exists.

## Official public releases used

- `2019-D01-01` — `M87 2017`, HOPS netcal Stokes I
- `2024-D01-01` — `M87 2018`, HOPS netcal 10 s Stokes I
- `2020-D01-01` — `3C279 2017`, HOPS netcal Stokes I
- `2021-D03-01` — `Centaurus A 2017`, HOPS netcal Stokes I

## Public protocol families

- `baseline_track_blocks` as the primary public benchmark family
- `station_dropout` as the structured sensitivity check

Support fractions are fixed at `80%`, `60%`, `40%`, and `20%`.

## Comparator set on the public suite

- Dirty
- Tikhonov
- `eht-imaging bridge`
- baseline 3D U-Net
- standalone visibility-conditioned model
- residual refinement
- CCRR
- EMC

The `eht-imaging bridge` is a frozen support-only wrapper that maps the benchmark's deterministic gridded support measurements into a pseudo-`Obsdata` interface so that `ehtim` can be scored by the same held-out evaluator. It is a fair benchmark bridge, not a claim of telescope-accurate raw-observation parity.

## One-command reproduction

```bash
python3.11 scripts/run_emc_public_eht_suite.py --skip-existing
python3.11 scripts/generate_public_eht_suite_artifacts.py
```

This path:

- downloads or reuses the official public EHT release repositories
- prepares deterministic public-EHT support-target datasets
- runs the public suite across releases and targets
- includes the frozen `eht-imaging bridge` comparator
- exports paper-facing tables and figures

## Core outputs

- suite manifest:
  - [`../outputs/public_eht_suite/suite_manifest.json`](../outputs/public_eht_suite/suite_manifest.json)
- benchmark matrix:
  - [`../outputs/public_eht_suite_artifacts/tables/emc_public_eht_matrix.md`](../outputs/public_eht_suite_artifacts/tables/emc_public_eht_matrix.md)
- release robustness summary:
  - [`../outputs/public_eht_suite_artifacts/tables/emc_public_eht_release_robustness.md`](../outputs/public_eht_suite_artifacts/tables/emc_public_eht_release_robustness.md)
- paired-bootstrap summary:
  - [`../outputs/public_eht_suite_artifacts/tables/emc_public_eht_stats.md`](../outputs/public_eht_suite_artifacts/tables/emc_public_eht_stats.md)
- day/band stratified summary:
  - [`../outputs/public_eht_suite_artifacts/tables/emc_public_eht_day_band.md`](../outputs/public_eht_suite_artifacts/tables/emc_public_eht_day_band.md)

## Interpretation guardrail

The public suite is observation-domain only. It is intended to test whether the benchmark question remains meaningful on released EHT measurement products and to expose synthetic-to-public transfer behaviour. It does **not** provide image-domain morphology validation.
