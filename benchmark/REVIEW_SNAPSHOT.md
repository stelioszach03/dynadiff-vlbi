# Review Snapshot and Reproducibility Scope

This note defines the repository snapshot sufficient to inspect and reproduce the benchmark and public-EHT results reported in the MNRAS manuscript.

## Included source components

- core source code under [`../src/`](../src)
- resolved experiment configurations under [`../configs/`](../configs)
- benchmark and public-validation documentation under [`./`](.)
- run scripts and artifact builders under [`../scripts/`](../scripts)
- regression tests under [`../tests/`](../tests)

## Included paper-facing artifacts

- synthetic benchmark artifacts under [`../outputs/emc_benchmark_artifacts/`](../outputs/emc_benchmark_artifacts)
- public-EHT suite artifacts under [`../outputs/public_eht_suite_artifacts/`](../outputs/public_eht_suite_artifacts)
- MNRAS real-data and ablation artifacts under [`../outputs/mnras_real_data_artifacts/`](../outputs/mnras_real_data_artifacts)
- manuscript source and exported PDF under [`../paper/`](../paper)

## Public real-data releases used

- `2019-D01-01` — `M87 2017`
- `2024-D01-01` — `M87 2018`
- `2020-D01-01` — `3C279 2017`
- `2021-D03-01` — `Centaurus A 2017`

## Reproduction commands

Synthetic benchmark:

```bash
python3.11 scripts/run_emc_benchmark.py --target all --skip-existing
python3.11 scripts/generate_emc_benchmark_artifacts.py
```

Public-EHT suite:

```bash
python3.11 scripts/run_emc_public_eht_suite.py --skip-existing
python3.11 scripts/generate_public_eht_suite_artifacts.py
```

MNRAS paper artifacts and PDF:

```bash
python3.11 scripts/run_emc_ablation_protocol.py --skip-existing
python3.11 scripts/generate_mnras_real_data_artifacts.py
python3.11 scripts/export_paper_pdf.py --input paper/manuscript.md --bibliography paper/references.bib --output paper/manuscript.pdf
```

## Scope note

The public-EHT evaluation is observation-domain only. The snapshot is intended to make the benchmark, manifests, figures, tables, and manuscript claims inspectable by editors and referees without implying a telescope-accurate end-to-end EHT pipeline.
