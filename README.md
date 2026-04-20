# dynadiff-vlbi

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

Evaluation framework and score-based diffusion reference implementation for **earned versus enforced measurement consistency** in dynamic VLBI imaging.

---

## Overview

`dynadiff-vlbi` accompanies the manuscript *"Earned versus enforced measurement consistency in dynamic VLBI imaging: an evaluation framework with score-based posterior sampling"*.

The repository provides:

- A deterministic **benchmark protocol** that partitions observed visibilities into support and target sets, restricting both model input and data-consistency projection to the support set and evaluating on genuinely unseen measurements.
- **DynaDiff**, a conditional score-based diffusion model for dynamic image sequences that conditions on sparse Fourier measurements through cross-attention and applies measurement-consistency guidance exclusively on the support partition.
- Benchmark runs at 128×128 resolution across three structured holdout families and four support fractions.
- Public-release validation on four official EHT calibrated-data releases: M87 (2017, 2018), 3C 279 (2017), and Centaurus A (2017).

The manuscript and compiled PDFs live under [`paper/`](paper/).

## Installation

Requires Python 3.10+. Validated with Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Quick start

Run the benchmark end-to-end:

```bash
python scripts/run_emc_benchmark.py --target all --skip-existing
```

Regenerate every figure and table in the paper from CSV/JSON artifacts:

```bash
python scripts/generate_emc_benchmark_artifacts.py
python scripts/generate_public_eht_suite_artifacts.py
```

## Project structure

```text
dynadiff-vlbi/
├── src/dynadiff_vlbi/     Library: models, physics, training, evaluation, utils
├── scripts/               CLI entry points for datasets, training, benchmarks, artifacts
├── configs/               YAML presets for every benchmark and training run
├── tests/                 Pytest suite
├── notebooks/             Colab walkthroughs
├── benchmark/             Benchmark release notes and reproducibility snapshot
├── paper/                 Manuscript (Markdown + rendered PDFs), figures, tables
└── theory/                Consistency-bound proof sketch and numerical verification
```

Generated artifacts (`outputs/`, `data/generated/`, `data/external/`, `data/real/`) are git-ignored and reproduced locally by the scripts above.

## Data availability

Public-release validation uses official EHT calibrated-data releases:

| Release | Target | DOI |
|:--|:--|:--|
| `2019-D01-01` | M87 (2017) | [`10.25739/g85n-f134`](https://doi.org/10.25739/g85n-f134) |
| `2024-D01-01` | M87 (2018 calibrated data) | [`10.25739/epm5-r371`](https://doi.org/10.25739/epm5-r371) |
| `2020-D01-01` | 3C 279 (2017) | [`10.25739/vty0-ve39`](https://doi.org/10.25739/vty0-ve39) |
| `2021-D03-01` | Centaurus A (2017) | [`10.25739/kejs-2n22`](https://doi.org/10.25739/kejs-2n22) |

## License

Released under the [MIT License](LICENSE).
