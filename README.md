# dynadiff-vlbi

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

Benchmark-first dynamic VLBI evaluation framework for **earned versus enforced measurement consistency**, with public Event Horizon Telescope release validation.

---

## Overview

`dynadiff-vlbi` accompanies the accompanying manuscript on dynamic Very Long Baseline Interferometry (VLBI) imaging. It packages:

- A released **64×64 EMC benchmark** with deterministic support / target split manifests.
- A compact reference implementation of **Earned Measurement Consistency (EMC)** alongside baseline models.
- **Public EHT validation** over the official calibrated-data releases for M87 (2017, 2018), 3C279 (2017), and Centaurus A (2017).
- The paper-facing artifact builders that regenerate every figure and table in the manuscript directly from CSV/JSON sources.

The goal is a clean, reproducible infrastructure for benchmark-style evaluation — not a full production EHT or ngEHT pipeline.

## Installation

Requires Python 3.10+. Validated with Python 3.11 on CPU and CUDA.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Quick start

Run the EMC benchmark end-to-end (support-target sweep, structured holdouts, challenge-inspired realism):

```bash
python scripts/run_emc_benchmark.py --target all --skip-existing
```

Regenerate every figure and table in the paper from the produced CSV / JSON artifacts:

```bash
python scripts/generate_emc_benchmark_artifacts.py
python scripts/generate_public_eht_suite_artifacts.py
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the end-to-end reproduction recipe, including seed control, compute requirements, and expected wall-clock times.

## Project structure

```text
dynadiff-vlbi/
├── src/dynadiff_vlbi/     # Library: models, physics, training, evaluation, EMC
├── scripts/               # CLI entry points for datasets, training, benchmarks, artifacts
├── configs/               # YAML configs for every preset (32x32, 64x64, EMC, Phase 2, CCRR)
├── tests/                 # Pytest suite
├── notebooks/             # Colab-friendly walkthroughs
├── benchmark/             # EMC benchmark release notes and reproducibility snapshot
├── paper/                 # Manuscript PDF, LaTeX sources, figures, tables, references
└── theory/                # Consistency-bound derivations and supporting figures
```

Generated artifacts (`outputs/`, `data/generated/`, `data/external/`, `data/real/`) are intentionally git-ignored and reproduced locally by the scripts above.

## Documentation

- [`benchmark/README.md`](benchmark/README.md) — EMC benchmark release, splits, and evaluation protocol.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — exact reproduction steps, seeds, and expected runtimes.
- [`paper/manuscript.pdf`](paper/manuscript.pdf) — current manuscript draft.

## Data availability

The public validation uses official EHT calibrated-data releases:

| Release | Target | DOI |
|:--|:--|:--|
| `2019-D01-01` | M87 (2017) | [`10.25739/g85n-f134`](https://doi.org/10.25739/g85n-f134) |
| `2024-D01-01` | M87 (2018 calibrated data) | [`10.25739/epm5-r371`](https://doi.org/10.25739/epm5-r371) |
| `2020-D01-01` | 3C279 (2017) | [`10.25739/vty0-ve39`](https://doi.org/10.25739/vty0-ve39) |
| `2021-D03-01` | Centaurus A (2017) | [`10.25739/kejs-2n22`](https://doi.org/10.25739/kejs-2n22) |

## Citation

```bibtex
@unpublished{zacharioudakis2026earned,
  author = {Zacharioudakis, Stylianos Georgios},
  title  = {A reproducible earned-versus-enforced benchmark for dynamic VLBI},
  year   = {2026},
  note   = {Manuscript}
}
```

Machine-readable metadata is also available in [`CITATION.cff`](CITATION.cff).

## License

Released under the [MIT License](LICENSE).
