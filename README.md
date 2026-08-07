# dynadiff-vlbi

An evaluation protocol for radio-interferometric imaging that holds out visibilities from **both** the model input **and** the data-consistency projection, so the score is on measurements the method never saw and was never forced to match — and a method that fails it on real data.

[![CI](https://github.com/stelioszach03/dynadiff-vlbi/actions/workflows/ci.yml/badge.svg)](https://github.com/stelioszach03/dynadiff-vlbi/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-f59e0b?style=flat-square)](LICENSE)

Solo research project. **Manuscript only — not published, not peer-reviewed, not submitted.**

## Results

Δ is the EMC-minus-comparator gap on held-out visibility RMSE; positive means EMC is better. `Δ_det` uses the deterministic partition, `Δ_adap` the adaptive one. Evidence for both tables: [`paper/tables/adaptive_partition_results.tex`](paper/tables/adaptive_partition_results.tex).

| Setting | Cells | Outcome |
|---|---:|---|
| Synthetic (`default32`), 3 holdout families × 4 support fractions | 12 | **All 12 positive on both partitions** (Δ_adap up to +0.1864) |
| Public EHT, 4 releases × 2 families × 4 support fractions | 32 | **25 negative on both.** Δ_det positive in 2, Δ_adap in 4; **no cell positive on both** |

All four positive `Δ_adap` cells come from a single release (M87 2017). The largest negative value in the table, `Δ_adap = −0.1116` on Cen A 2017, sits in a cell where `Δ_det` is positive. **The method does not transfer from the synthetic regime to real EHT data.**

That failing table is published because [`benchmark/BENCHMARK_CARD.md`](benchmark/BENCHMARK_CARD.md) contains a **Stop rules** section written before the results: *"If a method does not improve on the realism track, do not claim stronger realism performance; claim only the stronger evaluation principle if justified."* The claim this repo makes is about the evaluation protocol, not the method's imaging quality.

Validation uses four official EHT calibrated-data releases, none redistributed here: M87 2017 ([10.25739/g85n-f134](https://doi.org/10.25739/g85n-f134)), M87 2018 ([10.25739/epm5-r371](https://doi.org/10.25739/epm5-r371)), 3C 279 2017 ([10.25739/vty0-ve39](https://doi.org/10.25739/vty0-ve39)), Cen A 2017 ([10.25739/kejs-2n22](https://doi.org/10.25739/kejs-2n22)).

## Run

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest -q      # 99 passed, 1 skipped in 6.7 s — CPU only, no data download
```

The suite runs on tiny synthetic fixtures and includes a numerical check of the partial-DFT consistency bound against `theory/`, a measurement-leakage audit, and a holdout-protocol check. The benchmark itself needs a GPU and the real data:

```bash
python scripts/run_emc_benchmark.py --target all --skip-existing
python scripts/generate_public_eht_suite_artifacts.py
```

## Limitations

- **It does not work on real EHT data.** 25 of 32 public-release cells are negative on both partition strategies. Nothing here supports a claim of better imaging quality on real interferometry.
- **The synthetic benchmark is not telescope-accurate** — the benchmark card says so directly: it is "a synthetic VLBI-inspired benchmark", not an EHT or ngEHT evaluation suite.
- **Held-out visibility RMSE is a proxy**, not image fidelity, and not a science-grade metric a radio astronomer would accept alone.
- **Uncertainty is not calibrated**, and per the pre-registered stop rules is kept out of the main claim.
- **No comparison against established VLBI pipelines** (CLEAN, RML, `eht-imaging`). The comparators are this repo's own baseline, residual-refinement and CCRR variants.
- **No error bars on the 44 cells** — each is one number from one run per condition.
- Reproducing the benchmark is expensive and needs the four EHT releases downloaded separately; only aggregated tables, figures and manifests are tracked.

## License

MIT — see [LICENSE](LICENSE).
