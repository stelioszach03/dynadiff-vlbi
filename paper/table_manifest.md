# Table Manifest

## Table 1: Default64 Benchmark Matrix

- Artifact file: [`paper/tables/table01_default64_benchmark_matrix.tex`](./tables/table01_default64_benchmark_matrix.tex)
- CSV source: [`outputs/emc_benchmark_artifacts/tables/emc_benchmark_matrix.csv`](../outputs/emc_benchmark_artifacts/tables/emc_benchmark_matrix.csv)
- Caption:

  Default64 synthetic benchmark matrix across the three structured holdout families and four support fractions. The table reports held-out visibility RMSE for EMC and the main learned comparators under identical support-target partitions, adds the bounded DPS comparator on the rerun baseline-track family only, and reports EMC conformal empirical coverage and mean interval width (MIW).

## Table 2: Public EHT Release Means

- Artifact file: [`paper/tables/table02_public_eht_release_means.tex`](./tables/table02_public_eht_release_means.tex)
- CSV source: [`paper/tables/table02_public_eht_release_robustness.csv`](./tables/table02_public_eht_release_robustness.csv)
- Caption:

  Release-level public-EHT baseline-track summary with test-time optimization, DPS, and conformal MIW. The table reports mean held-out visibility RMSE for EMC, EMC-TTO, and DPS on each official release, together with MIW for EMC-family uncertainty intervals and the best baseline-track comparator on that release.

## Table 3: Public Family Robustness

- Artifact file: [`paper/tables/table03_public_family_robustness.tex`](./tables/table03_public_family_robustness.tex)
- CSV source: [`paper/tables/table03_public_release_gaps.csv`](./tables/table03_public_release_gaps.csv)
- Caption:

  Release-level robustness across the public baseline-track and station-dropout families. The table makes the release-by-release heterogeneity explicit and records which model is best under each split family.

## Table 4: Public Bootstrap Summary

- Artifact file: [`paper/tables/table04_public_bootstrap.tex`](./tables/table04_public_bootstrap.tex)
- CSV source: [`paper/tables/table04_public_bootstrap_stats.csv`](./tables/table04_public_bootstrap_stats.csv)
- Caption:

  Selected pooled public-EHT paired-bootstrap comparisons. Positive mean deltas indicate lower held-out visibility RMSE for the candidate method than for the reference method.

## Table 5: Seed Robustness

- Artifact file: [`paper/tables/table05_seed_robustness.tex`](./tables/table05_seed_robustness.tex)
- CSV source: [`paper/tables/table05_seed_robustness.csv`](./tables/table05_seed_robustness.csv)
- Caption:

  Bounded five-seed robustness study on the default64 baseline-track family over seeds 7, 19, 31, 42, and 137. This table is included because it tempers the single released breadth matrix and shows that residual refinement is the strongest mean learned comparator on this core family.

## Notes

- Table 1 remains the central synthetic breadth table.
- Tables 2 through 4 make the public-EHT story release-aware rather than pooled-only.
- DPS is intentionally bounded to the baseline-track add-on path and is left as `n/a` outside the rerun family rather than imputed.
- Public UQ reports MIW only, because released public-EHT data do not provide image-domain ground truth for coverage claims.
- Table 5 is essential to the final paper because it narrows the method claim and strengthens reviewer confidence in the benchmark framing.
- The legacy lower-resolution ablation table remains in the release package for auditability, but it is not part of the main `manuscript_v2` evidence stack.
