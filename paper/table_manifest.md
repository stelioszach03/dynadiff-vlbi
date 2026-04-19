# Table Manifest

## Table 1: EMC Benchmark Matrix

- Artifact file: [`outputs/emc_benchmark_artifacts/tables/emc_benchmark_matrix.md`](../outputs/emc_benchmark_artifacts/tables/emc_benchmark_matrix.md)
- CSV source: [`outputs/emc_benchmark_artifacts/tables/emc_benchmark_matrix.csv`](../outputs/emc_benchmark_artifacts/tables/emc_benchmark_matrix.csv)
- Caption:

  Default32 benchmark matrix across the three structured holdout families and the four support fractions. The table reports held-out visibility RMSE for EMC and the main learned comparators under identical support-target partitions. It is the core tabular evidence that earned measurement consistency is not a single-geometry effect.

## Table 2: Public EHT Observation-Domain Benchmark Matrix

- Artifact file: [`outputs/public_eht_suite_artifacts/tables/emc_public_eht_matrix.md`](../outputs/public_eht_suite_artifacts/tables/emc_public_eht_matrix.md)
- CSV source: [`outputs/public_eht_suite_artifacts/tables/emc_public_eht_matrix.csv`](../outputs/public_eht_suite_artifacts/tables/emc_public_eht_matrix.csv)
- Caption:

  Public EHT observation-domain benchmark matrix across the official calibrated-data releases used in the paper: M87 2017, M87 2018, 3C279 2017, and Centaurus A 2017. The table reports held-out visibility recovery on real measured coefficients withheld from both model input and support-set data consistency. It is included to test whether the benchmark question remains meaningful on released measurements, not to claim image-domain ground-truth recovery.

## Table 3: Synthetic-to-Public Robustness Summary

- Artifact file: [`outputs/public_eht_suite_artifacts/tables/emc_public_eht_release_robustness.md`](../outputs/public_eht_suite_artifacts/tables/emc_public_eht_release_robustness.md)
- CSV source: [`outputs/public_eht_suite_artifacts/tables/emc_public_eht_release_robustness.csv`](../outputs/public_eht_suite_artifacts/tables/emc_public_eht_release_robustness.csv)
- Caption:

  Public-EHT release robustness summary across the baseline-track and station-dropout families. The table exposes release-level heterogeneity, the EMC family gap between the two deterministic public holdout families, and the best comparator on each release under each family.

## Table 4: EMC Component Ablations

- Artifact file: [`outputs/mnras_real_data_artifacts/tables/emc_component_ablations.md`](../outputs/mnras_real_data_artifacts/tables/emc_component_ablations.md)
- CSV source: [`outputs/mnras_real_data_artifacts/tables/emc_component_ablations.csv`](../outputs/mnras_real_data_artifacts/tables/emc_component_ablations.csv)
- Caption:

  Compact ablation summary for the default32 baseline-track EMC protocol, averaged over the full support sweep. The table isolates the support-target holdout objective, support-set data consistency, closure-aware supervision, metadata conditioning, and uncertainty head. In the current regime, closure remains secondary rather than dominant.

## Notes

- Table 1 and Figure 2 remain the central synthetic benchmark evidence.
- Table 2 is the main astronomy-facing validation table because it uses official public EHT released measurements rather than synthetic data.
- Table 3 exists to make the release-level public robustness picture explicit rather than leaving it implicit in pooled numbers alone.
- Table 4 clarifies attribution and keeps closure from being overstated.
