# Synthetic-to-Public Robustness Summary

| Cohort | Comparison | Metric | Pairs | Candidate mean | Reference mean | Mean delta | 95% CI | Win rate | p-value |
|---|---|---|---|---|---|---|---|---|---|
| Synthetic benchmark breadth | EMC vs CCRR | heldout_visibility_rmse | 768 | 0.071403 | 0.098044 | 0.026641 | [0.024840, 0.028423] | 0.856771 | 0.000000 |
| Synthetic benchmark breadth | EMC vs Residual Refinement | heldout_visibility_rmse | 768 | 0.071403 | 0.095372 | 0.023970 | [0.022219, 0.025705] | 0.824219 | 0.000000 |
| Synthetic benchmark breadth | EMC vs Baseline 3D U-Net | heldout_visibility_rmse | 768 | 0.071403 | 0.099447 | 0.028044 | [0.026182, 0.029912] | 0.873698 | 0.000000 |
| Public EHT baseline-track suite | EMC vs CCRR | heldout_visibility_rmse | 120 | 0.399286 | 0.510094 | 0.110808 | [0.092877, 0.129450] | 0.891667 | 0.000000 |
| Public EHT baseline-track suite | EMC vs Residual Refinement | heldout_visibility_rmse | 120 | 0.399286 | 0.454671 | 0.055385 | [0.046190, 0.064860] | 0.858333 | 0.000000 |
| Public EHT baseline-track suite | EMC vs Baseline 3D U-Net | heldout_visibility_rmse | 120 | 0.399286 | 0.375102 | -0.024183 | [-0.064826, 0.015730] | 0.425000 | 0.253906 |
| Public EHT baseline-track suite | EMC vs eht-imaging bridge | heldout_visibility_rmse | 120 | 0.399286 | 0.375086 | -0.024200 | [-0.064845, 0.015698] | 0.425000 | 0.252930 |
| Public EHT baseline-track suite | EMC vs Tikhonov | heldout_visibility_rmse | 120 | 0.399286 | 0.374779 | -0.024507 | [-0.065222, 0.015375] | 0.400000 | 0.247070 |
| Public EHT baseline-track suite | EMC vs Tikhonov | heldout_reduced_chi2 | 120 | 15430204.6 | 7639983.0 | -7790221.6 | [-12532577.7, -3808916.8] | 0.441667 | 0.000000 |
| Public EHT baseline-track suite | EMC-TTO vs EMC | heldout_visibility_rmse | 120 | 0.365779 | 0.399286 | 0.033507 | [-0.003409, 0.071913] | 0.583333 | 0.082520 |
| Public EHT baseline-track suite | EMC-TTO vs CCRR | heldout_visibility_rmse | 120 | 0.365779 | 0.510094 | 0.144315 | [0.110256, 0.180924] | 0.750000 | 0.000000 |
| Public EHT baseline-track suite | EMC-TTO vs Residual Refinement | heldout_visibility_rmse | 120 | 0.365779 | 0.454671 | 0.088892 | [0.052414, 0.127293] | 0.675000 | 0.000000 |
| Public EHT baseline-track suite | EMC-TTO vs Baseline 3D U-Net | heldout_visibility_rmse | 120 | 0.365779 | 0.375102 | 0.009323 | [-0.000319, 0.019758] | 0.666667 | 0.065918 |
| Public EHT baseline-track suite | EMC-TTO vs eht-imaging bridge | heldout_visibility_rmse | 120 | 0.365779 | 0.375086 | 0.009307 | [-0.000337, 0.019731] | 0.650000 | 0.066406 |
| Public EHT baseline-track suite | EMC-TTO vs Tikhonov | heldout_visibility_rmse | 120 | 0.365779 | 0.374779 | 0.009000 | [-0.000652, 0.019446] | 0.633333 | 0.077637 |
| Public EHT baseline-track suite | EMC-TTO vs Tikhonov | heldout_reduced_chi2 | 120 | 6933615.8 | 7639983.0 | 706367.2 | [306642.1, 1201533.5] | 0.616667 | 0.001465 |

## Notes

- Mean deltas are direction-aware: positive means the candidate is better under the metric direction.
- The public rows pool all sample-support cases within the named public-EHT cohort.
