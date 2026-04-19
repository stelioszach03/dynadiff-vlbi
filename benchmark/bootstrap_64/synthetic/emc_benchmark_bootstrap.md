# Synthetic-to-Public Robustness Summary

| Cohort | Comparison | Metric | Pairs | Candidate mean | Reference mean | Mean delta | 95% CI | Win rate | p-value |
|---|---|---|---|---|---|---|---|---|---|
| Synthetic benchmark breadth | EMC vs CCRR | heldout_visibility_rmse | 768 | 0.071403 | 0.098044 | 0.026641 | [0.024840, 0.028423] | 0.856771 | 0.000000 |
| Synthetic benchmark breadth | EMC vs Residual Refinement | heldout_visibility_rmse | 768 | 0.071403 | 0.095372 | 0.023970 | [0.022219, 0.025705] | 0.824219 | 0.000000 |
| Synthetic benchmark breadth | EMC vs Baseline 3D U-Net | heldout_visibility_rmse | 768 | 0.071403 | 0.099447 | 0.028044 | [0.026182, 0.029912] | 0.873698 | 0.000000 |

## Notes

- Mean deltas are direction-aware: positive means the candidate is better under the metric direction.
- The public rows pool all sample-support cases within the named public-EHT cohort.
