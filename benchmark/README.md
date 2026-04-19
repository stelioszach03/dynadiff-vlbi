# EMC Benchmark Release

This directory documents the benchmark-first release path for `dynadiff-vlbi`.

The benchmark is built around one scientific distinction:

- **enforced measurement consistency**: coefficients the model was directly given or that a support-set data-consistency layer forced it to match
- **earned measurement consistency**: coefficients withheld from both the model input and the support-set data-consistency layer, then evaluated as unseen targets

EMC is the current learned reference implementation, but the benchmark is larger than one model. The release fixes the support-target logic, support fractions, comparator protocol, and artifact layout so future methods can be compared on the same structured dynamic VLBI-style task.

## Fixed benchmark scope

- project seed: `7`
- support fractions: `80%`, `60%`, `40%`, `20%`
- structured holdout families:
  - `baseline_track_blocks`
  - `scan_segment_blocks`
  - `station_dropout`
- challenge-inspired realism track:
  - `challenge_inspired_realism`
- learned comparator core:
  - baseline 3D U-Net
  - residual refinement
  - CCRR
  - EMC

The fixed benchmark release is synthetic and protocol-first. The public-EHT suite is complementary astronomy-facing validation, documented separately.

## One-command benchmark reproduction

```bash
python3.11 scripts/run_emc_benchmark.py --target all --skip-existing
python3.11 scripts/generate_emc_benchmark_artifacts.py
```

This path:

- runs the three benchmark families
- runs the challenge-inspired realism track
- exports deterministic split manifests
- exports resolved config manifests
- writes benchmark tables and figures

## Complementary public-EHT suite

For the multi-release public-EHT observation-domain suite, including the frozen `eht-imaging bridge`, use:

- [`PUBLIC_EHT_VALIDATION.md`](PUBLIC_EHT_VALIDATION.md)
- [`REVIEW_SNAPSHOT.md`](REVIEW_SNAPSHOT.md)

## Required reporting

At minimum, a method should report:

- support fraction
- holdout family
- held-out visibility RMSE
- held-out closure-phase MAE when all-target triangle support is sufficient
- support visibility RMSE separately from held-out performance
- MSE
- SSIM
- temporal consistency
- exact config manifest
- exact split manifest

## Key files

- Benchmark card: [`BENCHMARK_CARD.md`](BENCHMARK_CARD.md)
- Protocol card: [`PROTOCOL_CARD.md`](PROTOCOL_CARD.md)
- Expected outputs: [`EXPECTED_OUTPUTS.md`](EXPECTED_OUTPUTS.md)
- Release checklist: [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md)
- Leaderboard template: [`leaderboard_template.csv`](leaderboard_template.csv)
