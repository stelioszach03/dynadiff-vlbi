# EMC Benchmark Release

This directory documents the benchmark-first paper path for `dynadiff-vlbi`.

The benchmark is built around one scientific distinction:

- enforced measurement consistency: coefficients the model was directly given or that a data-consistency layer forced it to match
- earned measurement consistency: coefficients withheld from both the model input and the support-set data-consistency layer, then evaluated as unseen targets

The current learned reference implementation is EMC, but the benchmark is larger than one model. The release fixes the holdout logic, support fractions, comparator protocol, and artifact layout so future methods can be compared on the same structured sparse VLBI-inspired task.

The benchmark release is synthetic and protocol-first. A separate public-EHT observation-domain validation path exists elsewhere in the repository and should be read as complementary astronomy-facing validation rather than as part of the fixed benchmark release.

## One-command reproduction

From the repository root:

```bash
python3.11 scripts/run_emc_benchmark.py --target all --skip-existing && \
python3.11 scripts/generate_emc_benchmark_artifacts.py
```

This command:

- runs the benchmark families
- runs the challenge-inspired realism track
- writes deterministic split manifests
- writes resolved config manifests
- exports benchmark tables and figures

For the complementary public-M87 validation and the supplementary strengthening artifacts, use the top-level README commands instead of treating that path as part of the fixed benchmark release.

For the expanded multi-release public-EHT suite, including the frozen `eht-imaging bridge`, see:

- [`PUBLIC_EHT_VALIDATION.md`](../benchmark/PUBLIC_EHT_VALIDATION.md)
- [`REVIEW_SNAPSHOT.md`](../benchmark/REVIEW_SNAPSHOT.md)

## Fixed benchmark factors

- Project seed: `7`
- Support fractions: `0.8`, `0.6`, `0.4`, `0.2`
- Structured holdout families:
  - `baseline_track_blocks`
  - `scan_segment_blocks`
  - `station_dropout`
- Realism track:
  - `challenge_inspired_realism`
- Comparator set:
  - dirty
  - Tikhonov
  - baseline 3D U-Net
  - residual refinement
  - CCRR
  - EMC

## What must be reported

At minimum, a method should report:

- held-out visibility RMSE
- held-out closure-phase MAE when all-target triangle support is sufficient
- support-set visibility RMSE separately from held-out performance
- MSE
- SSIM
- temporal consistency
- exact config manifest
- exact split manifest

## Key files

- Public-EHT validation: [`PUBLIC_EHT_VALIDATION.md`](../benchmark/PUBLIC_EHT_VALIDATION.md)
- Review snapshot: [`REVIEW_SNAPSHOT.md`](../benchmark/REVIEW_SNAPSHOT.md)
- Benchmark card: [`BENCHMARK_CARD.md`](../benchmark/BENCHMARK_CARD.md)
- Protocol card: [`PROTOCOL_CARD.md`](../benchmark/PROTOCOL_CARD.md)
- Expected outputs: [`EXPECTED_OUTPUTS.md`](../benchmark/EXPECTED_OUTPUTS.md)
- Release checklist: [`RELEASE_CHECKLIST.md`](../benchmark/RELEASE_CHECKLIST.md)
- Leaderboard template: [`leaderboard_template.csv`](../benchmark/leaderboard_template.csv)
