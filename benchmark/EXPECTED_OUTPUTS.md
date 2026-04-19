# Expected Outputs

A successful benchmark release run should produce the following top-level files and directories.

## Release manifest root

- [`outputs/emc_benchmark_release/benchmark_output_manifest.json`](/Users/stelioszacharioudakis/Documents/Papers/DynaDiff-VLBI/outputs/emc_benchmark_release/benchmark_output_manifest.json)
- `outputs/emc_benchmark_release/config_manifests/`
- `outputs/emc_benchmark_release/split_manifests/`

## Config manifests

- `outputs/emc_benchmark_release/config_manifests/baseline_tracks.json`
- `outputs/emc_benchmark_release/config_manifests/scan_segments.json`
- `outputs/emc_benchmark_release/config_manifests/station_dropout.json`
- `outputs/emc_benchmark_release/config_manifests/challenge_inspired_realism.json`

## Split manifests

Each family directory should contain:

- `split_manifest.json`
- `support_80_split_manifest.npz`
- `support_60_split_manifest.npz`
- `support_40_split_manifest.npz`
- `support_20_split_manifest.npz`

## Benchmark artifact outputs

- `outputs/emc_benchmark_artifacts/tables/emc_benchmark_long.csv`
- `outputs/emc_benchmark_artifacts/tables/emc_benchmark_matrix.md`
- `outputs/emc_benchmark_artifacts/tables/emc_challenge_inspired_realism.md`
- [`outputs/emc_benchmark_artifacts/leaderboard_template.csv`](/Users/stelioszacharioudakis/Documents/Papers/DynaDiff-VLBI/outputs/emc_benchmark_artifacts/leaderboard_template.csv)
- `outputs/emc_benchmark_artifacts/summaries/emc_benchmark_artifact_manifest.json`

## Paper figures

- `paper/figures/fig02_emc_benchmark_support_curve.png`
- `paper/figures/fig03_emc_challenge_inspired_realism.png`
- `paper/figures/fig04_emc_benchmark_representative.png`
- `paper/figures/fig05_emc_realism_hard_example.png`

## Protocol run outputs

Each protocol run directory should contain:

- `logs/emc_protocol_summary.json`
- `logs/support_fraction_metrics.csv`
- `predictions/support_80.npz`
- `predictions/support_60.npz`
- `predictions/support_40.npz`
- `predictions/support_20.npz`
