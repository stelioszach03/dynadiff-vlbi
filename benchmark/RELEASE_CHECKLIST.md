# Release Checklist

- [ ] `python3.11 -m pytest` passes
- [ ] `python3.11 scripts/run_emc_benchmark.py --target all --skip-existing` completes
- [ ] `python3.11 scripts/generate_emc_benchmark_artifacts.py` completes
- [ ] `outputs/emc_benchmark_release/benchmark_output_manifest.json` exists
- [ ] deterministic split manifests exist for all benchmark families
- [ ] resolved config manifests exist for all benchmark families
- [ ] benchmark matrix and realism-track tables exist
- [ ] central benchmark support-curve figure exists
- [ ] realism-track figure exists
- [ ] README benchmark section matches the current commands
- [ ] manuscript describes the realism track as challenge-inspired, not private challenge data
- [ ] manuscript does not present uncertainty as calibrated
- [ ] manuscript does not present EMC as a universal winner on every metric
