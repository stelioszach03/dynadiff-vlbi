# Benchmark Card

## Name

Earned Measurement Consistency (EMC) benchmark for dynamic black-hole imaging under structured sparse VLBI-inspired sampling.

## Purpose

The benchmark asks whether a method can recover measurements it never saw and was never forced to match. It is designed to separate:

- observation-domain agreement created by support-set data consistency
- observation-domain agreement earned on structured held-out measurements

## Scope

This is a synthetic VLBI-inspired benchmark with a strong learned reference implementation. It is not a telescope-accurate EHT or ngEHT evaluation suite.

## Release contents

- deterministic support-target split manifests
- resolved config manifests
- support-fraction protocol outputs
- benchmark matrix tables
- challenge-inspired realism-track outputs
- leaderboard template

## Benchmark claim

We propose EMC as a reproducible benchmark protocol for earned versus enforced measurement consistency under structured sparse dynamic VLBI-inspired sampling.

## Stop rules

- If a method does not improve on the realism track, do not claim stronger realism performance; claim only the stronger evaluation principle if justified.
- If gains appear only in one holdout family, report partial robustness rather than broad generality.
- If uncertainty is not calibrated, keep it out of the main benchmark claim.
- If external challenge data are unavailable, do not imply use of private challenge assets.

## Current reference implementation

EMC is the current learned reference implementation for this benchmark. It keeps the existing 3D U-Net backbone and residual-refinement philosophy intact while training on structured held-out measurement targets.
