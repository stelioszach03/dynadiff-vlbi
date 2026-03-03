# dynadiff-vlbi

`dynadiff-vlbi` is a small, Colab-first research prototype for uncertainty-aware dynamic black-hole-like imaging from synthetic sparse Fourier measurements. The project now contains two additive research paths:

- Phase 1 baseline: a reproducible dirty-image-first pipeline with classical baselines and a compact 3D temporal U-Net.
- Phase 2 upgrade: a visibility-conditioned spatiotemporal model that ingests sparse complex Fourier measurements more directly while keeping the Phase 1 baseline intact as the reference comparator.

The goal is still not a full EHT or ngEHT pipeline. This repository is synthetic VLBI-inspired research infrastructure with runnable baselines and clean extension points.

## What the repository includes

- Synthetic grayscale black-hole-like movie generation with a bright ring, azimuthal asymmetry, moving hotspot, optional faint jet, and temporal variability.
- A sparse noisy Fourier measurement operator with configurable uv coverage, missing coverage, and Gaussian noise.
- Classical baselines: dirty image reconstruction and a lightweight Tikhonov-style iterative refinement.
- A compact 3D temporal U-Net baseline trained on dirty reconstructions.
- A compact visibility-conditioned model with a visibility encoder, optional dirty-image branch, spatiotemporal fusion, and optional heteroscedastic uncertainty head.
- Monte Carlo dropout uncertainty for the baseline and heteroscedastic predictive uncertainty for the Phase 2 model.
- Metrics for MSE, PSNR, SSIM, temporal consistency, ring-radius error, and hotspot localization error.
- CLI scripts, tests, configs, figures, checkpoints, logs, comparison tables, and Colab notebooks.

## Project layout

```text
dynadiff-vlbi/
├── configs/
├── notebooks/
├── scripts/
├── src/dynadiff_vlbi/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── physics/
│   ├── training/
│   └── utils/
└── tests/
```

## Local setup

Use Python 3.10 or newer. The current repository was validated with `python3.11`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the baseline smoke test:

```bash
python scripts/run_demo.py --preset smoke
```

Run the default 32x32 baseline:

```bash
python scripts/generate_toy_dataset.py --preset default32
python scripts/train_baseline.py --preset default32 --run-name train_default32
python scripts/evaluate_model.py --preset default32 --run-name train_default32
```

Run the default 32x32 Phase 2 visibility-conditioned path:

```bash
python scripts/generate_toy_dataset.py --base-config configs/phase2_visibility_default32.yaml
python scripts/train_baseline.py --preset default32 --data-dir data/generated/phase2_visibility_default32 --run-name phase2_default32_baseline_ref --epochs 2
python scripts/train_baseline.py --base-config configs/phase2_visibility_default32.yaml --run-name phase2_default32_visibility --epochs 2
python scripts/evaluate_model.py --base-config configs/phase2_visibility_default32.yaml --run-name phase2_default32_visibility --reference-baseline-checkpoint outputs/phase2_default32_baseline_ref/checkpoints/best.pt
```

Run a small Phase 2 smoke path:

```bash
python scripts/generate_toy_dataset.py --base-config configs/phase2_visibility_default32.yaml --preset smoke
python scripts/train_baseline.py --preset smoke --data-dir data/generated/smoke_phase2_visibility_default32 --run-name phase2_smoke_baseline_ref --epochs 1
python scripts/train_baseline.py --base-config configs/phase2_visibility_default32.yaml --preset smoke --run-name phase2_smoke_visibility --epochs 1
python scripts/evaluate_model.py --base-config configs/phase2_visibility_default32.yaml --preset smoke --run-name phase2_smoke_visibility --reference-baseline-checkpoint outputs/phase2_smoke_baseline_ref/checkpoints/best.pt
```

Optional 64x64 experiments:

```bash
python scripts/generate_toy_dataset.py --preset exp64
python scripts/train_baseline.py --preset exp64 --run-name train_exp64
python scripts/evaluate_model.py --preset exp64 --run-name train_exp64
```

```bash
python scripts/generate_toy_dataset.py --base-config configs/phase2_visibility_exp64.yaml
python scripts/train_baseline.py --base-config configs/phase2_visibility_exp64.yaml --run-name phase2_exp64_visibility
python scripts/evaluate_model.py --base-config configs/phase2_visibility_exp64.yaml --run-name phase2_exp64_visibility
```

Run tests:

```bash
python -m pytest
```

## Google Colab steps

1. Open a new Colab notebook and enable a GPU if available.
2. Clone or upload this repository into the Colab runtime.
3. For the Phase 1 baseline, from the repository root run:

```python
%pip install -e .
!python scripts/generate_toy_dataset.py --preset default32
!python scripts/train_baseline.py --preset default32 --run-name colab_default32_demo --epochs 2
!python scripts/evaluate_model.py --preset default32 --run-name colab_default32_demo
```

4. For the Phase 2 visibility-conditioned path, run:

```python
%pip install -e .
!python scripts/generate_toy_dataset.py --base-config configs/phase2_visibility_default32.yaml --preset smoke
!python scripts/train_baseline.py --preset smoke --data-dir data/generated/smoke_phase2_visibility_default32 --run-name colab_phase2_baseline_smoke --epochs 1
!python scripts/train_baseline.py --base-config configs/phase2_visibility_default32.yaml --preset smoke --run-name colab_phase2_visibility_smoke --epochs 2
!python scripts/evaluate_model.py --base-config configs/phase2_visibility_default32.yaml --preset smoke --run-name colab_phase2_visibility_smoke --reference-baseline-checkpoint outputs/colab_phase2_baseline_smoke/checkpoints/best.pt
```

5. Open the figures and summaries under `outputs/<run_name>/`.
6. Ready-made notebooks:
   - [`notebooks/01_colab_quickstart.ipynb`](/Users/stelioszacharioudakis/Documents/Papers/DynaDiff-VLBI/notebooks/01_colab_quickstart.ipynb)
   - [`notebooks/03_phase2_visibility_quickstart.ipynb`](/Users/stelioszacharioudakis/Documents/Papers/DynaDiff-VLBI/notebooks/03_phase2_visibility_quickstart.ipynb)

## Config presets

- `smoke`: very small end-to-end smoke test used by `scripts/run_demo.py`
- `default32`: main 32x32 Colab-friendly baseline
- `exp64`: optional 64x64 experiment with slightly larger model width

Phase 2 adds experiment-specific base configs:

- [`configs/phase2_visibility_default32.yaml`](/Users/stelioszacharioudakis/Documents/Papers/DynaDiff-VLBI/configs/phase2_visibility_default32.yaml)
- [`configs/phase2_visibility_exp64.yaml`](/Users/stelioszacharioudakis/Documents/Papers/DynaDiff-VLBI/configs/phase2_visibility_exp64.yaml)

The shared defaults live in [`configs/base.yaml`](/Users/stelioszacharioudakis/Documents/Papers/DynaDiff-VLBI/configs/base.yaml), while preset overrides live in [`configs/train.yaml`](/Users/stelioszacharioudakis/Documents/Papers/DynaDiff-VLBI/configs/train.yaml). When both `--base-config` and `--preset` are provided, custom config values take precedence over overlapping preset keys.

## Phase 2 model path

The Phase 2 upgrade adds a visibility-conditioned reconstruction model without removing the existing dirty-image baseline path. The new model:

- consumes sparse complex visibilities through a real and imaginary channel representation
- can also ingest dirty reconstructions through an optional image-domain branch
- fuses visibility and image features in a compact spatiotemporal encoder-decoder
- can predict a per-pixel log-variance map for a lightweight heteroscedastic uncertainty baseline

Evaluation can compare:

- dirty reconstruction
- Tikhonov refinement
- the original 3D U-Net baseline
- the new visibility-conditioned model

This remains a synthetic research prototype. The new path is meant to study whether direct visibility conditioning helps under sparse Fourier sampling, not to claim telescope-accurate VLBI performance.

## Outputs

Each run writes to:

```text
outputs/<run_name>/
├── checkpoints/
├── figures/
├── logs/
└── predictions/
```

Typical output files include:

- `checkpoints/best.pt`
- `checkpoints/latest.pt`
- `logs/config_snapshot.yaml`
- `logs/history.csv`
- `logs/training_summary.json`
- `logs/evaluation_summary.json`
- `logs/comparison_metrics.csv` for Phase 2 evaluation
- `predictions/test_predictions.npz`
- `figures/sample_*.png`

Synthetic datasets are written to `data/generated/<experiment_label>/`. For example:

- baseline smoke: `data/generated/smoke/`
- baseline default32: `data/generated/default32/`
- Phase 2 smoke: `data/generated/smoke_phase2_visibility_default32/`
- Phase 2 default32: `data/generated/phase2_visibility_default32/`

## Current limitations

- The forward model is VLBI-inspired sparse Fourier sampling, not a telescope-array-accurate EHT or ngEHT simulator.
- The Phase 1 learned baseline only consumes dirty reconstructions.
- The Phase 2 model is still compact and synthetic-data-focused. It is not a diffusion model and does not ingest telescope metadata beyond simple uv-style feature formatting.
- Uncertainty remains lightweight: MC dropout for the Phase 1 model and a heteroscedastic head for the Phase 2 model. Neither should be treated as calibrated posterior uncertainty without further study.
- The classical refinement baseline is intentionally lightweight and should be treated as a sanity-check comparator, not a strong inverse solver.
- The synthetic generator is designed for controllable experiments, not astrophysical realism.

## Roadmap

- Real EHT or ngEHT-compatible data loaders and measurement conventions
- Stronger diffusion-based spatiotemporal priors
- Posterior sampling and calibration-focused uncertainty analysis
- Polarization-aware and multi-frequency extensions
