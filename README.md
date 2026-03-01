# dynadiff-vlbi

`dynadiff-vlbi` is a small, Colab-first research prototype for uncertainty-aware dynamic black-hole-like imaging from synthetic sparse Fourier measurements. The goal for v1 is not a full EHT or ngEHT pipeline. It is a reproducible baseline that runs end-to-end on synthetic sequences, exposes clean abstractions for future work, and stays honest about its scope.

## What v1 includes

- Synthetic grayscale black-hole-like movie generation with a bright ring, azimuthal asymmetry, moving hotspot, optional faint jet, and temporal variability.
- A sparse noisy Fourier measurement operator with configurable uv coverage, missing coverage, and Gaussian noise.
- Classical baselines: dirty image reconstruction and a lightweight Tikhonov-style iterative refinement.
- A compact 3D temporal U-Net baseline trained on dirty reconstructions.
- Monte Carlo dropout uncertainty maps.
- Metrics for MSE, PSNR, SSIM, temporal consistency, ring-radius error, and hotspot localization error.
- CLI scripts, tests, configs, figures, checkpoints, logs, and Colab notebooks.

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

Run the smoke test:

```bash
python scripts/run_demo.py --preset smoke
```

Run the full default 32x32 baseline:

```bash
python scripts/generate_toy_dataset.py --preset default32
python scripts/train_baseline.py --preset default32 --run-name train_default32
python scripts/evaluate_model.py --preset default32 --run-name train_default32
```

Optional 64x64 experiment:

```bash
python scripts/generate_toy_dataset.py --preset exp64
python scripts/train_baseline.py --preset exp64 --run-name train_exp64
python scripts/evaluate_model.py --preset exp64 --run-name train_exp64
```

Run tests:

```bash
python -m pytest
```

## Google Colab steps

1. Open a new Colab notebook and enable a GPU if available.
2. Clone or upload this repository into the Colab runtime.
3. From the repository root, run:

```python
%pip install -e .
!python scripts/generate_toy_dataset.py --preset default32
!python scripts/train_baseline.py --preset default32 --run-name colab_default32_demo --epochs 2
!python scripts/evaluate_model.py --preset default32 --run-name colab_default32_demo
```

4. Open `outputs/colab_default32_demo/figures/` and `outputs/colab_default32_demo/logs/evaluation_summary.json`.
5. The repository also ships a ready-made notebook at [`notebooks/01_colab_quickstart.ipynb`](/Users/stelioszacharioudakis/Documents/Papers/DynaDiff-VLBI/notebooks/01_colab_quickstart.ipynb).

## Config presets

- `smoke`: very small end-to-end smoke test used by `scripts/run_demo.py`
- `default32`: main 32x32 Colab-friendly baseline
- `exp64`: optional 64x64 experiment with slightly larger model width

The shared defaults live in [`configs/base.yaml`](/Users/stelioszacharioudakis/Documents/Papers/DynaDiff-VLBI/configs/base.yaml), while preset overrides live in [`configs/train.yaml`](/Users/stelioszacharioudakis/Documents/Papers/DynaDiff-VLBI/configs/train.yaml).

## Outputs

Each run writes to:

```text
outputs/<run_name>/
├── checkpoints/
├── figures/
├── logs/
└── predictions/
```

Synthetic datasets are written to `data/generated/<preset>/`.

## Current limitations

- The forward model is VLBI-inspired sparse Fourier sampling, not a telescope-array-accurate EHT or ngEHT simulator.
- The learned baseline only consumes dirty reconstructions in v1. It does not operate directly on raw visibilities.
- Uncertainty is approximated with Monte Carlo dropout, not a calibrated Bayesian posterior.
- The classical refinement baseline is intentionally lightweight and should be treated as a sanity-check comparator, not a strong inverse solver.
- The synthetic generator is designed for controllable experiments, not astrophysical realism.

## Roadmap

- Real EHT or ngEHT-compatible data loaders and measurement conventions
- Stronger diffusion-based spatiotemporal priors
- Posterior sampling and calibration-focused uncertainty analysis
- Polarization-aware and multi-frequency extensions
