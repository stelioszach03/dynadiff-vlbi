"""Configuration loading and typed access."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_BASE_CONFIG_PATH = Path("configs/base.yaml")


@dataclass
class ProjectConfig:
    name: str
    seed: int


@dataclass
class PathsConfig:
    data_root: str
    output_root: str


@dataclass
class OutputConfig:
    checkpoint_subdir: str
    figure_subdir: str
    log_subdir: str
    prediction_subdir: str


@dataclass
class SyntheticSequenceConfig:
    train_size: int
    val_size: int
    test_size: int
    image_size: int
    sequence_length: int
    ring_radius: float
    ring_width: float
    asymmetry_strength: float
    hotspot_intensity: float
    hotspot_width: float
    hotspot_speed: float
    hotspot_radius: float
    second_hotspot_probability: float
    jet_intensity: float
    temporal_variability: float
    background_level: float


@dataclass
class SamplingConfig:
    coverage: float
    radial_exponent: float
    missing_fraction: float
    hermitian_symmetric: bool
    include_dc: bool
    mode: str = "random_radial"
    station_count: int = 0
    earth_rotation_degrees: float = 0.0
    station_jitter: float = 0.0
    scan_gap_probability: float = 0.0
    scan_gap_length: int = 0


@dataclass
class NoiseConfig:
    noise_std: float
    baseline_noise_jitter: float = 0.0
    gain_amplitude_std: float = 0.0
    gain_phase_std: float = 0.0


@dataclass
class ModelConfig:
    in_channels: int
    out_channels: int
    base_channels: int
    dropout: float
    model_type: str = "baseline"
    include_dirty_input: bool = False
    visibility_representation: str = "real_imag"
    include_uv_coords: bool = False
    include_mask_channel: bool = True
    include_observation_metadata: bool = False
    uncertainty_head: bool = False
    refinement_channels: int = 8
    residual_scale: float = 0.25
    freeze_backbone: bool = False
    dc_enabled: bool = False
    dc_weight: float = 1.0
    dc_learnable: bool = False
    num_levels: int = 2


@dataclass
class TrainingConfig:
    batch_size: int
    num_workers: int
    epochs: int
    learning_rate: float
    weight_decay: float
    temporal_loss_weight: float
    grad_clip_norm: float
    heteroscedastic_loss_weight: float = 0.0
    reconstruction_warmup_epochs: int = 0
    backbone_checkpoint: str | None = None
    visibility_loss_weight: float = 0.0
    closure_loss_weight: float = 0.0
    closure_max_triangles: int = 24
    target_visibility_loss_weight: float = 0.0
    target_closure_loss_weight: float = 0.0


@dataclass
class EvalConfig:
    mc_samples: int
    num_visualizations: int
    frames_to_plot: int
    figure_dpi: int
    save_prediction_arrays: bool
    tikhonov_lambda: float
    tikhonov_iterations: int
    tikhonov_step_size: float
    compare_reference_baseline: bool = False
    reference_baseline_run_name: str | None = None
    reference_visibility_run_name: str | None = None
    reference_residual_run_name: str | None = None
    reference_ccrr_run_name: str | None = None
    save_comparison_csv: bool = True


@dataclass
class HoldoutConfig:
    enabled: bool = False
    strategy: str = "none"
    support_fraction: float = 1.0
    train_support_fractions: tuple[float, ...] = (1.0,)
    eval_support_fractions: tuple[float, ...] = (1.0,)
    max_triangles: int = 24
    min_eval_closure_triangles: int = 24
    # Phase 4: partition mode and optional oracle checkpoint for adaptive mode.
    # partition_mode="deterministic" (default) keeps the legacy 3-strategy path.
    # partition_mode="adaptive" requires oracle_checkpoint to point at a trained
    # HeavyHitterOracle .ckpt and switches `strategy` to "learned_oracle_importance".
    partition_mode: str = "deterministic"
    oracle_checkpoint: str | None = None


@dataclass
class ExperimentConfig:
    project: ProjectConfig
    paths: PathsConfig
    outputs: OutputConfig
    dataset: SyntheticSequenceConfig
    sampling: SamplingConfig
    noise: NoiseConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvalConfig
    holdout: HoldoutConfig
    preset_name: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full configuration to a dictionary."""

        return asdict(self)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge nested dictionaries."""

    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_config(payload: dict[str, Any], preset_name: str) -> ExperimentConfig:
    holdout_payload = dict(payload.get("holdout", {}))
    train_support_fractions = tuple(
        float(value)
        for value in holdout_payload.get(
            "train_support_fractions",
            [holdout_payload.get("support_fraction", 1.0)],
        )
    )
    eval_support_fractions = tuple(
        float(value)
        for value in holdout_payload.get(
            "eval_support_fractions",
            [holdout_payload.get("support_fraction", 1.0)],
        )
    )
    holdout_payload["train_support_fractions"] = train_support_fractions
    holdout_payload["eval_support_fractions"] = eval_support_fractions
    return ExperimentConfig(
        project=ProjectConfig(**payload["project"]),
        paths=PathsConfig(**payload["paths"]),
        outputs=OutputConfig(**payload["outputs"]),
        dataset=SyntheticSequenceConfig(**payload["dataset"]),
        sampling=SamplingConfig(**payload["sampling"]),
        noise=NoiseConfig(**payload["noise"]),
        model=ModelConfig(**payload["model"]),
        training=TrainingConfig(**payload["training"]),
        evaluation=EvalConfig(**payload["evaluation"]),
        holdout=HoldoutConfig(**holdout_payload),
        preset_name=preset_name,
    )


def _resolve_inheritance(yaml_path: Path) -> dict[str, Any]:
    """Load a YAML file and fold in any ``_inherit_from:`` parent chain.

    Supports the thesis-extension overlay convention: a config can start
    with

        _inherit_from: configs/base.yaml

    and only override the fields that differ from its parent. Parent paths
    are resolved relative to the repo root (parent of src/ in the install
    layout). Circular inheritance is guarded with a visited set; missing
    parents raise FileNotFoundError.
    """
    repo_root = Path(__file__).resolve().parents[3]

    def _load(path: Path, seen: set[Path]) -> dict[str, Any]:
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError(f"Cyclic _inherit_from while loading {resolved}")
        seen = seen | {resolved}
        payload = load_yaml(resolved)
        if not isinstance(payload, dict):
            return payload
        parent_ref = payload.pop("_inherit_from", None)
        if parent_ref is None:
            return payload
        parent_path = Path(parent_ref)
        if not parent_path.is_absolute():
            parent_path = repo_root / parent_path
        parent_payload = _load(parent_path, seen)
        return deep_update(parent_payload, payload)

    return _load(yaml_path, seen=set())


def load_experiment_config(
    base_path: str | Path,
    train_path: str | Path,
    eval_path: str | Path,
    preset: str | None = None,
    default_base_path: str | Path | None = None,
) -> ExperimentConfig:
    """Load and merge the default base, evaluation, optional preset, and custom config files.

    Supports a ``_inherit_from: <path>`` directive at the top of either
    the ``base`` or ``eval`` YAMLs; the referenced parent (resolved
    relative to the repo root) is loaded first and then the current file
    overlaid on top. Applied recursively.
    """

    default_base_resolved = Path(default_base_path or base_path).resolve()
    base_path_resolved = Path(base_path).resolve()
    default_base_cfg = _resolve_inheritance(default_base_resolved)
    base_cfg = _resolve_inheritance(base_path_resolved)
    eval_cfg = _resolve_inheritance(Path(eval_path))
    merged = deep_update(default_base_cfg, eval_cfg)
    if preset is not None:
        train_cfg = load_yaml(train_path)
        presets = train_cfg.get("presets", {})
        if preset not in presets:
            available = ", ".join(sorted(presets))
            raise KeyError(f"Unknown preset '{preset}'. Available presets: {available}")
        merged = deep_update(merged, presets[preset])
    if base_path_resolved != default_base_resolved:
        merged = deep_update(merged, base_cfg)
    return _build_config(merged, preset_name=preset or "custom")
