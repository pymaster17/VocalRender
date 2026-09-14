from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import argbind
import yaml


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must contain a top-level mapping.")
    return data


def parse_args_with_config(config_path: str | Path | None = None):
    """Backward-compatible argbind helper used by legacy scripts."""
    cli_args = argbind.parse_args()
    if config_path is None:
        return cli_args

    yaml_args = load_yaml_config(config_path)
    with argbind.scope(cli_args):
        yaml_args = argbind.parse_args(yaml_args=yaml_args, argv=[])
    cli_args.update(yaml_args)
    return cli_args


@dataclass
class ModelConfig:
    pretrained_path: str
    lora: Optional[dict] = None
    hf_model_id: str = ""
    distribute: bool = False
    activation_checkpointing: bool = False


@dataclass
class SVSDataConfig:
    preprocessed_data_path: str
    train_datasets: Optional[list] = None
    cloudmusic_max_hours: float = 0.0
    dataset_upsample: Optional[dict] = None
    val_datasets: Optional[list] = None
    val_songs: Optional[list] = None
    val_samples: int = 0
    svs_mask_strategy: Any = "none"
    svs_mask_params: Optional[dict] = None
    prompt_audio_prob: float = 0.0
    prompt_max_frames: int = 50


@dataclass
class TrainConfig:
    sample_rate: int = 44100
    batch_size: int = 1
    grad_accum_steps: int = 1
    num_workers: int = 2
    # DataLoader prefetch depth per worker (passed through to torch
    # DataLoader). Only honored when num_workers > 0.
    prefetch_factor: int = 4
    num_iters: int = 100_000
    log_interval: int = 100
    valid_interval: int = 1_000
    audio_eval_interval: int = -1
    validate_at_step_zero: bool = False
    audio_validate_at_step_zero: bool = False
    save_interval: int = 10_000
    transient_save_interval: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 1_000
    max_steps: int = 100_000
    max_epochs: float = 0.0
    lr_scheduler: str = "cosine"
    max_batch_tokens: int = 0
    val_max_batch_tokens: int = -1
    val_batch_size: int = -1
    lambdas: Dict[str, float] = field(default_factory=lambda: {"loss/diff": 1.0, "loss/stop": 1.0})


@dataclass
class EvalConfig:
    val_audio_samples: int = -1
    val_tb_max_samples: int = 5
    val_max_len: int = 300
    eval_metrics: Optional[dict] = None


@dataclass
class RuntimeConfig:
    save_path: str = "checkpoints"
    tensorboard: str = ""
    timing_enabled: bool = False
    per_rank_diagnostics: bool = False


@dataclass
class DistConfig:
    train_precision: str = "amp_bf16"
    zero_stage: int = 0
    fsdp_sharding_group_size: int = 1


@dataclass
class SVSTrainConfig:
    model: ModelConfig
    data: SVSDataConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    dist: DistConfig = field(default_factory=DistConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SVS_GROUP_FIELDS = {
    "model": {
        "pretrained_path",
        "lora",
        "hf_model_id",
        "distribute",
        "activation_checkpointing",
    },
    "data": {
        "preprocessed_data_path",
        "train_datasets",
        "cloudmusic_max_hours",
        "dataset_upsample",
        "val_datasets",
        "val_songs",
        "val_samples",
        "svs_mask_strategy",
        "svs_mask_params",
        "prompt_audio_prob",
        "prompt_max_frames",
    },
    "train": {
        "sample_rate",
        "batch_size",
        "grad_accum_steps",
        "num_workers",
        "num_iters",
        "log_interval",
        "valid_interval",
        "audio_eval_interval",
        "validate_at_step_zero",
        "audio_validate_at_step_zero",
        "save_interval",
        "transient_save_interval",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "max_steps",
        "max_epochs",
        "lr_scheduler",
        "max_batch_tokens",
        "val_max_batch_tokens",
        "val_batch_size",
        "lambdas",
    },
    "eval": {
        "val_audio_samples",
        "val_tb_max_samples",
        "val_max_len",
        "eval_metrics",
    },
    "runtime": {
        "save_path",
        "tensorboard",
        "timing_enabled",
        "per_rank_diagnostics",
    },
    "dist": {
        "train_precision",
        "zero_stage",
        "fsdp_sharding_group_size",
    },
}


def _ensure_mapping(value: Any, name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{name}' to be a mapping, got {type(value).__name__}.")
    return deepcopy(value)


def _normalize_grouped_config(raw_config: Dict[str, Any], group_fields: Dict[str, set[str]]) -> Dict[str, Dict[str, Any]]:
    raw = deepcopy(raw_config)
    normalized = {group: _ensure_mapping(raw.get(group), group) for group in group_fields}
    consumed = {group for group in group_fields if group in raw}

    for key, value in raw.items():
        if key in consumed:
            continue
        assigned = False
        for group, keys in group_fields.items():
            if key in keys:
                normalized[group][key] = deepcopy(value)
                assigned = True
                break
        if not assigned:
            raise ValueError(f"Unknown configuration key '{key}'.")
    return normalized


def _build_model_config(data: Dict[str, Any]) -> ModelConfig:
    return ModelConfig(**data)


def _build_train_config(data: Dict[str, Any]) -> TrainConfig:
    return TrainConfig(**data)


def _build_eval_config(data: Dict[str, Any]) -> EvalConfig:
    return EvalConfig(**data)


def _build_runtime_config(data: Dict[str, Any]) -> RuntimeConfig:
    return RuntimeConfig(**data)


def _build_dist_config(data: Dict[str, Any]) -> DistConfig:
    return DistConfig(**data)


def load_svs_train_config(source: str | Path | Dict[str, Any]) -> SVSTrainConfig:
    raw = load_yaml_config(source) if isinstance(source, (str, Path)) else deepcopy(source)
    normalized = _normalize_grouped_config(raw, _SVS_GROUP_FIELDS)
    try:
        return SVSTrainConfig(
            model=_build_model_config(normalized["model"]),
            data=SVSDataConfig(**normalized["data"]),
            train=_build_train_config(normalized["train"]),
            eval=_build_eval_config(normalized["eval"]),
            runtime=_build_runtime_config(normalized["runtime"]),
            dist=_build_dist_config(normalized["dist"]),
        )
    except TypeError as exc:
        raise ValueError(f"Invalid SVS training configuration: {exc}") from exc


def _apply_dotted_override(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        elif not isinstance(child, dict):
            raise ValueError(f"Cannot set nested override '{dotted_key}': '{part}' is not a mapping.")
        cursor = child
    cursor[parts[-1]] = value


def apply_cli_overrides(config: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    merged = deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected KEY=VALUE.")
        key, raw_value = item.split("=", 1)
        _apply_dotted_override(merged, key.strip(), yaml.safe_load(raw_value))
    return merged


def parse_script_config(
    *,
    kind: str,
    argv: Optional[list[str]] = None,
) -> SVSTrainConfig:
    parser = argparse.ArgumentParser(description=f"VocalRender {kind.upper()} training")
    parser.add_argument("--config_path", type=str, default="", help="YAML config path")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values with dotted keys, e.g. train.batch_size=2",
    )
    args = parser.parse_args(argv)

    raw = load_yaml_config(args.config_path) if args.config_path else {}
    raw = apply_cli_overrides(raw, args.set)
    if kind == "svs":
        return load_svs_train_config(raw)
    raise ValueError(f"Unsupported config kind '{kind}'.")
