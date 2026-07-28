"""Typed configuration objects with YAML (de)serialisation."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

VALID_METHODS = ("simclr", "byol", "swav")
VALID_BACKBONES = ("resnet18", "resnet50", "vit", "vit_tiny")


@dataclass
class DataConfig:
    """Dataset location, split ratios and input pipeline settings."""

    root: str = "data/fabric"
    image_size: int = 224
    # (train, val, test); the paper compares 75/15/10, 80/10/10 and 70/20/10.
    split: Tuple[float, float, float] = (0.75, 0.15, 0.10)
    num_workers: int = 2
    balanced_sampler: bool = True
    sobel_prob: float = 0.0
    seed: int = 42


@dataclass
class ModelConfig:
    """Encoder / head architecture."""

    backbone: str = "resnet50"
    pretrained: bool = False
    projection_dim: int = 128
    hidden_dim: int = 2048
    # BYOL specific
    byol_momentum: float = 0.996
    # SwAV specific
    num_prototypes: int = 128
    sinkhorn_iterations: int = 3
    sinkhorn_epsilon: float = 0.05


@dataclass
class OptimConfig:
    """Optimiser / schedule settings shared by every training stage."""

    optimizer: str = "adam"
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-6
    batch_size: int = 64
    epochs: int = 50
    scheduler: str = "cosine"
    warmup_epochs: int = 0
    grad_clip: float = 1.0
    amp: bool = False
    patience: int = 10


@dataclass
class SSLConfig:
    """Self-supervised pretraining stage."""

    method: str = "simclr"
    temperature: float = 0.1
    optim: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class EvalConfig:
    """Linear probe and fine-tuning stages."""

    linear_epochs: int = 20
    finetune_epochs: int = 20
    linear_lr: float = 1.0e-3
    finetune_lr: float = 1.0e-4
    batch_size: int = 64
    label_fraction: float = 1.0


@dataclass
class Config:
    """Top-level experiment configuration."""

    experiment: str = "simclr_resnet50_t0.1"
    output_dir: str = "outputs"
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ------------------------------------------------------------------ #
    # (de)serialisation helpers
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
        return path

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Config:
        return _build(cls, data or {})

    @classmethod
    def load(cls, path: str | Path) -> Config:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls.from_dict(payload)

    def validate(self) -> Config:
        if self.ssl.method not in VALID_METHODS:
            raise ValueError(
                f"ssl.method must be one of {VALID_METHODS}, got {self.ssl.method!r}"
            )
        if self.model.backbone not in VALID_BACKBONES:
            raise ValueError(
                f"model.backbone must be one of {VALID_BACKBONES}, got {self.model.backbone!r}"
            )
        if self.ssl.temperature <= 0:
            raise ValueError("ssl.temperature must be > 0")
        if len(self.data.split) != 3:
            raise ValueError("data.split must contain exactly three ratios")
        total = float(sum(self.data.split))
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"data.split must sum to 1.0, got {total}")
        if any(ratio <= 0 for ratio in self.data.split):
            raise ValueError("every entry of data.split must be > 0")
        if not 0.0 < self.eval.label_fraction <= 1.0:
            raise ValueError("eval.label_fraction must lie in (0, 1]")
        return self


def _build(cls: type, payload: Dict[str, Any]) -> Any:
    """Recursively instantiate nested dataclasses from a plain dict."""
    if not isinstance(payload, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(payload).__name__}")

    known = {f.name: f for f in fields(cls)}
    unknown = set(payload) - set(known)
    if unknown:
        raise ValueError(f"unknown key(s) for {cls.__name__}: {sorted(unknown)}")

    kwargs: Dict[str, Any] = {}
    for name, value in payload.items():
        field_type = known[name].type
        resolved = _resolve(field_type)
        if is_dataclass(resolved):
            kwargs[name] = _build(resolved, value)
        elif name == "split":
            kwargs[name] = tuple(float(v) for v in value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


_DATACLASS_TYPES = {
    "DataConfig": DataConfig,
    "ModelConfig": ModelConfig,
    "OptimConfig": OptimConfig,
    "SSLConfig": SSLConfig,
    "EvalConfig": EvalConfig,
    "Config": Config,
}


def _resolve(field_type: Any) -> Any:
    """Map a (possibly string) dataclass annotation to the actual class."""
    if isinstance(field_type, str):
        return _DATACLASS_TYPES.get(field_type, field_type)
    return field_type


def load_config(path: Optional[str | Path] = None, **overrides: Any) -> Config:
    """Load a config from YAML (or defaults) and apply dotted-key overrides.

    >>> cfg = load_config(None, **{"ssl.temperature": 0.5})
    >>> cfg.ssl.temperature
    0.5
    """
    cfg = Config.load(path) if path else Config()
    for key, value in overrides.items():
        if value is None:
            continue
        _set_dotted(cfg, key, value)
    return cfg.validate()


def _set_dotted(cfg: Any, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target = cfg
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"unknown config section {part!r} in {dotted_key!r}")
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise ValueError(f"unknown config key {dotted_key!r}")
    setattr(target, leaf, value)
