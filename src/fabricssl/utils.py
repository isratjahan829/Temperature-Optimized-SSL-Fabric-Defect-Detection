"""Small shared helpers: seeding, device selection, logging, checkpoints, meters."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

LOGGER_NAME = "fabricssl"


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (CPU + CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """Resolve ``auto`` / ``cuda`` / ``mps`` / ``cpu`` into a real device.

    Falls back to CPU whenever the requested accelerator is unavailable, so the
    same config runs unchanged on a laptop and on a GPU node.
    """
    preference = (preference or "auto").lower()
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preference.startswith("cuda") and not torch.cuda.is_available():
        get_logger().warning("CUDA requested but unavailable - falling back to CPU.")
        return torch.device("cpu")
    if preference == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            get_logger().warning("MPS requested but unavailable - falling back to CPU.")
            return torch.device("cpu")
    return torch.device(preference)


def get_logger(name: str = LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    """Return a process-wide logger with a single stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


class AverageMeter:
    """Running mean of a scalar stream."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    ``mode='min'`` treats lower as better (loss); ``mode='max'`` the opposite.
    """

    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 0.0) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = max(1, int(patience))
        self.mode = mode
        self.min_delta = abs(float(min_delta))
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Feed the latest metric; returns ``True`` when it is a new best."""
        metric = float(metric)
        if self.best is None:
            self.best = metric
            return True
        improved = (
            metric < self.best - self.min_delta
            if self.mode == "min"
            else metric > self.best + self.min_delta
        )
        if improved:
            self.best = metric
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return path


def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    # weights_only=False: our checkpoints also carry the (plain-dict) config.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only kwarg
        return torch.load(path, map_location=map_location)


def save_json(payload: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serialisable")


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    params: Iterable[torch.nn.Parameter] = module.parameters()
    return sum(p.numel() for p in params if p.requires_grad or not trainable_only)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
