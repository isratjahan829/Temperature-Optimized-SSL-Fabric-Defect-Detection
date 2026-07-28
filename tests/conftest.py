"""Shared fixtures: a tiny synthetic dataset and a fast CPU config."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fabricssl.config import Config  # noqa: E402
from fabricssl.data import make_synthetic_dataset  # noqa: E402

CLASSES = ("baekra", "contamination", "cut", "stain")


@pytest.fixture(scope="session")
def torch_single_thread():
    torch.set_num_threads(1)
    return True


@pytest.fixture(scope="session")
def sample_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("fabric_sample")
    make_synthetic_dataset(root, CLASSES, images_per_class=12, image_size=48, seed=0, overwrite=True)
    return root


@pytest.fixture()
def cpu_config(sample_root: Path, tmp_path: Path) -> Config:
    cfg = Config()
    cfg.experiment = "pytest"
    cfg.output_dir = str(tmp_path / "outputs")
    cfg.device = "cpu"
    cfg.seed = 0
    cfg.data.root = str(sample_root)
    cfg.data.image_size = 32
    cfg.data.num_workers = 0
    cfg.data.split = (0.6, 0.2, 0.2)
    cfg.model.backbone = "resnet18"
    cfg.model.hidden_dim = 128
    cfg.model.projection_dim = 32
    cfg.model.num_prototypes = 16
    cfg.ssl.method = "simclr"
    cfg.ssl.temperature = 0.1
    cfg.ssl.optim.epochs = 1
    cfg.ssl.optim.batch_size = 8
    cfg.ssl.optim.patience = 2
    cfg.ssl.optim.amp = False
    cfg.eval.linear_epochs = 1
    cfg.eval.finetune_epochs = 1
    cfg.eval.batch_size = 8
    return cfg.validate()
