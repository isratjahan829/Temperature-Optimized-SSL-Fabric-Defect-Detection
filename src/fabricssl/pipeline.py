"""End-to-end experiment orchestration: pretrain -> probe -> fine-tune -> analyse."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .analysis.clustering import clustering_metrics
from .config import Config
from .data import FabricDataModule
from .engine.downstream import extract_features, finetune, linear_probe, load_trained_classifier
from .engine.pretrain import pretrain
from .utils import ensure_dir, get_device, get_logger, save_json, set_seed

LOGGER = get_logger()


def run_experiment(
    cfg: Config,
    do_pretrain: bool = True,
    do_linear: bool = True,
    do_finetune: bool = True,
    do_clustering: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Run every stage for a single configuration and persist a summary."""
    cfg.validate()
    set_seed(cfg.seed)
    device = device or get_device(cfg.device)
    datamodule = FabricDataModule(cfg.data, method=cfg.ssl.method)
    out_dir = ensure_dir(Path(cfg.output_dir) / cfg.experiment)

    summary: Dict[str, object] = {
        "experiment": cfg.experiment,
        "method": cfg.ssl.method,
        "backbone": cfg.model.backbone,
        "temperature": cfg.ssl.temperature,
        "device": str(device),
        "data": datamodule.summary(),
    }

    checkpoint: Optional[str] = None
    if do_pretrain:
        pretrain_result = pretrain(cfg, datamodule=datamodule, device=device)
        checkpoint = str(pretrain_result["checkpoint"])
        summary["pretrain_loss"] = pretrain_result["best_loss"]
        summary["pretrain_history"] = pretrain_result["history"]
        summary["encoder_checkpoint"] = checkpoint

    if do_linear:
        linear_result = linear_probe(cfg, checkpoint=checkpoint, datamodule=datamodule, device=device)
        summary["linear"] = _stage_digest(linear_result)

    finetune_checkpoint: Optional[str] = None
    if do_finetune:
        finetune_result = finetune(cfg, checkpoint=checkpoint, datamodule=datamodule, device=device)
        summary["finetune"] = _stage_digest(finetune_result)
        summary["finetune_full"] = finetune_result["test_metrics"]
        finetune_checkpoint = str(finetune_result["checkpoint"])

    if do_clustering and finetune_checkpoint is not None:
        model, _classes = load_trained_classifier(finetune_checkpoint, device=device)
        features, targets = extract_features(model, datamodule.eval_loader("test", cfg.eval.batch_size), device)
        summary["clustering"] = clustering_metrics(features, targets, seed=cfg.seed)
        np.savez(out_dir / "test_features.npz", features=features, targets=targets)

    save_json(summary, out_dir / "summary.json")
    return summary


def _stage_digest(result: Dict[str, object]) -> Dict[str, float]:
    metrics = result["test_metrics"]
    assert isinstance(metrics, dict)
    return {
        "accuracy": float(metrics["accuracy"]),
        "top1": float(metrics["top1"]),
        "top5": float(metrics["top5"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "mAP": float(metrics.get("mAP", float("nan"))),
        "best_val_accuracy": float(result["best_val_accuracy"]),
    }


def temperature_sweep(
    base_cfg: Config,
    temperatures: Sequence[float] = (0.1, 0.5, 0.8),
    methods: Sequence[str] = ("simclr",),
    backbones: Sequence[str] = ("resnet50",),
    device: Optional[torch.device] = None,
) -> List[Dict[str, object]]:
    """Reproduce the paper's temperature x method x backbone grid."""
    rows: List[Dict[str, object]] = []
    for method in methods:
        for backbone in backbones:
            for temperature in temperatures:
                cfg = copy.deepcopy(base_cfg)
                cfg.ssl.method = method
                cfg.ssl.temperature = float(temperature)
                cfg.model.backbone = backbone
                cfg.experiment = f"{method}_{backbone}_T{temperature}"
                LOGGER.info("=== sweep run: %s ===", cfg.experiment)
                summary = run_experiment(cfg, device=device)
                rows.append(
                    {
                        "method": method,
                        "backbone": backbone,
                        "temperature": float(temperature),
                        "pretrain_loss": summary.get("pretrain_loss"),
                        "linear_accuracy": _get(summary, "linear", "accuracy"),
                        "finetune_accuracy": _get(summary, "finetune", "accuracy"),
                        "top1": _get(summary, "finetune", "top1"),
                        "top5": _get(summary, "finetune", "top5"),
                        "macro_f1": _get(summary, "finetune", "macro_f1"),
                        "weighted_f1": _get(summary, "finetune", "weighted_f1"),
                    }
                )
    out_path = Path(base_cfg.output_dir) / "temperature_sweep.json"
    save_json(rows, out_path)
    return rows


def split_sweep(
    base_cfg: Config,
    splits: Sequence[Sequence[float]] = ((0.75, 0.15, 0.10), (0.80, 0.10, 0.10), (0.70, 0.20, 0.10)),
    device: Optional[torch.device] = None,
) -> List[Dict[str, object]]:
    """Compare the three train/val/test splits studied in the paper (Table 10)."""
    rows: List[Dict[str, object]] = []
    for split in splits:
        cfg = copy.deepcopy(base_cfg)
        cfg.data.split = tuple(float(v) for v in split)
        tag = "-".join(str(int(round(v * 100))) for v in cfg.data.split)
        cfg.experiment = f"{base_cfg.experiment}_split{tag}"
        LOGGER.info("=== split run: %s ===", cfg.experiment)
        summary = run_experiment(cfg, device=device)
        rows.append(
            {
                "split": list(cfg.data.split),
                "pretrain_loss": summary.get("pretrain_loss"),
                "linear_accuracy": _get(summary, "linear", "accuracy"),
                "finetune_accuracy": _get(summary, "finetune", "accuracy"),
                "top1": _get(summary, "finetune", "top1"),
                "top5": _get(summary, "finetune", "top5"),
            }
        )
    save_json(rows, Path(base_cfg.output_dir) / "split_sweep.json")
    return rows


def label_efficiency_sweep(
    base_cfg: Config,
    fractions: Sequence[float] = (0.1, 0.25, 0.5, 1.0),
    encoder_checkpoint: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> List[Dict[str, object]]:
    """Fine-tune with progressively fewer labels - the data-efficiency claim."""
    device = device or get_device(base_cfg.device)
    datamodule = FabricDataModule(base_cfg.data, method=base_cfg.ssl.method)

    checkpoint = encoder_checkpoint
    if checkpoint is None:
        checkpoint = str(pretrain(base_cfg, datamodule=datamodule, device=device)["checkpoint"])

    rows: List[Dict[str, object]] = []
    for fraction in fractions:
        cfg = copy.deepcopy(base_cfg)
        cfg.eval.label_fraction = float(fraction)
        cfg.experiment = f"{base_cfg.experiment}_labels{int(round(fraction * 100))}"
        result = finetune(cfg, checkpoint=checkpoint, datamodule=datamodule, device=device)
        metrics = result["test_metrics"]
        assert isinstance(metrics, dict)
        rows.append(
            {
                "label_fraction": float(fraction),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
            }
        )
        LOGGER.info("labels %.0f%% -> accuracy %.4f", fraction * 100, metrics["accuracy"])
    save_json(rows, Path(base_cfg.output_dir) / "label_efficiency.json")
    return rows


def _get(summary: Dict[str, object], stage: str, key: str) -> Optional[float]:
    stage_data = summary.get(stage)
    if isinstance(stage_data, dict) and key in stage_data:
        return float(stage_data[key])
    return None
