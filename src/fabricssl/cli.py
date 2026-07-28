"""Command line interface: ``python -m fabricssl.cli <command> [options]``.

Commands
--------
make-sample-data   Generate a synthetic stand-in dataset (CI / smoke tests).
pretrain           Stage 1 - self-supervised pretraining.
linear             Stage 2a - linear evaluation of a frozen encoder.
finetune           Stage 2b - supervised fine-tuning.
run                Full pipeline (pretrain -> linear -> finetune -> clustering).
sweep              Temperature / method / backbone grid.
explain            CAM heatmaps + quantitative explainability metrics.
predict            Classify a single image with a trained checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from . import DEFECT_CLASSES, __version__
from .config import Config, load_config
from .utils import ensure_dir, get_device, get_logger, save_json

LOGGER = get_logger()


# ---------------------------------------------------------------------- #
# argument plumbing
# ---------------------------------------------------------------------- #
def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--data-root", dest="data.root", type=str, default=None)
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None)
    parser.add_argument("--experiment", dest="experiment", type=str, default=None)
    parser.add_argument("--device", dest="device", type=str, default=None)
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("--image-size", dest="data.image_size", type=int, default=None)
    parser.add_argument("--num-workers", dest="data.num_workers", type=int, default=None)
    parser.add_argument(
        "--method", dest="ssl.method", type=str, default=None, choices=["simclr", "byol", "swav"]
    )
    parser.add_argument(
        "--backbone",
        dest="model.backbone",
        type=str,
        default=None,
        choices=["resnet18", "resnet50", "vit", "vit_tiny"],
    )
    parser.add_argument("--temperature", dest="ssl.temperature", type=float, default=None)
    parser.add_argument("--epochs", dest="ssl.optim.epochs", type=int, default=None)
    parser.add_argument("--batch-size", dest="ssl.optim.batch_size", type=int, default=None)
    parser.add_argument("--lr", dest="ssl.optim.lr", type=float, default=None)
    parser.add_argument("--optimizer", dest="ssl.optim.optimizer", type=str, default=None)
    parser.add_argument("--linear-epochs", dest="eval.linear_epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", dest="eval.finetune_epochs", type=int, default=None)
    parser.add_argument("--label-fraction", dest="eval.label_fraction", type=float, default=None)


_OVERRIDE_KEYS = {
    "output_dir",
    "experiment",
    "device",
    "seed",
    "data.root",
    "data.image_size",
    "data.num_workers",
    "ssl.method",
    "ssl.temperature",
    "ssl.optim.epochs",
    "ssl.optim.batch_size",
    "ssl.optim.lr",
    "ssl.optim.optimizer",
    "model.backbone",
    "eval.linear_epochs",
    "eval.finetune_epochs",
    "eval.label_fraction",
}


def _config_from_args(args: argparse.Namespace) -> Config:
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key in _OVERRIDE_KEYS and value is not None
    }
    return load_config(args.config, **overrides)


# ---------------------------------------------------------------------- #
# commands
# ---------------------------------------------------------------------- #
def cmd_make_sample_data(args: argparse.Namespace) -> int:
    from .data import make_synthetic_dataset

    root = make_synthetic_dataset(
        root=args.root,
        classes=DEFECT_CLASSES,
        images_per_class=args.images_per_class,
        image_size=args.image_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    total = sum(1 for _ in Path(root).rglob("*.png"))
    LOGGER.info("wrote %d synthetic images to %s", total, root)
    return 0


def cmd_pretrain(args: argparse.Namespace) -> int:
    from .engine.pretrain import pretrain

    cfg = _config_from_args(args)
    result = pretrain(cfg)
    LOGGER.info("best pretrain loss %.4f -> %s", result["best_loss"], result["checkpoint"])
    return 0


def cmd_linear(args: argparse.Namespace) -> int:
    from .engine.downstream import linear_probe

    cfg = _config_from_args(args)
    result = linear_probe(cfg, checkpoint=args.checkpoint)
    _print_metrics("linear evaluation", result["test_metrics"])
    return 0


def cmd_finetune(args: argparse.Namespace) -> int:
    from .engine.downstream import finetune

    cfg = _config_from_args(args)
    result = finetune(cfg, checkpoint=args.checkpoint)
    _print_metrics("fine-tuning", result["test_metrics"])
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_experiment

    cfg = _config_from_args(args)
    summary = run_experiment(
        cfg,
        do_pretrain=not args.skip_pretrain,
        do_linear=not args.skip_linear,
        do_finetune=not args.skip_finetune,
        do_clustering=not args.skip_clustering,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "pretrain_history"}, indent=2))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from .pipeline import temperature_sweep

    cfg = _config_from_args(args)
    rows = temperature_sweep(
        cfg,
        temperatures=args.temperatures,
        methods=args.methods,
        backbones=args.backbones,
    )
    print(json.dumps(rows, indent=2))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    from .data import FabricDataModule, denormalize
    from .engine.downstream import load_trained_classifier
    from .xai.cam import CAM_METHODS, build_cam
    from .xai.metrics import evaluate_cam_methods

    cfg = _config_from_args(args)
    device = get_device(cfg.device)
    model, classes = load_trained_classifier(args.checkpoint, device=device)
    datamodule = FabricDataModule(cfg.data, method=cfg.ssl.method)
    loader = datamodule.eval_loader("test", batch_size=1)

    samples = []
    images_for_plot: List[np.ndarray] = []
    titles: List[str] = []
    for index, (image, target) in enumerate(loader):
        if index >= args.num_samples:
            break
        samples.append((image[0], int(target[0].item()), None))
        images_for_plot.append(denormalize(image[0]).permute(1, 2, 0).cpu().numpy())
        titles.append(classes[int(target[0].item())])

    if not samples:
        LOGGER.error("test split is empty - nothing to explain")
        return 1

    methods = args.methods or list(CAM_METHODS)
    results = evaluate_cam_methods(
        model,
        samples,
        cam_factory=lambda name: build_cam(name, model),
        methods=methods,
        device=device,
    )

    out_dir = ensure_dir(Path(cfg.output_dir) / cfg.experiment)
    save_json(results, out_dir / "explainability_metrics.json")
    print(json.dumps(results, indent=2))

    if args.plot:
        from .analysis.plots import plot_cam_grid

        heatmaps: Dict[str, List[np.ndarray]] = {}
        for method in methods:
            with build_cam(method, model) as cam:
                heatmaps[method] = [cam(image.to(device), target) for image, target, _ in samples]
        plot_cam_grid(images_for_plot, heatmaps, titles, path=out_dir / "cam_grid.png")
        LOGGER.info("wrote %s", out_dir / "cam_grid.png")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .data.transforms import build_eval_transform, pil_loader
    from .engine.downstream import load_trained_classifier

    device = get_device(args.device or "auto")
    model, classes = load_trained_classifier(args.checkpoint, device=device)
    transform = build_eval_transform(args.image_size)
    tensor = transform(pil_loader(args.image)).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    order = np.argsort(-probabilities)[: min(args.topk, len(classes))]
    payload = {
        "image": args.image,
        "prediction": classes[int(order[0])],
        "confidence": float(probabilities[int(order[0])]),
        "topk": [{"class": classes[int(i)], "probability": float(probabilities[int(i)])} for i in order],
    }
    print(json.dumps(payload, indent=2))
    return 0


def _print_metrics(title: str, metrics: object) -> None:
    assert isinstance(metrics, dict)
    print(f"\n=== {title} (test split) ===")
    for key in ("accuracy", "top1", "top5", "macro_f1", "weighted_f1", "macro_auc", "mAP"):
        if key in metrics:
            print(f"{key:>12}: {metrics[key]:.4f}")


# ---------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fabricssl",
        description="Temperature-optimized SSL for fabric defect detection",
    )
    parser.add_argument("--version", action="version", version=f"fabricssl {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("make-sample-data", help="generate a synthetic dataset")
    sample.add_argument("--root", type=str, default="data/sample")
    sample.add_argument("--images-per-class", type=int, default=24)
    sample.add_argument("--image-size", type=int, default=96)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--overwrite", action="store_true")
    sample.set_defaults(func=cmd_make_sample_data)

    pre = subparsers.add_parser("pretrain", help="self-supervised pretraining")
    _add_common(pre)
    pre.set_defaults(func=cmd_pretrain)

    linear = subparsers.add_parser("linear", help="linear evaluation of a frozen encoder")
    _add_common(linear)
    linear.add_argument("--checkpoint", type=str, required=True)
    linear.set_defaults(func=cmd_linear)

    fine = subparsers.add_parser("finetune", help="supervised fine-tuning")
    _add_common(fine)
    fine.add_argument("--checkpoint", type=str, default=None)
    fine.set_defaults(func=cmd_finetune)

    run = subparsers.add_parser("run", help="run the full pipeline")
    _add_common(run)
    run.add_argument("--skip-pretrain", action="store_true")
    run.add_argument("--skip-linear", action="store_true")
    run.add_argument("--skip-finetune", action="store_true")
    run.add_argument("--skip-clustering", action="store_true")
    run.set_defaults(func=cmd_run)

    sweep = subparsers.add_parser("sweep", help="temperature / method / backbone grid")
    _add_common(sweep)
    sweep.add_argument("--temperatures", type=float, nargs="+", default=[0.1, 0.5, 0.8])
    sweep.add_argument("--methods", type=str, nargs="+", default=["simclr"])
    sweep.add_argument("--backbones", type=str, nargs="+", default=["resnet50"])
    sweep.set_defaults(func=cmd_sweep)

    explain = subparsers.add_parser("explain", help="CAM heatmaps and explainability metrics")
    _add_common(explain)
    explain.add_argument("--checkpoint", type=str, required=True)
    explain.add_argument("--num-samples", type=int, default=8)
    explain.add_argument("--methods", type=str, nargs="+", default=None)
    explain.add_argument("--plot", action="store_true")
    explain.set_defaults(func=cmd_explain)

    predict = subparsers.add_parser("predict", help="classify one image")
    predict.add_argument("--checkpoint", type=str, required=True)
    predict.add_argument("--image", type=str, required=True)
    predict.add_argument("--image-size", type=int, default=224)
    predict.add_argument("--topk", type=int, default=3)
    predict.add_argument("--device", type=str, default=None)
    predict.set_defaults(func=cmd_predict)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
