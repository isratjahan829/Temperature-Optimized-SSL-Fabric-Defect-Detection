"""Driver for the paper's experiment matrix.

Examples
--------
    # single best configuration
    python scripts/run_experiments.py --config configs/simclr_resnet50_t01.yaml

    # full temperature x method x backbone grid
    python scripts/run_experiments.py --config configs/simclr_resnet50_t01.yaml --mode sweep

    # data-split study / label-efficiency study
    python scripts/run_experiments.py --config configs/simclr_resnet50_t01.yaml --mode splits
    python scripts/run_experiments.py --config configs/simclr_resnet50_t01.yaml --mode labels
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fabricssl.config import load_config  # noqa: E402
from fabricssl.pipeline import (  # noqa: E402
    label_efficiency_sweep,
    run_experiment,
    split_sweep,
    temperature_sweep,
)
from fabricssl.utils import get_logger  # noqa: E402

LOGGER = get_logger()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="YAML config (defaults to library defaults)")
    parser.add_argument(
        "--mode",
        choices=["single", "sweep", "splits", "labels"],
        default="single",
        help="which experiment family to run",
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.1, 0.5, 0.8])
    parser.add_argument("--methods", type=str, nargs="+", default=["simclr", "byol", "swav"])
    parser.add_argument("--backbones", type=str, nargs="+", default=["resnet18", "resnet50", "vit"])
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    overrides = {}
    if args.data_root:
        overrides["data.root"] = args.data_root
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.device:
        overrides["device"] = args.device
    cfg = load_config(args.config, **overrides)

    if args.mode == "single":
        result = run_experiment(cfg)
        LOGGER.info("done: %s", json.dumps(result.get("finetune", {}), indent=2))
    elif args.mode == "sweep":
        rows = temperature_sweep(
            cfg, temperatures=args.temperatures, methods=args.methods, backbones=args.backbones
        )
        LOGGER.info("%d sweep runs complete", len(rows))
    elif args.mode == "splits":
        rows = split_sweep(cfg)
        LOGGER.info("%d split runs complete", len(rows))
    else:
        rows = label_efficiency_sweep(cfg, fractions=args.fractions)
        LOGGER.info("%d label-fraction runs complete", len(rows))

    LOGGER.info("artefacts written under %s", Path(cfg.output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
