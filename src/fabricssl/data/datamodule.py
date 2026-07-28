"""Stratified splitting and dataloader construction."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..config import DataConfig
from .datasets import (
    ContrastivePairDataset,
    FabricDefectDataset,
    MultiCropDataset,
    multicrop_collate,
)
from .transforms import (
    build_eval_transform,
    build_multicrop_transforms,
    build_ssl_transform,
    build_train_transform,
)


def stratified_split(
    targets: Sequence[int],
    ratios: Tuple[float, float, float],
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Split indices per class so every subset keeps the class distribution.

    Guarantees at least one sample per class in train/val/test whenever the
    class has three or more samples.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")
    rng = np.random.default_rng(seed)

    per_class: Dict[int, List[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        per_class[int(target)].append(index)

    train: List[int] = []
    val: List[int] = []
    test: List[int] = []
    for target in sorted(per_class):
        indices = np.array(per_class[target])
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_val = min(max(n_val, 1), n - n_train - 1)
        n_train = max(0, min(n_train, n))
        n_val = max(0, min(n_val, n - n_train))
        train.extend(indices[:n_train].tolist())
        val.extend(indices[n_train : n_train + n_val].tolist())
        test.extend(indices[n_train + n_val :].tolist())

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def label_subset(targets: Sequence[int], fraction: float, seed: int = 42) -> List[int]:
    """Keep a stratified ``fraction`` of indices (data-efficiency experiments)."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    if fraction == 1.0:
        return list(range(len(targets)))
    rng = np.random.default_rng(seed)
    per_class: Dict[int, List[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        per_class[int(target)].append(index)
    keep: List[int] = []
    for target in sorted(per_class):
        indices = np.array(per_class[target])
        rng.shuffle(indices)
        n_keep = max(1, int(round(fraction * len(indices))))
        keep.extend(indices[:n_keep].tolist())
    rng.shuffle(keep)
    return keep


def class_balanced_sampler(targets: Sequence[int], num_classes: int) -> WeightedRandomSampler:
    """Inverse-frequency sampler that counteracts the stain-heavy imbalance."""
    counts = np.bincount(np.asarray(targets, dtype=np.int64), minlength=num_classes)
    counts = np.maximum(counts, 1)
    class_weights = 1.0 / counts
    weights = class_weights[np.asarray(targets, dtype=np.int64)]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(targets),
        replacement=True,
    )


def class_weights(targets: Sequence[int], num_classes: int) -> torch.Tensor:
    """Normalised inverse-frequency weights for a weighted cross-entropy."""
    counts = np.bincount(np.asarray(targets, dtype=np.int64), minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32)


class FabricDataModule:
    """Builds the train/val/test datasets and every dataloader flavour."""

    def __init__(self, cfg: DataConfig, method: str = "simclr") -> None:
        self.cfg = cfg
        self.method = method
        self.image_size = cfg.image_size

        base = FabricDefectDataset(root=Path(cfg.root), transform=None)
        self.classes = base.classes
        self.num_classes = len(self.classes)

        train_idx, val_idx, test_idx = stratified_split(base.targets, tuple(cfg.split), seed=cfg.seed)
        eval_tf = build_eval_transform(self.image_size)
        self.train_labelled = base.subset(train_idx, transform=build_train_transform(self.image_size))
        self.train_plain = base.subset(train_idx, transform=eval_tf)
        self.val = base.subset(val_idx, transform=eval_tf)
        self.test = base.subset(test_idx, transform=eval_tf)
        self._pretrain_base = base.subset(train_idx, transform=None)

    # ------------------------------------------------------------------ #
    @property
    def class_distribution(self) -> Dict[str, int]:
        return self.train_labelled.class_counts()

    def _loader_kwargs(self) -> Dict[str, object]:
        workers = max(0, int(self.cfg.num_workers))
        kwargs: Dict[str, object] = {"num_workers": workers, "pin_memory": torch.cuda.is_available()}
        if workers > 0:
            kwargs["persistent_workers"] = True
        return kwargs

    # ------------------------------------------------------------------ #
    def pretrain_loader(self, batch_size: int, drop_last: bool = True) -> DataLoader:
        """Unlabelled (label-agnostic) loader for the SSL pretext task."""
        if self.method == "swav":
            transforms = build_multicrop_transforms(self.image_size, sobel_prob=self.cfg.sobel_prob)
            dataset = MultiCropDataset(self._pretrain_base, transforms)
            collate = multicrop_collate
        else:
            transform = build_ssl_transform(self.image_size, sobel_prob=self.cfg.sobel_prob)
            dataset = ContrastivePairDataset(self._pretrain_base, transform)
            collate = None
        effective_drop_last = drop_last and len(dataset) > batch_size
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=effective_drop_last,
            collate_fn=collate,
            **self._loader_kwargs(),
        )

    def train_loader(
        self,
        batch_size: int,
        label_fraction: float = 1.0,
        balanced: Optional[bool] = None,
        augment: bool = True,
    ) -> DataLoader:
        dataset = self.train_labelled if augment else self.train_plain
        indices = label_subset(dataset.targets, label_fraction, seed=self.cfg.seed)
        if len(indices) != len(dataset):
            dataset = dataset.subset(indices)
        use_balanced = self.cfg.balanced_sampler if balanced is None else balanced
        sampler = class_balanced_sampler(dataset.targets, self.num_classes) if use_balanced else None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=False,
            **self._loader_kwargs(),
        )

    def eval_loader(self, split: str = "val", batch_size: int = 64) -> DataLoader:
        datasets = {"train": self.train_plain, "val": self.val, "test": self.test}
        if split not in datasets:
            raise ValueError(f"split must be one of {sorted(datasets)}, got {split!r}")
        return DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **self._loader_kwargs(),
        )

    def train_class_weights(self) -> torch.Tensor:
        return class_weights(self.train_labelled.targets, self.num_classes)

    def summary(self) -> Dict[str, object]:
        return {
            "root": str(self.cfg.root),
            "classes": self.classes,
            "split": list(self.cfg.split),
            "num_train": len(self.train_labelled),
            "num_val": len(self.val),
            "num_test": len(self.test),
            "train_class_counts": self.class_distribution,
        }
