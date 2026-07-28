"""Dataset objects for the fabric defect corpus.

Expected on-disk layout (one folder per defect class)::

    data/fabric/
        baekra/*.jpg
        color_issues/*.jpg
        contamination/*.jpg
        cut/*.jpg
        gray_stitch/*.jpg
        selvet/*.jpg
        stain/*.jpg

The real corpus is available at https://tinyurl.com/488uhkhy. When it is not
present, :func:`make_synthetic_dataset` writes a small procedurally generated
stand-in with the same structure so every stage of the pipeline is runnable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .transforms import pil_loader

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(frozen=True)
class Sample:
    """One labelled image on disk."""

    path: Path
    target: int


class FabricDefectDataset(Dataset):
    """Folder-per-class image dataset returning ``(image, target)`` pairs."""

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        classes: Optional[Sequence[str]] = None,
        samples: Optional[Sequence[Sample]] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        if samples is None:
            self.classes, self.samples = self._scan(self.root, classes)
        else:
            if classes is None:
                raise ValueError("`classes` is required when `samples` is provided")
            self.classes = list(classes)
            self.samples = list(samples)
        self.class_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self.classes)}
        self.targets: List[int] = [s.target for s in self.samples]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _scan(root: Path, classes: Optional[Sequence[str]]) -> Tuple[List[str], List[Sample]]:
        if not root.exists():
            raise FileNotFoundError(
                f"dataset root {root} does not exist. Download the corpus or run "
                "`python -m fabricssl.cli make-sample-data --root <root>`."
            )
        found = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        class_names = list(classes) if classes is not None else found
        if not class_names:
            raise RuntimeError(f"no class sub-directories found under {root}")

        samples: List[Sample] = []
        for index, name in enumerate(class_names):
            class_dir = root / name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append(Sample(path=path, target=index))
        if not samples:
            raise RuntimeError(f"no images with extensions {IMAGE_EXTENSIONS} found under {root}")
        return class_names, samples

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = pil_loader(str(sample.path))
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.target

    # ------------------------------------------------------------------ #
    def subset(self, indices: Sequence[int], transform: Optional[Callable] = None) -> "FabricDefectDataset":
        """Return a new dataset over ``indices`` (optionally re-transformed)."""
        chosen = [self.samples[i] for i in indices]
        return FabricDefectDataset(
            root=self.root,
            transform=self.transform if transform is None else transform,
            classes=self.classes,
            samples=chosen,
        )

    def class_counts(self) -> Dict[str, int]:
        counts = {name: 0 for name in self.classes}
        for target in self.targets:
            counts[self.classes[target]] += 1
        return counts


class ContrastivePairDataset(Dataset):
    """Wrap a labelled dataset and emit two independently augmented views."""

    def __init__(self, base: FabricDefectDataset, transform: Callable) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.base.samples[index]
        image = pil_loader(str(sample.path))
        return self.transform(image), self.transform(image), sample.target


class MultiCropDataset(Dataset):
    """Wrap a labelled dataset and emit a list of crops (used by SwAV)."""

    def __init__(self, base: FabricDefectDataset, transforms: Sequence[Callable]) -> None:
        if not transforms:
            raise ValueError("at least one crop transform is required")
        self.base = base
        self.transforms = list(transforms)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], int]:
        sample = self.base.samples[index]
        image = pil_loader(str(sample.path))
        return [transform(image) for transform in self.transforms], sample.target


def multicrop_collate(batch):
    """Collate crops of possibly different resolutions into a list of tensors."""
    num_crops = len(batch[0][0])
    crops = [torch.stack([item[0][i] for item in batch]) for i in range(num_crops)]
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return crops, targets


# ---------------------------------------------------------------------- #
# Synthetic stand-in data
# ---------------------------------------------------------------------- #
def make_synthetic_dataset(
    root: str | Path,
    classes: Sequence[str],
    images_per_class: int | Sequence[int] = 24,
    image_size: int = 96,
    seed: int = 42,
    overwrite: bool = False,
) -> Path:
    """Generate a tiny procedural fabric-like dataset for smoke tests and CI.

    Each class gets a distinct weave colour plus a class-specific defect
    primitive (blob, line, patch, ...), so a classifier can reach well above
    chance while the whole set stays a few megabytes.
    """
    root = Path(root)
    if root.exists() and any(root.iterdir()) and not overwrite:
        return root

    rng = np.random.default_rng(seed)
    if isinstance(images_per_class, int):
        counts = [images_per_class] * len(classes)
    else:
        counts = list(images_per_class)
        if len(counts) != len(classes):
            raise ValueError("images_per_class must match the number of classes")

    for class_index, (name, count) in enumerate(zip(classes, counts)):
        class_dir = root / name
        class_dir.mkdir(parents=True, exist_ok=True)
        hue = 40 + (class_index * 180) % 200
        for i in range(count):
            image = _synthetic_image(rng, image_size, hue, class_index)
            image.save(class_dir / f"{name}_{i:04d}.png")
    return root


def _synthetic_image(rng: np.random.Generator, size: int, hue: int, class_index: int) -> Image.Image:
    """Woven-looking background plus one class-specific defect primitive."""
    base = np.full((size, size, 3), fill_value=140, dtype=np.float32)
    base[..., class_index % 3] = hue
    # Weave pattern: alternating warp/weft stripes.
    xs = np.arange(size)
    warp = 12.0 * np.sin(xs * (0.7 + 0.05 * class_index))[None, :, None]
    weft = 12.0 * np.cos(xs * (0.5 + 0.05 * class_index))[:, None, None]
    base = base + warp + weft + rng.normal(0.0, 6.0, size=(size, size, 3))
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")

    draw = ImageDraw.Draw(image)
    cx, cy = rng.integers(size // 4, 3 * size // 4, size=2).tolist()
    radius = int(rng.integers(max(3, size // 12), max(5, size // 6)))
    colour = (int(rng.integers(0, 90)), int(rng.integers(0, 90)), int(rng.integers(0, 90)))
    kind = class_index % 5
    if kind == 0:
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=colour)
    elif kind == 1:
        draw.line([cx - 2 * radius, cy, cx + 2 * radius, cy], fill=colour, width=3)
    elif kind == 2:
        draw.rectangle([cx - radius, cy - radius, cx + radius, cy + radius], outline=colour, width=3)
    elif kind == 3:
        draw.line([cx, cy - 2 * radius, cx, cy + 2 * radius], fill=colour, width=3)
    else:
        draw.arc([cx - radius, cy - radius, cx + radius, cy + radius], 0, 250, fill=colour, width=3)
    return image
