"""Augmentation pipelines used for SSL pretraining and supervised evaluation.

The SimCLR/BYOL family of views follows the paper: random resized crop,
horizontal/vertical flip, rotation, colour jitter, colour drop (grayscale),
Gaussian blur and Gaussian noise, plus an optional Sobel edge channel that
emphasises defects close to fabric edges.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from PIL import Image
from torchvision import transforms as T

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class GaussianNoise:
    """Add zero-mean Gaussian noise to a normalised tensor image."""

    def __init__(self, std: float = 0.02, p: float = 0.5) -> None:
        self.std = float(std)
        self.p = float(p)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0 or torch.rand(1).item() > self.p:
            return tensor
        return tensor + torch.randn_like(tensor) * self.std

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{type(self).__name__}(std={self.std}, p={self.p})"


class SobelEdge:
    """Blend a Sobel edge map into the image with probability ``p``.

    Operates on a CHW float tensor in [0, 1] *before* normalisation.
    """

    _KX = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    _KY = _KX.t().contiguous()

    def __init__(self, p: float = 0.2, blend: float = 0.5) -> None:
        self.p = float(p)
        self.blend = float(min(max(blend, 0.0), 1.0))

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.p <= 0 or torch.rand(1).item() > self.p:
            return tensor
        edges = sobel_magnitude(tensor)
        return (1.0 - self.blend) * tensor + self.blend * edges.expand_as(tensor)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{type(self).__name__}(p={self.p}, blend={self.blend})"


def sobel_magnitude(tensor: torch.Tensor) -> torch.Tensor:
    """Return the single-channel Sobel gradient magnitude of a CHW tensor."""
    if tensor.dim() != 3:
        raise ValueError(f"expected a CHW tensor, got shape {tuple(tensor.shape)}")
    gray = tensor.mean(dim=0, keepdim=True).unsqueeze(0)  # 1x1xHxW
    kx = SobelEdge._KX.to(tensor.dtype).view(1, 1, 3, 3)
    ky = SobelEdge._KY.to(tensor.dtype).view(1, 1, 3, 3)
    gx = torch.nn.functional.conv2d(gray, kx, padding=1)
    gy = torch.nn.functional.conv2d(gray, ky, padding=1)
    magnitude = torch.sqrt(gx.pow(2) + gy.pow(2)).squeeze(0)
    peak = magnitude.max()
    if peak > 0:
        magnitude = magnitude / peak
    return magnitude


def _odd_kernel(image_size: int) -> int:
    """SimCLR uses a blur kernel of ~10% of the image size (must be odd)."""
    kernel = max(3, int(0.1 * image_size))
    return kernel if kernel % 2 == 1 else kernel + 1


def build_ssl_transform(image_size: int = 224, sobel_prob: float = 0.0) -> T.Compose:
    """Stochastic augmentation producing one contrastive view."""
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.3, 1.0), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomApply([T.RandomRotation(degrees=30)], p=0.5),
            T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(_odd_kernel(image_size), sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            SobelEdge(p=sobel_prob),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            GaussianNoise(std=0.02, p=0.3),
        ]
    )


def build_train_transform(image_size: int = 224) -> T.Compose:
    """Mild augmentation for supervised fine-tuning."""
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.RandomRotation(degrees=15)], p=0.3),
            T.ColorJitter(0.2, 0.2, 0.2, 0.05),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224) -> T.Compose:
    """Deterministic pipeline for validation / test / inference."""
    resize = int(round(image_size * 1.14))
    return T.Compose(
        [
            T.Resize(resize, antialias=True),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_multicrop_transforms(
    image_size: int = 224,
    small_size: int = 96,
    num_small: int = 2,
    sobel_prob: float = 0.0,
) -> List[T.Compose]:
    """SwAV multi-crop: two global views plus ``num_small`` local views."""
    global_view = build_ssl_transform(image_size, sobel_prob=sobel_prob)
    local_view = T.Compose(
        [
            T.RandomResizedCrop(small_size, scale=(0.05, 0.4), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return [global_view, build_ssl_transform(image_size, sobel_prob=sobel_prob)] + [
        local_view for _ in range(max(0, num_small))
    ]


def denormalize(
    tensor: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """Undo normalisation for visualisation; accepts CHW or NCHW."""
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)
    shape = (1, -1, 1, 1) if tensor.dim() == 4 else (-1, 1, 1)
    return (tensor * std_t.view(*shape) + mean_t.view(*shape)).clamp(0.0, 1.0)


def pil_loader(path: str) -> Image.Image:
    """Load an image file as RGB (robust to palette/grayscale sources)."""
    with open(path, "rb") as handle:
        image = Image.open(handle)
        return image.convert("RGB")
