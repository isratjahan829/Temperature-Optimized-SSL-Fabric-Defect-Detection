"""Figure helpers (Agg backend, so they work headless in CI and notebooks)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend switch)
import numpy as np  # noqa: E402

from ..utils import ensure_dir  # noqa: E402


def _finish(fig: plt.Figure, path: Optional[str | Path]) -> plt.Figure:
    fig.tight_layout()
    if path is not None:
        path = Path(path)
        ensure_dir(path.parent)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curve(
    history: Sequence[Dict[str, float]],
    keys: Sequence[str] = ("loss",),
    title: str = "Training curve",
    path: Optional[str | Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = [row.get("epoch", i + 1) for i, row in enumerate(history)]
    for key in keys:
        values = [row[key] for row in history if key in row]
        if values:
            ax.plot(epochs[: len(values)], values, marker="o", markersize=3, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(fig, path)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    normalize: bool = False,
    title: str = "Confusion matrix",
    path: Optional[str | Path] = None,
) -> plt.Figure:
    matrix = np.asarray(matrix, dtype=np.float64)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / np.clip(row_sums, 1e-12, None)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    fmt = "{:.2f}" if normalize else "{:.0f}"
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                fmt.format(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[i, j] > threshold else "black",
            )
    return _finish(fig, path)


def plot_roc_curves(
    curves: Dict[int, Dict[str, Sequence[float]]],
    class_names: Sequence[str],
    title: str = "Multi-class ROC",
    path: Optional[str | Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    for class_index, curve in sorted(curves.items()):
        name = class_names[class_index] if class_index < len(class_names) else str(class_index)
        ax.plot(curve["fpr"], curve["tpr"], lw=1.6, label=f"{name} (AUC={curve['auc']:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    return _finish(fig, path)


def plot_class_distribution(
    counts: Dict[str, int],
    title: str = "Class distribution",
    path: Optional[str | Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(counts)
    values = [counts[name] for name in names]
    ax.bar(names, values, color="#4C78A8")
    ax.set_ylabel("Samples")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    for index, value in enumerate(values):
        ax.text(index, value, str(value), ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _finish(fig, path)


def plot_tsne(
    embedding: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    title: str = "t-SNE of encoder embeddings",
    path: Optional[str | Path] = None,
) -> plt.Figure:
    embedding = np.asarray(embedding)
    labels = np.asarray(labels).reshape(-1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colours = plt.cm.tab10(np.linspace(0, 1, max(1, len(class_names))))
    for class_index, name in enumerate(class_names):
        mask = labels == class_index
        if mask.any():
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=14,
                alpha=0.75,
                color=colours[class_index % len(colours)],
                label=name,
            )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, markerscale=1.5)
    return _finish(fig, path)


def plot_temperature_sweep(
    results: Dict[float, Dict[str, float]],
    metrics: Sequence[str] = ("linear_accuracy", "finetune_accuracy"),
    title: str = "Accuracy vs contrastive temperature",
    path: Optional[str | Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    temperatures = sorted(results)
    for metric in metrics:
        values = [results[t].get(metric, float("nan")) for t in temperatures]
        ax.plot(temperatures, values, marker="o", label=metric.replace("_", " "))
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(fig, path)


def plot_cam_grid(
    images: Sequence[np.ndarray],
    heatmaps: Dict[str, Sequence[np.ndarray]],
    titles: Optional[Sequence[str]] = None,
    path: Optional[str | Path] = None,
) -> plt.Figure:
    """Rows = CAM methods (plus the original image), columns = samples."""
    from ..xai.cam import overlay_heatmap

    methods = list(heatmaps)
    n_rows = len(methods) + 1
    n_cols = len(images)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows), squeeze=False)

    for col, image in enumerate(images):
        axes[0][col].imshow(np.clip(image, 0, 1))
        axes[0][col].set_title(titles[col] if titles is not None else f"sample {col}", fontsize=9)
        axes[0][col].axis("off")
        for row, method in enumerate(methods, start=1):
            axes[row][col].imshow(overlay_heatmap(image, heatmaps[method][col]))
            axes[row][col].axis("off")
            if col == 0:
                axes[row][col].set_ylabel(method)
                axes[row][col].set_title(method, fontsize=9, loc="left")
    return _finish(fig, path)
