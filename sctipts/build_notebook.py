"""Generate ``notebooks/Fabric_Defect_SSL_Pipeline.ipynb`` from this script.

Keeping the notebook's source in a plain ``.py`` file makes it reviewable in
git and lets CI regenerate + execute it to prove every cell still runs::

    python scripts/build_notebook.py            # write the notebook
    python scripts/build_notebook.py --execute  # write and run it end-to-end
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "Fabric_Defect_SSL_Pipeline.ipynb"

MD: List[str] = []
CODE: List[str] = []
CELLS: List[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# --------------------------------------------------------------------- #
md(
    """
# Temperature-Optimized Self-Supervised Contrastive Learning for Fabric Defect Detection

Reference implementation of **SimCLR / BYOL / SwAV** pretraining with
**ResNet-18 / ResNet-50 / ViT** backbones, linear evaluation, supervised
fine-tuning, explainability (Grad-CAM family) and latent-space analysis.

This notebook runs **end-to-end on CPU in a few minutes** using a small
procedurally generated stand-in dataset, so every cell is executable without
the private industrial corpus. To reproduce the paper's numbers, point
`DATA_ROOT` at the real dataset and switch to the full settings marked
`# PAPER SETTING` in the config cell.

**Pipeline:** data -> augmentation -> SSL pretraining -> linear probe ->
fine-tuning -> metrics -> XAI -> clustering -> statistics.
"""
)

md("## 1. Environment and imports")

code(
    """
import os
import sys
from pathlib import Path

# Make `src/` importable when the notebook runs from a fresh clone.
REPO_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("OMP_NUM_THREADS", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import fabricssl
from fabricssl.config import Config
from fabricssl.utils import get_device, set_seed

print("fabricssl", fabricssl.__version__)
print("torch", torch.__version__)
print("device", get_device("auto"))
"""
)

md("## 2. Configuration\n\nEvery hyperparameter of the study lives in one typed config object.")

code(
    """
SMOKE = True  # set to False to use the paper's full settings

cfg = Config()
cfg.experiment = "notebook_simclr_resnet18" if SMOKE else "simclr_resnet50_T0.1"
cfg.output_dir = str(REPO_ROOT / "outputs")
cfg.device = "auto"
cfg.seed = 42

cfg.data.root = str(REPO_ROOT / "data" / "sample")
cfg.data.split = (0.75, 0.15, 0.10)   # PAPER SETTING: also 80/10/10 and 70/20/10
cfg.data.num_workers = 0
cfg.data.sobel_prob = 0.2

cfg.ssl.method = "simclr"             # simclr | byol | swav
cfg.ssl.temperature = 0.1             # PAPER SETTING: best of {0.1, 0.5, 0.8}
cfg.ssl.optim.lr = 3e-4
cfg.ssl.optim.optimizer = "adam"
cfg.model.projection_dim = 128

if SMOKE:
    cfg.data.image_size = 64
    cfg.model.backbone = "resnet18"
    cfg.model.hidden_dim = 512
    cfg.ssl.optim.epochs = 3
    cfg.ssl.optim.batch_size = 16
    cfg.eval.linear_epochs = 3
    cfg.eval.finetune_epochs = 3
    cfg.eval.batch_size = 16
else:
    cfg.data.root = str(REPO_ROOT / "data" / "fabric")   # real corpus
    cfg.data.image_size = 224
    cfg.model.backbone = "resnet50"
    cfg.model.hidden_dim = 2048
    cfg.ssl.optim.epochs = 50
    cfg.ssl.optim.batch_size = 64
    cfg.eval.linear_epochs = 30
    cfg.eval.finetune_epochs = 30
    cfg.eval.batch_size = 64

cfg.validate()
set_seed(cfg.seed)
DEVICE = get_device(cfg.device)
pd.Series(
    {
        "method": cfg.ssl.method,
        "backbone": cfg.model.backbone,
        "temperature": cfg.ssl.temperature,
        "epochs": cfg.ssl.optim.epochs,
        "batch_size": cfg.ssl.optim.batch_size,
        "lr": cfg.ssl.optim.lr,
        "image_size": cfg.data.image_size,
        "split": str(cfg.data.split),
    },
    name="value",
).to_frame()
"""
)

md(
    """
## 3. Dataset

The real corpus (7 defect classes: *baekra, color issues, contamination, cut,
gray stitch, selvet, stain*) is described in the paper and hosted at
<https://tinyurl.com/488uhkhy>. Expected layout:

```
data/fabric/
    baekra/*.jpg
    color_issues/*.jpg
    ...
```

If that folder is absent we generate a synthetic stand-in with the same
structure and a deliberately imbalanced class distribution.
"""
)

code(
    """
from fabricssl import DEFECT_CLASSES
from fabricssl.data import FabricDataModule, make_synthetic_dataset

data_root = Path(cfg.data.root)
if not data_root.exists() or not any(data_root.iterdir()):
    print(f"{data_root} not found - generating a synthetic stand-in dataset")
    # Imbalanced on purpose: 'stain' dominates, mirroring the real corpus.
    make_synthetic_dataset(
        root=data_root,
        classes=DEFECT_CLASSES,
        images_per_class=[30, 34, 40, 46, 30, 28, 90],
        image_size=cfg.data.image_size,
        seed=cfg.seed,
    )

datamodule = FabricDataModule(cfg.data, method=cfg.ssl.method)
summary = datamodule.summary()
print(f"classes : {summary['classes']}")
print(f"train/val/test : {summary['num_train']}/{summary['num_val']}/{summary['num_test']}")
pd.Series(summary["train_class_counts"], name="train samples").to_frame()
"""
)

code(
    """
from fabricssl.analysis.plots import plot_class_distribution

plot_class_distribution(datamodule.class_distribution, title="Training class distribution")
plt.show()
"""
)

md(
    """
## 4. Augmentation pipeline

Two independently augmented views of the same image form a positive pair
(crop, flip, rotation, colour jitter, colour drop, blur, Gaussian noise and an
optional Sobel edge blend).
"""
)

code(
    """
from fabricssl.data import build_ssl_transform, denormalize
from fabricssl.data.datasets import FabricDefectDataset
from fabricssl.data.transforms import pil_loader

raw = FabricDefectDataset(cfg.data.root)
sample_path = raw.samples[0].path
image = pil_loader(str(sample_path))
ssl_transform = build_ssl_transform(cfg.data.image_size, sobel_prob=cfg.data.sobel_prob)

fig, axes = plt.subplots(1, 6, figsize=(15, 2.8))
axes[0].imshow(image.resize((cfg.data.image_size, cfg.data.image_size)))
axes[0].set_title("original", fontsize=9)
axes[0].axis("off")
for i in range(1, 6):
    view = denormalize(ssl_transform(image)).permute(1, 2, 0).numpy()
    axes[i].imshow(view)
    axes[i].set_title(f"view {i}", fontsize=9)
    axes[i].axis("off")
plt.tight_layout()
plt.show()
"""
)

md(
    """
## 5. Stage 1 - self-supervised pretraining

SimCLR minimises the NT-Xent loss

$$\\mathcal{L}_{i,j} = -\\log \\frac{\\exp(\\mathrm{sim}(z_i,z_j)/T)}{\\sum_{k \\neq i}\\exp(\\mathrm{sim}(z_i,z_k)/T)}$$

where the temperature $T$ is exactly the quantity this study optimises.
BYOL replaces negatives with an EMA target network; SwAV uses swapped
prototype assignments computed with Sinkhorn-Knopp.
"""
)

code(
    """
from fabricssl.engine.pretrain import pretrain

pretrain_result = pretrain(cfg, datamodule=datamodule, device=DEVICE)
encoder_ckpt = pretrain_result["checkpoint"]
print(f"best pretraining loss: {pretrain_result['best_loss']:.4f}")
print(f"encoder checkpoint  : {encoder_ckpt}")
pd.DataFrame(pretrain_result["history"])
"""
)

code(
    """
from fabricssl.analysis.plots import plot_training_curve

plot_training_curve(pretrain_result["history"], keys=("loss",), title="Contrastive pretraining loss")
plt.show()
"""
)

md(
    """
## 6. Stage 2a - linear evaluation

The encoder is frozen and only a linear classifier is trained. This measures
how *transferable* the learned representation is - the metric where SimCLR at
$T=0.1$ leads in the paper.
"""
)

code(
    """
from fabricssl.engine.downstream import linear_probe

linear_result = linear_probe(cfg, checkpoint=encoder_ckpt, datamodule=datamodule, device=DEVICE)
linear_metrics = linear_result["test_metrics"]
print(f"linear accuracy : {linear_metrics['accuracy']:.4f}")
print(f"macro F1        : {linear_metrics['macro_f1']:.4f}")
"""
)

md("## 7. Stage 2b - supervised fine-tuning")

code(
    """
from fabricssl.engine.downstream import finetune

finetune_result = finetune(cfg, checkpoint=encoder_ckpt, datamodule=datamodule, device=DEVICE)
metrics = finetune_result["test_metrics"]
classifier_ckpt = finetune_result["checkpoint"]

pd.Series(
    {
        "accuracy": metrics["accuracy"],
        "top-1": metrics["top1"],
        "top-5": metrics["top5"],
        "macro F1": metrics["macro_f1"],
        "weighted F1": metrics["weighted_f1"],
        "macro AUC": metrics["macro_auc"],
        "mAP": metrics["mAP"],
    },
    name="fine-tuned",
).to_frame().round(4)
"""
)

md("## 8. Class-wise analysis, confusion matrix and ROC curves")

code(
    """
per_class = pd.DataFrame(metrics["per_class"]).T
per_class.index.name = "defect class"
per_class.round(3)
"""
)

code(
    """
from fabricssl.analysis.plots import plot_confusion_matrix, plot_roc_curves
from fabricssl.engine.metrics import roc_curves

plot_confusion_matrix(
    np.array(metrics["confusion_matrix"]), datamodule.classes, title="Confusion matrix (test)"
)
plt.show()

logits_file = np.load(Path(cfg.output_dir) / cfg.experiment / "finetune_test_logits.npz")
curves = roc_curves(logits_file["logits"], logits_file["targets"], datamodule.num_classes)
plot_roc_curves(curves, datamodule.classes, title="One-vs-rest ROC (test)")
plt.show()
"""
)

md(
    """
## 9. Temperature sweep

The paper's headline result: lower temperatures give lower pretraining loss and
better transfer. Here we sweep a short schedule; with `SMOKE = False` this
reproduces the full $T \\in \\{0.1, 0.5, 0.8\\}$ grid.
"""
)

code(
    """
import copy

from fabricssl.engine.downstream import linear_probe as _linear_probe

sweep_rows = []
for temperature in [0.1, 0.5, 0.8]:
    sweep_cfg = copy.deepcopy(cfg)
    sweep_cfg.ssl.temperature = temperature
    sweep_cfg.experiment = f"{cfg.experiment}_T{temperature}"
    result = pretrain(sweep_cfg, datamodule=datamodule, device=DEVICE)
    probe = _linear_probe(sweep_cfg, checkpoint=result["checkpoint"], datamodule=datamodule, device=DEVICE)
    sweep_rows.append(
        {
            "temperature": temperature,
            "pretrain_loss": result["best_loss"],
            "linear_accuracy": probe["test_metrics"]["accuracy"],
            "macro_f1": probe["test_metrics"]["macro_f1"],
        }
    )

sweep_df = pd.DataFrame(sweep_rows)
sweep_df.round(4)
"""
)

code(
    """
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(sweep_df["temperature"], sweep_df["pretrain_loss"], marker="o", color="#b3261e")
axes[0].set_xlabel("temperature T")
axes[0].set_ylabel("pretraining loss")
axes[0].set_title("Contrastive loss vs temperature")
axes[0].grid(alpha=0.3)

axes[1].plot(sweep_df["temperature"], sweep_df["linear_accuracy"], marker="o", label="linear acc")
axes[1].plot(sweep_df["temperature"], sweep_df["macro_f1"], marker="s", label="macro F1")
axes[1].set_xlabel("temperature T")
axes[1].set_title("Transfer quality vs temperature")
axes[1].grid(alpha=0.3)
axes[1].legend()
plt.tight_layout()
plt.show()
"""
)

md(
    """
## 10. Explainability

Six CAM variants are evaluated with four quantitative metrics:

| metric | meaning |
| --- | --- |
| IoU-Heat | overlap of the thresholded heatmap with the annotated defect region |
| PointAcc | does the peak activation land inside the defect region? |
| Relevance | share of heatmap mass inside the defect region |
| Fidelity | correlation between CAM importance and the confidence drop under occlusion |

> With no hand-drawn masks available, a Grad-CAM-derived pseudo-mask is used, so
> the numbers are only comparable *between methods*, not against the paper's
> annotated IoU of 0.92.
"""
)

code(
    """
from fabricssl.engine.downstream import load_trained_classifier
from fabricssl.xai import CAM_METHODS, build_cam, evaluate_cam_methods

model, class_names = load_trained_classifier(classifier_ckpt, device=DEVICE)

test_loader = datamodule.eval_loader("test", batch_size=1)
samples, preview_images, preview_titles = [], [], []
for index, (img, target) in enumerate(test_loader):
    if index >= 4:
        break
    samples.append((img[0], int(target[0]), None))
    preview_images.append(denormalize(img[0]).permute(1, 2, 0).numpy())
    preview_titles.append(class_names[int(target[0])])

cam_scores = evaluate_cam_methods(
    model,
    samples,
    cam_factory=lambda name: build_cam(name, model),
    methods=list(CAM_METHODS),
    device=DEVICE,
)
pd.DataFrame(cam_scores).T.round(3).sort_values("iou", ascending=False)
"""
)

code(
    """
from fabricssl.analysis.plots import plot_cam_grid

heatmaps = {}
for method in ["gradcam", "gradcam++", "layercam", "hirescam"]:
    with build_cam(method, model) as cam:
        heatmaps[method] = [cam(img.to(DEVICE), target) for img, target, _ in samples]

plot_cam_grid(preview_images, heatmaps, preview_titles)
plt.show()
"""
)

md("## 11. Latent-space analysis: clustering quality and t-SNE")

code(
    """
from fabricssl.analysis.clustering import clustering_metrics, tsne_embedding
from fabricssl.analysis.plots import plot_tsne
from fabricssl.engine.downstream import extract_features

features, targets = extract_features(model, datamodule.eval_loader("test", cfg.eval.batch_size), DEVICE)
cluster_scores = clustering_metrics(features, targets, seed=cfg.seed)
print({k: round(v, 4) for k, v in cluster_scores.items()})

embedding = tsne_embedding(features, seed=cfg.seed)
plot_tsne(embedding, targets, class_names, title="t-SNE of encoder embeddings")
plt.show()
"""
)

md(
    """
## 12. Statistical validation

The paper validates its gains with paired t-tests across folds. The cell below
shows the exact procedure; replace the illustrative per-fold scores with your
own cross-validation results.
"""
)

code(
    """
from fabricssl.analysis.stats import compare_models, summarize_folds

fold_scores = {
    "SimCLR-ResNet50": [0.947, 0.951, 0.943, 0.949, 0.945],
    "BYOL-ResNet50": [0.931, 0.928, 0.935, 0.930, 0.933],
    "SwAV-ResNet50": [0.905, 0.899, 0.911, 0.902, 0.907],
}

print(pd.DataFrame({k: summarize_folds(v) for k, v in fold_scores.items()}).T.round(4))
pd.DataFrame(compare_models(fold_scores, baseline="SimCLR-ResNet50")).round(4)
"""
)

md(
    """
## 13. Inference and deployment

The saved classifier checkpoint is all the web app needs:

```bash
python app/app.py --checkpoint outputs/<experiment>/finetune_best.pt
# then open http://127.0.0.1:5000
```

Or from the CLI:

```bash
python -m fabricssl.cli predict --checkpoint outputs/<experiment>/finetune_best.pt --image path/to/fabric.jpg
```
"""
)

code(
    """
import torch.nn.functional as F

image_tensor, true_label, _ = samples[0]
with torch.no_grad():
    probabilities = F.softmax(model(image_tensor.unsqueeze(0).to(DEVICE)), dim=1)[0].cpu().numpy()

order = np.argsort(-probabilities)[:3]
print(f"ground truth : {class_names[true_label]}")
for rank, index in enumerate(order, start=1):
    print(f"  {rank}. {class_names[index]:<15} {probabilities[index]:.3f}")

print(f"\\nclassifier checkpoint : {classifier_ckpt}")
print(f"encoder checkpoint    : {encoder_ckpt}")
"""
)

md(
    """
## Summary

| stage | artefact |
| --- | --- |
| SSL pretraining | `outputs/<experiment>/encoder_best.pt` |
| Linear probe | `outputs/<experiment>/linear_results.json` |
| Fine-tuning | `outputs/<experiment>/finetune_best.pt`, `finetune_results.json` |
| Explainability | `outputs/<experiment>/explainability_metrics.json` |
| Full summary | `outputs/<experiment>/summary.json` |

To reproduce the paper end-to-end on the real corpus:

```bash
python -m fabricssl.cli run --config configs/simclr_resnet50_t01.yaml --data-root data/fabric
python -m fabricssl.cli sweep --config configs/simclr_resnet50_t01.yaml \\
    --temperatures 0.1 0.5 0.8 --methods simclr byol swav --backbones resnet18 resnet50 vit
```
"""
)


def build_notebook() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(body) if kind == "markdown" else nbf.v4.new_code_cell(body)
        for kind, body in CELLS
    ]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run every cell after writing")
    parser.add_argument("--output", type=str, default=str(NOTEBOOK_PATH))
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    notebook = build_notebook()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.execute:
        from nbclient import NotebookClient

        client = NotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(REPO_ROOT)}},
        )
        client.execute()

    nbf.write(notebook, str(output))
    print(f"wrote {output} ({len(notebook.cells)} cells, executed={args.execute})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
