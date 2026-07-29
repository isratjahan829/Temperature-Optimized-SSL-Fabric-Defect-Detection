# Reproducing the paper

Every number in the paper comes from the same three stages: contrastive
pretraining, transfer (linear probe / fine-tuning) and post-hoc analysis. This
guide lists the exact commands.

## 0. Setup

```bash
pip install -e ".[app,dev]"
# place the corpus at data/fabric/<class>/*.jpg  (see docs/DATASET.md)
```

Hyperparameters used throughout (`configs/simclr_resnet50_t01.yaml`):

| Setting | Value |
| --- | --- |
| Optimiser | Adam (RMSprop / Nadam / SGD / LARS also supported) |
| Learning rate | 3 × 10⁻⁴, cosine annealing with 5 warm-up epochs |
| Batch size | 64 |
| Epochs | 50 (20 and 30 also reported) |
| Projection dim | 128 |
| Temperature | 0.1 / 0.5 / 0.8 |
| Image size | 224 × 224 |
| Split | 75/15/10 (also 80/10/10, 70/20/10) |
| Early stopping | patience 10 on the monitored metric |

## 1. Best configuration — SimCLR + ResNet-50, T = 0.1

```bash
python -m fabricssl.cli run --config configs/simclr_resnet50_t01.yaml --data-root data/fabric
```

Writes to `outputs/simclr_resnet50_T0.1/`:

| File | Contents |
| --- | --- |
| `encoder_best.pt` | Pretrained encoder weights |
| `pretrain_history.json` | Per-epoch contrastive loss |
| `linear_results.json` | Linear-evaluation metrics |
| `finetune_best.pt` | Fine-tuned classifier |
| `finetune_results.json` | Test metrics, per-class P/R/F1, confusion matrix |
| `summary.json` | Everything above, plus clustering scores |

## 2. Temperature × method × backbone grid (Tables 6–9)

```bash
python -m fabricssl.cli sweep \
    --config configs/simclr_resnet50_t01.yaml \
    --data-root data/fabric \
    --temperatures 0.1 0.5 0.8 \
    --methods simclr byol swav \
    --backbones resnet18 resnet50 vit
```

Results land in `outputs/temperature_sweep.json` (27 runs — budget GPU time
accordingly). Individual configs are also available:
`configs/byol_resnet50_t01.yaml`, `configs/swav_resnet50_t01.yaml`,
`configs/simclr_vit_t01.yaml`, `configs/simclr_resnet18_t01.yaml`.

Epoch budgets of 20 / 30 / 50 are reproduced with `--epochs`.

## 3. Data-split study (Table 10)

```python
from fabricssl.config import Config
from fabricssl.pipeline import split_sweep

cfg = Config.load("configs/simclr_resnet50_t01.yaml")
split_sweep(cfg, splits=[(0.75, 0.15, 0.10), (0.80, 0.10, 0.10), (0.70, 0.20, 0.10)])
```

## 4. Label-efficiency study

```python
from fabricssl.pipeline import label_efficiency_sweep

label_efficiency_sweep(cfg, fractions=[0.1, 0.25, 0.5, 1.0])
```

This is the data-efficiency claim: the SSL encoder retains most of its accuracy
when only a fraction of the labels is used for fine-tuning.

## 5. Explainability (Tables 14–15)

```bash
python -m fabricssl.cli explain \
    --config configs/simclr_resnet50_t01.yaml \
    --checkpoint outputs/simclr_resnet50_T0.1/finetune_best.pt \
    --methods gradcam gradcam++ hirescam layercam scorecam eigencam \
    --num-samples 50 --plot
```

Produces `explainability_metrics.json` (IoU-Heat, PointAcc, Relevance,
Fidelity per method) and `cam_grid.png`.

**Annotations.** With no per-image defect masks, the CLI falls back to a
Grad-CAM-derived pseudo-mask, which makes the scores comparable *between
methods* but not against the paper's annotated IoU of 0.92. To use real masks,
pass them yourself:

```python
from fabricssl.xai import build_cam, evaluate_cam_methods

samples = [(image_tensor, class_index, boolean_mask), ...]   # mask: HxW numpy array
scores = evaluate_cam_methods(model, samples, lambda n: build_cam(n, model),
                              methods=["layercam", "gradcam++"])
```

## 6. Clustering and t-SNE (Table 16, Figure 18)

```python
from fabricssl.analysis.clustering import clustering_metrics, tsne_embedding
from fabricssl.analysis.plots import plot_tsne
from fabricssl.engine.downstream import extract_features, load_trained_classifier

model, classes = load_trained_classifier("outputs/simclr_resnet50_T0.1/finetune_best.pt")
features, targets = extract_features(model, datamodule.eval_loader("test", 64), device)

print(clustering_metrics(features, targets))          # silhouette / DBI / ARI
plot_tsne(tsne_embedding(features), targets, classes, path="tsne.png")
```

## 7. Statistical validation (Table 12)

Collect per-fold scores (e.g. by re-running with `--seed 0..9`), then:

```python
from fabricssl.analysis.stats import compare_models, summarize_folds

scores = {
    "SimCLR-ResNet50": [...],   # one mAP per fold
    "BYOL-ResNet50":   [...],
    "SwAV-ResNet50":   [...],
}
print(summarize_folds(scores["SimCLR-ResNet50"]))     # mean ± SD, 95 % CI
print(compare_models(scores, baseline="SimCLR-ResNet50"))  # t, p, Cohen's d
```

## 8. Deployment

```bash
python app/app.py --checkpoint outputs/simclr_resnet50_T0.1/finetune_best.pt
# http://127.0.0.1:5000 — upload an image, get the class + Grad-CAM overlay
```

## Notes on exact reproduction

- Seeds are fixed (`utils.set_seed`) and cuDNN runs deterministically, but GPU
  kernel non-determinism and library versions still cause small drifts.
- The paper's accuracies come from the full industrial corpus; the synthetic
  stand-in dataset cannot approximate them.
- `amp: true` (mixed precision) only activates on CUDA; CPU runs stay in fp32.
- Full-grid pretraining is the expensive part — 27 sweep runs at 50 epochs each.
  Start with a single configuration to validate the setup.
