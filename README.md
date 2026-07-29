# Temperature-Optimized SSL for Fabric Defect Detection

Reference implementation of **"Temperature-Optimized Self-Supervised Contrastive
Learning Enables Data-Efficient and Explainable Fabric Defect Detection"**.

Fabric inspection data is abundant but *labels* are scarce. This repository
pretrains an encoder on unlabeled fabric images with contrastive
(**SimCLR**, **BYOL**) and clustering-based (**SwAV**) objectives, then transfers
it to seven-class defect classification with a linear probe or full fine-tuning.
The contrastive **temperature `T`** is treated as a first-class hyperparameter —
the paper's best configuration is **SimCLR + ResNet-50 at `T = 0.1`**.

Everything is runnable end-to-end on CPU: `make demo` generates a synthetic
stand-in dataset and walks the full pipeline in a few minutes.

---

## Contents

| Path | What it is |
| --- | --- |
| `src/fabricssl/` | The library (data, models, losses, engines, XAI, analysis, CLI) |
| `notebooks/Fabric_Defect_SSL_Pipeline.ipynb` | Executable end-to-end walkthrough with figures |
| `app/` | Flask web app: upload an image, get a prediction + Grad-CAM overlay |
| `configs/` | Ready-made YAML configs for each method/backbone/temperature |
| `scripts/` | Reproduction driver and the notebook generator |
| `tests/` | 96 pytest cases covering every module |
| `docs/` | Dataset notes and reproduction guide |

## Features

- **Three SSL methods** — SimCLR (NT-Xent), BYOL (EMA target, no negatives), SwAV
  (Sinkhorn-Knopp swapped prediction with multi-crop).
- **Three backbones** — ResNet-18, ResNet-50, ViT-B/16 (plus a `vit_tiny` for CPU runs).
- **Temperature study** — sweep `T ∈ {0.1, 0.5, 0.8}` across methods and backbones.
- **Class-imbalance handling** — inverse-frequency sampling, weighted loss, focal
  loss, and an augmentation stack (rotation, flips, colour jitter/drop, blur,
  Gaussian noise, Sobel edge blending).
- **Six CAM explainers written from scratch** — Grad-CAM, Grad-CAM++, HiResCAM,
  LayerCAM, ScoreCAM, EigenCAM — working on both CNN and ViT backbones, with
  quantitative **IoU / PointAcc / Relevance / Fidelity** scoring.
- **Latent analysis** — silhouette, Davies-Bouldin, ARI, t-SNE.
- **Statistics** — paired t-tests, Cohen's *d*, 95 % CIs over folds.
- **Deployment** — CLI (`fabricssl …`) and a Flask app with heatmap overlays.

## Installation

```bash
git clone https://github.com/isratjahan829/Temperature-Optimized-SSL-Fabric-Defect-Detection.git
cd Temperature-Optimized-SSL-Fabric-Defect-Detection

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[app,dev]"        # or: pip install -r requirements-dev.txt
```

Python ≥ 3.9, PyTorch ≥ 2.0. A GPU is optional — everything falls back to CPU.

## Quickstart (no dataset needed)

```bash
# 1. Generate a synthetic stand-in corpus (7 classes)
python -m fabricssl.cli make-sample-data --root data/sample --images-per-class 40 --image-size 96

# 2. Pretrain -> linear probe -> fine-tune -> clustering
python -m fabricssl.cli run --config configs/smoke_cpu.yaml

# 3. Explain the fine-tuned model
python -m fabricssl.cli explain \
    --config configs/smoke_cpu.yaml \
    --checkpoint outputs/smoke_simclr_resnet18/finetune_best.pt --plot

# 4. Serve it
python app/app.py --checkpoint outputs/smoke_simclr_resnet18/finetune_best.pt --image-size 64
```

Or simply `make demo`.

## Using the real dataset

The industrial corpus (~2 800 images from the Chenab Textile production line)
is published by the authors at <https://tinyurl.com/488uhkhy>. Arrange it as one
folder per class:

```
data/fabric/
├── baekra/
├── color_issues/
├── contamination/
├── cut/
├── gray_stitch/
├── selvet/
└── stain/
```

Then reproduce the paper's best configuration:

```bash
python -m fabricssl.cli run --config configs/simclr_resnet50_t01.yaml --data-root data/fabric
```

See [`docs/DATASET.md`](docs/DATASET.md) for class definitions and
[`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the full experiment matrix.

## CLI reference

| Command | Purpose |
| --- | --- |
| `make-sample-data` | Generate the synthetic stand-in dataset |
| `pretrain` | Stage 1 — self-supervised pretraining |
| `linear` | Stage 2a — linear evaluation of a frozen encoder |
| `finetune` | Stage 2b — supervised fine-tuning |
| `run` | Full pipeline + clustering analysis |
| `sweep` | Temperature × method × backbone grid |
| `explain` | CAM heatmaps and explainability metrics |
| `predict` | Classify a single image |

Any config field can be overridden on the command line:

```bash
python -m fabricssl.cli pretrain \
    --config configs/simclr_resnet50_t01.yaml \
    --method byol --backbone resnet18 --temperature 0.5 --epochs 30 --batch-size 128
```

## Library usage

```python
from fabricssl.config import Config
from fabricssl.pipeline import run_experiment

cfg = Config()
cfg.data.root = "data/fabric"
cfg.ssl.method = "simclr"          # simclr | byol | swav
cfg.ssl.temperature = 0.1          # the optimised hyperparameter
cfg.model.backbone = "resnet50"    # resnet18 | resnet50 | vit
cfg.validate()

summary = run_experiment(cfg)
print(summary["finetune"]["accuracy"], summary["finetune"]["macro_f1"])
```

## Method

**Stage 1 — pretext task.** Two augmented views `x_i, x_j` of each image are
encoded and projected to `z_i, z_j`. SimCLR minimises

```
L = -log( exp(sim(z_i, z_j)/T) / Σ_{k≠i} exp(sim(z_i, z_k)/T) )
```

Lower `T` sharpens the similarity distribution, which the paper links to faster
convergence and better transfer. BYOL drops negatives and regresses an EMA
target; SwAV matches Sinkhorn-balanced prototype assignments across views.

**Stage 2 — downstream task.** The encoder is either frozen (linear evaluation,
measuring representation quality) or fine-tuned end-to-end (peak accuracy), with
class-weighted cross-entropy and balanced sampling to counter the stain-heavy
label distribution.

**Stage 3 — explanation.** CAM heatmaps localise the defect and are scored for
spatial alignment (IoU, PointAcc), mass concentration (Relevance) and causal
faithfulness (occlusion-based Fidelity).

## Reported results (paper, real corpus)

| Model | Linear acc. | Fine-tuned acc. | Top-1 | Top-5 | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| **SimCLR + ResNet-50, T = 0.1** | **72.50 %** | **91.94 %** | **91.97 %** | **98.5 %** | **0.98** |
| BYOL + ResNet-50, T = 0.1 | lower | 91.75 % | — | — | 0.96 |
| SwAV + ViT, T = 0.8 | 53 % | — | — | — | 0.60 |

LayerCAM gave the best explanation quality (IoU 0.92, PointAcc 0.91,
Relevance 0.92, Fidelity 0.93). Paired t-tests confirmed SimCLR-ResNet-50's
advantage over BYOL (t = 8.35, p < 0.001, d = 3.20) and SwAV
(t = 9.25, p < 0.001, d = 4.00).

> Numbers produced by the quickstart use synthetic data and are **not**
> comparable to the table above — they exist to prove the pipeline runs.

## Development

```bash
make test        # pytest (96 tests)
make lint        # ruff
make notebook    # regenerate + execute the notebook
```

The notebook is generated from `scripts/build_notebook.py`, so it stays
reviewable in git and CI can re-execute it to prove every cell still works.

## Citation

```bibtex
@article{islam2026temperature,
  title   = {Temperature-Optimized Self-Supervised Contrastive Learning Enables
             Data-Efficient and Explainable Fabric Defect Detection},
  author  = {Islam, Raihan Ul and Jahan, Israt and Islam, Md. Samiul and
             Pantho, Tomal Ahmed and Ahammed, Md. Atik and
             Rashid, Mohammad Rifat Ahmmad and Reza, Ahmed Wasif and Ripon, Shamim H.},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {57256--57280},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3678495}
}
```

## License

MIT — see [LICENSE](LICENSE).
