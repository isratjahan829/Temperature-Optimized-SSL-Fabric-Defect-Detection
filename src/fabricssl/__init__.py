"""Temperature-Optimized Self-Supervised Contrastive Learning for Fabric Defect Detection.

Reference implementation of the paper:
    "Temperature-Optimized Self-Supervised Contrastive Learning Enables
     Data-Efficient and Explainable Fabric Defect Detection"

The package provides:
    * SSL pretraining (SimCLR, BYOL, SwAV) with ResNet18 / ResNet50 / ViT backbones
    * Linear evaluation and supervised fine-tuning of the pretrained encoder
    * Classification metrics (top-1/top-5, macro/weighted F1, ROC, confusion matrix)
    * Explainability (Grad-CAM, Grad-CAM++, LayerCAM, HiResCAM, ScoreCAM, EigenCAM)
      with quantitative IoU / PointAcc / Relevance / Explanation-Fidelity metrics
    * Latent-space analysis (t-SNE, silhouette, Davies-Bouldin, ARI)
    * Statistical validation (paired t-test, Cohen's d)
"""

__version__ = "1.0.0"

DEFECT_CLASSES = (
    "baekra",
    "color_issues",
    "contamination",
    "cut",
    "gray_stitch",
    "selvet",
    "stain",
)

__all__ = ["__version__", "DEFECT_CLASSES"]
