"""Post-hoc analysis: clustering, statistics and figures."""

from .clustering import clustering_metrics, tsne_embedding
from .stats import (
    cohens_d_paired,
    compare_models,
    effect_size_label,
    paired_ttest,
    significance_label,
    summarize_folds,
)

__all__ = [
    "clustering_metrics",
    "cohens_d_paired",
    "compare_models",
    "effect_size_label",
    "paired_ttest",
    "significance_label",
    "summarize_folds",
    "tsne_embedding",
]
