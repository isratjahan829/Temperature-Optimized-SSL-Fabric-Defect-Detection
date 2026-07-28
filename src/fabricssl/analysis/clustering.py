"""Latent-space analysis: t-SNE plus silhouette / Davies-Bouldin / ARI (Table 16)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score


def clustering_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    num_clusters: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, float]:
    """Cluster embeddings with k-means and score them against the true labels.

    Silhouette and Davies-Bouldin are computed on the *true* labels (how well
    the classes separate in feature space); ARI compares the k-means partition
    against the labels.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {features.shape}")
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")

    unique = np.unique(labels)
    result: Dict[str, float] = {
        "silhouette": float("nan"),
        "davies_bouldin": float("nan"),
        "ari": float("nan"),
    }
    if len(unique) < 2 or len(features) <= len(unique):
        return result

    result["silhouette"] = float(silhouette_score(features, labels))
    result["davies_bouldin"] = float(davies_bouldin_score(features, labels))

    k = int(num_clusters or len(unique))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed)
    predicted = kmeans.fit_predict(features)
    result["ari"] = float(adjusted_rand_score(labels, predicted))
    return result


def tsne_embedding(
    features: np.ndarray,
    perplexity: float = 30.0,
    seed: int = 42,
    n_components: int = 2,
) -> np.ndarray:
    """2-D t-SNE projection; perplexity is clamped to the sample count."""
    features = np.asarray(features, dtype=np.float64)
    n_samples = len(features)
    if n_samples < 3:
        raise ValueError("t-SNE needs at least three samples")
    safe_perplexity = float(max(2.0, min(perplexity, (n_samples - 1) / 3.0)))
    tsne = TSNE(
        n_components=n_components,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(features)
