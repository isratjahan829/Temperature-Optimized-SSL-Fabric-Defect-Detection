"""Statistical validation: paired t-tests, Cohen's d and k-fold aggregation (Table 12)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy import stats


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d for paired samples (mean difference / SD of differences)."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if diff.size < 2:
        return float("nan")
    sd = diff.std(ddof=1)
    if sd < 1e-12:
        return float("inf") if abs(diff.mean()) > 0 else 0.0
    return float(diff.mean() / sd)


def effect_size_label(d: float) -> str:
    """Conventional descriptor for an effect size magnitude."""
    magnitude = abs(d)
    if not np.isfinite(magnitude):
        return "undefined"
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "moderate"
    if magnitude < 1.2:
        return "large"
    return "very large"


def significance_label(p_value: float) -> str:
    if p_value < 0.001:
        return "p < 0.001 (highly significant)"
    if p_value < 0.01:
        return "p < 0.01 (very significant)"
    if p_value < 0.05:
        return "p < 0.05 (significant)"
    return "not significant"


def paired_ttest(
    model_a: Sequence[float],
    model_b: Sequence[float],
    name_a: str = "model_a",
    name_b: str = "model_b",
) -> Dict[str, object]:
    """Two-sided paired t-test between per-fold scores of two models."""
    a = np.asarray(model_a, dtype=np.float64)
    b = np.asarray(model_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"score arrays must match: {a.shape} vs {b.shape}")
    if a.size < 2:
        raise ValueError("paired t-test needs at least two folds")

    t_stat, p_value = stats.ttest_rel(a, b)
    d = cohens_d_paired(a, b)
    return {
        "comparison": f"{name_a} vs {name_b}",
        "mean_a": float(a.mean()),
        "std_a": float(a.std(ddof=1)),
        "mean_b": float(b.mean()),
        "std_b": float(b.std(ddof=1)),
        "mean_difference": float(a.mean() - b.mean()),
        "t_value": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(d),
        "effect_size": effect_size_label(d),
        "significance": significance_label(float(p_value)),
        "folds": int(a.size),
    }


def compare_models(scores: Dict[str, Sequence[float]], baseline: str) -> List[Dict[str, object]]:
    """Paired t-test of every model in ``scores`` against ``baseline``."""
    if baseline not in scores:
        raise KeyError(f"baseline {baseline!r} not present in scores ({sorted(scores)})")
    rows = []
    for name, values in scores.items():
        if name == baseline:
            continue
        rows.append(paired_ttest(scores[baseline], values, baseline, name))
    return rows


def summarize_folds(scores: Iterable[float]) -> Dict[str, float]:
    """Mean +/- SD and 95% CI across cross-validation folds."""
    values = np.asarray(list(scores), dtype=np.float64)
    if values.size == 0:
        raise ValueError("no fold scores provided")
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
    if values.size > 1:
        half_width = float(stats.t.ppf(0.975, values.size - 1) * sd / np.sqrt(values.size))
    else:
        half_width = 0.0
    return {
        "mean": mean,
        "std": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "folds": int(values.size),
    }
