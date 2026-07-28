import numpy as np
import pytest
import torch

from fabricssl.analysis.clustering import clustering_metrics, tsne_embedding
from fabricssl.analysis.stats import compare_models, effect_size_label, paired_ttest, summarize_folds
from fabricssl.engine.metrics import classification_metrics, roc_curves, softmax, text_report, topk_accuracy
from fabricssl.engine.optim import LARS, build_optimizer, build_scheduler


def test_topk_accuracy():
    logits = np.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
    targets = np.array([1, 0, 1])
    assert topk_accuracy(logits, targets, k=1) == pytest.approx(2 / 3)
    assert topk_accuracy(logits, targets, k=2) == pytest.approx(1.0)
    # k larger than the number of classes is clipped, never an error
    assert topk_accuracy(logits, targets, k=5) == pytest.approx(1.0)


def test_softmax_rows_sum_to_one():
    probs = softmax(np.random.randn(5, 4) * 10)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0)


def test_classification_metrics_perfect_prediction():
    targets = np.array([0, 1, 2, 0, 1, 2])
    logits = np.eye(3)[targets] * 10
    metrics = classification_metrics(logits, targets, ["a", "b", "c"])

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["weighted_f1"] == pytest.approx(1.0)
    assert metrics["per_class"]["a"]["support"] == 2
    assert np.trace(np.array(metrics["confusion_matrix"])) == 6
    assert metrics["macro_auc"] == pytest.approx(1.0)


def test_classification_metrics_handles_missing_class():
    targets = np.array([0, 0, 1, 1])
    logits = np.eye(3)[np.array([0, 0, 1, 2])] * 5
    metrics = classification_metrics(logits, targets, ["a", "b", "c"])
    assert metrics["per_class"]["c"]["support"] == 0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_roc_curves_and_report():
    targets = np.array([0, 1, 0, 1])
    logits = np.array([[3.0, 0.0], [0.0, 3.0], [2.0, 0.5], [0.2, 2.0]])
    curves = roc_curves(logits, targets, num_classes=2)
    assert set(curves) == {0, 1}
    assert curves[0]["auc"] == pytest.approx(1.0)
    assert "precision" in text_report(logits, targets, ["a", "b"])


def test_optimizer_and_scheduler_factories():
    params = [torch.nn.Parameter(torch.randn(4, 4))]
    for name in ["adam", "adamw", "nadam", "sgd", "rmsprop", "lars"]:
        assert build_optimizer(params, name=name, lr=1e-3) is not None
    with pytest.raises(ValueError):
        build_optimizer(params, name="lion")

    optimizer = build_optimizer(params, "adam", lr=1e-2)
    assert build_scheduler(optimizer, "none") is None
    assert build_scheduler(optimizer, "step", epochs=9) is not None
    assert build_scheduler(optimizer, "cosine", epochs=10, warmup_epochs=2) is not None
    with pytest.raises(ValueError):
        build_scheduler(optimizer, "exponential")


def test_lars_step_updates_parameters():
    param = torch.nn.Parameter(torch.ones(3, 3))
    optimizer = LARS([param], lr=0.1)
    (param.sum()).backward()
    optimizer.step()
    assert not torch.allclose(param.data, torch.ones(3, 3))


def test_cosine_scheduler_decays_lr():
    param = [torch.nn.Parameter(torch.randn(2, 2))]
    optimizer = build_optimizer(param, "sgd", lr=1.0)
    scheduler = build_scheduler(optimizer, "cosine", epochs=10)
    start = optimizer.param_groups[0]["lr"]
    for _ in range(5):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] < start


def test_clustering_metrics_on_separated_blobs():
    rng = np.random.default_rng(0)
    features = np.vstack([rng.normal(0, 0.1, (20, 4)), rng.normal(6, 0.1, (20, 4))])
    labels = np.array([0] * 20 + [1] * 20)
    metrics = clustering_metrics(features, labels, seed=0)
    assert metrics["silhouette"] > 0.8
    assert metrics["davies_bouldin"] < 0.5
    assert metrics["ari"] == pytest.approx(1.0)


def test_clustering_metrics_degenerate_input():
    metrics = clustering_metrics(np.zeros((3, 2)), np.zeros(3))
    assert np.isnan(metrics["silhouette"])


def test_tsne_embedding_shape():
    rng = np.random.default_rng(0)
    embedding = tsne_embedding(rng.normal(size=(30, 8)), perplexity=30.0, seed=0)
    assert embedding.shape == (30, 2)


def test_paired_ttest_detects_improvement():
    a = [0.94, 0.95, 0.93, 0.96, 0.94]
    b = [0.90, 0.91, 0.89, 0.92, 0.90]
    result = paired_ttest(a, b, "simclr_resnet50", "byol_resnet50")

    assert result["mean_difference"] > 0
    assert result["p_value"] < 0.05
    assert result["cohens_d"] > 0.8
    assert "significant" in result["significance"]
    assert result["folds"] == 5


def test_compare_models_against_baseline():
    scores = {
        "simclr": [0.94, 0.95, 0.93],
        "byol": [0.92, 0.92, 0.91],
        "swav": [0.89, 0.90, 0.88],
    }
    rows = compare_models(scores, baseline="simclr")
    assert len(rows) == 2
    assert all(row["mean_difference"] > 0 for row in rows)
    with pytest.raises(KeyError):
        compare_models(scores, baseline="dino")


def test_effect_size_labels():
    assert effect_size_label(0.1) == "negligible"
    assert effect_size_label(0.35) == "small"
    assert effect_size_label(0.6) == "moderate"
    assert effect_size_label(3.2) == "very large"


def test_summarize_folds_ci():
    summary = summarize_folds([0.90, 0.92, 0.91, 0.93, 0.89])
    assert summary["mean"] == pytest.approx(0.91, abs=1e-6)
    assert summary["ci95_low"] < summary["mean"] < summary["ci95_high"]
    assert summary["folds"] == 5
