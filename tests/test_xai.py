import numpy as np
import pytest
import torch

from fabricssl.models import build_classifier
from fabricssl.xai import (
    CAM_METHODS,
    build_cam,
    evaluate_cam_methods,
    explanation_fidelity,
    iou_heat,
    normalize_heatmap,
    overlay_heatmap,
    point_accuracy,
    pseudo_mask_from_heatmap,
    relevance_score,
)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_classifier(4, "resnet18", image_size=32).eval()


@pytest.fixture(scope="module")
def image():
    torch.manual_seed(1)
    return torch.randn(1, 3, 32, 32)


@pytest.mark.parametrize("method", list(CAM_METHODS))
def test_cam_methods_return_valid_maps(model, image, method):
    with build_cam(method, model) as cam:
        heatmap = cam(image, target_class=1)
    assert heatmap.shape == (32, 32)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0
    assert np.isfinite(heatmap).all()
    assert heatmap.std() > 0  # not a constant map


def test_cam_accepts_chw_input(model):
    with build_cam("gradcam", model) as cam:
        assert cam(torch.randn(3, 32, 32)).shape == (32, 32)


def test_cam_rejects_batches(model):
    with build_cam("gradcam", model) as cam:
        with pytest.raises(ValueError):
            cam(torch.randn(2, 3, 32, 32))


def test_cam_works_for_vit_backbone():
    vit_model = build_classifier(4, "vit_tiny", image_size=32).eval()
    with build_cam("layercam", vit_model) as cam:
        heatmap = cam(torch.randn(1, 3, 32, 32), target_class=0)
    assert heatmap.shape == (32, 32)
    assert heatmap.std() > 0


def test_unknown_cam_rejected(model):
    with pytest.raises(ValueError):
        build_cam("magiccam", model)


def test_hooks_are_removed(model, image):
    cam = build_cam("gradcam", model)
    cam(image)
    cam.remove_hooks()
    assert cam._handles == []


def test_cam_does_not_leave_model_in_train_mode(model, image):
    model.train()
    with build_cam("gradcam", model) as cam:
        cam(image)
    assert model.training
    model.eval()


def test_normalize_heatmap_edge_cases():
    assert np.all(normalize_heatmap(np.zeros((4, 4))) == 0)
    scaled = normalize_heatmap(np.array([[1.0, 3.0], [5.0, 7.0]]))
    assert scaled.min() == 0.0 and scaled.max() == 1.0


def test_iou_point_relevance_on_synthetic_heatmap():
    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[2:6, 2:6] = 1.0
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 2:6] = True

    assert iou_heat(heatmap, mask) == pytest.approx(1.0)
    assert point_accuracy(heatmap, mask) == 1.0
    assert relevance_score(heatmap, mask) == pytest.approx(1.0)

    disjoint = np.zeros((10, 10), dtype=bool)
    disjoint[7:9, 7:9] = True
    assert iou_heat(heatmap, disjoint) == 0.0
    assert point_accuracy(heatmap, disjoint) == 0.0


def test_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError):
        iou_heat(np.zeros((4, 4)), np.zeros((5, 5), dtype=bool))


def test_pseudo_mask_selects_top_pixels():
    rng = np.random.default_rng(0)
    heatmap = rng.random((20, 20))
    mask = pseudo_mask_from_heatmap(heatmap, quantile=0.9)
    assert mask.dtype == bool
    assert 0 < mask.sum() <= 60


def test_explanation_fidelity_range(model, image):
    with build_cam("gradcam", model) as cam:
        heatmap = cam(image, target_class=0)
    score = explanation_fidelity(model, image, heatmap, target_class=0, grid=2)
    assert -1.0 <= score <= 1.0


def test_overlay_heatmap_shape_and_range():
    rgb = np.random.rand(8, 8, 3)
    blended = overlay_heatmap(rgb, np.random.rand(8, 8))
    assert blended.shape == (8, 8, 3)
    assert blended.min() >= 0.0 and blended.max() <= 1.0
    with pytest.raises(ValueError):
        overlay_heatmap(np.random.rand(8, 8), np.random.rand(8, 8))


def test_evaluate_cam_methods_returns_all_metrics(model):
    samples = [(torch.randn(3, 32, 32), i % 4, None) for i in range(2)]
    results = evaluate_cam_methods(
        model,
        samples,
        cam_factory=lambda name: build_cam(name, model),
        methods=["gradcam", "layercam"],
        fidelity_grid=2,
    )
    assert set(results) == {"gradcam", "layercam"}
    for scores in results.values():
        assert set(scores) == {"iou", "point_acc", "relevance", "fidelity"}
        assert all(np.isfinite(v) for v in scores.values())
