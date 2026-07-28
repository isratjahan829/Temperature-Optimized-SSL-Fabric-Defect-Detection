import math

import pytest
import torch

from fabricssl.config import ModelConfig
from fabricssl.losses import BYOLLoss, FocalLoss, NTXentLoss, SwAVLoss, build_ssl_loss, sinkhorn
from fabricssl.models import BYOL, SimCLR, SwAV, build_backbone, build_classifier, build_ssl_model


@pytest.mark.parametrize("name, dim", [("resnet18", 512), ("resnet50", 2048), ("vit_tiny", 192)])
def test_backbone_shapes(name, dim):
    backbone = build_backbone(name, image_size=32)
    x = torch.randn(2, 3, 32, 32)
    assert backbone.feature_dim == dim
    assert backbone(x).shape == (2, dim)
    feature_map = backbone.feature_map(x)
    assert feature_map.dim() == 4 and feature_map.shape[1] == dim


def test_unknown_backbone_rejected():
    with pytest.raises(ValueError):
        build_backbone("resnet101")


def test_ntxent_identical_views_beat_random():
    torch.manual_seed(0)
    loss_fn = NTXentLoss(temperature=0.1)
    z = torch.randn(8, 16)
    aligned = loss_fn(z, z.clone())
    misaligned = loss_fn(z, torch.randn(8, 16))
    assert aligned < misaligned
    assert aligned >= 0


def test_ntxent_temperature_changes_loss():
    torch.manual_seed(0)
    z_i, z_j = torch.randn(8, 16), torch.randn(8, 16)
    assert NTXentLoss(0.1)(z_i, z_j) != NTXentLoss(0.8)(z_i, z_j)


def test_ntxent_rejects_bad_inputs():
    with pytest.raises(ValueError):
        NTXentLoss(temperature=0.0)
    with pytest.raises(ValueError):
        NTXentLoss(0.1)(torch.randn(1, 4), torch.randn(1, 4))
    with pytest.raises(ValueError):
        NTXentLoss(0.1)(torch.randn(4, 4), torch.randn(3, 4))


def test_byol_loss_is_zero_for_identical_embeddings():
    z = torch.randn(4, 8)
    assert float(BYOLLoss()(z, z.clone())) == pytest.approx(0.0, abs=1e-5)


def test_sinkhorn_produces_row_stochastic_assignments():
    scores = torch.randn(16, 8)
    q = sinkhorn(scores, epsilon=0.05, iterations=3)
    assert q.shape == (16, 8)
    assert torch.allclose(q.sum(dim=1), torch.ones(16), atol=1e-4)
    assert torch.all(q >= 0)


def test_swav_loss_finite_and_positive():
    loss = SwAVLoss(0.1)([torch.randn(8, 16) for _ in range(3)])
    assert torch.isfinite(loss) and loss > 0

    with pytest.raises(ValueError):
        SwAVLoss(0.1)([torch.randn(8, 16)])


def test_focal_loss_downweights_easy_examples():
    logits = torch.tensor([[8.0, 0.0], [0.1, 0.0]])
    targets = torch.tensor([0, 0])
    per_sample = FocalLoss(gamma=2.0, reduction="none")(logits, targets)
    assert per_sample[0] < per_sample[1]
    assert float(FocalLoss(gamma=2.0)(logits, targets)) == pytest.approx(float(per_sample.mean()))


def test_build_ssl_loss_factory():
    assert isinstance(build_ssl_loss("simclr", 0.1), NTXentLoss)
    assert isinstance(build_ssl_loss("byol", 0.1), BYOLLoss)
    assert isinstance(build_ssl_loss("swav", 0.1), SwAVLoss)
    with pytest.raises(ValueError):
        build_ssl_loss("mocov2", 0.1)


@pytest.fixture()
def small_model_cfg():
    cfg = ModelConfig()
    cfg.backbone = "resnet18"
    cfg.hidden_dim = 64
    cfg.projection_dim = 16
    cfg.num_prototypes = 8
    return cfg


def test_simclr_step_produces_gradients(small_model_cfg):
    model = build_ssl_model("simclr", small_model_cfg, temperature=0.1, image_size=32)
    assert isinstance(model, SimCLR)
    loss = model.training_step([torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)])
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())


def test_byol_target_network_is_frozen_and_updates(small_model_cfg):
    model = build_ssl_model("byol", small_model_cfg, image_size=32)
    assert isinstance(model, BYOL)
    assert all(not p.requires_grad for p in model.target_backbone.parameters())

    before = next(model.target_backbone.parameters()).clone()
    loss = model.training_step([torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)])
    loss.backward()
    for param in model.backbone.parameters():
        param.data.add_(torch.randn_like(param) * 0.1)
    model.on_after_step()
    assert not torch.allclose(before, next(model.target_backbone.parameters()))


def test_byol_momentum_schedule(small_model_cfg):
    model = build_ssl_model("byol", small_model_cfg, image_size=32)
    assert isinstance(model, BYOL)
    model.set_momentum_from_progress(0.0)
    start = model.momentum
    model.set_momentum_from_progress(1.0)
    assert math.isclose(start, model.base_momentum, rel_tol=1e-6)
    assert model.momentum == pytest.approx(1.0)


def test_swav_prototypes_are_normalized(small_model_cfg):
    model = build_ssl_model("swav", small_model_cfg, temperature=0.1, image_size=32)
    assert isinstance(model, SwAV)
    crops = [torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32), torch.randn(4, 3, 16, 16)]
    loss = model.training_step([crops])
    loss.backward()
    model.on_after_step()
    norms = model.prototypes.prototypes.weight.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_classifier_freezing():
    model = build_classifier(4, "resnet18", freeze_backbone=True, image_size=32)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    model.train()
    assert not model.backbone.training  # BN stays in eval mode while probing

    model.set_backbone_frozen(False)
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_classifier_rejects_mismatched_checkpoint(tmp_path):
    torch.save({"backbone": build_backbone("resnet18", image_size=32).state_dict()}, tmp_path / "enc.pt")
    with pytest.raises(RuntimeError):
        build_classifier(4, "resnet50", checkpoint=tmp_path / "enc.pt", image_size=32)
