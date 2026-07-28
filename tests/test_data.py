import numpy as np
import pytest
import torch

from fabricssl.data import (
    ContrastivePairDataset,
    FabricDataModule,
    FabricDefectDataset,
    MultiCropDataset,
    build_eval_transform,
    build_multicrop_transforms,
    build_ssl_transform,
    denormalize,
    label_subset,
    sobel_magnitude,
    stratified_split,
)
from fabricssl.data.datasets import multicrop_collate


def test_dataset_scan(sample_root):
    dataset = FabricDefectDataset(sample_root, transform=build_eval_transform(32))
    assert len(dataset) == 48
    assert dataset.classes == ["baekra", "contamination", "cut", "stain"]
    image, target = dataset[0]
    assert image.shape == (3, 32, 32)
    assert 0 <= target < 4
    assert sum(dataset.class_counts().values()) == len(dataset)


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FabricDefectDataset(tmp_path / "does-not-exist")


def test_stratified_split_is_disjoint_and_stratified():
    targets = [0] * 20 + [1] * 10 + [2] * 6
    train, val, test = stratified_split(targets, (0.7, 0.2, 0.1), seed=1)

    assert len(train) + len(val) + len(test) == len(targets)
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    for subset in (train, val, test):
        assert len({targets[i] for i in subset}) == 3


def test_label_subset_keeps_every_class():
    targets = [0] * 20 + [1] * 20
    keep = label_subset(targets, 0.25, seed=0)
    assert len(keep) == 10
    assert {targets[i] for i in keep} == {0, 1}


def test_contrastive_views_differ(sample_root):
    base = FabricDefectDataset(sample_root)
    dataset = ContrastivePairDataset(base, build_ssl_transform(32))
    view_a, view_b, target = dataset[0]
    assert view_a.shape == view_b.shape == (3, 32, 32)
    assert isinstance(target, int)
    assert not torch.allclose(view_a, view_b)


def test_multicrop_collate(sample_root):
    base = FabricDefectDataset(sample_root)
    dataset = MultiCropDataset(base, build_multicrop_transforms(32, small_size=16, num_small=2))
    crops, targets = multicrop_collate([dataset[0], dataset[1]])
    assert len(crops) == 4
    assert crops[0].shape == (2, 3, 32, 32)
    assert crops[-1].shape == (2, 3, 16, 16)
    assert targets.shape == (2,)


def test_sobel_magnitude_detects_edges():
    image = torch.zeros(3, 16, 16)
    image[:, :, 8:] = 1.0
    edges = sobel_magnitude(image)
    assert edges.shape == (1, 16, 16)
    assert edges.max() == pytest.approx(1.0)
    assert edges[0, 8, 7] > edges[0, 8, 1]


def test_denormalize_round_trip():
    tensor = torch.rand(3, 8, 8)
    from fabricssl.data.transforms import IMAGENET_MEAN, IMAGENET_STD

    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    normalized = (tensor - mean) / std
    assert torch.allclose(denormalize(normalized), tensor, atol=1e-5)


def test_datamodule_loaders(cpu_config):
    dm = FabricDataModule(cpu_config.data, method="simclr")
    assert dm.num_classes == 4
    assert len(dm.train_labelled) + len(dm.val) + len(dm.test) == 48

    views_a, views_b, _ = next(iter(dm.pretrain_loader(batch_size=4)))
    assert views_a.shape == views_b.shape == (4, 3, 32, 32)

    images, targets = next(iter(dm.train_loader(batch_size=4)))
    assert images.shape == (4, 3, 32, 32)
    assert targets.dtype == torch.long

    weights = dm.train_class_weights()
    assert weights.shape == (4,)
    assert torch.all(weights > 0)


def test_datamodule_swav_multicrop(cpu_config):
    dm = FabricDataModule(cpu_config.data, method="swav")
    crops, targets = next(iter(dm.pretrain_loader(batch_size=4)))
    assert isinstance(crops, list) and len(crops) == 4
    assert targets.shape == (4,)


def test_eval_loader_rejects_unknown_split(cpu_config):
    dm = FabricDataModule(cpu_config.data)
    with pytest.raises(ValueError):
        dm.eval_loader("holdout")


def test_synthetic_images_are_class_separable(sample_root):
    dataset = FabricDefectDataset(sample_root, transform=build_eval_transform(32))
    means = {}
    for index in range(len(dataset)):
        image, target = dataset[index]
        means.setdefault(target, []).append(float(image.mean()))
    averages = np.array([np.mean(v) for v in means.values()])
    assert averages.std() > 0.01
