import pytest

from fabricssl.config import Config, load_config


def test_defaults_match_paper_hyperparameters():
    cfg = Config()
    assert cfg.ssl.method == "simclr"
    assert cfg.model.backbone == "resnet50"
    assert cfg.ssl.temperature == 0.1
    assert cfg.ssl.optim.lr == pytest.approx(3e-4)
    assert cfg.ssl.optim.batch_size == 64
    assert cfg.ssl.optim.epochs == 50
    assert cfg.model.projection_dim == 128


def test_yaml_roundtrip(tmp_path):
    cfg = Config()
    cfg.ssl.temperature = 0.5
    cfg.data.split = (0.8, 0.1, 0.1)
    path = cfg.save(tmp_path / "cfg.yaml")

    restored = Config.load(path)
    assert restored.ssl.temperature == 0.5
    assert restored.data.split == (0.8, 0.1, 0.1)
    assert restored.to_dict() == cfg.to_dict()


def test_dotted_overrides():
    cfg = load_config(None, **{"ssl.temperature": 0.8, "model.backbone": "resnet18", "seed": 7})
    assert cfg.ssl.temperature == 0.8
    assert cfg.model.backbone == "resnet18"
    assert cfg.seed == 7


@pytest.mark.parametrize(
    "key, value",
    [
        ("ssl.method", "moco"),
        ("model.backbone", "efficientnet"),
        ("ssl.temperature", 0.0),
        ("eval.label_fraction", 1.5),
    ],
)
def test_invalid_values_rejected(key, value):
    with pytest.raises(ValueError):
        load_config(None, **{key: value})


def test_split_must_sum_to_one():
    cfg = Config()
    cfg.data.split = (0.5, 0.2, 0.2)
    with pytest.raises(ValueError):
        cfg.validate()


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        Config.from_dict({"nonsense": 1})
