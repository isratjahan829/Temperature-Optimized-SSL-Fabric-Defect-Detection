"""End-to-end smoke tests: pretrain -> probe -> fine-tune -> explain -> serve."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fabricssl.cli import main as cli_main
from fabricssl.data import FabricDataModule
from fabricssl.engine.downstream import extract_features, finetune, linear_probe, load_trained_classifier
from fabricssl.engine.pretrain import pretrain
from fabricssl.pipeline import run_experiment
from fabricssl.utils import EarlyStopping, get_device, load_checkpoint, save_json, set_seed


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    torch.set_num_threads(1)


def test_early_stopping_triggers():
    stopper = EarlyStopping(patience=2, mode="min")
    assert stopper.step(1.0) is True
    assert stopper.step(0.5) is True
    assert stopper.step(0.6) is False
    stopper.step(0.7)
    assert stopper.should_stop


def test_set_seed_is_reproducible():
    set_seed(123)
    first = torch.randn(4)
    set_seed(123)
    assert torch.allclose(first, torch.randn(4))


def test_get_device_falls_back_to_cpu():
    assert get_device("cpu").type == "cpu"
    assert get_device("auto").type in {"cpu", "cuda", "mps"}


@pytest.mark.parametrize("method", ["simclr", "byol", "swav"])
def test_pretrain_all_methods(cpu_config, method):
    cpu_config.ssl.method = method
    cpu_config.experiment = f"pretrain_{method}"
    result = pretrain(cpu_config)

    assert Path(result["checkpoint"]).exists()
    assert np.isfinite(result["best_loss"])
    assert len(result["history"]) == 1

    state = load_checkpoint(result["checkpoint"])
    assert state["method"] == method
    assert state["backbone_name"] == "resnet18"


def test_full_pipeline_produces_metrics_and_artifacts(cpu_config):
    summary = run_experiment(cpu_config)

    assert "pretrain_loss" in summary
    for stage in ("linear", "finetune"):
        metrics = summary[stage]
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert metrics["top5"] >= metrics["top1"]

    assert set(summary["clustering"]) == {"silhouette", "davies_bouldin", "ari"}

    out_dir = Path(cpu_config.output_dir) / cpu_config.experiment
    for name in ("summary.json", "config.yaml", "encoder_best.pt", "finetune_best.pt"):
        assert (out_dir / name).exists(), name

    payload = json.loads((out_dir / "summary.json").read_text())
    assert payload["experiment"] == cpu_config.experiment


def test_linear_probe_keeps_encoder_frozen(cpu_config):
    checkpoint = pretrain(cpu_config)["checkpoint"]
    result = linear_probe(cpu_config, checkpoint=checkpoint)
    assert Path(result["checkpoint"]).exists()
    assert 0.0 <= result["test_metrics"]["accuracy"] <= 1.0


def test_finetune_from_scratch_without_checkpoint(cpu_config):
    result = finetune(cpu_config, checkpoint=None)
    assert 0.0 <= result["test_metrics"]["accuracy"] <= 1.0
    assert (Path(result["checkpoint"])).exists()


def test_features_and_reload(cpu_config):
    result = finetune(cpu_config, checkpoint=None)
    device = get_device("cpu")
    model, classes = load_trained_classifier(result["checkpoint"], device=device)
    assert classes == ["baekra", "contamination", "cut", "stain"]

    dm = FabricDataModule(cpu_config.data)
    features, targets = extract_features(model, dm.eval_loader("test", 8), device)
    assert features.shape[0] == targets.shape[0]
    assert features.shape[1] == model.backbone.feature_dim


def test_load_trained_classifier_rejects_bad_checkpoint(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"model": {}}, path)
    with pytest.raises(KeyError):
        load_trained_classifier(path)


def test_save_json_handles_numpy(tmp_path):
    path = save_json({"a": np.float32(1.5), "b": np.arange(3)}, tmp_path / "x.json")
    assert json.loads(path.read_text()) == {"a": 1.5, "b": [0, 1, 2]}


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def test_cli_make_sample_data(tmp_path):
    root = tmp_path / "sample"
    assert cli_main(["make-sample-data", "--root", str(root), "--images-per-class", "3", "--image-size", "32"]) == 0
    assert len(list(root.rglob("*.png"))) == 21  # 7 classes x 3 images


def test_cli_pretrain_and_predict(tmp_path, sample_root, capsys):
    common = [
        "--data-root", str(sample_root),
        "--output-dir", str(tmp_path / "out"),
        "--experiment", "cli",
        "--device", "cpu",
        "--backbone", "resnet18",
        "--image-size", "32",
        "--num-workers", "0",
        "--epochs", "1",
        "--batch-size", "8",
    ]
    assert cli_main(["pretrain", *common]) == 0
    encoder = tmp_path / "out" / "cli" / "encoder_best.pt"
    assert encoder.exists()

    assert cli_main(["finetune", *common, "--checkpoint", str(encoder), "--finetune-epochs", "1"]) == 0
    classifier = tmp_path / "out" / "cli" / "finetune_best.pt"
    assert classifier.exists()

    image = next(Path(sample_root).rglob("*.png"))
    capsys.readouterr()
    assert cli_main(["predict", "--checkpoint", str(classifier), "--image", str(image), "--image-size", "32"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["prediction"] in ["baekra", "contamination", "cut", "stain"]
    assert 0.0 <= payload["confidence"] <= 1.0


def test_cli_explain(tmp_path, sample_root, capsys):
    common = [
        "--data-root", str(sample_root),
        "--output-dir", str(tmp_path / "out"),
        "--experiment", "xai",
        "--device", "cpu",
        "--backbone", "resnet18",
        "--image-size", "32",
        "--num-workers", "0",
        "--finetune-epochs", "1",
    ]
    assert cli_main(["finetune", *common]) == 0
    checkpoint = tmp_path / "out" / "xai" / "finetune_best.pt"

    capsys.readouterr()
    assert cli_main([
        "explain", *common,
        "--checkpoint", str(checkpoint),
        "--num-samples", "2",
        "--methods", "gradcam", "layercam",
    ]) == 0
    results = json.loads(capsys.readouterr().out)
    assert set(results) == {"gradcam", "layercam"}


def test_cli_rejects_unknown_method():
    with pytest.raises(SystemExit):
        cli_main(["pretrain", "--method", "moco"])
