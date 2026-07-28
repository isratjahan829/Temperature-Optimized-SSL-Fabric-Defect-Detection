"""Flask application tests (route contracts, validation, inference path)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

flask = pytest.importorskip("flask")

from app import DefectInspector, create_app  # noqa: E402

from fabricssl.engine.downstream import finetune  # noqa: E402


@pytest.fixture()
def demo_client():
    return create_app(None).test_client()


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory, sample_root):
    from fabricssl.config import Config

    cfg = Config()
    cfg.experiment = "app"
    cfg.output_dir = str(tmp_path_factory.mktemp("app_out"))
    cfg.device = "cpu"
    cfg.data.root = str(sample_root)
    cfg.data.image_size = 32
    cfg.data.num_workers = 0
    cfg.data.split = (0.6, 0.2, 0.2)
    cfg.model.backbone = "resnet18"
    cfg.eval.finetune_epochs = 1
    cfg.eval.batch_size = 8
    cfg.validate()
    return finetune(cfg, checkpoint=None)["checkpoint"]


def _png_bytes(size: int = 48) -> io.BytesIO:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), (120, 130, 140)).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_index_renders(demo_client):
    response = demo_client.get("/")
    assert response.status_code == 200
    assert b"Fabric Defect Inspection" in response.data


def test_health_without_model(demo_client):
    payload = demo_client.get("/health").get_json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is False


def test_predict_without_model_returns_503(demo_client):
    response = demo_client.post(
        "/predict", data={"image": (_png_bytes(), "fabric.png")}, content_type="multipart/form-data"
    )
    assert response.status_code == 503
    assert "error" in response.get_json()


def test_predict_requires_a_file(demo_client):
    assert demo_client.post("/predict", data={}, content_type="multipart/form-data").status_code in (400, 503)


def test_predict_with_model(trained_checkpoint):
    inspector = DefectInspector(trained_checkpoint, image_size=32, device="cpu")
    client = create_app(inspector).test_client()

    health = client.get("/health").get_json()
    assert health["model_loaded"] is True
    assert len(health["classes"]) == 4

    response = client.post(
        "/predict",
        data={"image": (_png_bytes(), "fabric.png"), "heatmap": "true"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["prediction"] in health["classes"]
    assert 0.0 <= payload["confidence"] <= 1.0
    assert len(payload["topk"]) == 3
    assert payload["heatmap"].startswith("data:image/png;base64,")


def test_predict_without_heatmap(trained_checkpoint):
    inspector = DefectInspector(trained_checkpoint, image_size=32, device="cpu")
    client = create_app(inspector).test_client()
    payload = client.post(
        "/predict",
        data={"image": (_png_bytes(), "fabric.png"), "heatmap": "false"},
        content_type="multipart/form-data",
    ).get_json()
    assert "heatmap" not in payload


def test_rejects_unsupported_extension(trained_checkpoint):
    inspector = DefectInspector(trained_checkpoint, image_size=32, device="cpu")
    client = create_app(inspector).test_client()
    response = client.post(
        "/predict",
        data={"image": (io.BytesIO(b"not an image"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "unsupported file type" in response.get_json()["error"]
