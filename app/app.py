"""Flask web application for interactive fabric defect inspection.

Run with::

    python app/app.py --checkpoint outputs/<experiment>/finetune_best.pt

Upload a fabric image and the app returns the predicted defect class, the
class-probability ranking and a Grad-CAM overlay that localises the defect.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

# Allow `python app/app.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fabricssl.data.transforms import build_eval_transform, denormalize  # noqa: E402
from fabricssl.engine.downstream import load_trained_classifier  # noqa: E402
from fabricssl.utils import get_device, get_logger  # noqa: E402
from fabricssl.xai.cam import build_cam, overlay_heatmap  # noqa: E402

LOGGER = get_logger("fabricssl.app")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB


class DefectInspector:
    """Loads a trained checkpoint once and serves predictions + heatmaps."""

    def __init__(self, checkpoint: str, image_size: int = 224, device: str = "auto", cam: str = "gradcam") -> None:
        self.device = get_device(device)
        self.model, self.classes = load_trained_classifier(checkpoint, device=self.device)
        self.transform = build_eval_transform(image_size)
        self.cam_name = cam
        LOGGER.info("loaded %d-class model from %s on %s", len(self.classes), checkpoint, self.device)

    def predict(self, image: Image.Image, topk: int = 3, with_cam: bool = True) -> Dict[str, object]:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()

        order = np.argsort(-probabilities)[: min(topk, len(self.classes))]
        payload: Dict[str, object] = {
            "prediction": self.classes[int(order[0])],
            "confidence": float(probabilities[int(order[0])]),
            "topk": [
                {"class": self.classes[int(i)], "probability": float(probabilities[int(i)])}
                for i in order
            ],
        }
        if with_cam:
            payload["heatmap"] = self._heatmap_data_uri(tensor, int(order[0]))
        return payload

    def _heatmap_data_uri(self, tensor: torch.Tensor, target_class: int) -> str:
        with build_cam(self.cam_name, self.model) as cam:
            heatmap = cam(tensor, target_class)
        base = denormalize(tensor[0]).permute(1, 2, 0).cpu().numpy()
        overlay = (overlay_heatmap(base, heatmap) * 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(overlay).save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def create_app(inspector: Optional[DefectInspector] = None) -> Flask:
    """Application factory - ``inspector`` may be ``None`` for a demo-only UI."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["INSPECTOR"] = inspector

    @app.get("/")
    def index():
        classes = inspector.classes if inspector is not None else []
        return render_template("index.html", classes=classes, model_ready=inspector is not None)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "model_loaded": inspector is not None,
                "classes": inspector.classes if inspector is not None else [],
                "device": str(inspector.device) if inspector is not None else None,
            }
        )

    @app.post("/predict")
    def predict():
        current: Optional[DefectInspector] = app.config["INSPECTOR"]
        if current is None:
            return jsonify({"error": "no model loaded; start the server with --checkpoint"}), 503

        image, error = _read_upload(request.files.get("image"))
        if error is not None:
            return jsonify({"error": error}), 400
        assert image is not None

        try:
            with_cam = request.form.get("heatmap", "true").lower() != "false"
            result = current.predict(image, with_cam=with_cam)
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.exception("inference failed")
            return jsonify({"error": f"inference failed: {exc}"}), 500
        return jsonify(result)

    return app


def _read_upload(file_storage) -> Tuple[Optional[Image.Image], Optional[str]]:
    if file_storage is None or not file_storage.filename:
        return None, "no image uploaded (field name must be 'image')"
    suffix = Path(file_storage.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return None, f"unsupported file type {suffix!r}; allowed: {sorted(ALLOWED_EXTENSIONS)}"
    try:
        image = Image.open(file_storage.stream).convert("RGB")
    except Exception as exc:
        return None, f"could not decode image: {exc}"
    return image, None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fabric defect detection web app")
    parser.add_argument("--checkpoint", type=str, default=os.environ.get("FABRICSSL_CHECKPOINT"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cam", type=str, default="gradcam")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    inspector = None
    if args.checkpoint:
        inspector = DefectInspector(
            args.checkpoint, image_size=args.image_size, device=args.device, cam=args.cam
        )
    else:
        LOGGER.warning("no --checkpoint given: starting in demo mode, /predict will return 503")
    create_app(inspector).run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
