"""/api/vision: states, queries, profiles and real errors - with stand-in models (no weights needed)."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.runtime_config import load_config
from app.services.inference_service import FEATURES_ID, VisionService
from app.services.ontology import load_ontology
from app.vision.registry import REGISTRY_DIR, ModelRegistry
from app.vision.types import Detection, ImageInfo, InferenceResult
from tests.conftest import make_settings


def _jpeg(w: int = 320, h: int = 240, value: int | None = None, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = np.full((h, w, 3), value, np.uint8) if value is not None else (rng.random((h, w, 3)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class FakeDetector:
    model_id, model_version = "yolox_s", "test"

    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections

    def detect(self, image_bgr: np.ndarray) -> InferenceResult:
        h, w = image_bgr.shape[:2]
        return InferenceResult(self.model_id, self.model_version, ImageInfo(w, h), [Detection(**vars_(d)) for d in self.detections],
                               config={"input_size": 640}, timings_ms={"total": 1.0})


def vars_(d: Detection) -> dict[str, Any]:
    return {"class_id": d.class_id, "class_name": d.class_name, "confidence": d.confidence, "box": d.box,
            "uncertainty": dict(d.uncertainty or {}), "extra": dict(d.extra)}


class FakeExtractor:
    """Patch features = the mean colour of each 8x8 cell, so a painted square is a real anomaly."""

    size = 64

    def __call__(self, images: list[np.ndarray]) -> np.ndarray:
        import cv2

        out = [cv2.resize(im, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 for im in images]
        return np.stack(out)


def _registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, present: bool) -> ModelRegistry:
    """A registry whose yolox_s / feature weights are tiny stand-in files with matching checksums."""
    reg_dir, weights = tmp_path / "registry", tmp_path / "weights"
    reg_dir.mkdir()
    weights.mkdir()
    for model_id in ("yolox_s", FEATURES_ID):
        entry = json.loads((REGISTRY_DIR / f"{model_id}.json").read_text())
        blob = f"stand-in weights for {model_id}".encode()
        entry["status"] = "fetchable"
        entry["weights"] = {**entry["weights"], "file": f"{model_id}.bin", "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
        (reg_dir / f"{model_id}.json").write_text(json.dumps(entry))
        if present:
            (weights / f"{model_id}.bin").write_bytes(blob)
    monkeypatch.setenv("RD_MODEL_DIR", str(weights))
    monkeypatch.setattr("app.vision.registry._weight_dirs", lambda: [weights])
    return ModelRegistry(reg_dir)


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detections: list[Detection] | None = None, *, present: bool = True) -> TestClient:
    settings = make_settings()
    app = create_app(settings)
    registry = _registry(tmp_path, monkeypatch, present=present)
    loaders = {"yolox": lambda e, p: FakeDetector(detections or []), "resnet": lambda e, p: FakeExtractor()}
    app.state.vision = VisionService(settings, load_config(), load_ontology(), registry=registry, loaders=loaders)
    return TestClient(app)


DETS = [
    Detection(0, "person", 0.91, (10, 10, 100, 200), uncertainty={"second_class_score": 0.05}),
    Detection(56, "chair", 0.62, (120, 80, 200, 220), uncertainty={"second_class_score": 0.48}, extra={"second_class": "couch"}),
    Detection(39, "bottle", 0.18, (220, 40, 240, 90), uncertainty={"second_class_score": 0.01}),
]


def test_status_lists_models_without_loading_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch, DETS)
    body = client.get("/api/vision/status").json()
    assert {m["id"] for m in body["models"]} == {"yolox_s", FEATURES_ID}
    assert all(m["available"] and not m["loaded"] for m in body["models"])
    assert body["profiles"]["medical_research"]["status"] == "unavailable"


def test_detect_reports_object_states_and_query_answers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch, DETS)
    r = client.post("/api/vision/detect", files={"image": ("a.jpg", _jpeg(), "image/jpeg")},
                    data={"queries": "person, sofa chair, bottle, sports car, forklift"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["state"] == "complete" and body["coordinate_system"] == "image_px_xyxy"
    states = {o["label"]: o["state"] for o in body["objects"]}
    assert states == {"person": "detected", "chair": "ambiguous", "bottle": "tentative"}
    assert next(o for o in body["objects"] if o["label"] == "chair")["alternative_label"] == "couch"
    answers = {q["query"]: q for q in body["queries"]}
    assert answers["person"]["state"] == "detected" and answers["person"]["match"] == "exact"
    assert answers["bottle"]["state"] == "tentative"
    assert answers["sports car"]["state"] == "not_detected" and answers["sports car"]["labels"] == ["car"]
    assert answers["forklift"]["state"] == "unsupported_category" and answers["forklift"]["labels"] == []


def test_not_detected_is_downgraded_on_a_poor_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch, [])
    r = client.post("/api/vision/detect", files={"image": ("dark.jpg", _jpeg(value=3), "image/jpeg")}, data={"queries": "car"})
    body = r.json()
    assert body["state"] == "insufficient_image_quality"
    assert {i["code"] for i in body["quality"]["issues"]} >= {"too_dark"}
    assert body["queries"][0]["state"] == "insufficient_image_quality"


def test_missing_weights_are_a_real_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch, DETS, present=False)
    r = client.post("/api/vision/detect", files={"image": ("a.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 503
    err = r.json()["error"]
    assert err["code"] == "MODEL_UNAVAILABLE" and "not present" in err["hint"]


def test_unknown_and_unavailable_profiles_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch, DETS)
    for profile in ("no_such_profile", "medical_research"):
        r = client.post("/api/vision/detect", files={"image": ("a.jpg", _jpeg(), "image/jpeg")}, data={"profile": profile})
        assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_VISION_REQUEST"


def test_compare_flags_a_painted_defect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    refs = [("references", (f"r{i}.jpg", _jpeg(value=128 + i), "image/jpeg")) for i in range(4)]
    good = client.post("/api/vision/compare", files=[*refs, ("image", ("q.jpg", _jpeg(value=129), "image/jpeg"))]).json()
    assert good["verdict"] == "within_reference_variation" and good["regions"] == []
    bad_img = np.full((240, 320, 3), 129, np.uint8)
    bad_img[100:160, 140:200] = 20
    buf = io.BytesIO()
    Image.fromarray(bad_img).save(buf, format="PNG")
    refs = [("references", (f"r{i}.jpg", _jpeg(value=128 + i), "image/jpeg")) for i in range(4)]
    bad = client.post("/api/vision/compare", files=[*refs, ("image", ("q.png", buf.getvalue(), "image/png"))]).json()
    assert bad["verdict"] == "anomalous" and bad["score"] > bad["threshold"]
    x1, y1, x2, y2 = bad["regions"][0]["box"]
    assert x1 < 200 and x2 > 140 and y1 < 160 and y2 > 100  # region overlaps the painted square
    assert bad["heatmap"]["encoding"] == "png-base64" and bad["heatmap"]["data"]


def test_compare_needs_enough_references(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    r = client.post("/api/vision/compare", files=[("references", ("r.jpg", _jpeg(), "image/jpeg")), ("image", ("q.jpg", _jpeg(), "image/jpeg"))])
    assert r.status_code == 422


def test_vision_can_be_turned_off() -> None:
    client = TestClient(create_app(make_settings(vision_backend=False)))
    r = client.get("/api/vision/status")
    assert r.status_code == 503 and r.json()["error"]["code"] == "VISION_OFF"
