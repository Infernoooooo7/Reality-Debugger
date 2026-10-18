"""Workspace reference comparison (app/vision/workspace.py): registration, object diff, changed regions."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from app.runtime_config import load_config
from app.vision.types import Detection
from app.vision.workspace import WorkspaceConfig, compare_workspace, pair_objects, register, small_gray
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.inference_service import VisionService
from app.services.ontology import load_ontology
from tests.conftest import make_settings
from tests.test_vision_api import FakeDetector, FakeExtractor, _registry

CFG = WorkspaceConfig.from_config(load_config().get)


def scene_image(seed: int = 0, w: int = 960, h: int = 720) -> np.ndarray:
    """A cluttered, textured stand-in for a desk photo: random rectangles and discs plus sensor noise."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 120, np.uint8)
    for _ in range(160):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(10, 90)), y + int(rng.integers(10, 90))), color, -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(5, 45)), color, -1)
    return np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)


def camera_move(w: int, h: int, angle_deg: float = 3.0, scale: float = 1.04, shift: tuple[float, float] = (25, -18),
                perspective: float = 2e-5) -> np.ndarray:
    """Reference -> current pixel mapping of a small hand-held camera change."""
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, scale)
    hmat = np.vstack([m, [0, 0, 1]])
    hmat[0, 2] += shift[0]
    hmat[1, 2] += shift[1]
    hmat[2, 0] = perspective
    return hmat


def corner_error(h_est: np.ndarray, h_true: np.ndarray, w: int, h: int) -> float:
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]]).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(pts, h_est).reshape(-1, 2)
    b = cv2.perspectiveTransform(pts, h_true).reshape(-1, 2)
    return float(np.linalg.norm(a - b, axis=1).max())


def test_registration_recovers_a_known_camera_move() -> None:
    ref = scene_image(1)
    h_rc = camera_move(960, 720)
    cur = cv2.warpPerspective(ref, h_rc, (960, 720))
    reg = register(small_gray(ref, CFG.max_side)[0], small_gray(cur, CFG.max_side)[0], CFG)
    assert reg.ok and reg.homography is not None, reg.reason
    assert corner_error(reg.homography, np.linalg.inv(h_rc), 960, 720) < 2.0  # current -> reference, in px
    assert reg.inliers >= CFG.min_inliers and reg.overlap is not None and reg.overlap > 0.8


def test_unrelated_photos_are_not_compared_by_position() -> None:
    out = compare_workspace(scene_image(2), scene_image(3), [], [], CFG)
    assert out["registration"]["ok"] is False and out["registration"]["reason"]
    assert out["objects"] is None and out["changes"] is None
    assert "could not be aligned" in out["summary"]


def test_a_new_object_is_a_changed_region_and_exposure_is_not() -> None:
    ref = scene_image(4)
    h_rc = camera_move(960, 720, angle_deg=-2.0, shift=(-15, 10))
    cur = cv2.warpPerspective(ref, h_rc, (960, 720))
    darker = np.clip(cur.astype(np.float32) * 0.75 + 12, 0, 255).astype(np.uint8)  # the room got darker
    quiet = compare_workspace(ref, darker, [], [], CFG)
    assert quiet["registration"]["ok"]
    assert quiet["changes"]["changed_fraction"] < 0.01 and not quiet["changes"]["regions"]

    changed = darker.copy()
    cv2.rectangle(changed, (600, 300), (700, 380), (40, 220, 40), -1)  # something new on the desk
    cv2.rectangle(changed, (620, 320), (680, 360), (250, 250, 250), -1)
    out = compare_workspace(ref, changed, [], [], CFG)
    regions = out["changes"]["regions"]
    assert regions, out["changes"]
    # The region, in reference pixels, sits where the new object lands after undoing the camera move.
    centre = cv2.perspectiveTransform(np.float32([[[650, 340]]]), np.linalg.inv(h_rc)).reshape(2)
    x1, y1, x2, y2 = regions[0]["box"]
    assert x1 - 5 <= centre[0] <= x2 + 5 and y1 - 5 <= centre[1] <= y2 + 5
    assert regions[0]["explained_by"] == []  # no detector box explains it
    cx1, cy1, cx2, cy2 = regions[0]["current_box"]  # drawn on the current photo
    assert cx1 - 5 <= 650 <= cx2 + 5 and cy1 - 5 <= 340 <= cy2 + 5


def obj(label: str, box: list[float], conf: float = 0.9) -> dict:
    return {"label": label, "confidence": conf, "box": box}


def test_objects_are_paired_by_category_and_place() -> None:
    reference = [obj("cup", [100, 100, 140, 150]), obj("book", [300, 300, 400, 340]), obj("laptop", [500, 200, 800, 400]),
                 obj("mouse", [820, 380, 860, 410], 0.4)]
    current = [obj("cup", [102, 101, 142, 152]),            # unchanged
               obj("laptop", [430, 230, 730, 430]),          # moved left
               obj("bottle", [850, 100, 880, 200]),          # added
               obj("cell phone", [50, 600, 90, 680], 0.35)]  # weak: uncertain, never "added"
    out = pair_objects(reference, current, diagonal=float(np.hypot(960, 720)), cfg=CFG)
    assert [o["label"] for o in out["unchanged"]] == ["cup"]
    assert [o["label"] for o in out["moved"]] == ["laptop"] and 0.05 < out["moved"][0]["moved_by"] < 0.1
    assert [o["label"] for o in out["added"]] == ["bottle"]
    assert [o["label"] for o in out["removed"]] == ["book"]
    assert sorted(o["label"] for o in out["uncertain"]) == ["cell phone", "mouse"]
    far = pair_objects([obj("cup", [0, 0, 40, 40])], [obj("cup", [900, 650, 940, 700])], diagonal=float(np.hypot(960, 720)), cfg=CFG)
    assert [o["label"] for o in far["removed"]] == ["cup"] and [o["label"] for o in far["added"]] == ["cup"]


def test_detections_are_mapped_through_the_registration() -> None:
    ref = scene_image(5)
    h_rc = camera_move(960, 720, angle_deg=0.0, scale=1.0, shift=(60, 0), perspective=0.0)  # camera moved: content shifts 60 px right
    cur = cv2.warpPerspective(ref, h_rc, (960, 720))
    out = compare_workspace(ref, cur, [obj("cup", [300, 300, 360, 380])], [obj("cup", [360, 300, 420, 380])], CFG)
    assert out["registration"]["ok"]
    assert [o["label"] for o in out["objects"]["unchanged"]] == ["cup"]  # same place once the camera move is undone


def _jpeg(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class SequenceDetector(FakeDetector):
    """Returns the next list of detections on each call (reference first, then current)."""

    def __init__(self, per_call: list[list[Detection]]) -> None:
        super().__init__([])
        self.per_call = per_call
        self.calls = 0

    def detect(self, image_bgr: np.ndarray):  # type: ignore[override]
        self.detections = self.per_call[min(self.calls, len(self.per_call) - 1)]
        self.calls += 1
        return super().detect(image_bgr)


def test_workspace_diff_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    detector = SequenceDetector([
        [Detection(41, "cup", 0.9, (100, 100, 150, 160)), Detection(73, "book", 0.8, (400, 400, 520, 450))],
        [Detection(41, "cup", 0.88, (101, 99, 151, 161)), Detection(39, "bottle", 0.7, (700, 100, 740, 220))],
    ])
    settings = make_settings()
    app = create_app(settings)
    loaders = {"yolox": lambda e, p: detector, "resnet": lambda e, p: FakeExtractor()}
    app.state.vision = VisionService(settings, load_config(), load_ontology(), registry=_registry(tmp_path, monkeypatch, present=True), loaders=loaders)
    client = TestClient(app)
    ref = scene_image(6)
    cur = ref.copy()
    cv2.rectangle(cur, (700, 100), (740, 220), (30, 200, 230), -1)  # a bottle appears
    cv2.rectangle(cur, (400, 400), (520, 450), (128, 128, 128), -1)  # the book is gone
    res = client.post("/api/vision/workspace-diff", files={"reference": ("ref.jpg", _jpeg(ref), "image/jpeg"),
                                                           "current": ("cur.jpg", _jpeg(cur), "image/jpeg")})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["registration"]["ok"] and body["state"] == "complete"
    assert [o["label"] for o in body["objects"]["added"]] == ["bottle"]
    assert [o["label"] for o in body["objects"]["removed"]] == ["book"]
    assert [o["label"] for o in body["objects"]["unchanged"]] == ["cup"]
    assert "1 added (bottle)" in body["summary"] and "1 removed (book)" in body["summary"]
    assert any("parallax" in item for item in body["limitations"])
    assert body["counts"] == {"book": {"reference": 1, "current": 0}, "bottle": {"reference": 0, "current": 1}}


def test_object_changes_need_changed_pixels() -> None:
    """The same photo twice, but the detector reports a bottle in one of them only: a detector disagreement."""
    ref = scene_image(7)
    out = compare_workspace(ref, ref.copy(), [obj("cup", [100, 100, 150, 160])],
                            [obj("cup", [100, 100, 150, 160]), obj("bottle", [700, 100, 740, 220])], CFG)
    assert out["objects"]["added"] == []
    weak = out["objects"]["uncertain"]
    assert [o["label"] for o in weak] == ["bottle"] and weak[0]["pixel_change"] == 0.0
    assert "did not change" in weak[0]["why"]


def test_moves_pair_each_object_with_its_nearest_counterpart() -> None:
    reference = [obj("cup", [100, 100, 140, 150]), obj("cup", [400, 100, 440, 150])]
    current = [obj("cup", [130, 100, 170, 150]), obj("cup", [430, 100, 470, 150])]  # both nudged 30 px right
    out = pair_objects(reference, current, diagonal=float(np.hypot(960, 720)), cfg=CFG)
    assert len(out["moved"]) == 2 and all(o["moved_by"] < 0.03 for o in out["moved"])
    assert {(o["reference_box"][0], o["current_box"][0]) for o in out["moved"]} == {(100, 130), (400, 430)}


def test_objects_outside_the_shared_view_are_not_called_removed() -> None:
    ref = scene_image(8)
    h_rc = camera_move(960, 720, angle_deg=0.0, scale=1.0, shift=(-400, 0), perspective=0.0)  # camera turned right: left part leaves the view
    cur = cv2.warpPerspective(ref, h_rc, (960, 720))
    out = compare_workspace(ref, cur, [obj("book", [40, 300, 160, 380]), obj("cup", [700, 300, 760, 380])],
                            [obj("cup", [300, 300, 360, 380])], CFG)
    assert out["registration"]["ok"], out["registration"]
    assert out["objects"]["removed"] == []
    gone = next(o for o in out["objects"]["uncertain"] if o["label"] == "book")
    assert "out of view" in gone["why"]
    assert [o["label"] for o in out["objects"]["unchanged"]] == ["cup"]
