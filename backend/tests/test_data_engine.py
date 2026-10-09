"""Data engine: pre-annotation export (COCO / YOLO), review queue, dataset versions, promotion gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from pytest import approx

from app.datasets import annotation as ann

ROOT = Path(__file__).resolve().parents[2]

IMG = ann.ImageDetections("a.jpg", 200, 100, [
    {"label": "cup", "box": [10, 20, 50, 80], "confidence": 0.9, "state": "detected"},
    {"label": "bottle", "box": [150, 0, 210, 40], "confidence": 0.5, "state": "detected"},  # crosses the right border
    {"label": "forklift", "box": [0, 0, 5, 5], "confidence": 0.9, "state": "detected"},  # not a model category
])
CATS = ["bottle", "cup"]


def test_coco_export_keeps_boxes_and_marks_them_for_review() -> None:
    coco = ann.to_coco([IMG], CATS)
    assert [c["name"] for c in coco["categories"]] == CATS
    a = next(a for a in coco["annotations"] if a["category_id"] == 2)
    assert a["bbox"] == [10.0, 20.0, 40.0, 60.0] and a["area"] == 2400.0 and a["review"] == "pending"
    assert len(coco["annotations"]) == 2  # the unknown label is not exported


def test_yolo_export_round_trips_and_clips_to_the_image(tmp_path: Path) -> None:
    ann.to_yolo([IMG], CATS, tmp_path)
    lines = (tmp_path / "a.txt").read_text().splitlines()
    assert len(lines) == 2 and (tmp_path / "classes.txt").read_text().split() == CATS
    back = ann.from_yolo(tmp_path / "a.txt", CATS, 200, 100)
    cup = next(o for o in back if o["label"] == "cup")
    assert cup["box"] == approx([10, 20, 50, 80], abs=1e-3)
    bottle = next(o for o in back if o["label"] == "bottle")
    assert bottle["box"] == approx([150, 0, 200, 40], abs=1e-3)  # clipped at x = 200


def test_uncertainty_ranks_ambiguous_above_confident() -> None:
    confident = [{"state": "detected", "confidence": 0.95, "uncertainty": {"second_class_score": 0.01}}]
    ambiguous = [{"state": "ambiguous", "confidence": 0.6, "uncertainty": {"second_class_score": 0.5}}]
    borderline = [{"state": "tentative", "confidence": 0.29}]
    assert ann.uncertainty(ambiguous) > ann.uncertainty(borderline) > ann.uncertainty(confident)
    assert ann.uncertainty([]) > 0  # empty images still get reviewed eventually


def test_review_queue_pushes_near_duplicates_back() -> None:
    items = [
        {"file": "a", "uncertainty": 3.0, "dhash": "ffff0000ffff0000"},
        {"file": "a-again", "uncertainty": 2.9, "dhash": "ffff0000ffff0001"},  # 1 bit away: same shot
        {"file": "b", "uncertainty": 1.0, "dhash": "0000ffff0000ffff"},
    ]
    queue = ann.review_queue(items, limit=3)
    assert [q["file"] for q in queue] == ["a", "b", "a-again"]
    assert queue[-1]["near_duplicate_of_queued"] is True


def test_version_manifest_is_content_addressed(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "a.png"), np.zeros((4, 4, 3), np.uint8))
    coco = tmp_path / "ann.json"
    coco.write_text(json.dumps({"images": [{"id": 1, "file_name": "a.png"}], "annotations": [{"image_id": 1, "category_id": 1}],
                                "categories": [{"id": 1, "name": "cup"}]}))
    m1 = ann.version_manifest(tmp_path, name="t", version="1", coco_json=coco, splits={"train": ["a.png"]})
    m2 = ann.version_manifest(tmp_path, name="t", version="1", coco_json=coco, splits={"train": ["a.png"]})
    assert m1["content_sha256"] == m2["content_sha256"] and m1["labels_per_split"] == {"train": {"cup": 1}}
    (tmp_path / "images" / "a.png").write_bytes(b"changed")
    assert ann.version_manifest(tmp_path, name="t", version="1")["content_sha256"] != m1["content_sha256"]


def test_extract_frames_drops_static_duplicates(tmp_path: Path) -> None:
    video = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    rng = np.random.default_rng(0)
    still = (rng.random((48, 64, 3)) * 255).astype(np.uint8)
    for i in range(40):  # 2 s of one still shot, then 2 s of changing content
        writer.write(still if i < 20 else (rng.random((48, 64, 3)) * 255).astype(np.uint8))
    writer.release()
    kept = ann.extract_frames(video, tmp_path / "frames", every_s=0.5)
    assert 1 + 2 <= len(kept) <= 1 + 4  # the still shot is kept once
    assert all((tmp_path / "frames" / k["file"]).exists() for k in kept)


def _promote():
    spec = importlib.util.spec_from_file_location("promote", ROOT / "tools" / "promote.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _record(ap: float, per_class: dict[str, float], ece: float = 0.02, ms: float = 100.0) -> dict:
    return {"dataset": {"id": "d", "split": "test", "images": 10, "evaluated_categories": ["a", "b"]},
            "metrics": {"coco": {"AP": ap}, "calibration": {"ece": ece}},
            "per_class": {k: {"AP": v} for k, v in per_class.items()}, "performance": {"model_ms": {"median": ms}}}


def test_promotion_gate_rejects_a_hidden_class_regression() -> None:
    promote = _promote()
    base = _record(40.0, {"a": 40.0, "b": 40.0})
    better = _record(42.0, {"a": 43.0, "b": 41.0})
    regress = _record(42.0, {"a": 50.0, "b": 34.0})
    kwargs = {"min_ap_gain": 1.0, "max_class_drop": 2.0, "max_ece_increase": 0.02, "max_latency_ratio": 1.5}
    assert all(ok for _, ok, _ in promote.check(base, better, **kwargs))
    failed = [name for name, ok, _ in promote.check(base, regress, **kwargs) if not ok]
    assert failed == ["no per-class regression"]
