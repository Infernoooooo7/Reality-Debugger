#!/usr/bin/env python3
"""Evaluate the local detectors on a labelled image set and pick score thresholds.

Runs the fast detector (EfficientDet-Lite0 through MediaPipe Tasks - the same
C++ graph and post-processing the browser uses) and the deep detector (YOLOX-S
through ONNX Runtime with the official pre/post-processing) on a YOLO-format
dataset, then reports:

  * COCO AP / AP50 / AP75 / APs / APm / APl / AR100 (pycocotools COCOeval);
  * precision / recall / F1 at IoU 0.5 for a sweep of score thresholds, used to
    choose the operating points written to config/detection.json and
    config/tracking.json;
  * per-image inference latency on this machine's CPU;
  * for the fast detector, square letterboxing vs. MediaPipe's default stretch.

Dataset used for the numbers in docs/COMPUTER_VISION_RESEARCH.md: "coco128"
(the first 128 COCO train2017 images with labels, as packaged by Ultralytics:
https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip).
It is only downloaded locally for measurement and is not part of this
repository. Because both detectors were trained on COCO train2017, absolute AP
on these images is optimistic; the numbers are used to compare configurations
and to locate precision/recall trade-offs, not as accuracy claims.

Usage:  python tools/eval_detectors.py --data /path/to/coco128 --out docs/benchmarks/detector_eval_coco128.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "frontend" / "public" / "models"
COCO_META = ROOT / "tools" / "data" / "coco_panoptic_categories.json"
THRESHOLDS = [round(t, 2) for t in np.arange(0.10, 0.81, 0.05)]


# ---------------------------------------------------------------- dataset

def load_dataset(root: Path, labels80: list[str]):
    things = [c for c in json.loads(COCO_META.read_text()) if c["isthing"]]
    name_to_id = {c["name"]: c["id"] for c in things}
    cat_ids = [name_to_id[name] for name in labels80]  # YOLO class index -> COCO category id
    images, annotations = [], []
    for i, img_path in enumerate(sorted((root / "images").rglob("*.jpg"))):
        h, w = cv2.imread(str(img_path)).shape[:2]
        images.append({"id": i, "file_name": str(img_path), "width": w, "height": h})
        label_path = root / "labels" / img_path.parent.name / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        for line in label_path.read_text().splitlines():
            cls, cx, cy, bw, bh = line.split()
            bw_px, bh_px = float(bw) * w, float(bh) * h
            x, y = float(cx) * w - bw_px / 2, float(cy) * h - bh_px / 2
            annotations.append({"id": len(annotations) + 1, "image_id": i, "category_id": cat_ids[int(cls)],
                                "bbox": [x, y, bw_px, bh_px], "area": bw_px * bh_px, "iscrowd": 0})
    gt = {"images": images, "annotations": annotations, "categories": [{"id": c["id"], "name": c["name"]} for c in things]}
    return gt, {name: name_to_id[name] for name in labels80}


# ---------------------------------------------------------------- detectors

class FastDetector:
    """EfficientDet-Lite0 via MediaPipe Tasks (same options the app uses)."""

    def __init__(self, letterbox: bool, score_floor: float):
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp_image = __import__("mediapipe").Image
        self.mp_format = __import__("mediapipe").ImageFormat
        options = vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "efficientdet_lite0.tflite")),
            running_mode=vision.RunningMode.IMAGE,
            score_threshold=score_floor,
            max_results=100,
        )
        t0 = time.perf_counter()
        self.detector = vision.ObjectDetector.create_from_options(options)
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.letterbox = letterbox

    def close(self) -> None:
        self.detector.close()

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.letterbox:  # centre on a square grey canvas, like frontend/src/vision/engine.ts
            side = max(h, w)
            canvas = np.full((side, side, 3), 114, np.uint8)
            ox, oy = (side - w) // 2, (side - h) // 2
            canvas[oy:oy + h, ox:ox + w] = rgb
            rgb = canvas
        else:
            ox = oy = 0
        result = self.detector.detect(self.mp_image(image_format=self.mp_format.SRGB, data=np.ascontiguousarray(rgb)))
        out = []
        for d in result.detections:
            b = d.bounding_box
            c = d.categories[0]
            out.append((c.category_name, c.score, b.origin_x - ox, b.origin_y - oy, b.width, b.height))
        return out


class DeepDetector:
    """YOLOX-S via ONNX Runtime with the official demo pre/post-processing."""

    def __init__(self, nms_iou: float, score_floor: float):
        manifest = json.loads((MODELS / "yolox_s.manifest.json").read_text())
        self.labels = manifest["labels"]
        self.size = manifest["input"]["width"]
        self.nms_iou, self.score_floor = nms_iou, score_floor
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(str(MODELS / manifest["file"]), so, providers=["CPUExecutionProvider"])
        self.load_ms = (time.perf_counter() - t0) * 1000
        grids, strides = [], []
        for s in manifest["decode"]["strides"]:
            n = self.size // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grid, self.stride = np.concatenate(grids), np.concatenate(strides)

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        r = min(self.size / h, self.size / w)
        pad = np.full((self.size, self.size, 3), 114, np.uint8)
        pad[: int(h * r), : int(w * r)] = cv2.resize(bgr, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        x = pad.transpose(2, 0, 1)[None].astype(np.float32)
        p = self.session.run(None, {self.session.get_inputs()[0].name: x})[0][0]
        xy = (p[:, :2] + self.grid) * self.stride
        wh = np.exp(p[:, 2:4]) * self.stride
        scores = p[:, 4:5] * p[:, 5:]
        cls, sc = scores.argmax(1), scores.max(1)
        keep = sc > self.score_floor
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1)[keep] / r
        cls, sc = cls[keep], sc[keep]
        order = nms(boxes, sc, self.nms_iou)
        return [(self.labels[cls[i]], float(sc[i]), boxes[i, 0], boxes[i, 1], boxes[i, 2] - boxes[i, 0], boxes[i, 3] - boxes[i, 1]) for i in order]


def nms(boxes, scores, iou_thr):
    order, keep = scores.argsort()[::-1], []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0]); yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2]); yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


# ---------------------------------------------------------------- metrics

def coco_metrics(gt: dict, dets: list[dict]) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets) if dets else None
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    names = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    return {n: round(float(v) * 100, 1) for n, v in zip(names, ev.stats)}


def iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax2, bx2) - max(a[0], b[0])), max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def operating_points(gt: dict, dets: list[dict]) -> list[dict]:
    by_img_gt: dict[int, list] = {}
    for a in gt["annotations"]:
        by_img_gt.setdefault(a["image_id"], []).append(a)
    total_gt = len(gt["annotations"])
    rows = []
    for t in THRESHOLDS:
        tp = fp = 0
        for img in gt["images"]:
            cands = sorted((d for d in dets if d["image_id"] == img["id"] and d["score"] >= t), key=lambda d: -d["score"])
            used = set()
            for d in cands:
                best, best_j = 0.5, None
                for j, g in enumerate(by_img_gt.get(img["id"], [])):
                    if j in used or g["category_id"] != d["category_id"]:
                        continue
                    v = iou_xywh(d["bbox"], g["bbox"])
                    if v >= best:
                        best, best_j = v, j
                if best_j is None:
                    fp += 1
                else:
                    tp += 1
                    used.add(best_j)
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / total_gt if total_gt else 0.0
        rows.append({"threshold": t, "precision": round(p, 3), "recall": round(r, 3),
                     "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0, "tp": tp, "fp": fp})
    return rows


def run(name: str, detector, gt: dict, name_to_cat: dict) -> dict:
    dets, latencies = [], []
    for img in gt["images"]:
        bgr = cv2.imread(img["file_name"])
        t0 = time.perf_counter()
        out = detector(bgr)
        latencies.append((time.perf_counter() - t0) * 1000)
        for label, score, x, y, w, h in out:
            if label in name_to_cat:
                dets.append({"image_id": img["id"], "category_id": name_to_cat[label], "bbox": [float(x), float(y), float(w), float(h)], "score": float(score)})
    if hasattr(detector, "close"):
        detector.close()
    ops = operating_points(gt, dets)
    best = max(ops, key=lambda r: r["f1"])
    result = {
        "detector": name,
        "load_ms": round(detector.load_ms),
        "latency_ms_median": round(float(np.median(latencies)), 1),
        "latency_ms_p90": round(float(np.percentile(latencies, 90)), 1),
        "detections": len(dets),
        "coco": coco_metrics(gt, dets),
        "operating_points": ops,
        "f1_optimal": best,
    }
    print(f"\n{name}: load {result['load_ms']} ms, median {result['latency_ms_median']} ms/img")
    print("  COCO:", result["coco"])
    print("  thr   P     R     F1")
    for r in ops:
        print(f"  {r['threshold']:.2f}  {r['precision']:.3f} {r['recall']:.3f} {r['f1']:.3f}{'  <- F1 max' if r is best else ''}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "benchmarks" / "detector_eval_coco128.json")
    args = parser.parse_args()

    labels80 = json.loads((MODELS / "yolox_s.manifest.json").read_text())["labels"]
    gt, name_to_cat = load_dataset(args.data, labels80)
    print(f"dataset: {len(gt['images'])} images, {len(gt['annotations'])} boxes")
    results = [
        run("efficientdet_lite0 (letterboxed, as in the app)", FastDetector(letterbox=True, score_floor=0.05), gt, name_to_cat),
        run("efficientdet_lite0 (MediaPipe default stretch)", FastDetector(letterbox=False, score_floor=0.05), gt, name_to_cat),
        run("yolox_s (official preprocessing, NMS IoU 0.45)", DeepDetector(nms_iou=0.45, score_floor=0.05), gt, name_to_cat),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "dataset": "coco128 (first 128 COCO train2017 images, Ultralytics packaging) - local measurement only",
        "caveat": "Both detectors were trained on COCO train2017, so absolute AP here is optimistic; use for relative comparison.",
        "images": len(gt["images"]), "boxes": len(gt["annotations"]),
        "machine": "Linux x86_64 CPU, onnxruntime 4 intra-op threads, MediaPipe Tasks Python CPU",
        "results": results,
    }, indent=1) + "\n")
    print(f"\nwrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
