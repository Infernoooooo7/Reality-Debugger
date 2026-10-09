"""YOLOX (official ONNX exports) on ONNX Runtime.

Pre/post-processing follows the official demo (demo/ONNXRuntime,
yolox/data/data_augment.py preproc, yolox/utils/demo_utils.py): resize
keeping aspect ratio, paste top-left on a canvas filled with 114, BGR,
float32 0-255, NCHW; decode per stride-8/16/32 grid; score = objectness x
class score; NMS. The same decoding runs in the browser
(frontend/src/vision/deep/yolox.ts).

Two configurations matter and are reported separately:
* ``protocol``: the settings YOLOX uses for its published COCO numbers
  (score >= 0.01, class-aware NMS IoU 0.65) - measures the model.
* ``deployed``: what the app uses (pre-NMS 0.1, class-agnostic NMS 0.45) -
  measures the product.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.vision.nms import batched_nms
from app.vision.types import Detection, ImageInfo, InferenceResult


@dataclass(slots=True)
class YoloxConfig:
    score_floor: float = 0.1  # pre-NMS score (objectness x class)
    nms_iou: float = 0.45
    class_agnostic: bool = True
    max_detections: int = 100
    input_size: int | None = None  # None = the model's native size
    keep_candidates: bool = False  # keep pre-NMS candidates (for error analysis)

    @classmethod
    def protocol(cls) -> YoloxConfig:
        return cls(score_floor=0.01, nms_iou=0.65, class_agnostic=False, max_detections=100)

    @classmethod
    def deployed(cls) -> YoloxConfig:
        return cls(score_floor=0.1, nms_iou=0.45, class_agnostic=True, max_detections=100)


def _session(path: Path, threads: int | None) -> Any:
    from app.vision.runtime import ort_session

    return ort_session(path, threads=threads)


class YoloxDetector:
    def __init__(self, entry: dict[str, Any], weights: Path, labels: list[str | None], *, config: YoloxConfig | None = None, threads: int | None = None) -> None:
        self.model_id = entry["id"]
        self.model_version = entry.get("version", "")
        self._labels = labels
        self.config = config or YoloxConfig.deployed()
        t0 = time.perf_counter()
        self.session = _session(weights, threads)
        self.load_ms = (time.perf_counter() - t0) * 1000
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.native_size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else int(entry["input"]["width"])
        self.strides = [int(s) for s in entry.get("decode", {}).get("strides", [8, 16, 32])]
        self._grids: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def labels(self) -> list[str | None]:
        return self._labels

    def _grid(self, size: int) -> tuple[np.ndarray, np.ndarray]:
        if size not in self._grids:
            grids, strides = [], []
            for s in self.strides:
                n = size // s
                xv, yv = np.meshgrid(np.arange(n), np.arange(n))
                grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
                strides.append(np.full((n * n, 1), s))
            self._grids[size] = (np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32))
        return self._grids[size]

    def _forward(self, image_bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, dict[str, float]]:
        h, w = image_bgr.shape[:2]
        t0 = time.perf_counter()
        r = min(size / h, size / w)
        nh, nw = int(h * r), int(w * r)
        canvas = np.full((size, size, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        t1 = time.perf_counter()
        out = self.session.run(None, {self.input_name: x})[0][0]
        t2 = time.perf_counter()
        return out, r, {"preprocess": (t1 - t0) * 1000, "inference": (t2 - t1) * 1000}

    def _postprocess(self, out: np.ndarray, r: float, size: int, cfg: YoloxConfig, w: int, h: int) -> tuple[list[Detection], dict[str, Any] | None]:
        grid, stride = self._grid(size)
        xy = (out[:, :2] + grid) * stride
        wh = np.exp(out[:, 2:4]) * stride
        cls_scores = out[:, 4:5] * out[:, 5:]
        cls = cls_scores.argmax(1)
        score = cls_scores[np.arange(len(cls)), cls]
        keep = score >= cfg.score_floor
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1)[keep] / r
        cls, score, objectness = cls[keep], score[keep], out[keep, 4]
        # Second-best class score: a cheap ambiguity signal (not a calibrated uncertainty).
        ranked = np.argsort(cls_scores[keep], axis=1)
        sorted_scores = np.take_along_axis(cls_scores[keep], ranked, axis=1)
        second = sorted_scores[:, -2] if sorted_scores.shape[1] > 1 else np.zeros(len(sorted_scores), np.float32)
        second_cls = ranked[:, -2] if ranked.shape[1] > 1 else np.full(len(ranked), -1)
        margin = sorted_scores[:, -1] - second
        order = batched_nms(boxes, score, cls, cfg.nms_iou, cfg.class_agnostic)[: cfg.max_detections]
        dets = []
        for i in order:
            label = self._labels[int(cls[i])] if int(cls[i]) < len(self._labels) else None
            if label is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            dets.append(Detection(
                class_id=int(cls[i]), class_name=label, confidence=float(score[i]),
                box=(max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)),
                uncertainty={"class_margin": round(float(margin[i]), 4), "second_class_score": round(float(second[i]), 4)},
                extra={"objectness": round(float(objectness[i]), 4),
                       "second_class": self._labels[int(second_cls[i])] if 0 <= int(second_cls[i]) < len(self._labels) else None},
            ))
        candidates = {"boxes": boxes, "scores": score, "classes": cls} if cfg.keep_candidates else None
        return dets, candidates

    def detect_many(self, image_bgr: np.ndarray, configs: list[YoloxConfig], *, frame_id: int | None = None, timestamp_ms: float | None = None) -> list[InferenceResult]:
        """One network pass, several post-processing configurations (same input size)."""
        sizes = {c.input_size or self.native_size for c in configs}
        if len(sizes) != 1:
            raise ValueError("detect_many needs one input size")
        size = sizes.pop()
        h, w = image_bgr.shape[:2]
        out, r, timings = self._forward(image_bgr, size)
        results = []
        for cfg in configs:
            t0 = time.perf_counter()
            dets, candidates = self._postprocess(out, r, size, cfg, w, h)
            post = (time.perf_counter() - t0) * 1000
            result = InferenceResult(
                model_id=self.model_id, model_version=self.model_version,
                image=ImageInfo(w, h, frame_id, timestamp_ms), detections=dets,
                config={**asdict(cfg), "input_size": size, "runtime": "onnxruntime-cpu"},
                timings_ms={**timings, "postprocess": post, "total": timings["preprocess"] + timings["inference"] + post},
            )
            if candidates is not None:
                result.config["_candidates"] = candidates
            results.append(result)
        return results

    def detect(self, image_bgr: np.ndarray, *, frame_id: int | None = None, timestamp_ms: float | None = None, config: YoloxConfig | None = None) -> InferenceResult:
        return self.detect_many(image_bgr, [config or self.config], frame_id=frame_id, timestamp_ms=timestamp_ms)[0]


def load(entry: dict[str, Any], weights: Path, labels: list[str | None] | None = None) -> YoloxDetector:
    if labels is None:
        from app.vision.registry import ModelRegistry

        labels = ModelRegistry().labels(entry["id"])
    return YoloxDetector(entry, weights, labels)
