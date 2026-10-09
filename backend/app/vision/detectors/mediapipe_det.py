"""EfficientDet-Lite (TFLite) through MediaPipe Tasks - evaluation only.

The browser runs the same model through MediaPipe Tasks (WASM). The Python
MediaPipe package runs the same C++ graph and post-processing, so its outputs
are the closest offline proxy for what the app sees. Requires the optional
``mediapipe`` package (tools/requirements-eval.txt); the backend never imports it.

``letterbox=True`` reproduces the app: the frame is centred on a square grey
canvas before MediaPipe resizes it to the model input (frontend/src/vision/engine.ts).
MediaPipe's own NMS (class-agnostic, IoU 0.3) is not configurable through the
Tasks API.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.vision.types import Detection, ImageInfo, InferenceResult


@dataclass(slots=True)
class MediaPipeConfig:
    score_floor: float = 0.15
    max_results: int = 30
    letterbox: bool = True

    @classmethod
    def protocol(cls) -> MediaPipeConfig:
        """Lowest floor / most results MediaPipe allows: measures the model (COCO protocol caps at 100 per image)."""
        return cls(score_floor=0.01, max_results=100, letterbox=True)

    @classmethod
    def deployed(cls) -> MediaPipeConfig:
        """config/detection.json fast.lowScoreFloor / fast.maxResults, letterboxed as in the app."""
        return cls(score_floor=0.15, max_results=30, letterbox=True)


class MediaPipeDetector:
    def __init__(self, entry: dict[str, Any], weights: Path, labels: list[str | None], *, config: MediaPipeConfig | None = None) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.model_id = entry["id"]
        self.model_version = entry.get("version", "")
        self._labels = labels
        self.config = config or MediaPipeConfig.deployed()
        self._mp = mp
        options = vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(weights)),
            running_mode=vision.RunningMode.IMAGE,
            score_threshold=self.config.score_floor,
            max_results=self.config.max_results,
        )
        t0 = time.perf_counter()
        self.detector = vision.ObjectDetector.create_from_options(options)
        self.load_ms = (time.perf_counter() - t0) * 1000
        self._name_to_id = {name: i for i, name in enumerate(labels) if name}

    @property
    def labels(self) -> list[str | None]:
        return self._labels

    def close(self) -> None:
        self.detector.close()

    def detect(self, image_bgr: np.ndarray, *, frame_id: int | None = None, timestamp_ms: float | None = None) -> InferenceResult:
        cfg = self.config
        h, w = image_bgr.shape[:2]
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ox = oy = 0
        if cfg.letterbox:
            side = max(h, w)
            canvas = np.full((side, side, 3), 114, np.uint8)
            ox, oy = (side - w) // 2, (side - h) // 2
            canvas[oy : oy + h, ox : ox + w] = rgb
            rgb = canvas
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        t1 = time.perf_counter()
        result = self.detector.detect(image)
        t2 = time.perf_counter()
        dets = []
        for d in result.detections:
            c = d.categories[0]
            b = d.bounding_box
            x1, y1 = b.origin_x - ox, b.origin_y - oy
            dets.append(Detection(
                class_id=self._name_to_id.get(c.category_name, c.index), class_name=c.category_name, confidence=float(c.score),
                box=(max(0.0, float(x1)), max(0.0, float(y1)), min(float(w), float(x1 + b.width)), min(float(h), float(y1 + b.height))),
            ))
        t3 = time.perf_counter()
        return InferenceResult(
            model_id=self.model_id, model_version=self.model_version, image=ImageInfo(w, h, frame_id, timestamp_ms),
            detections=dets, config={**asdict(cfg), "runtime": "mediapipe-python", "nms": "MediaPipe class-agnostic IoU 0.3 (fixed)"},
            timings_ms={"preprocess": (t1 - t0) * 1000, "inference": (t2 - t1) * 1000, "postprocess": (t3 - t2) * 1000, "total": (t3 - t0) * 1000},
        )
