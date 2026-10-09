"""Image-quality gate: is the image good enough for "not detected" to mean anything?

Brightness and sharpness use the browser's signal definitions (mean
luminance and the log-scaled variance of the Laplacian of a 128x96 grey
thumbnail, config/vision.json "signals"), so the app and the server judge an
image the same way. The thresholds are in config/inference.json "quality".

A failed gate does not hide detections: the analysis still runs, but its
state becomes ``insufficient_image_quality`` and absent objects are reported
as unreliable rather than absent. Upscaling is never used to "recover"
detail that the image does not contain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.runtime_config import RuntimeConfig
from app.services.vision_service import gray_signals, thumbnail_gray


@dataclass(slots=True)
class QualityReport:
    brightness: float
    sharpness: float
    width: int
    height: int
    issues: list[dict[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "brightness": self.brightness, "sharpness": self.sharpness,
                "width": self.width, "height": self.height, "issues": self.issues}


def signals(image_bgr: np.ndarray, config: RuntimeConfig) -> tuple[float, float]:
    """(brightness, sharpness) exactly as app.services.vision_service.frame_signals computes them."""
    tw, th = int(config.get("vision.signals.thumbWidth")), int(config.get("vision.signals.thumbHeight"))
    gray = thumbnail_gray(Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)), tw, th)
    return gray_signals(gray, log_offset=float(config.get("vision.signals.sharpnessLogOffset")),
                        log_span=float(config.get("vision.signals.sharpnessLogSpan")))


def assess(image_bgr: np.ndarray, config: RuntimeConfig) -> QualityReport:
    h, w = image_bgr.shape[:2]
    brightness, sharpness = signals(image_bgr, config)
    report = QualityReport(brightness=brightness, sharpness=sharpness, width=w, height=h)
    min_b = float(config.get("inference.quality.minBrightness"))
    min_s = float(config.get("inference.quality.minSharpness"))
    min_side = int(config.get("inference.quality.minShortSide"))
    if min(w, h) < min_side:
        report.issues.append({"code": "too_small", "message": f"The image is {w}x{h} px; below {min_side} px on its short side small objects cannot be found."})
    if brightness < min_b:
        report.issues.append({"code": "too_dark", "message": f"Mean brightness {brightness:.2f} is below {min_b:.2f}: detection degrades sharply in this light."})
    if sharpness < min_s:
        report.issues.append({"code": "too_blurred", "message": f"Sharpness {sharpness:.2f} is below {min_s:.2f}: motion or focus blur hides object detail."})
    return report
