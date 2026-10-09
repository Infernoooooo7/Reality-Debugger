"""Normalised results shared by every model adapter.

A detection always says which model (and version) produced it, in which
coordinate system its box is, and for which image size. Values a model does
not provide (masks, track ids, uncertainty) are ``None`` - unavailable, never
approximated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

COORDINATES = "image_px_xyxy"  # x1, y1, x2, y2 in source-image pixels, origin top-left


class AnalysisState(StrEnum):
    """Outcome of one analysis request (not of one object)."""

    COMPLETE = "complete"
    INCOMPLETE = "analysis_incomplete"  # e.g. a time budget stopped tiles or keyframes early
    MODEL_UNAVAILABLE = "model_unavailable"  # weights missing, failed checksum, or failed to load
    INSUFFICIENT_IMAGE_QUALITY = "insufficient_image_quality"  # too dark / blurred / small to trust


class ObjectState(StrEnum):
    """Evidence level of one reported object."""

    DETECTED = "detected"  # above the model's operating threshold
    TENTATIVE = "tentative"  # below it but above the low floor (or a track not yet confirmed)
    AMBIGUOUS = "ambiguous"  # models disagree on the class, or two classes score alike


class QueryState(StrEnum):
    """Answer to "is there an X?" for a category."""

    DETECTED = "detected"
    TENTATIVE = "tentative"  # only candidates below the operating threshold
    AMBIGUOUS = "ambiguous"  # found, but the model nearly as strongly suggests another class
    NOT_DETECTED = "not_detected"  # supported but not found: NOT the same as confirmed absent
    UNSUPPORTED_CATEGORY = "unsupported_category"  # no loaded model can name it
    INSUFFICIENT_IMAGE_QUALITY = "insufficient_image_quality"  # not found, but the image is too poor to rule it out
    ANALYSIS_INCOMPLETE = "analysis_incomplete"  # not found, but part of the image was not analysed


@dataclass(slots=True)
class ImageInfo:
    width: int
    height: int
    frame_id: int | None = None
    timestamp_ms: float | None = None


@dataclass(slots=True)
class Detection:
    class_id: int  # index in the producing model's own label list
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]  # COORDINATES
    mask: np.ndarray | None = None  # bool HxW in source-image coordinates, only from segmentation models
    track_id: int | None = None
    uncertainty: dict[str, float] | None = None  # e.g. {"class_margin": 0.12}; None = not estimated
    extra: dict[str, Any] = field(default_factory=dict)  # model-specific metadata (objectness, tile index...)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def state(self, operating_threshold: float) -> ObjectState:
        if self.uncertainty and self.uncertainty.get("ambiguous"):
            return ObjectState.AMBIGUOUS
        return ObjectState.DETECTED if self.confidence >= operating_threshold else ObjectState.TENTATIVE


@dataclass(slots=True)
class InferenceResult:
    model_id: str
    model_version: str
    image: ImageInfo
    detections: list[Detection]
    config: dict[str, Any]  # the inference configuration actually used
    timings_ms: dict[str, float]
    state: AnalysisState = AnalysisState.COMPLETE
    notes: list[str] = field(default_factory=list)
    coordinate_system: str = COORDINATES

    def to_dict(self, *, include_masks: bool = False) -> dict[str, Any]:
        w, h = self.image.width, self.image.height
        out = []
        for d in self.detections:
            x1, y1, x2, y2 = d.box
            item: dict[str, Any] = {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": round(float(d.confidence), 4),
                "box": [round(float(v), 2) for v in d.box],
                "box_normalized": {"x": x1 / w, "y": y1 / h, "w": (x2 - x1) / w, "h": (y2 - y1) / h} if w and h else None,
                "mask": None,
                "track_id": d.track_id,
                "uncertainty": d.uncertainty,
            }
            if d.mask is not None:
                item["mask"] = mask_to_rle(d.mask) if include_masks else {"available": True, "area_px": int(d.mask.sum())}
            out.append(item)
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "coordinate_system": self.coordinate_system,
            "image": {"width": w, "height": h, "frame_id": self.image.frame_id, "timestamp_ms": self.image.timestamp_ms},
            "state": self.state.value,
            "config": self.config,
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
            "notes": self.notes,
            "detections": out,
        }


def mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """Uncompressed COCO-style RLE (column-major counts starting with zeros)."""
    flat = np.asfortranarray(mask.astype(np.uint8)).reshape(-1, order="F")
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat.size and flat[0] == 1:
        counts = [0, *counts]
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def rle_to_mask(rle: dict[str, Any]) -> np.ndarray:
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos, value = 0, 0
    for count in rle["counts"]:
        if value:
            flat[pos : pos + count] = 1
        pos += count
        value ^= 1
    return flat.reshape((h, w), order="F").astype(bool)
