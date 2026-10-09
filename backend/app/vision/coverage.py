"""How much of an image's visible structure lies outside every recognised object.

Detection count alone cannot tell whether a scan looked at "everything": a
closed-vocabulary detector recognises only its own categories (COCO-80 here),
and small objects disappear when a large photo is downscaled. This module
measures the visible structure (intensity edges) that no detection box
explains. It does not count objects and it is not a recall estimate; it is
evidence that the image contains detail the detectors did not account for.

Definition (identical in frontend/src/vision/coverage.ts; parity-tested):
1. luminance 0.299 R + 0.587 G + 0.114 B on a 0..1 scale, image reduced so its
   long side is at most ``STRUCTURE_SIDE`` px (area averaging);
2. central-difference gradients (zero on the border rows/columns);
   a pixel is "structure" when the gradient magnitude >= ``EDGE_THRESHOLD``;
3. the boxes of all recognised objects (normalised x, y, w, h) are rasterised
   with floor/ceil to cover every touched pixel;
4. ``unexplained_share`` = structure pixels outside every box / structure pixels,
   ``edge_density`` = structure pixels / all pixels,
   ``box_coverage`` = pixels inside any box / all pixels.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

STRUCTURE_SIDE = 256
EDGE_THRESHOLD = 0.05


@dataclass(slots=True)
class StructureStats:
    edge_density: float
    unexplained_share: float | None  # None when the image has no structure at all
    box_coverage: float
    width: int
    height: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {"edge_density": round(self.edge_density, 4),
                "unexplained_share": None if self.unexplained_share is None else round(self.unexplained_share, 4),
                "box_coverage": round(self.box_coverage, 4), "width": self.width, "height": self.height}


def luminance_small(image_bgr: np.ndarray, side: int = STRUCTURE_SIDE) -> np.ndarray:
    import cv2

    h, w = image_bgr.shape[:2]
    scale = min(1.0, side / max(h, w))
    small = image_bgr if scale >= 1.0 else cv2.resize(image_bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    rgb = small[..., ::-1].astype(np.float32) / 255.0
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def structure_mask(gray: np.ndarray, threshold: float = EDGE_THRESHOLD) -> np.ndarray:
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
    return np.sqrt(gx * gx + gy * gy) >= threshold


def box_mask(shape: tuple[int, int], boxes: Iterable[Mapping[str, float]]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for b in boxes:
        x1 = max(0, math.floor(b["x"] * w))
        y1 = max(0, math.floor(b["y"] * h))
        x2 = min(w, math.ceil((b["x"] + b["w"]) * w))
        y2 = min(h, math.ceil((b["y"] + b["h"]) * h))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask


def structure_stats(gray: np.ndarray, boxes: Iterable[Mapping[str, float]], threshold: float = EDGE_THRESHOLD) -> StructureStats:
    """``gray``: 0..1 luminance at analysis size; ``boxes``: normalised {x, y, w, h}."""
    edges = structure_mask(gray, threshold)
    covered = box_mask(gray.shape, boxes)
    n_edges = int(edges.sum())
    total = edges.size
    return StructureStats(
        edge_density=n_edges / total if total else 0.0,
        unexplained_share=float((edges & ~covered).sum()) / n_edges if n_edges else None,
        box_coverage=float(covered.sum()) / total if total else 0.0,
        width=int(gray.shape[1]),
        height=int(gray.shape[0]),
    )


def test_pattern(w: int = 64, h: int = 48) -> np.ndarray:
    """Deterministic pattern shared with the TypeScript parity test: a bright square,
    a diagonal ramp band and a checkerboard patch on a flat background."""
    ys, xs = np.mgrid[0:h, 0:w]
    img = np.full((h, w), 0.2, dtype=np.float32)
    img[8:20, 6:22] = 0.9
    band = (xs + ys >= 40) & (xs + ys < 52)
    img[band] = 0.2 + 0.6 * ((xs[band] + ys[band] - 40) / 12.0)
    checker = ((xs // 2 + ys // 2) % 2 == 0) & (xs >= 44) & (xs < 60) & (ys >= 28) & (ys < 44)
    img[checker] = 0.7
    return img
