"""Global (camera-induced) image motion - port of frontend/src/vision/motion.ts.

Coarse-to-fine block matching (mean absolute difference) on a grey
thumbnail with a parabolic sub-pixel fit. Used offline to give the tracker
the same camera-motion estimate the browser computes, so the tracking
evaluation measures the shipped algorithm. Both implementations are checked
against a shared test pattern (backend/tests/test_vision_core.py and
frontend/src/vision/__tests__/motion.test.ts).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.vision.interfaces import GlobalMotion

THUMB_W, THUMB_H = 128, 96  # config/vision.json signals.thumbWidth / thumbHeight


@dataclass(slots=True)
class MotionOptions:
    max_shift: float = 0.12
    border: float = 0.08


def _downsample2(img: np.ndarray) -> np.ndarray:
    h2, w2 = img.shape[0] // 2, img.shape[1] // 2
    a = img[: 2 * h2, : 2 * w2]
    return 0.25 * (a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2])


def _sad(prev: np.ndarray, cur: np.ndarray, sx: int, sy: int, margin: int) -> float:
    h, w = cur.shape
    y0, y1 = max(margin, sy), min(h - margin, h + sy)
    x0, x1 = max(margin, sx), min(w - margin, w + sx)
    if y1 <= y0 or x1 <= x0:
        return float("inf")
    c = cur[y0:y1, x0:x1]
    p = prev[y0 - sy : y1 - sy, x0 - sx : x1 - sx]
    return float(np.mean(np.abs(c - p)))


def _search(prev: np.ndarray, cur: np.ndarray, cx: int, cy: int, radius: int, margin: int) -> tuple[int, int, float, dict[tuple[int, int], float]]:
    best, bx, by = float("inf"), cx, cy
    costs: dict[tuple[int, int], float] = {}
    for sy in range(cy - radius, cy + radius + 1):
        for sx in range(cx - radius, cx + radius + 1):
            c = _sad(prev, cur, sx, sy, margin)
            costs[(sx, sy)] = c
            if c < best:
                best, bx, by = c, sx, sy
    return bx, by, best, costs


def _parabola(cm: float, c0: float, cp: float) -> float:
    denom = cm - 2 * c0 + cp
    return max(-0.5, min(0.5, 0.5 * (cm - cp) / denom)) if denom > 1e-9 else 0.0


def estimate_shift(prev: np.ndarray, cur: np.ndarray, opts: MotionOptions | None = None) -> tuple[float, float, float]:
    """(dx, dy, confidence): shift of the content from prev to cur in frame widths/heights."""
    o = opts or MotionOptions()
    h, w = cur.shape
    margin = max(1, round(o.border * w))
    p2, c2 = _downsample2(prev), _downsample2(cur)
    r2 = max(1, round(o.max_shift * w / 2))
    cbx, cby, cbest, ccosts = _search(p2, c2, 0, 0, r2, max(1, margin >> 1))
    fbx, fby, fbest, fcosts = _search(prev, cur, cbx * 2, cby * 2, 1, margin)

    def get(x: int, y: int) -> float:
        return fcosts.get((x, y), _sad(prev, cur, x, y, margin))

    fx = fbx + _parabola(get(fbx - 1, fby), fbest, get(fbx + 1, fby))
    fy = fby + _parabola(get(fbx, fby - 1), fbest, get(fbx, fby + 1))
    values = sorted(v for v in ccosts.values() if np.isfinite(v))
    median = values[len(values) // 2] if values else 0.0
    confidence = max(0.0, min(1.0, 1 - cbest / median)) if median > 1e-6 else 0.0
    return fx / w, fy / h, round(confidence, 3)


def thumbnail_gray(image_bgr: np.ndarray) -> np.ndarray:
    """Same luminance and size as the browser's SignalAnalyzer (values 0..1)."""
    small = cv2.resize(image_bgr, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    b, g, r = small[..., 0], small[..., 1], small[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


class BlockMatchingMotion:
    """MotionEstimator over full frames (thumbnails are computed internally)."""

    method = "coarse-to-fine block matching (translation), 128x96 thumbnail"

    def __init__(self, opts: MotionOptions | None = None) -> None:
        self.opts = opts or MotionOptions()

    def estimate(self, previous_gray: np.ndarray, current_gray: np.ndarray) -> GlobalMotion:
        dx, dy, conf = estimate_shift(previous_gray, current_gray, self.opts)
        return GlobalMotion(dx=dx, dy=dy, confidence=conf, method=self.method)


def _hash32(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """32-bit integer hash of pixel coordinates (same arithmetic as Math.imul / >>> in TypeScript)."""
    m = 0xFFFFFFFF
    h = (x.astype(np.int64) * 374761393 + y.astype(np.int64) * 668265263) & m
    h = ((h ^ (h >> 13)) * 1274126177) & m
    return (h ^ (h >> 16)) & m


def test_pattern(w: int = THUMB_W, h: int = THUMB_H, shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    """Deterministic textured pattern shared with the TypeScript tests (hashed noise, then a 3x3 box blur)."""
    ys, xs = np.mgrid[0:h, 0:w]
    base = (_hash32(xs - shift_x, ys - shift_y) & 255).astype(np.float64) / 255.0
    out = np.empty_like(base)
    for y in range(h):
        for x in range(w):
            y0, y1, x0, x1 = max(0, y - 1), min(h, y + 2), max(0, x - 1), min(w, x + 2)
            out[y, x] = base[y0:y1, x0:x1].mean()
    return out
