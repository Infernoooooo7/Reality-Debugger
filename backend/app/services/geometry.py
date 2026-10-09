"""Box geometry for the local diagnostic engine.

All boxes are normalised ``Box(x, y, w, h)`` (0..1, origin top-left). These
are plain measurements - nothing here knows what the objects are.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from app.schemas.common import Box


def area(b: Box) -> float:
    return b.w * b.h


def intersection(a: Box, b: Box) -> float:
    iw = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    ih = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    return iw * ih if iw > 0 and ih > 0 else 0.0


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(inner: Box, outer: Box) -> float:
    """Share of ``inner`` that lies inside ``outer``."""
    a = area(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def edge_gap(a: Box, b: Box) -> float:
    """Euclidean distance between the closest edges (0 when the boxes touch or overlap)."""
    dx = max(0.0, b.x - (a.x + a.w), a.x - (b.x + b.w))
    dy = max(0.0, b.y - (a.y + a.h), a.y - (b.y + b.h))
    return math.hypot(dx, dy)


def reference_size(a: Box, b: Box) -> float:
    """Square root of the larger box area: a scale-invariant yardstick for 'near'."""
    return math.sqrt(max(area(a), area(b)))


def center(b: Box) -> tuple[float, float]:
    return b.x + b.w / 2, b.y + b.h / 2


def base_inside(inner: Box, outer: Box, tolerance: float = 0.02) -> bool:
    """Whether the bottom-centre point of ``inner`` (where an object stands) lies within ``outer``."""
    bx, by = inner.x + inner.w / 2, inner.y + inner.h
    return outer.x <= bx <= outer.x + outer.w and outer.y <= by <= outer.y + outer.h + tolerance


def near_edge(b: Box, margin: float) -> bool:
    """Whether the box comes within ``margin`` of any frame edge."""
    return b.x <= margin or b.y <= margin or b.x + b.w >= 1 - margin or b.y + b.h >= 1 - margin


def union_box(boxes: Iterable[Box]) -> Box | None:
    items = list(boxes)
    if not items:
        return None
    x1 = min(b.x for b in items)
    y1 = min(b.y for b in items)
    x2 = max(b.x + b.w for b in items)
    y2 = max(b.y + b.h for b in items)
    return Box(x=round(x1, 4), y=round(y1, 4), w=round(max(0.005, x2 - x1), 4), h=round(max(0.005, y2 - y1), 4))


def occupancy(boxes: Iterable[Box], within: Box | None = None, cells: int = 64) -> float:
    """Fraction of ``within`` (default: the whole frame) covered by the union of ``boxes``.

    Rasterised on a ``cells`` x ``cells`` grid, so overlapping boxes are not double counted.
    """
    region = within or Box(x=0, y=0, w=1, h=1)
    items = list(boxes)
    if not items:
        return 0.0
    covered = 0
    for gy in range(cells):
        py = region.y + (gy + 0.5) * region.h / cells
        for gx in range(cells):
            px = region.x + (gx + 0.5) * region.w / cells
            if any(b.x <= px <= b.x + b.w and b.y <= py <= b.y + b.h for b in items):
                covered += 1
    return covered / (cells * cells)


@dataclass(frozen=True, slots=True)
class Proximity:
    gap: float
    relative_gap: float
    overlap_iou: float
    touching: bool
    near: bool


def proximity(a: Box, b: Box, *, touch_gap: float, near_relative_gap: float) -> Proximity:
    gap = edge_gap(a, b)
    ref = reference_size(a, b) or 1e-6
    rel = gap / ref
    ov = iou(a, b)
    touching = ov > 0 or gap <= touch_gap
    return Proximity(gap=gap, relative_gap=rel, overlap_iou=ov, touching=touching, near=touching or rel <= near_relative_gap)


def direction(a: Box, b: Box) -> str:
    """Where ``a`` is relative to ``b`` (dominant axis of the centre offset)."""
    (ax, ay), (bx, by) = center(a), center(b)
    dx, dy = ax - bx, ay - by
    if abs(dx) >= abs(dy):
        return "left of" if dx < 0 else "right of"
    return "above" if dy < 0 else "below"
