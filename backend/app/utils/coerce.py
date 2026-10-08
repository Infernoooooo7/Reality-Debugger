"""Lenient coercion helpers used to sanitise model output and client context.

Vision models occasionally return percentages instead of fractions, unknown
enum spellings, pixel boxes or overly long prose. These helpers normalise what
can be normalised and return ``None``/defaults for what cannot, so a single bad
field never takes down a whole diagnosis.
"""

from __future__ import annotations

import math
import re
from typing import Any

from app.schemas.common import Box, Category, Severity

_WS = re.compile(r"\s+")

SEVERITY_ALIASES = {
    "CRIT": "CRITICAL",
    "BLOCKER": "CRITICAL",
    "SEVERE": "HIGH",
    "MAJOR": "HIGH",
    "WARN": "MEDIUM",
    "WARNING": "MEDIUM",
    "MODERATE": "MEDIUM",
    "MED": "MEDIUM",
    "MINOR": "LOW",
    "TRIVIAL": "LOW",
    "NOTE": "INFO",
    "INFORMATIONAL": "INFO",
    "NONE": "INFO",
}

CATEGORY_ALIASES = {
    "INEFFICIENCY": "EFFICIENCY",
    "PERFORMANCE": "EFFICIENCY",
    "ORGANISATION": "ORGANIZATION",
    "CLUTTER": "ORGANIZATION",
    "STORAGE": "ORGANIZATION",
    "INCONSISTENCY": "CONSISTENCY",
    "VISUAL": "CONSISTENCY",
    "VISUAL_INCONSISTENCY": "CONSISTENCY",
    "PROCESS": "WORKFLOW",
    "HAZARD": "SAFETY",
    "ERGONOMIC": "ERGONOMICS",
    "POSTURE": "ERGONOMICS",
    "AESTHETIC": "AESTHETICS",
    "STYLE": "AESTHETICS",
    "LAYOUT": "SPATIAL",
    "SPACE": "SPATIAL",
    "SPATIAL_RELATIONSHIP": "SPATIAL",
    "TECHNICAL_DEBT": "TECH_DEBT",
    "DEBT": "TECH_DEBT",
    "HUMOR": "ABSURD",
    "HUMOUR": "ABSURD",
    "FUNNY": "ABSURD",
    "MISC": "ABSURD",
}


def text(value: Any, max_len: int) -> str:
    """Collapse whitespace, strip and clip to ``max_len`` characters."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    cleaned = _WS.sub(" ", value).strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1].rstrip() + "…"
    return cleaned


def unit_float(value: Any, default: float = 0.5) -> float:
    """Coerce to 0..1. Values in (1, 100] are treated as percentages."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    if 1.0 < number <= 100.0:
        number /= 100.0
    return min(1.0, max(0.0, number))


def score(value: Any) -> int | None:
    """Coerce a 0..100 score. Fractions (<= 1) are scaled up."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    if 0 < number <= 1.0:
        number *= 100
    return int(round(min(100.0, max(0.0, number))))


def severity(value: Any) -> Severity:
    key = text(value, 24).upper().replace(" ", "_")
    key = SEVERITY_ALIASES.get(key, key)
    try:
        return Severity(key)
    except ValueError:
        return Severity.MEDIUM


def category(value: Any) -> Category:
    key = text(value, 40).upper().replace(" ", "_").replace("-", "_")
    key = CATEGORY_ALIASES.get(key, key)
    try:
        return Category(key)
    except ValueError:
        return Category.WORKFLOW


def box(value: Any) -> Box | None:
    """Accept {x,y,w,h} or [x,y,w,h] in fractions or percentages."""
    if value is None:
        return None
    try:
        if isinstance(value, dict):
            raw = [value.get("x"), value.get("y"), value.get("w"), value.get("h")]
        elif isinstance(value, (list, tuple)) and len(value) == 4:
            raw = list(value)
        else:
            return None
        nums = [float(v) for v in raw]  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if any(math.isnan(n) or math.isinf(n) for n in nums):
        return None
    if max(nums) > 1.5:
        if max(nums) <= 100.0:
            nums = [n / 100.0 for n in nums]
        else:
            return None  # pixel coordinates without a reference size
    x, y, w, h = nums
    x = min(1.0, max(0.0, x))
    y = min(1.0, max(0.0, y))
    w = min(1.0 - x, max(0.0, w))
    h = min(1.0 - y, max(0.0, h))
    if w < 0.005 or h < 0.005:
        return None
    return Box(x=round(x, 4), y=round(y, 4), w=round(w, 4), h=round(h, 4))


def identifier(value: Any, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.#:-]", "", text(value, 48))
    return cleaned or fallback


def token_set(value: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", value.lower()) if len(t) > 2}


def similarity(a: str, b: str) -> float:
    """Jaccard similarity of word sets (cheap fuzzy match for finding titles)."""
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def iou(a: Box | None, b: Box | None) -> float:
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0
