"""Workspace reference comparison: what measurably changed between a reference photo of a workspace and a
current photo of the same workspace.

This is separate from generic detection and makes no judgement about what a good workspace looks like: the
reference is the user's own photo. Thresholds are in config/inference.json "workspace".

1. Registration (current -> reference): ORB features on both photos reduced to at most ``max_side`` px,
   Lowe's ratio test, RANSAC homography. It is accepted only with enough consistent matches, a plausible
   transform and enough overlap; otherwise positions and pixels are not compared (counts per category only).
2. Objects: the server detector's boxes from both photos are put in the reference frame and paired per
   category (greedy on IoU, then on centre distance): unchanged, moved, added, removed. Detections below the
   evidence threshold may pair with an object, but never count as added or removed on their own: they are
   listed as uncertain.
3. Changed regions: the registered current photo is compared with the reference after matching mean and
   contrast inside the overlap; connected regions of large luminance difference are reported, with the object
   changes that explain them, or none (a change the detector did not recognise, or lighting).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

Box = Sequence[float]  # x1, y1, x2, y2 in pixels


@dataclass(slots=True)
class WorkspaceConfig:
    max_side: int = 1024
    features: int = 3000
    ratio: float = 0.75
    reproj_px: float = 4.0
    min_inliers: int = 30
    min_inlier_ratio: float = 0.25
    min_overlap: float = 0.5
    max_area_change: float = 3.0
    unchanged_iou: float = 0.5
    max_move: float = 0.35
    min_evidence: float = 0.5
    diff_threshold: float = 0.15
    blur_sigma: float = 2.0
    min_region: float = 0.002
    min_object_change: float = 0.15

    @classmethod
    def from_config(cls, get: Any) -> WorkspaceConfig:
        p = "inference.workspace."
        return cls(
            max_side=int(get(p + "maxSide")), features=int(get(p + "orbFeatures")), ratio=float(get(p + "ratioTest")),
            reproj_px=float(get(p + "ransacReprojectionPx")), min_inliers=int(get(p + "minInliers")),
            min_inlier_ratio=float(get(p + "minInlierRatio")), min_overlap=float(get(p + "minOverlap")),
            max_area_change=float(get(p + "maxAreaChange")), unchanged_iou=float(get(p + "unchangedIou")),
            max_move=float(get(p + "maxMove")), min_evidence=float(get("diagnostics.confidence.minEvidenceDeep")),
            diff_threshold=float(get(p + "diffThreshold")), blur_sigma=float(get(p + "blurSigma")),
            min_region=float(get(p + "minRegionArea")), min_object_change=float(get(p + "minObjectChange")),
        )


@dataclass(slots=True)
class Registration:
    ok: bool
    homography: np.ndarray | None  # current (analysis px) -> reference (analysis px)
    matches: int
    inliers: int
    inlier_ratio: float
    overlap: float | None
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "method": "ORB features + RANSAC homography", "matches": self.matches, "inliers": self.inliers,
                "inlier_ratio": round(self.inlier_ratio, 3), "overlap": None if self.overlap is None else round(self.overlap, 3),
                "reason": self.reason}


def small_gray(image_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Luminance reduced so the long side is at most ``max_side`` (never enlarged), and the scale used."""
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if scale < 1.0:
        gray = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return gray, scale


def _frame_mask(src_shape: tuple[int, int], h: np.ndarray, dst_shape: tuple[int, int]) -> np.ndarray:
    ones = np.full(src_shape, 255, np.uint8)
    return cv2.warpPerspective(ones, h, (dst_shape[1], dst_shape[0]), flags=cv2.INTER_NEAREST) > 0


def register(ref_gray: np.ndarray, cur_gray: np.ndarray, cfg: WorkspaceConfig) -> Registration:
    def fail(reason: str, matches: int = 0, inliers: int = 0, ratio: float = 0.0) -> Registration:
        return Registration(False, None, matches, inliers, ratio, None, reason)

    orb = cv2.ORB_create(nfeatures=cfg.features)
    kc, dc = orb.detectAndCompute(cur_gray, None)
    kr, dr = orb.detectAndCompute(ref_gray, None)
    if dc is None or dr is None or len(kc) < 8 or len(kr) < 8:
        return fail("too few distinctive features to align the photos (plain or blurred surfaces)")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dc, dr, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < cfg.ratio * p[1].distance]
    if len(good) < cfg.min_inliers:
        return fail(f"only {len(good)} distinctive feature matches (need {cfg.min_inliers}): the photos may not show the same place",
                    matches=len(good))
    src = np.float32([kc[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, cfg.reproj_px)
    inliers = int(mask.sum()) if mask is not None else 0
    ratio = inliers / len(good)
    if h is None or inliers < cfg.min_inliers or ratio < cfg.min_inlier_ratio:
        return fail(f"{inliers} of {len(good)} matches agree on one alignment (need {cfg.min_inliers} and "
                    f"{cfg.min_inlier_ratio:.0%}): the viewpoint or the scene changed too much", len(good), inliers, ratio)
    ch, cw = cur_gray.shape
    corners = cv2.perspectiveTransform(np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]]).reshape(-1, 1, 2), h).reshape(-1, 2)
    area = abs(cv2.contourArea(corners))
    scale_area = area / (cw * ch * (ref_gray.size / max(1, cur_gray.size)))
    if not cv2.isContourConvex(corners.astype(np.float32)) or not (1 / cfg.max_area_change <= scale_area <= cfg.max_area_change):
        return fail("the estimated alignment is implausible (a folded or strongly rescaled view)", len(good), inliers, ratio)
    overlap = float(_frame_mask(cur_gray.shape, h, ref_gray.shape).mean())
    if overlap < cfg.min_overlap:
        return fail(f"the photos overlap on only {overlap:.0%} of the reference (need {cfg.min_overlap:.0%})", len(good), inliers, ratio)
    return Registration(True, h, len(good), inliers, ratio, overlap, None)


def map_box(box: Box, h: np.ndarray, src_scale: float, dst_scale: float, dst_size: tuple[int, int]) -> list[float]:
    """A box from the source photo (pixels) into the destination photo (pixels) through ``h`` (analysis px)."""
    x1, y1, x2, y2 = box
    pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]) * src_scale
    out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), h).reshape(-1, 2) / dst_scale
    w, hh = dst_size
    return [round(float(np.clip(out[:, 0].min(), 0, w)), 1), round(float(np.clip(out[:, 1].min(), 0, hh)), 1),
            round(float(np.clip(out[:, 0].max(), 0, w)), 1), round(float(np.clip(out[:, 1].max(), 0, hh)), 1)]


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _centre(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def pair_objects(reference: list[dict[str, Any]], current: list[dict[str, Any]], diagonal: float, cfg: WorkspaceConfig) -> dict[str, list[dict[str, Any]]]:
    """Pair objects of the same category. ``current`` boxes must already be in reference coordinates
    (key ``box``; the current photo's own box may be given as ``own_box``)."""
    out: dict[str, list[dict[str, Any]]] = {"unchanged": [], "moved": [], "added": [], "removed": [], "uncertain": []}
    used_r: set[int] = set()
    used_c: set[int] = set()
    candidates = []
    for i, r in enumerate(reference):
        for j, c in enumerate(current):
            if r["label"] != c["label"]:
                continue
            (rx, ry), (cx, cy) = _centre(r["box"]), _centre(c["box"])
            candidates.append((iou(r["box"], c["box"]), -math.hypot(rx - cx, ry - cy) / diagonal, i, j))
    # First objects that stayed in place (highest overlap), then the nearest same-category object as a move.
    for overlap, neg_dist, i, j in sorted(candidates, reverse=True):
        if i in used_r or j in used_c or overlap < cfg.unchanged_iou:
            continue
        used_r.add(i)
        used_c.add(j)
        out["unchanged"].append(_item(reference[i], current[j]))
    for overlap, neg_dist, i, j in sorted(candidates, key=lambda c: (-c[1], -c[0])):  # nearest first, then larger overlap
        if i in used_r or j in used_c or -neg_dist > cfg.max_move:
            continue
        used_r.add(i)
        used_c.add(j)
        out["moved"].append({**_item(reference[i], current[j]), "moved_by": round(-neg_dist, 3)})
    for i, r in enumerate(reference):
        if i not in used_r:
            key = "removed" if r["confidence"] >= cfg.min_evidence else "uncertain"
            out[key].append({**_item(r, None), "change": "removed", **({} if key == "removed" else {"why": f"detection confidence below {cfg.min_evidence:.1f}"})})
    for j, c in enumerate(current):
        if j not in used_c:
            key = "added" if c["confidence"] >= cfg.min_evidence else "uncertain"
            out[key].append({**_item(None, c), "change": "added", **({} if key == "added" else {"why": f"detection confidence below {cfg.min_evidence:.1f}"})})
    return out


def _item(r: dict[str, Any] | None, c: dict[str, Any] | None) -> dict[str, Any]:
    base = c or r
    assert base is not None
    return {
        "label": base["label"],
        "reference_box": r["box"] if r else None,
        "current_box": (c.get("own_box") or c["box"]) if c else None,
        "current_box_in_reference": c["box"] if c else None,
        "confidence": round(max(r["confidence"] if r else 0.0, c["confidence"] if c else 0.0), 3),
    }


def changed_regions(ref_gray: np.ndarray, cur_gray: np.ndarray, h: np.ndarray,
                    cfg: WorkspaceConfig) -> tuple[float, list[dict[str, Any]], np.ndarray | None, np.ndarray | None]:
    """Share of the overlap that changed, the changed regions (reference analysis px), and the change and
    overlap masks (reference analysis px; None without overlap)."""
    rh, rw = ref_gray.shape
    warped = cv2.warpPerspective(cur_gray, h, (rw, rh), flags=cv2.INTER_LINEAR)
    valid = _frame_mask(cur_gray.shape, h, ref_gray.shape).astype(np.uint8)
    valid = cv2.erode(valid, np.ones((7, 7), np.uint8)) > 0  # resampled borders are not changes
    if valid.sum() < 100:
        return 0.0, [], None, None
    ref = cv2.GaussianBlur(ref_gray.astype(np.float32) / 255.0, (0, 0), cfg.blur_sigma)
    cur = cv2.GaussianBlur(warped.astype(np.float32) / 255.0, (0, 0), cfg.blur_sigma)
    r, c = ref[valid], cur[valid]
    cur = (cur - c.mean()) / (c.std() + 1e-6) * r.std() + r.mean()  # global exposure / contrast change
    mask = ((np.abs(ref - cur) >= cfg.diff_threshold) & valid).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total = float(valid.sum())
    regions = []
    for k in range(1, n):
        x, y, w, hh, area = (int(v) for v in stats[k])
        if area / total >= cfg.min_region:
            regions.append({"box": [x, y, x + w, y + hh], "area": round(area / total, 4)})
    regions.sort(key=lambda g: -g["area"])
    return round(float(mask.sum()) / total, 4), regions, mask.astype(bool), valid


def box_change(mask: np.ndarray | None, valid: np.ndarray | None, box: Box, scale: float) -> float | None:
    """Share of the overlapping pixels inside ``box`` (reference px) that changed; None outside the overlap."""
    if mask is None or valid is None:
        return None
    h, w = mask.shape
    x1, y1 = max(0, int(box[0] * scale)), max(0, int(box[1] * scale))
    x2, y2 = min(w, int(math.ceil(box[2] * scale))), min(h, int(math.ceil(box[3] * scale)))
    if x2 <= x1 or y2 <= y1:
        return None
    inside = valid[y1:y2, x1:x2]
    if inside.sum() < 0.5 * inside.size:
        return None
    return float(mask[y1:y2, x1:x2][inside].mean())


def corroborate(objects: dict[str, list[dict[str, Any]]], mask: np.ndarray | None, valid: np.ndarray | None, scale: float,
                cfg: WorkspaceConfig) -> None:
    """An object change counts only where the pixels changed too. A box present in one photo only, over pixels
    that did not change, is a detector disagreement (flicker, a different crop of the same thing); a box outside
    the area both photos show cannot be checked at all (the object may simply be out of view). Both move to
    "uncertain" with the reason."""
    for key in ("added", "removed", "moved"):
        kept = []
        for o in objects[key]:
            places = [b for b in (o["reference_box"] if key != "added" else None,
                                  o["current_box_in_reference"] if key != "removed" else None) if b]
            shares = [box_change(mask, valid, b, scale) for b in places]
            measured = [v for v in shares if v is not None]
            o["pixel_change"] = round(min(measured), 3) if measured else None
            if not measured:
                objects["uncertain"].append({**o, "change": key, "why": "outside the area both photos show: it may simply be out of view"})
            elif min(measured) < cfg.min_object_change:
                why = ("detected at two places, but the pixels there did not change" if key == "moved"
                       else "detected in only one photo, but the pixels there did not change")
                objects["uncertain"].append({**o, "change": key, "why": why})
            else:
                kept.append(o)
        objects[key] = kept
    for o in objects["uncertain"]:
        if "pixel_change" not in o:
            box = o["current_box_in_reference"] if o.get("change") == "added" else o["reference_box"]
            v = box_change(mask, valid, box, scale) if box else None
            o["pixel_change"] = None if v is None else round(v, 3)


def compare_workspace(reference_bgr: np.ndarray, current_bgr: np.ndarray, reference_objects: list[dict[str, Any]],
                      current_objects: list[dict[str, Any]], cfg: WorkspaceConfig) -> dict[str, Any]:
    """Objects are {label, confidence, box: [x1, y1, x2, y2] in their own photo's pixels}."""
    ref_gray, s_ref = small_gray(reference_bgr, cfg.max_side)
    cur_gray, s_cur = small_gray(current_bgr, cfg.max_side)
    reg = register(ref_gray, cur_gray, cfg)
    rh, rw = reference_bgr.shape[:2]
    ch, cw = current_bgr.shape[:2]
    confident = lambda objs: Counter(o["label"] for o in objs if o["confidence"] >= cfg.min_evidence)  # noqa: E731
    ref_counts, cur_counts = confident(reference_objects), confident(current_objects)
    counts = {label: {"reference": ref_counts[label], "current": cur_counts[label]}
              for label in sorted(set(ref_counts) | set(cur_counts)) if ref_counts[label] != cur_counts[label]}
    result: dict[str, Any] = {
        "registration": reg.to_dict(),
        "reference": {"width": rw, "height": rh, "objects": len(reference_objects)},
        "current": {"width": cw, "height": ch, "objects": len(current_objects)},
        "counts": counts,
        "objects": None,
        "changes": None,
    }
    if not reg.ok or reg.homography is None:
        result["summary"] = (f"The photos could not be aligned ({reg.reason}), so positions and pixels were not compared. "
                             + _count_text(counts))
        return result
    hmat = reg.homography
    # current pixels -> reference pixels at full resolution
    full = np.diag([1 / s_ref, 1 / s_ref, 1.0]) @ hmat @ np.diag([s_cur, s_cur, 1.0])
    result["registration"]["current_to_reference"] = [[float(f"{v:.10g}") for v in row] for row in full / full[2, 2]]
    mapped = [{**o, "own_box": list(o["box"]), "box": map_box(o["box"], hmat, s_cur, s_ref, (rw, rh))} for o in current_objects]
    objects = pair_objects(reference_objects, mapped, math.hypot(rw, rh), cfg)
    fraction, regions, change_mask, overlap_mask = changed_regions(ref_gray, cur_gray, hmat, cfg)
    corroborate(objects, change_mask, overlap_mask, s_ref, cfg)
    inv = np.linalg.inv(hmat)
    changed_objects = [(o, o["reference_box"]) for o in objects["removed"] + objects["moved"] if o["reference_box"]]
    changed_objects += [(o, o["current_box_in_reference"]) for o in objects["added"] + objects["moved"] if o["current_box_in_reference"]]
    for g in regions:
        x1, y1, x2, y2 = (v / s_ref for v in g["box"])
        g["box"] = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
        g["current_box"] = map_box(g["box"], inv, s_ref, s_cur, (cw, ch))
        area = max(1e-6, (x2 - x1) * (y2 - y1))
        explained = []
        for o, b in changed_objects:
            ix = max(0.0, min(x2, b[2]) - max(x1, b[0]))
            iy = max(0.0, min(y2, b[3]) - max(y1, b[1]))
            if ix * iy / area >= 0.2 and o["label"] not in explained:
                explained.append(o["label"])
        g["explained_by"] = explained
        g["possibly"] = []
        for o in objects["uncertain"]:
            b = o["current_box_in_reference"] if o.get("change") == "added" else o["reference_box"]
            if not b or (o.get("pixel_change") or 0) < cfg.min_object_change:
                continue
            ix = max(0.0, min(x2, b[2]) - max(x1, b[0]))
            iy = max(0.0, min(y2, b[3]) - max(y1, b[1]))
            if ix * iy / area >= 0.2:
                g["possibly"].append(f"{o['label']} ({o['confidence'] * 100:.0f}%)")
    result["objects"] = objects
    result["changes"] = {"changed_fraction": fraction, "regions": regions[:12],
                         "unexplained_regions": sum(1 for g in regions if not g["explained_by"] and not g["possibly"])}
    result["summary"] = _summary(objects, fraction, regions)
    return result


def _count_text(counts: dict[str, dict[str, int]]) -> str:
    if not counts:
        return "Confident detections give the same count per category in both photos."
    parts = []
    for label, c in counts.items():
        d = c["current"] - c["reference"]
        parts.append(f"{abs(d)} {'more' if d > 0 else 'fewer'} {label}")
    return "Category counts (confident detections only): " + ", ".join(parts) + "."


def _summary(objects: dict[str, list[dict[str, Any]]], fraction: float, regions: list[dict[str, Any]]) -> str:
    parts = []
    for key in ("added", "removed", "moved"):
        items = objects[key]
        if items:
            names = ", ".join(sorted({o["label"] for o in items}))
            parts.append(f"{len(items)} {key} ({names})")
    head = "Compared with the reference: " + ("; ".join(parts) if parts else "no recognised object was added, removed or moved")
    unexplained = sum(1 for g in regions if not g["explained_by"] and not g.get("possibly"))
    possible = sorted({p.split(" (")[0] for g in regions if not g["explained_by"] for p in g.get("possibly", [])})
    tail = f". {fraction * 100:.1f}% of the overlapping area changed in brightness pattern"
    if possible:
        tail += f"; changed region(s) may be a weakly detected {', '.join(possible)}"
    if unexplained:
        tail += f"; {unexplained} changed region(s) are not explained by a recognised object (an unsupported object, or lighting)"
    if objects["uncertain"]:
        tail += f"; {len(objects['uncertain'])} unconfirmed difference(s) are listed as uncertain"
    return head + tail + "."
