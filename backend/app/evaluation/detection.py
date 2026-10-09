"""Detection evaluation: COCO metrics plus an explanation of every miss.

Metrics
* COCO AP / AP50 / AP75 / AP_small / AP_medium / AP_large / AR (pycocotools,
  official protocol: up to 100 detections per image, crowd regions ignored);
  per-class AP and AP50.
* Precision / recall / F1 at a sweep of score thresholds (IoU 0.5, greedy
  matching by score, detections on crowd regions ignored - as COCO).
* Miss causes at the operating threshold. Every unmatched ground-truth object
  gets one cause, in this precedence:
    lost_to_neighbour  a same-class box >= threshold covers it (IoU >= .5) but was matched to an adjacent object
    suppressed_by_nms  a same-class pre-NMS candidate >= threshold covered it and NMS removed it (adapters that expose candidates)
    wrong_class        a box >= threshold covers it with another class
    poor_localization  a same-class box >= threshold overlaps it with 0.1 <= IoU < 0.5
    below_threshold    a same-class box covers it but scored below the threshold
    undetected         nothing near it at all
  split by object size (COCO area ranges). Inclusive flags are also given.
* False-positive types at the operating threshold (TIDE-style): duplicate,
  wrong_class, localization, both (wrong class and poor box), background.
* Calibration: expected calibration error of the confidence as an estimate of
  precision (IoU 0.5), with the reliability table.

The cause analysis follows the error taxonomy of Hoiem et al. (ECCV 2012)
and TIDE (Bolya et al., ECCV 2020) but counts errors at one operating
threshold instead of computing AP deltas - it answers "why was this object
not shown to the user?".
"""

from __future__ import annotations

import contextlib
import io
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SIZE_SMALL = 32**2
SIZE_MEDIUM = 96**2
THRESHOLDS = [round(float(t), 2) for t in np.arange(0.05, 0.951, 0.05)]
MISS_CAUSES = ("lost_to_neighbour", "suppressed_by_nms", "wrong_class", "poor_localization", "below_threshold", "undetected")
FP_TYPES = ("duplicate", "wrong_class", "localization", "both", "background")


def size_bucket(area: float) -> str:
    return "small" if area < SIZE_SMALL else "medium" if area < SIZE_MEDIUM else "large"


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xyxy boxes a (N,4) and b (M,4)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def ioa_matrix(dets: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Intersection over detection area (COCO's overlap measure for crowd regions)."""
    if dets.size == 0 or regions.size == 0:
        return np.zeros((len(dets), len(regions)))
    x1 = np.maximum(dets[:, None, 0], regions[None, :, 0])
    y1 = np.maximum(dets[:, None, 1], regions[None, :, 1])
    x2 = np.minimum(dets[:, None, 2], regions[None, :, 2])
    y2 = np.minimum(dets[:, None, 3], regions[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    return inter / np.maximum(area[:, None], 1e-9)


@dataclass(slots=True)
class ImageEval:
    """Ground truth and predictions of one image, with overlaps precomputed."""

    image_id: Any
    gt_boxes: np.ndarray  # (G,4) xyxy, non-crowd
    gt_cats: np.ndarray  # (G,)
    gt_areas: np.ndarray  # (G,)
    crowd_boxes: np.ndarray  # (C,4)
    crowd_cats: np.ndarray  # (C,)
    pred_boxes: np.ndarray  # (P,4) sorted by score desc
    pred_cats: np.ndarray
    pred_scores: np.ndarray
    cand_boxes: np.ndarray | None = None  # pre-NMS candidates (optional)
    cand_cats: np.ndarray | None = None
    cand_scores: np.ndarray | None = None
    iou: np.ndarray = field(init=False)  # (P,G)
    crowd_overlap: np.ndarray = field(init=False)  # (P,C)

    def __post_init__(self) -> None:
        self.iou = iou_matrix(self.pred_boxes, self.gt_boxes)
        self.crowd_overlap = ioa_matrix(self.pred_boxes, self.crowd_boxes)

    def match(self, threshold: float, iou_thr: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """Greedy matching of predictions >= threshold. Returns (pred_state, gt_match).

        pred_state: 1 = TP, 0 = FP, -1 = ignored (on a crowd region) or below threshold.
        gt_match: index of the matched prediction or -1.
        """
        P, G = len(self.pred_scores), len(self.gt_cats)
        state = np.full(P, -1, dtype=np.int8)
        gt_match = np.full(G, -1, dtype=np.int64)
        for p in range(P):
            if self.pred_scores[p] < threshold:
                continue  # sorted desc: the rest are below too, but keep -1
            best, best_iou = -1, iou_thr
            if G:
                same = (self.gt_cats == self.pred_cats[p]) & (gt_match < 0)
                cand = np.flatnonzero(same & (self.iou[p] >= best_iou))
                if cand.size:
                    best = int(cand[np.argmax(self.iou[p, cand])])
            if best >= 0:
                state[p] = 1
                gt_match[best] = p
            elif len(self.crowd_cats) and np.any((self.crowd_cats == self.pred_cats[p]) & (self.crowd_overlap[p] >= iou_thr)):
                state[p] = -1  # ignored, as COCO does for crowd regions
            else:
                state[p] = 0
        return state, gt_match


def build_image_evals(gt: dict[str, Any], results: Iterable[dict[str, Any]], candidates: dict[Any, dict[str, np.ndarray]] | None = None,
                      cat_ids: set[int] | None = None) -> list[ImageEval]:
    gts: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for a in gt["annotations"]:
        if cat_ids is None or a["category_id"] in cat_ids or a.get("iscrowd"):
            gts[a["image_id"]].append(a)
    preds: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        if cat_ids is None or r["category_id"] in cat_ids:
            preds[r["image_id"]].append(r)
    out = []
    for img in gt["images"]:
        iid = img["id"]
        g = [a for a in gts.get(iid, []) if not a.get("iscrowd")]
        c = [a for a in gts.get(iid, []) if a.get("iscrowd")]
        p = sorted(preds.get(iid, []), key=lambda r: -r["score"])
        xyxy = lambda b: [b[0], b[1], b[0] + b[2], b[1] + b[3]]  # noqa: E731
        cand = (candidates or {}).get(iid)
        out.append(ImageEval(
            image_id=iid,
            gt_boxes=np.array([xyxy(a["bbox"]) for a in g], dtype=np.float64).reshape(-1, 4),
            gt_cats=np.array([a["category_id"] for a in g], dtype=np.int64),
            gt_areas=np.array([a.get("area", a["bbox"][2] * a["bbox"][3]) for a in g], dtype=np.float64),
            crowd_boxes=np.array([xyxy(a["bbox"]) for a in c], dtype=np.float64).reshape(-1, 4),
            # A crowd/ignore region without a usable class ignores detections of every class (VisDrone "ignored regions").
            crowd_cats=np.array([a["category_id"] for a in c], dtype=np.int64),
            pred_boxes=np.array([xyxy(r["bbox"]) for r in p], dtype=np.float64).reshape(-1, 4),
            pred_cats=np.array([r["category_id"] for r in p], dtype=np.int64),
            pred_scores=np.array([r["score"] for r in p], dtype=np.float64),
            cand_boxes=None if cand is None else cand["boxes"],
            cand_cats=None if cand is None else cand["cats"],
            cand_scores=None if cand is None else cand["scores"],
        ))
    return out


def coco_metrics(gt: dict[str, Any], results: list[dict[str, Any]], *, iou_type: str = "bbox", cat_ids: list[int] | None = None,
                 category_names: dict[int, str] | None = None, max_dets: tuple[int, int, int] = (1, 10, 100)) -> dict[str, Any]:
    """COCO metrics. ``max_dets`` is (1, 10, 100) for COCO; VisDrone's protocol counts up to 500 (AP@500)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    names = ["AP", "AP50", "AP75", "APs", "APm", "APl", f"AR{max_dets[0]}", f"AR{max_dets[1]}", f"AR{max_dets[2]}", "ARs", "ARm", "ARl"]
    if not results:
        return {"summary": {n: 0.0 for n in names}, "per_class": {}}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = {k: v for k, v in gt.items() if k in ("images", "annotations", "categories")}
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(results)
        ev = COCOeval(coco_gt, coco_dt, iou_type)
        if cat_ids is not None:
            ev.params.catIds = cat_ids
        ev.params.maxDets = list(max_dets)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    summary = {n: round(float(v) * 100, 2) for n, v in zip(names, ev.stats, strict=True)}
    precision = ev.eval["precision"]  # [T, R, K, A, M]
    per_class = {}
    for k, cat_id in enumerate(ev.params.catIds):
        pr = precision[:, :, k, 0, -1]
        pr50 = precision[0, :, k, 0, -1]
        ap = float(np.mean(pr[pr > -1])) * 100 if np.any(pr > -1) else None
        ap50 = float(np.mean(pr50[pr50 > -1])) * 100 if np.any(pr50 > -1) else None
        per_class[(category_names or {}).get(cat_id, str(cat_id))] = {
            "AP": None if ap is None else round(ap, 2), "AP50": None if ap50 is None else round(ap50, 2)}
    return {"summary": summary, "per_class": per_class}


def operating_points(images: list[ImageEval], thresholds: list[float] = THRESHOLDS) -> list[dict[str, Any]]:
    total_gt = int(sum(len(im.gt_cats) for im in images))
    rows = []
    for t in thresholds:
        tp = fp = 0
        for im in images:
            state, _ = im.match(t)
            tp += int((state == 1).sum())
            fp += int((state == 0).sum())
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / total_gt if total_gt else 0.0
        rows.append({"threshold": t, "precision": round(p, 4), "recall": round(r, 4),
                     "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "tp": tp, "fp": fp, "fn": total_gt - tp})
    return rows


def _covers(boxes: np.ndarray, gt_box: np.ndarray, lo: float, hi: float = 1.01) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros(0, dtype=bool)
    ious = iou_matrix(boxes, gt_box[None])[:, 0]
    return (ious >= lo) & (ious < hi)


def miss_and_fp_analysis(images: list[ImageEval], threshold: float, category_names: dict[int, str]) -> dict[str, Any]:
    exclusive: dict[str, Counter[str]] = {s: Counter() for s in ("small", "medium", "large", "all")}
    inclusive: Counter[str] = Counter()
    per_class_miss: dict[str, Counter[str]] = defaultdict(Counter)
    per_class_gt: Counter[str] = Counter()
    fp_types: Counter[str] = Counter()
    worst_misses: list[tuple[int, Any]] = []
    worst_fps: list[tuple[int, Any]] = []
    confusions: Counter[tuple[str, str]] = Counter()
    candidates_available = any(im.cand_boxes is not None for im in images)
    for im in images:
        state, gt_match = im.match(threshold)
        above = im.pred_scores >= threshold
        misses = 0
        for g in range(len(im.gt_cats)):
            cat = int(im.gt_cats[g])
            cname = category_names.get(cat, str(cat))
            per_class_gt[cname] += 1
            if gt_match[g] >= 0:
                continue
            misses += 1
            box = im.gt_boxes[g]
            same = im.pred_cats == cat
            flags = {
                "lost_to_neighbour": bool(np.any(above & same & (im.iou[:, g] >= 0.5) & (state == 1))),
                "suppressed_by_nms": False,
                "wrong_class": bool(np.any(above & ~same & (im.iou[:, g] >= 0.5))),
                "poor_localization": bool(np.any(above & same & (im.iou[:, g] >= 0.1) & (im.iou[:, g] < 0.5))),
                "below_threshold": bool(np.any(~above & same & (im.iou[:, g] >= 0.5))),
            }
            if im.cand_boxes is not None and len(im.cand_boxes):
                cand_ok = (im.cand_cats == cat) & (im.cand_scores >= threshold)
                flags["suppressed_by_nms"] = bool(np.any(_covers(im.cand_boxes[cand_ok], box, 0.5))) and not flags["lost_to_neighbour"]
            cause = next((c for c in MISS_CAUSES[:-1] if flags[c]), "undetected")
            for c, on in flags.items():
                if on:
                    inclusive[c] += 1
            if cause == "undetected":
                inclusive["undetected"] += 1
            bucket = size_bucket(float(im.gt_areas[g]))
            exclusive[bucket][cause] += 1
            exclusive["all"][cause] += 1
            per_class_miss[cname][cause] += 1
            if cause == "wrong_class":
                hit = np.flatnonzero(above & ~same & (im.iou[:, g] >= 0.5))
                best = hit[np.argmax(im.iou[hit, g])]
                confusions[(cname, category_names.get(int(im.pred_cats[best]), "?"))] += 1
        fps = 0
        for p in np.flatnonzero(state == 0):
            fps += 1
            ious = im.iou[p] if len(im.gt_cats) else np.zeros(0)
            same = im.gt_cats == im.pred_cats[p]
            if np.any(same & (ious >= 0.5)):
                fp_types["duplicate"] += 1
            elif np.any(~same & (ious >= 0.5)):
                fp_types["wrong_class"] += 1
            elif np.any(same & (ious >= 0.1)):
                fp_types["localization"] += 1
            elif np.any(~same & (ious >= 0.1)):
                fp_types["both"] += 1
            else:
                fp_types["background"] += 1
        worst_misses.append((misses, im.image_id))
        worst_fps.append((fps, im.image_id))
    total_missed = sum(exclusive["all"].values())
    per_class = {}
    for cname, n in per_class_gt.items():
        missed = sum(per_class_miss[cname].values())
        per_class[cname] = {"gt": n, "recall": round(1 - missed / n, 4) if n else None, "misses": dict(per_class_miss[cname])}
    return {
        "threshold": threshold,
        "candidates_available": candidates_available,
        "missed_total": total_missed,
        "miss_causes": {k: dict(v) for k, v in exclusive.items()},
        "miss_causes_share": {c: round(exclusive["all"][c] / total_missed, 4) if total_missed else 0.0 for c in MISS_CAUSES},
        "miss_cause_flags_inclusive": dict(inclusive),
        "false_positive_types": dict(fp_types),
        "top_confusions": [{"ground_truth": a, "predicted": b, "count": n} for (a, b), n in confusions.most_common(10)],
        "per_class": per_class,
        "error_examples": {
            "most_missed_images": [i for n, i in sorted(worst_misses, key=lambda x: -x[0])[:5] if n],
            "most_false_positive_images": [i for n, i in sorted(worst_fps, key=lambda x: -x[0])[:5] if n],
        },
    }


def calibration(images: list[ImageEval], floor: float, bins: int = 10) -> dict[str, Any]:
    scores, correct = [], []
    for im in images:
        state, _ = im.match(floor)
        keep = state >= 0
        scores.extend(im.pred_scores[keep].tolist())
        correct.extend((state[keep] == 1).astype(int).tolist())
    if not scores:
        return {"ece": None, "bins": []}
    s, c = np.asarray(scores), np.asarray(correct)
    edges = np.linspace(floor, 1.0, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (s >= lo) & ((s < hi) if hi < 1.0 else (s <= hi))
        if not m.any():
            continue
        conf, prec = float(s[m].mean()), float(c[m].mean())
        ece += m.sum() / len(s) * abs(conf - prec)
        rows.append({"range": [round(float(lo), 3), round(float(hi), 3)], "count": int(m.sum()), "mean_confidence": round(conf, 4), "precision": round(prec, 4)})
    return {"ece": round(float(ece), 4), "detections": len(s), "iou": 0.5, "bins": rows,
            "note": "Confidence read as an estimate of precision at IoU 0.5 (detection calibration, cf. Kuppers et al. 2020)."}


def latency_summary(totals_ms: list[float], wall_ms: list[float]) -> dict[str, Any]:
    t, w = np.asarray(totals_ms), np.asarray(wall_ms)
    return {
        "model_ms": {"median": round(float(np.median(t)), 2), "p90": round(float(np.percentile(t, 90)), 2), "p99": round(float(np.percentile(t, 99)), 2), "mean": round(float(t.mean()), 2)},
        "wall_ms_incl_decode": {"median": round(float(np.median(w)), 2), "p90": round(float(np.percentile(w, 90)), 2)},
        "throughput_images_per_s": round(1000.0 / float(w.mean()), 2) if w.size else None,
        "images": int(t.size),
    }
