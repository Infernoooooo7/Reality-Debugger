"""Multi-object tracking metrics.

* HOTA, DetA, AssA, LocA, DetRe/DetPr, AssRe/AssPr (Luiten et al., IJCV 2021),
  implemented after the algorithm of the official TrackEval code (hota.py):
  global alignment scores between ground-truth and tracker ids, per-frame
  Hungarian matching on alignment x IoU, averaged over IoU thresholds
  alpha = 0.05..0.95.
* CLEAR MOT (MOTA, MOTP, ID switches, fragmentations, mostly tracked / lost)
  and IDF1 (Ristani et al., 2016) via py-motmetrics (MIT licence).

Input per sequence: a list of frames; each frame has ground truth
``[(gt_id, box_xyxy)]``, ignore regions ``[box_xyxy]`` and tracker output
``[(track_id, box_xyxy)]``. Tracker boxes that lie mostly inside an ignore
region (intersection over tracker box >= 0.5) and match no ground truth are
removed before scoring, as benchmark evaluation kits do for DontCare areas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.evaluation.detection import ioa_matrix, iou_matrix

ALPHAS = np.arange(0.05, 0.96, 0.05)


@dataclass(slots=True)
class Frame:
    gt: list[tuple[int, tuple[float, float, float, float]]]
    tracks: list[tuple[int, tuple[float, float, float, float]]]
    ignore: list[tuple[float, float, float, float]]


def _arr(boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def preprocess(frames: list[Frame]) -> list[Frame]:
    """Drop tracker boxes on ignore regions that do not match any ground truth (IoU >= 0.5)."""
    out = []
    for f in frames:
        if not f.ignore or not f.tracks:
            out.append(f)
            continue
        tb = _arr([b for _, b in f.tracks])
        gb = _arr([b for _, b in f.gt])
        matched_gt = (iou_matrix(tb, gb) >= 0.5).any(axis=1) if len(gb) else np.zeros(len(tb), bool)
        on_ignore = (ioa_matrix(tb, _arr(f.ignore)) >= 0.5).any(axis=1)
        keep = [t for t, m, ig in zip(f.tracks, matched_gt, on_ignore, strict=True) if m or not ig]
        out.append(Frame(f.gt, keep, f.ignore))
    return out


def hota(frames: list[Frame]) -> dict[str, Any]:
    from scipy.optimize import linear_sum_assignment

    gt_ids = sorted({g for f in frames for g, _ in f.gt})
    tr_ids = sorted({t for f in frames for t, _ in f.tracks})
    gmap = {g: i for i, g in enumerate(gt_ids)}
    tmap = {t: i for i, t in enumerate(tr_ids)}
    ng, nt = len(gt_ids), len(tr_ids)
    potential = np.zeros((ng, nt))
    gt_count = np.zeros((ng, 1))
    tr_count = np.zeros((1, nt))
    sims = []
    for f in frames:
        gi = np.array([gmap[g] for g, _ in f.gt], dtype=np.int64)
        ti = np.array([tmap[t] for t, _ in f.tracks], dtype=np.int64)
        sim = iou_matrix(_arr([b for _, b in f.gt]), _arr([b for _, b in f.tracks]))
        sims.append((gi, ti, sim))
        if len(gi) and len(ti):
            denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            sim_iou = np.zeros_like(sim)
            mask = denom > np.finfo(float).eps
            sim_iou[mask] = sim[mask] / denom[mask]
            potential[gi[:, None], ti[None, :]] += sim_iou
        gt_count[gi] += 1
        tr_count[0, ti] += 1
    global_alignment = potential / np.maximum(gt_count + tr_count - potential, np.finfo(float).eps)
    n_alpha = len(ALPHAS)
    tp, fn, fp, loc = np.zeros(n_alpha), np.zeros(n_alpha), np.zeros(n_alpha), np.zeros(n_alpha)
    matches = [np.zeros((ng, nt)) for _ in range(n_alpha)]
    for gi, ti, sim in sims:
        if len(gi) == 0:
            fp += len(ti)
            continue
        if len(ti) == 0:
            fn += len(gi)
            continue
        score = global_alignment[gi[:, None], ti[None, :]] * sim
        rows, cols = linear_sum_assignment(-score)
        for a, alpha in enumerate(ALPHAS):
            ok = sim[rows, cols] >= alpha - np.finfo(float).eps
            r, c = rows[ok], cols[ok]
            tp[a] += len(r)
            fn[a] += len(gi) - len(r)
            fp[a] += len(ti) - len(r)
            if len(r):
                loc[a] += sim[r, c].sum()
                matches[a][gi[r], ti[c]] += 1
    ass_a, ass_re, ass_pr = np.zeros(n_alpha), np.zeros(n_alpha), np.zeros(n_alpha)
    for a in range(n_alpha):
        m = matches[a]
        ass_a[a] = np.sum(m * (m / np.maximum(1, gt_count + tr_count - m))) / max(1.0, tp[a])
        ass_re[a] = np.sum(m * (m / np.maximum(1, gt_count))) / max(1.0, tp[a])
        ass_pr[a] = np.sum(m * (m / np.maximum(1, tr_count))) / max(1.0, tp[a])
    loc_a = np.maximum(1e-10, loc) / np.maximum(1e-10, tp)
    det_re = tp / np.maximum(1, tp + fn)
    det_pr = tp / np.maximum(1, tp + fp)
    det_a = tp / np.maximum(1, tp + fn + fp)
    hota_alpha = np.sqrt(det_a * ass_a)
    mean = lambda v: round(float(np.mean(v)) * 100, 2)  # noqa: E731
    return {"HOTA": mean(hota_alpha), "DetA": mean(det_a), "AssA": mean(ass_a), "LocA": mean(loc_a),
            "DetRe": mean(det_re), "DetPr": mean(det_pr), "AssRe": mean(ass_re), "AssPr": mean(ass_pr),
            "HOTA(0)": round(float(hota_alpha[0]) * 100, 2), "gt_ids": ng, "tracker_ids": nt,
            "gt_dets": int(gt_count.sum()), "tracker_dets": int(tr_count.sum())}


def clear_and_identity(frames: list[Frame]) -> dict[str, Any]:
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    for f in frames:
        gb, tb = _arr([b for _, b in f.gt]), _arr([b for _, b in f.tracks])
        # motmetrics' convention: distance = 1 - IoU, NaN when IoU < 0.5 (no match possible).
        # (Computed here: motmetrics' own iou_matrix uses np.asfarray, removed in NumPy 2.)
        dist = 1.0 - iou_matrix(gb, tb)
        dist[dist > 0.5] = np.nan
        acc.update([g for g, _ in f.gt], [t for t, _ in f.tracks], dist)
    mh = mm.metrics.create()
    names = ["mota", "motp", "idf1", "idp", "idr", "num_switches", "num_fragmentations", "mostly_tracked", "mostly_lost",
             "num_false_positives", "num_misses", "num_objects", "num_unique_objects"]
    summary = mh.compute(acc, metrics=names, name="seq")
    row = summary.loc["seq"]
    out: dict[str, Any] = {}
    for n in names:
        v = row[n]
        if n in ("mota", "idf1", "idp", "idr"):
            out[n.upper()] = round(float(v) * 100, 2)
        elif n == "motp":
            out["MOTP_iou"] = round((1 - float(v)) * 100, 2) if np.isfinite(v) else None  # motmetrics reports 1 - IoU
        else:
            out[n] = int(v)
    return out


def evaluate(frames: list[Frame]) -> dict[str, Any]:
    frames = preprocess(frames)
    return {**hota(frames), **clear_and_identity(frames)}
