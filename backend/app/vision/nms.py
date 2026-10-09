"""Box suppression and merging (numpy).

* ``nms`` / ``batched_nms``: greedy non-maximum suppression, class-agnostic or
  per class.
* ``greedy_nmm``: greedy non-maximum *merging* as used by SAHI for sliced
  inference: overlapping boxes (by IoU or by IoS, intersection over the
  smaller box) are merged into their union instead of being dropped, which
  re-assembles objects cut at tile borders.
"""

from __future__ import annotations

import numpy as np


def box_area(b: np.ndarray) -> np.ndarray:
    return np.maximum(0, b[..., 2] - b[..., 0]) * np.maximum(0, b[..., 3] - b[..., 1])


def overlap(a: np.ndarray, b: np.ndarray, metric: str = "iou") -> np.ndarray:
    """Pairwise IoU or IoS between boxes a (N,4) and b (M,4) -> (N,M)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a, area_b = box_area(a)[:, None], box_area(b)[None, :]
    if metric == "ios":
        denom = np.minimum(area_a, area_b)
    else:
        denom = area_a + area_b - inter
    return inter / np.maximum(denom, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Indices kept, highest score first."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = overlap(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_threshold: float, class_agnostic: bool) -> np.ndarray:
    if class_agnostic or boxes.size == 0:
        return nms(boxes, scores, iou_threshold)
    # Offset boxes per class so different classes never overlap (standard trick).
    offset = classes.astype(np.float64)[:, None] * (float(boxes.max()) + 1.0)
    keep = nms(boxes + offset, scores, iou_threshold)
    return keep


def greedy_nmm(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    *,
    metric: str = "ios",
    threshold: float = 0.5,
    class_agnostic: bool = False,
    return_index: bool = False,
) -> tuple[np.ndarray, ...]:
    """Greedy non-maximum merging. Returns (boxes, scores, classes) after merging
    (plus, with ``return_index``, the index of each group's leading box).

    Highest-scoring box first; the remaining boxes of the same class (or any
    class if class_agnostic) whose overlap with it is >= threshold are taken
    out, and each is merged into the growing union box if it still overlaps
    that union by >= threshold (SAHI's GreedyNMMPostprocess: union box, max
    score, class of the higher score). Matched boxes that no longer overlap
    the union are dropped, as in SAHI.
    """
    if boxes.size == 0:
        empty = (boxes.reshape(0, 4), scores, classes)
        return (*empty, np.zeros(0, np.int64)) if return_index else empty
    order = list(np.argsort(-scores, kind="stable"))
    out_boxes, out_scores, out_classes, leaders = [], [], [], []
    alive = np.ones(len(boxes), dtype=bool)
    for i in order:
        if not alive[i]:
            continue
        alive[i] = False
        merged = boxes[i].astype(np.float64).copy()
        candidates = np.flatnonzero(alive)
        if not class_agnostic:
            candidates = candidates[classes[candidates] == classes[i]]
        if candidates.size:
            ov = overlap(boxes[i : i + 1], boxes[candidates], metric)[0]
            matched = candidates[ov >= threshold]
            for j in matched[np.argsort(-scores[matched], kind="stable")]:
                alive[j] = False
                if overlap(merged[None], boxes[j : j + 1], metric)[0, 0] >= threshold:
                    merged[:2] = np.minimum(merged[:2], boxes[j, :2])
                    merged[2:] = np.maximum(merged[2:], boxes[j, 2:])
        out_boxes.append(merged)
        out_scores.append(scores[i])
        out_classes.append(classes[i])
        leaders.append(i)
    result = (np.asarray(out_boxes), np.asarray(out_scores), np.asarray(out_classes))
    return (*result, np.asarray(leaders, np.int64)) if return_index else result
