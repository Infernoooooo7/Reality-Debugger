"""Anomaly detection and segmentation metrics, and the VisA benchmark.

Metrics (as in the MVTec AD / VisA literature):
* image AUROC and image average precision of the image scores;
* pixel AUROC over all test pixels;
* AUPRO@0.3: area under the per-region-overlap curve up to a pixel false
  positive rate of 0.3, normalised to 0..1 - the algorithm of the official
  MVTec AD evaluation code (generic_util.compute_pro / trapezoid): connected
  components of each ground-truth mask (8-connectivity), PRO averaged over
  regions, FPR over all defect-free pixels, ties resolved by score.
* the operating point a deployment would actually use: the threshold set on
  held-out *normal* training images (no anomalous data), and the resulting
  defect recall and false-alarm rate on the test images.

Protocol on VisA (official 1cls split): for each category a seeded random
permutation of the training (normal) images gives 20 held-out images for
calibration and, disjoint from them, the k reference images for each
configuration. Maps are evaluated at 224 x 224 against masks resized with
nearest-neighbour interpolation.
"""

from __future__ import annotations

import time
from bisect import bisect
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

EVAL_SIZE = 224
HOLDOUT = 20


def average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given their average rank (as scipy.stats.rankdata)."""
    x = np.asarray(x).ravel()
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    bounds = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1], True])
    starts, ends = bounds[:-1], bounds[1:]
    ranks = np.empty(len(x), np.float64)
    ranks[order] = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    return ranks


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties count one half)."""
    scores, labels = np.asarray(scores, np.float64).ravel(), np.asarray(labels, bool).ravel()
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (area under the precision-recall step curve), ties grouped."""
    scores, labels = np.asarray(scores, np.float64).ravel(), np.asarray(labels, bool).ravel()
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    tp, fp = np.cumsum(y), np.cumsum(~y)
    last = np.r_[np.diff(s) != 0, True]  # evaluate only at the end of each tie group
    tp, fp = tp[last], fp[last]
    precision, recall = tp / (tp + fp), tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _trapezoid(x: np.ndarray, y: np.ndarray, x_max: float) -> float:
    correction = 0.0
    if x_max not in x:
        ins = bisect(x.tolist(), x_max)
        y_interp = y[ins - 1] + (y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1])
        correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])
    keep = x <= x_max
    x, y = x[keep], y[keep]
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction)


def aupro(maps: np.ndarray, masks: np.ndarray, fpr_limit: float = 0.3) -> float:
    """Normalised area under the PRO curve up to ``fpr_limit`` (maps, masks: N x H x W)."""
    fp_change = np.zeros(maps.shape, np.float64)
    pro_change = np.zeros(maps.shape, np.float64)
    n_ok, n_regions = 0, 0
    for i, mask in enumerate(masks):
        n, labelled = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        ok = labelled == 0
        n_ok += int(ok.sum())
        fp_change[i][ok] = 1.0
        for k in range(1, n):
            region = labelled == k
            pro_change[i][region] = 1.0 / region.sum()
        n_regions += n - 1
    if n_regions == 0 or n_ok == 0:
        return float("nan")
    order = np.argsort(-maps.ravel(), kind="mergesort")
    s = maps.ravel()[order]
    fprs = np.cumsum(fp_change.ravel()[order]) / n_ok
    pros = np.cumsum(pro_change.ravel()[order]) / n_regions
    keep = np.r_[np.diff(s) != 0, True]
    fprs = np.clip(np.r_[0.0, fprs[keep], 1.0], None, 1.0)
    pros = np.clip(np.r_[0.0, pros[keep], 1.0], None, 1.0)
    return _trapezoid(fprs, pros, fpr_limit) / fpr_limit


def load_mask(path: Any, size: int = EVAL_SIZE) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize((m > 0).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)


def evaluate_maps(scores: np.ndarray, anomalous: np.ndarray, maps: np.ndarray, masks: np.ndarray,
                  threshold: float | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "image_auroc": round(auroc(scores, anomalous) * 100, 2),
        "image_ap": round(average_precision(scores, anomalous) * 100, 2),
        "pixel_auroc": round(auroc(maps, masks) * 100, 2),
        "aupro_0.3": round(aupro(maps, masks) * 100, 2),
        "n_test": int(len(scores)), "n_anomalous": int(anomalous.sum()),
    }
    if threshold is not None:
        flagged = scores > threshold
        out["at_calibrated_threshold"] = {
            "threshold": round(float(threshold), 4),
            "defect_recall": round(float(flagged[anomalous].mean()), 4),
            "false_alarm_rate": round(float(flagged[~anomalous].mean()), 4),
        }
    return out


def run_visa(categories: list[str], configs: dict[str, dict[str, Any]], *, extractor_weights: Any, size: int = EVAL_SIZE,
             seed: int = 0, progress: Callable[[str], None] = print) -> dict[str, dict[str, Any]]:
    """Run every configuration on every category; returns {config: {category: metrics}}.

    configs: name -> {"method": "patchcore" | "reference-difference", "k": references,
                      "coreset_ratio": float (patchcore)}.
    """
    from app.datasets.adapters import VisaAdapter
    from app.vision.anomaly import PatchCore, PatchCoreConfig, PatchFeatureExtractor, ReferenceDifference, leave_one_out_scores

    visa = VisaAdapter()
    extractor = PatchFeatureExtractor(extractor_weights, size=size)
    results: dict[str, dict[str, Any]] = {name: {} for name in configs}
    # score_ms_per_image is the full per-image cost (PatchCore: feature extraction + nearest-neighbour search).
    timing: dict[str, dict[str, float]] = {name: {"fit_s": 0.0, "score_ms_per_image": 0.0} for name in configs}
    k_max = max(c["k"] for c in configs.values())
    for cat in categories:
        train, test = visa.items(cat, "train"), visa.items(cat, "test")
        order = np.random.default_rng(seed).permutation(len(train))
        holdout = [train[i] for i in order[:HOLDOUT]]
        pool = [train[i] for i in order[HOLDOUT : HOLDOUT + k_max]]
        read = lambda items: [cv2.imread(str(it["image"])) for it in items]  # noqa: E731
        t0 = time.perf_counter()
        test_imgs = read(test)
        anomalous = np.array([it["anomalous"] for it in test])
        masks = np.stack([load_mask(it["mask"], size) if it["mask"] else np.zeros((size, size), bool) for it in test])
        t_feat = time.perf_counter()
        test_feats = extractor(test_imgs)
        feature_ms = 1000 * (time.perf_counter() - t_feat) / len(test)
        hold_imgs = read(holdout)
        hold_feats = extractor(hold_imgs)
        pool_imgs = read(pool)
        pool_feats = extractor(pool_imgs)
        progress(f"  {cat}: features for {len(test)} test / {len(pool)} reference / {len(holdout)} held-out images "
                 f"in {time.perf_counter() - t0:.0f}s")
        for name, cfg in configs.items():
            k = cfg["k"]
            t1 = time.perf_counter()
            loo_threshold = None
            if cfg["method"] == "patchcore":
                model = PatchCore(extractor, PatchCoreConfig(coreset_ratio=cfg.get("coreset_ratio", 1.0), seed=seed))
                model.fit_features(pool_feats[:k])
                threshold = model.calibrate_features(hold_feats)
                if cfg.get("loo_threshold"):  # the rule the API uses when no extra normal images exist
                    loo_threshold = max(leave_one_out_scores(pool_feats[:k])[0])
                fit_s = time.perf_counter() - t1
                t2 = time.perf_counter()
                ps = [model.patch_scores(f) for f in test_feats]
                scores = np.array([p.max() for p in ps])
                maps = np.stack([model.anomaly_map(p) for p in ps])
                extra = {"memory_bank": int(len(model.bank))}
                fit_s += feature_ms * k / 1000  # features of the k references
            else:
                model = ReferenceDifference()
                model.fit(pool_imgs[:k])
                threshold = model.calibrate(hold_imgs)
                fit_s = time.perf_counter() - t1
                t2 = time.perf_counter()
                scored = [model.map_at_working_size(im) for im in test_imgs]
                scores = np.array([s for s, _ in scored])
                maps = np.stack([cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA) for _, m in scored])
                extra = {}
            score_ms = 1000 * (time.perf_counter() - t2) / len(test) + (feature_ms if cfg["method"] == "patchcore" else 0.0)
            metrics = evaluate_maps(scores, anomalous, maps, masks, threshold)
            if loo_threshold is not None:
                flagged = scores > loo_threshold
                metrics["at_leave_one_out_threshold"] = {"threshold": round(float(loo_threshold), 4),
                                                         "defect_recall": round(float(flagged[anomalous].mean()), 4),
                                                         "false_alarm_rate": round(float(flagged[~anomalous].mean()), 4)}
            results[name][cat] = {**metrics, **extra, "references": k}
            timing[name]["fit_s"] += fit_s
            timing[name]["score_ms_per_image"] += score_ms / len(categories)
            progress(f"    {name:28} image AUROC {metrics['image_auroc']:6.2f}  pixel AUROC {metrics['pixel_auroc']:6.2f}  "
                     f"AUPRO {metrics['aupro_0.3']:6.2f}  recall@thr {metrics['at_calibrated_threshold']['defect_recall']:.2f} "
                     f"false alarms {metrics['at_calibrated_threshold']['false_alarm_rate']:.2f}")
    for name in configs:
        per_cat = results[name]
        keys = ("image_auroc", "image_ap", "pixel_auroc", "aupro_0.3")
        mean = {k: round(float(np.mean([m[k] for m in per_cat.values()])), 2) for k in keys}
        mean["defect_recall_at_threshold"] = round(float(np.mean([m["at_calibrated_threshold"]["defect_recall"] for m in per_cat.values()])), 4)
        mean["false_alarm_rate_at_threshold"] = round(float(np.mean([m["at_calibrated_threshold"]["false_alarm_rate"] for m in per_cat.values()])), 4)
        if all("at_leave_one_out_threshold" in m for m in per_cat.values()):
            loo = [m["at_leave_one_out_threshold"] for m in per_cat.values()]
            mean["defect_recall_at_loo_threshold"] = round(float(np.mean([x["defect_recall"] for x in loo])), 4)
            mean["false_alarm_rate_at_loo_threshold"] = round(float(np.mean([x["false_alarm_rate"] for x in loo])), 4)
        results[name] = {"mean_over_categories": mean, "per_category": per_cat,
                         "timing": {k: round(v, 2) for k, v in timing[name].items()}}
    return results
