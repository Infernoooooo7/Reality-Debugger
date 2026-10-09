"""Run a detector over a dataset and write benchmark records.

One network pass per image serves several post-processing configurations
(e.g. "protocol" - the settings behind published numbers - and "deployed" -
what the app uses), so they are compared on identical inputs.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.evaluation import detection as det_eval
from app.evaluation import records
from app.vision.types import InferenceResult


@dataclass(slots=True)
class ConfigSpec:
    name: str
    config: Any  # adapter-specific config object (YoloxConfig, MediaPipeConfig, TilingConfig...)
    operating_threshold: float  # the threshold a user-facing view would apply
    calibration_floor: float
    keep_candidates: bool = False
    description: str = ""
    #: Derive this config from another config's predictions by filtering (MediaPipe floor/max-results only).
    derive_from: str | None = None
    derive_min_score: float = 0.0
    derive_max_results: int = 100


@dataclass(slots=True)
class DetectionTask:
    task_name: str
    dataset_id: str
    split: str
    gt: dict[str, Any]  # COCO-format ground truth (images carry absolute file_name paths or are resolved by image_path)
    image_path: Callable[[dict[str, Any]], str]
    eval_category_ids: list[int] | None = None
    notes: list[str] = field(default_factory=list)
    max_dets: tuple[int, int, int] = (1, 10, 100)


def _to_results(result: InferenceResult, image_id: Any, name_to_cat: dict[str, int]) -> tuple[list[dict[str, Any]], int]:
    out, unmapped = [], 0
    for d in result.detections:
        cat = name_to_cat.get(d.class_name)
        if cat is None:
            unmapped += 1
            continue
        x1, y1, x2, y2 = d.box
        row = {"image_id": image_id, "category_id": cat, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": round(float(d.confidence), 5)}
        if d.uncertainty and "second_class_score" in d.uncertainty:
            row["second_class_score"] = d.uncertainty["second_class_score"]  # ignored by COCOeval; used for the ambiguity analysis
        out.append(row)
    return out, unmapped


def run_detection(
    task: DetectionTask,
    *,
    model_entry: dict[str, Any],
    infer: Callable[[np.ndarray, list[ConfigSpec]], list[InferenceResult]],
    configs: Sequence[ConfigSpec],
    model_labels: Sequence[str | None] = (),
    load_ms: float | None = None,
    predictions_dir: Path | None = None,
    progress_every: int = 250,
    extra_record: dict[str, Any] | None = None,
) -> list[Path]:
    name_to_cat = {c["name"]: c["id"] for c in task.gt["categories"]}
    cat_names = {c["id"]: c["name"] for c in task.gt["categories"]}
    direct = [c for c in configs if c.derive_from is None]
    derived = [c for c in configs if c.derive_from is not None]
    results: dict[str, list[dict[str, Any]]] = {c.name: [] for c in configs}
    candidates: dict[str, dict[Any, dict[str, np.ndarray]]] = {c.name: {} for c in configs if c.keep_candidates}
    totals: dict[str, list[float]] = {c.name: [] for c in direct}
    wall: list[float] = []
    unmapped = 0
    images = task.gt["images"]
    started = time.perf_counter()
    for n, img in enumerate(images, 1):
        t0 = time.perf_counter()
        bgr = cv2.imread(task.image_path(img), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"unreadable image: {task.image_path(img)}")
        outs = infer(bgr, list(direct))
        wall.append((time.perf_counter() - t0) * 1000)
        by_name = {}
        for spec, res in zip(direct, outs, strict=True):
            r, u = _to_results(res, img["id"], name_to_cat)
            unmapped += u
            results[spec.name].extend(r)
            by_name[spec.name] = r
            totals[spec.name].append(res.timings_ms.get("total", 0.0))
            cand = res.config.pop("_candidates", None)
            if spec.keep_candidates and cand is not None:
                labels = {i: name_to_cat.get(nm or "") for i, nm in enumerate(model_labels)}
                cats = np.array([labels.get(int(c)) or -1 for c in cand["classes"]], dtype=np.int64)
                candidates[spec.name][img["id"]] = {"boxes": cand["boxes"].astype(np.float64), "cats": cats, "scores": cand["scores"].astype(np.float64)}
        for spec in derived:
            src = sorted(by_name[spec.derive_from], key=lambda r: -r["score"])
            kept = [r for r in src if r["score"] >= spec.derive_min_score][: spec.derive_max_results]
            results[spec.name].extend(kept)
        if progress_every and n % progress_every == 0:
            rate = n / (time.perf_counter() - started)
            print(f"    {n}/{len(images)} images ({rate:.1f} img/s)", flush=True)
    elapsed = time.perf_counter() - started
    env = records.environment()
    paths = []
    for spec in configs:
        res = results[spec.name]
        if predictions_dir:
            predictions_dir.mkdir(parents=True, exist_ok=True)
            (predictions_dir / f"{model_entry['id']}__{task.dataset_id}__{task.split}__{spec.name}.json").write_text(json.dumps(res))
        cat_ids = task.eval_category_ids
        coco = det_eval.coco_metrics(task.gt, res, cat_ids=cat_ids, category_names=cat_names, max_dets=task.max_dets)
        image_evals = det_eval.build_image_evals(task.gt, res, candidates.get(spec.name), set(cat_ids) if cat_ids else None)
        ops = det_eval.operating_points(image_evals)
        best = max(ops, key=lambda r: r["f1"])
        analysis = det_eval.miss_and_fp_analysis(image_evals, spec.operating_threshold, cat_names)
        calib = det_eval.calibration(image_evals, spec.calibration_floor)
        timing_src = spec.derive_from or spec.name
        record = {
            "task": task.task_name,
            "date": records.now(),
            "model": {"id": model_entry["id"], "name": model_entry.get("name"), "version": model_entry.get("version"),
                      "weights_sha256": model_entry.get("weights", {}).get("sha256"), "precision": model_entry.get("weights", {}).get("precision")},
            "dataset": {"id": task.dataset_id, "split": task.split, "images": len(images),
                        "instances": sum(1 for a in task.gt["annotations"] if not a.get("iscrowd") and (cat_ids is None or a["category_id"] in cat_ids)),
                        "evaluated_categories": [cat_names[c] for c in (cat_ids or sorted(cat_names))], "max_detections_per_image": task.max_dets[-1]},
            "config_name": spec.name,
            "config_description": spec.description,
            "config": dataclasses.asdict(spec.config) if dataclasses.is_dataclass(spec.config) else dict(spec.config),
            "operating_threshold": spec.operating_threshold,
            "environment": env,
            "inference_precision": model_entry.get("weights", {}).get("precision"),
            "performance": {
                **det_eval.latency_summary(totals[timing_src], wall),
                "model_load_ms": None if load_ms is None else round(load_ms, 1),
                "peak_rss_mb_process": records.peak_rss_mb(),
                "total_run_s": round(elapsed, 1),
                "note": "derived configuration: timing of its source run" if spec.derive_from else None,
            },
            "metrics": {
                "coco": coco["summary"],
                "f1_optimal": best,
                "at_operating_threshold": next((r for r in ops if abs(r["threshold"] - spec.operating_threshold) < 1e-6), None),
                "operating_points": ops,
                "calibration": calib,
            },
            "per_class": {name: {**coco["per_class"].get(name, {}), **analysis["per_class"].get(name, {})} for name in sorted(set(coco["per_class"]) | set(analysis["per_class"]))},
            "error_analysis": {k: v for k, v in analysis.items() if k != "per_class"},
            "unmapped_detections": unmapped,
            "notes": task.notes,
            **(extra_record or {}),
        }
        paths.append(records.write(record))
        s = coco["summary"]
        print(f"  [{spec.name}] AP {s['AP']} AP50 {s['AP50']} APs {s['APs']} APm {s['APm']} APl {s['APl']} | "
              f"F1-opt t={best['threshold']} P={best['precision']} R={best['recall']} | ECE {calib['ece']}", flush=True)
    return paths

