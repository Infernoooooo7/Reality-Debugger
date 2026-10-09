#!/usr/bin/env python3
"""Evaluation suite: reproducible benchmarks written to docs/benchmarks/records/.

    python tools/evaluate.py detection --model yolox_s --dataset coco_val2017
    python tools/evaluate.py detection --model efficientdet_lite0 --dataset coco_val2017 --limit 500
    python tools/evaluate.py detection --model yolox_s --dataset visdrone_det_val --split test

Datasets come from tools/datasets.py, models from tools/fetch_models.py.
Requirements: tools/requirements-eval.txt (pycocotools, onnxruntime, mediapipe).
See docs/EVALUATION_PROTOCOL.md for what each benchmark measures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.datasets import adapters, paths  # noqa: E402
from app.evaluation.runner import ConfigSpec, DetectionTask, run_detection  # noqa: E402
from app.vision.registry import ModelRegistry  # noqa: E402

DETECTION = json.loads((ROOT / "config" / "detection.json").read_text())


def _v(node: dict) -> float:
    return node["value"]


def detection_task(dataset_id: str, split: str | None, limit: int | None) -> DetectionTask:
    if dataset_id == "coco_val2017":
        ad = adapters.CocoAdapter()
        ids = sorted(i["id"] for i in ad.gt["images"])
        if limit:
            ids = ids[:limit]
        gt = ad.coco_gt(set(ids)) if limit else ad.gt
        by_id = {i["id"]: i for i in ad.gt["images"]}
        notes = ["COCO 2017 val: held out from the detectors' COCO train2017 training data."]
        if limit:
            notes.append(f"Subset: the first {limit} images by image id (deterministic).")
        return DetectionTask("detection", dataset_id, "val2017" + (f"-first{limit}" if limit else ""), gt,
                             lambda img: str(ad.image_path(by_id[img["id"]])), None, notes)
    if dataset_id == "visdrone_det_val":
        ad = adapters.VisDroneAdapter()
        ids = adapters.load_split(dataset_id, split) if split else None
        gt = ad.coco_gt(ids)
        if limit:
            keep = {i["id"] for i in sorted(gt["images"], key=lambda i: i["id"])[:limit]}
            gt = {**gt, "images": [i for i in gt["images"] if i["id"] in keep], "annotations": [a for a in gt["annotations"] if a["image_id"] in keep]}
        notes = ["VisDrone classes mapped to COCO (pedestrian/people->person, car/van->car, motor->motorcycle); "
                 "tricycles, 'others' and ignored regions are ignore regions for every class."]
        return DetectionTask("detection", dataset_id, split or "all", gt, lambda img: img["file_name"],
                             [c["id"] for c in gt["categories"]], notes + ["Up to 500 detections per image are evaluated (VisDrone protocol)."],
                             max_dets=(1, 10, 500))
    raise SystemExit(f"detection evaluation not set up for {dataset_id}")


def load_detector(model_id: str, registry: ModelRegistry):
    entry = registry.get(model_id)
    avail = registry.availability(model_id, verify=True)
    if not avail.available:
        raise SystemExit(f"{model_id}: {avail.reason}")
    labels = list(adapters.COCO80)
    family = entry.get("family")
    if family == "yolox":
        from app.vision.detectors.yolox import YoloxConfig, YoloxDetector

        det = YoloxDetector(entry, Path(avail.path), labels)
        thr = _v(DETECTION["deep"]["scoreThreshold"])
        configs = [
            ConfigSpec("protocol", YoloxConfig.protocol(), thr, 0.01,
                       description="YOLOX COCO-eval settings: score>=0.01, class-aware NMS IoU 0.65, 100 detections"),
            ConfigSpec("deployed", YoloxConfig(score_floor=_v(DETECTION["deep"]["preNmsScore"]), nms_iou=_v(DETECTION["deep"]["nmsIou"]),
                                                class_agnostic=_v(DETECTION["deep"]["classAgnosticNms"]), keep_candidates=True),
                       thr, _v(DETECTION["deep"]["preNmsScore"]),
                       keep_candidates=True, description="As in the app (config/detection.json deep.*)"),
            # One-variable ablation of the deployed config: class-aware instead of class-agnostic NMS.
            ConfigSpec("deployed-classaware", YoloxConfig(score_floor=_v(DETECTION["deep"]["preNmsScore"]), nms_iou=_v(DETECTION["deep"]["nmsIou"]),
                                                           class_agnostic=False, keep_candidates=True),
                       thr, _v(DETECTION["deep"]["preNmsScore"]), keep_candidates=True,
                       description="Deployed settings with class-aware NMS (ablation of deep.classAgnosticNms only)"),
        ]

        def infer(bgr, specs):
            return det.detect_many(bgr, [s.config for s in specs])

        return entry, infer, configs, det.load_ms, labels
    if family == "efficientdet-lite":
        from app.vision.detectors.mediapipe_det import MediaPipeConfig, MediaPipeDetector

        det = MediaPipeDetector(entry, Path(avail.path), labels, config=MediaPipeConfig.protocol())
        thr = _v(DETECTION["fast"]["scoreThreshold"])
        configs = [
            ConfigSpec("protocol", MediaPipeConfig.protocol(), thr, 0.01,
                       description="Letterboxed as in the app; lowest score floor (0.01) and 100 results"),
            ConfigSpec("deployed", MediaPipeConfig.deployed(), thr, _v(DETECTION["fast"]["lowScoreFloor"]),
                       derive_from="protocol", derive_min_score=_v(DETECTION["fast"]["lowScoreFloor"]), derive_max_results=int(_v(DETECTION["fast"]["maxResults"])),
                       description="As in the app: floor 0.15, 30 results (derived from the protocol run: MediaPipe applies the score floor before NMS and keeps the top results, so filtering is exact)"),
        ]

        def infer(bgr, specs):
            return [det.detect(bgr)]

        infer.close = det.close  # type: ignore[attr-defined]  # MediaPipe must be closed before interpreter shutdown
        return entry, infer, configs, det.load_ms, labels
    raise SystemExit(f"no evaluation adapter for model family {family}")


def cmd_detection(args: argparse.Namespace) -> int:
    registry = ModelRegistry()
    task = detection_task(args.dataset, args.split, args.limit)
    entry, infer, configs, load_ms, labels = load_detector(args.model, registry)
    print(f"{args.model} on {task.dataset_id}/{task.split}: {len(task.gt['images'])} images", flush=True)
    t0 = time.perf_counter()
    out = run_detection(task, model_entry=entry, infer=infer, configs=configs, model_labels=labels, load_ms=load_ms,
                        predictions_dir=paths.data_root() / "processed" / "predictions")
    getattr(infer, "close", lambda: None)()
    print(f"done in {time.perf_counter() - t0:.0f} s")
    for p in out:
        print("  wrote", p.relative_to(ROOT))
    return 0


def cmd_tiling(args: argparse.Namespace) -> int:
    """Sliced inference sweep for one YOLOX model: full-image baseline vs tile sizes/overlaps/merging."""
    from app.vision.detectors.yolox import YoloxConfig, YoloxDetector
    from app.vision.tiling import TilingConfig, sliced_detect

    registry = ModelRegistry()
    entry = registry.get(args.model)
    avail = registry.availability(args.model, verify=True)
    if not avail.available:
        raise SystemExit(f"{args.model}: {avail.reason}")
    labels = list(adapters.COCO80)
    det = YoloxDetector(entry, Path(avail.path), labels)
    task = detection_task(args.dataset, args.split, args.limit)
    task.task_name = "detection-tiling"
    max_det = task.max_dets[-1]
    per_pass = YoloxConfig(score_floor=0.01, nms_iou=0.65, class_agnostic=False, max_detections=max_det)
    thr = _v(DETECTION["deep"]["scoreThreshold"])
    specs = [ConfigSpec("full", per_pass, thr, 0.01, description=f"single full-image pass at {det.native_size}px (YOLOX eval NMS, {max_det} detections)")]
    for tile in [int(t) for t in args.tiles.split(",")]:
        for overlap in [float(o) for o in args.overlaps.split(",")]:
            for merge in args.merges.split(","):
                cfg = TilingConfig(tile=tile, overlap=overlap, full_image=not args.no_full, merge=merge,
                                   match_metric="ios" if merge == "nmm" else "iou", match_threshold=0.5)
                specs.append(ConfigSpec(f"tile{tile}-ov{overlap:g}-{merge}{'' if cfg.full_image else '-nofull'}", cfg, thr, 0.01,
                                        description=f"sliced inference: {tile}px tiles, {overlap:g} overlap, {merge} merge, full-image pass {cfg.full_image}"))

    def infer(bgr, run_specs):
        out = []
        for spec in run_specs:
            if isinstance(spec.config, TilingConfig):
                res = sliced_detect(lambda im: det.detect(im, config=per_pass), bgr, spec.config, model_id=det.model_id, model_version=det.model_version)
                res.detections = sorted(res.detections, key=lambda d: -d.confidence)[:max_det]
                out.append(res)
            else:
                out.append(det.detect(bgr, config=spec.config))
        return out

    print(f"{args.model} tiling sweep on {task.dataset_id}/{task.split}: {len(task.gt['images'])} images, {len(specs)} configurations", flush=True)
    paths_out = run_detection(task, model_entry=entry, infer=infer, configs=specs, model_labels=labels, load_ms=det.load_ms,
                              predictions_dir=paths.data_root() / "processed" / "predictions", progress_every=50)
    for p in paths_out:
        print("  wrote", p.relative_to(ROOT))
    return 0


def cmd_robustness(args: argparse.Namespace) -> int:
    """Accuracy under controlled corruptions (blur, motion blur, low light, JPEG, low resolution)."""
    import numpy as np

    from app.evaluation.corruptions import CORRUPTIONS

    registry = ModelRegistry()
    task = detection_task(args.dataset, None, args.limit)
    task.task_name = "detection-robustness"
    entry, base_infer, configs, load_ms, labels = load_detector(args.model, registry)
    deployed = next(c for c in configs if c.name == "deployed")
    protocol = next(c for c in configs if c.name == "protocol")
    names = ["clean", *[c for c in args.corruptions.split(",") if c]] if args.corruptions else ["clean", *CORRUPTIONS]
    specs = []
    for name in names:
        params = {} if name == "clean" else CORRUPTIONS[name][1]
        spec = ConfigSpec(name, {"corruption": name, **params, "detector_config": "protocol"}, protocol.operating_threshold, protocol.calibration_floor,
                          description=f"{name}: {params or 'no corruption'}; detector in its protocol configuration")
        specs.append(spec)

    def infer(bgr, run_specs):
        out = []
        for spec in run_specs:
            name = spec.config["corruption"]
            # Stable per-image, per-corruption seed (Python's hash() is salted per process).
            rng = np.random.default_rng(zlib.crc32(name.encode() + np.ascontiguousarray(bgr[::97, ::89]).tobytes()))
            img = bgr if name == "clean" else CORRUPTIONS[name][0](bgr, rng)
            out.append(base_infer(img, [protocol])[0])
        return out

    print(f"{args.model} robustness on {task.dataset_id}/{task.split}: {len(task.gt['images'])} images x {len(specs)} conditions", flush=True)
    out = run_detection(task, model_entry=entry, infer=infer, configs=specs, model_labels=labels, load_ms=load_ms, progress_every=100)
    for p in out:
        print("  wrote", p.relative_to(ROOT))
    _ = deployed
    return 0


def cmd_tracking(args: argparse.Namespace) -> int:
    """Tracker benchmark on KITTI tracking: oracle and detector inputs x tracker variants."""
    from app.evaluation.tracking_runner import run_kitti

    registry = ModelRegistry()
    labels = list(adapters.COCO80)
    detectors = {}
    for name in args.inputs.split(","):
        if name == "oracle":
            detectors["oracle"] = None
        elif name == "efficientdet_lite0":
            from app.vision.detectors.mediapipe_det import MediaPipeConfig, MediaPipeDetector

            entry = registry.get(name)
            det = MediaPipeDetector(entry, Path(registry.availability(name).path), labels, config=MediaPipeConfig.deployed())
            detectors[name] = lambda img, d=det: [{"label": x.class_name, "score": x.confidence, "box": x.box} for x in d.detect(img).detections]
        elif name.startswith("yolox"):
            from app.vision.detectors.yolox import YoloxConfig, YoloxDetector

            entry = registry.get(name)
            det = YoloxDetector(entry, Path(registry.availability(name).path), labels, config=YoloxConfig.deployed())
            detectors[name] = lambda img, d=det: [{"label": x.class_name, "score": x.confidence, "box": x.box} for x in d.detect(img).detections]
        else:
            raise SystemExit(f"unknown tracking input {name}")
    variants = {
        "cmc-off": {"cmc": False},
        "cmc-on": {"cmc": True},
        "cmc-on-no-low-score": {"cmc": True, "lowScore": 0.3},
    }
    if args.variants:
        variants = {k: v for k, v in variants.items() if k in args.variants.split(",")}
    out = run_kitti(detectors, variants, paths.data_root() / "processed" / "tracking")
    for p in out:
        print("  wrote", p.relative_to(ROOT))
    return 0


def cmd_vocabulary(args: argparse.Namespace) -> int:
    """Vocabulary gap: LVIS categories vs COCO-80 (with real detector coverage) and Open Images coverage."""
    from app.evaluation import records
    from app.evaluation.vocabulary import lvis_coverage, openimages_coverage

    pred = paths.data_root() / "processed" / "predictions" / f"{args.model}__coco_val2017__val2017__protocol.json"
    if not pred.exists():
        raise SystemExit(f"run first: python tools/evaluate.py detection --model {args.model} --dataset coco_val2017")
    thr = _v(DETECTION["deep"]["scoreThreshold"]) if args.model.startswith("yolox") else _v(DETECTION["fast"]["scoreThreshold"])
    lvis = lvis_coverage(pred, thr, args.wordnet_dir)
    oi = openimages_coverage()
    record = {
        "task": "vocabulary-coverage", "date": records.now(),
        "model": {"id": args.model, "name": args.model, "version": ModelRegistry().get(args.model).get("version")},
        "dataset": {"id": "lvis_v1_minival+openimages_v5_val", "split": "val"},
        "config": {"display_threshold": thr, "localization_iou": 0.5}, "config_name": "default",
        "environment": records.environment(),
        "metrics": {"lvis": lvis, "open_images": oi},
    }
    path = records.write(record)
    print(json.dumps({"lvis_instance_share": lvis["instance_share"], "lvis_localized": lvis["localized_rate_by_relation"],
                      "openimages_box_share_mapped": oi["box_share_mapped"]}, indent=1))
    print("  wrote", path.relative_to(ROOT))
    return 0


def cmd_anomaly(args: argparse.Namespace) -> int:
    """VisA anomaly detection/segmentation: PatchCore (ResNet-50 features) vs the reference-difference baseline."""
    from app.evaluation import records
    from app.evaluation.anomaly import EVAL_SIZE, HOLDOUT, run_visa

    registry = ModelRegistry()
    avail = registry.availability("resnet50_v1_patch_features", verify=True)
    if not avail.available:
        raise SystemExit(f"resnet50_v1_patch_features unavailable: {avail.reason}")
    configs = {
        "patchcore-k5": {"method": "patchcore", "k": 5, "coreset_ratio": 1.0, "loo_threshold": True},
        "patchcore-k10": {"method": "patchcore", "k": 10, "coreset_ratio": 1.0, "loo_threshold": True},
        "patchcore-k50-coreset10": {"method": "patchcore", "k": 50, "coreset_ratio": 0.1},
        "patchcore-k200-coreset1": {"method": "patchcore", "k": 200, "coreset_ratio": 0.01},
        "refdiff-k10": {"method": "reference-difference", "k": 10},
        "refdiff-k50": {"method": "reference-difference", "k": 50},
    }
    if args.configs:
        configs = {k: v for k, v in configs.items() if k in args.configs.split(",")}
    from app.datasets.adapters import VisaAdapter

    categories = args.categories.split(",") if args.categories else VisaAdapter().categories
    t0 = time.perf_counter()
    results = run_visa(categories, configs, extractor_weights=Path(avail.path), size=EVAL_SIZE, seed=args.seed)
    entry = registry.get("resnet50_v1_patch_features")
    for name, res in results.items():
        cfg = configs[name]
        method = "PatchCore (ResNet-50 v1 stages 2+3)" if cfg["method"] == "patchcore" else "Reference difference (ECC affine + CIE76 Delta E)"
        record = {
            "task": "anomaly-detection", "date": records.now(),
            "model": {"id": "patchcore-resnet50" if cfg["method"] == "patchcore" else "reference-difference", "name": method,
                      "version": entry.get("weights", {}).get("sha256", "")[:12] if cfg["method"] == "patchcore" else "n/a",
                      "feature_extractor": "resnet50_v1_patch_features" if cfg["method"] == "patchcore" else None},
            "dataset": {"id": "visa", "split": "official 1cls test", "categories": categories},
            "config_name": name,
            "config": {**cfg, "input_size": EVAL_SIZE, "eval_size": EVAL_SIZE, "holdout_normals_for_threshold": HOLDOUT, "seed": args.seed,
                       "reference_selection": "seeded random permutation of the official train (normal) images, disjoint from the held-out ones"},
            "environment": records.environment(),
            "performance": {**res["timing"], "peak_rss_mb": records.peak_rss_mb(), "wall_s_total_all_configs": round(time.perf_counter() - t0, 1)},
            "metrics": {"mean_over_categories": res["mean_over_categories"], "per_category": res["per_category"]},
            "notes": [
                "Image score = maximum of the anomaly map / patch scores; scores are distances, not probabilities.",
                "Threshold = maximum image score of held-out normal training images; no anomalous image is used for any choice.",
                "Pixel metrics at 224x224; ground-truth masks resized with nearest-neighbour interpolation (very thin defects can shrink or vanish).",
                "Deviations from published PatchCore: ResNet-50 v1 instead of WideResNet-50-2; full-image resize instead of resize+centre-crop; "
                "k reference images instead of the full training set.",
            ],
        }
        path = records.write(record)
        print("  wrote", path.relative_to(ROOT))
    return 0


def cmd_quality(args: argparse.Namespace) -> int:
    """Ground the image-quality gate: the app's brightness/sharpness signals vs measured accuracy under corruption.

    For each corruption family (low light -> brightness, blur -> sharpness), levels are ordered by their median
    signal; the threshold is the midpoint between the median signal of the last level that keeps at least half
    of the clean AP and the first level that does not. Uses the robustness records (same images, same seeds).
    """
    import cv2
    import numpy as np
    from PIL import Image

    from app.datasets.adapters import CocoAdapter
    from app.evaluation import records
    from app.evaluation.corruptions import CORRUPTIONS
    from app.vision.signals import gray_signals, thumbnail_gray

    vision = json.loads((ROOT / "config" / "vision.json").read_text())["signals"]
    tw, th = int(_v(vision["thumbWidth"])), int(_v(vision["thumbHeight"]))
    off, span = float(_v(vision["sharpnessLogOffset"])), float(_v(vision["sharpnessLogSpan"]))
    rob = {}
    for path in sorted((ROOT / "docs" / "benchmarks" / "records").glob(f"detection-robustness-{args.model.replace('_', '-')}-*.json")):
        r = json.loads(path.read_text())
        rob[r["config_name"]] = r["metrics"]["coco"]["AP"]
    if "clean" not in rob:
        raise SystemExit(f"run first: python tools/evaluate.py robustness --model {args.model}")
    ad = CocoAdapter()
    ids = sorted(i["id"] for i in ad.gt["images"])[: args.limit]
    by_id = {i["id"]: i for i in ad.gt["images"]}
    signals: dict[str, list[tuple[float, float]]] = {name: [] for name in ["clean", *CORRUPTIONS]}
    for image_id in ids:
        bgr = cv2.imread(str(ad.image_path(by_id[image_id])))
        for name in signals:
            rng = np.random.default_rng(zlib.crc32(name.encode() + np.ascontiguousarray(bgr[::97, ::89]).tobytes()))
            img = bgr if name == "clean" else CORRUPTIONS[name][0](bgr, rng)
            gray = thumbnail_gray(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), tw, th)
            signals[name].append(gray_signals(gray, log_offset=off, log_span=span))
    summary = {}
    for name, vals in signals.items():
        b, sh = np.array([v[0] for v in vals]), np.array([v[1] for v in vals])
        summary[name] = {"ap": rob.get(name), "relative_ap": round(rob[name] / rob["clean"], 3) if name in rob else None,
                         "brightness_median": round(float(np.median(b)), 4), "brightness_p10": round(float(np.percentile(b, 10)), 4),
                         "sharpness_median": round(float(np.median(sh)), 4), "sharpness_p10": round(float(np.percentile(sh, 10)), 4)}

    def threshold(family: list[str], key: str) -> dict:
        levels = sorted((n for n in family if summary[n]["relative_ap"] is not None), key=lambda n: -summary[n][f"{key}_median"])
        for prev, cur in zip(levels, levels[1:], strict=False):
            if summary[cur]["relative_ap"] < 0.5 <= summary[prev]["relative_ap"]:
                value = (summary[prev][f"{key}_median"] + summary[cur][f"{key}_median"]) / 2
                return {"value": round(value, 3), "between": [prev, cur], "levels": levels}
        return {"value": None, "reason": "no measured level halves the AP", "levels": levels}

    result = {
        "minBrightness": threshold(["clean", "low_light_x0.1", "low_light_x0.02"], "brightness"),
        "minSharpness": threshold(["clean", "gaussian_blur_s2", "gaussian_blur_s4", "motion_blur_15", "motion_blur_31"], "sharpness"),
    }
    record = {
        "task": "quality-gate", "date": records.now(),
        "model": {"id": args.model, "name": args.model, "version": ModelRegistry().get(args.model).get("version")},
        "dataset": {"id": "coco_val2017", "split": f"val2017-first{args.limit}"},
        "config": {"rule": "midpoint of median signals around the level where AP falls below half of clean", "signals": "config/vision.json signals"},
        "config_name": "default", "environment": records.environment(),
        "metrics": {"per_condition": summary, "thresholds": result},
        "notes": ["Thresholds describe synthetic corruptions; the gate's own decisions are not evaluated on separate data (not measured)."],
    }
    path = records.write(record)
    print(json.dumps(result, indent=1))
    print("  wrote", path.relative_to(ROOT))
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    """Ground the inspection-coverage rules on held-out annotations.

    For every COCO val2017 image (LVIS v1 minival annotates the same images with 1203 categories) the app's
    unexplained-structure measure is computed with the boxes the two deployed detectors return (>= 0.3,
    EfficientDet-Lite0 + YOLOX-S, as Image Debug fuses them), and compared with how many annotated objects
    were not recognised: COCO objects without a same-class detection at IoU >= 0.5, plus LVIS objects of
    categories outside COCO-80 without any detection at IoU >= 0.5. LVIS is federated, so the unrecognised
    share is a lower bound. Also measures detection precision per confidence band.
    """
    import collections

    import cv2
    import numpy as np

    from app.evaluation import records
    from app.evaluation.detection import iou_matrix
    from app.evaluation.vocabulary import classify_lvis_categories
    from app.vision.coverage import EDGE_THRESHOLD, STRUCTURE_SIDE, luminance_small, structure_stats

    data = paths.data_root()
    coco = adapters.CocoAdapter()
    lvis = adapters.LvisAdapter()
    cats = classify_lvis_categories(lvis.gt["categories"], args.wordnet_dir)
    gt = collections.defaultdict(list)
    for a in coco.gt["annotations"]:
        if not a.get("iscrowd"):
            gt[a["image_id"]].append(a)
    outside = collections.defaultdict(list)
    for a in lvis.gt["annotations"]:
        if cats[a["category_id"]]["relation"] == "outside":
            outside[a["image_id"]].append(a)
    preds: dict[int, list[dict]] = collections.defaultdict(list)
    for model in ("efficientdet_lite0", "yolox_s"):
        f = data / "processed" / "predictions" / f"{model}__coco_val2017__val2017__deployed.json"
        if not f.exists():
            raise SystemExit(f"run first: python tools/evaluate.py detection --model {model} --dataset coco_val2017")
        for p in json.loads(f.read_text()):
            p["model"] = model
            preds[p["image_id"]].append(p)
    xyxy = lambda b: [b[0], b[1], b[0] + b[2], b[1] + b[3]]  # noqa: E731
    rows = []
    bands = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.7), (0.7, 1.01)]
    band_tp = collections.Counter()
    band_n = collections.Counter()
    images = sorted(coco.gt["images"], key=lambda i: i["id"])[: args.limit or None]
    for img in images:
        iid, w, h = img["id"], img["width"], img["height"]
        kept = [p for p in preds[iid] if p["score"] >= args.threshold]
        boxes = [{"x": p["bbox"][0] / w, "y": p["bbox"][1] / h, "w": p["bbox"][2] / w, "h": p["bbox"][3] / h} for p in kept]
        bgr = cv2.imread(str(coco.image_path(img)))
        st = structure_stats(luminance_small(bgr), boxes)
        g, o = gt[iid], outside[iid]
        matched = 0
        if kept and g:
            pb = np.array([xyxy(p["bbox"]) for p in kept])
            for a in g:
                same = np.array([p["category_id"] == a["category_id"] for p in kept])
                if same.any() and (iou_matrix(np.array([xyxy(a["bbox"])]), pb[same])[0] >= 0.5).any():
                    matched += 1
        localized = 0
        if kept and o:
            localized = int((iou_matrix(np.array([xyxy(a["bbox"]) for a in o]), np.array([xyxy(p["bbox"]) for p in kept])).max(1) >= 0.5).sum())
        annotated = len(g) + len(o)
        rows.append({"image_id": iid, "unexplained": st.unexplained_share, "edge_density": st.edge_density, "detections": len(kept),
                     "annotated": annotated, "unrecognised_share": None if not annotated else 1 - (matched + localized) / annotated})
        # precision per confidence band (YOLOX-S only: the deep detector carries the analysis)
        ys = [p for p in kept if p["model"] == "yolox_s"]
        used = set()
        for p in sorted(ys, key=lambda x: -x["score"]):
            band = next((b for b in bands if b[0] <= p["score"] < b[1]), None)
            if band is None:
                continue
            band_n[band] += 1
            for j, a in enumerate(g):
                if j in used or a["category_id"] != p["category_id"]:
                    continue
                if iou_matrix(np.array([xyxy(p["bbox"])]), np.array([xyxy(a["bbox"])]))[0, 0] >= 0.5:
                    used.add(j)
                    band_tp[band] += 1
                    break
    valid = [r for r in rows if r["unexplained"] is not None and r["unrecognised_share"] is not None]
    u = np.array([r["unexplained"] for r in valid])
    gshare = np.array([r["unrecognised_share"] for r in valid])
    edges = np.quantile(u, np.linspace(0, 1, 11))
    deciles = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (u >= lo) & (u <= hi)
        deciles.append({"unexplained_from": round(float(lo), 3), "unexplained_to": round(float(hi), 3), "images": int(m.sum()),
                        "unrecognised_share_median": round(float(np.median(gshare[m])), 3),
                        "share_of_images_with_most_objects_unrecognised": round(float((gshare[m] > 0.5).mean()), 3)})
    # Decision rule: the lowest unexplained-share value from which, in every higher decile, most annotated
    # objects (median > 50%) were not recognised.
    threshold = None
    for i, d in enumerate(deciles):
        if all(x["unrecognised_share_median"] > 0.5 for x in deciles[i:]):
            threshold = d["unexplained_from"]
            break
    from scipy.stats import spearmanr

    rho = spearmanr(u, gshare)
    zero = [r for r in rows if r["detections"] == 0 and r["annotated"]]
    result = {
        "images": len(rows), "with_structure_and_annotations": len(valid),
        "spearman_unexplained_vs_unrecognised": {"rho": round(float(rho.statistic), 3), "p": float(rho.pvalue)},
        "deciles": deciles,
        "unexplained_threshold": threshold,
        "unexplained_threshold_rule": "lowest decile start from which every higher decile has median unrecognised share > 0.5",
        "images_without_any_detection": {"count": len(zero), "median_edge_density": round(float(np.median([r["edge_density"] for r in zero])), 4) if zero else None,
                                          "median_annotated_objects": float(np.median([r["annotated"] for r in zero])) if zero else None},
        "yolox_s_precision_by_confidence": {f"{lo:.1f}-{min(hi, 1.0):.1f}": {"detections": band_n[(lo, hi)],
                                                                             "precision": round(band_tp[(lo, hi)] / band_n[(lo, hi)], 3) if band_n[(lo, hi)] else None}
                                            for lo, hi in bands},
    }
    record = {
        "task": "inspection-coverage", "date": records.now(),
        "model": {"id": "efficientdet_lite0+yolox_s", "name": "deployed fast + deep detectors (fused boxes)", "version": "deployed"},
        "dataset": {"id": "coco_val2017+lvis_v1_minival", "split": "val2017" + (f"-first{args.limit}" if args.limit else "")},
        "config": {"structure_side_px": STRUCTURE_SIDE, "edge_threshold": EDGE_THRESHOLD, "detection_threshold": args.threshold,
                   "recognised": "COCO object: same-class detection IoU>=0.5; LVIS non-COCO object: any detection IoU>=0.5"},
        "config_name": "default", "environment": records.environment(), "metrics": result,
        "notes": ["LVIS is federated: unrecognised shares are lower bounds.",
                  "The measure is evidence of unexplained image structure, not an estimate of recall."],
    }
    path = records.write(record)
    print(json.dumps({k: v for k, v in result.items() if k != "deciles"}, indent=1))
    for d in deciles:
        print("  ", d)
    print("  wrote", path.relative_to(ROOT))
    return 0


DESK_CATEGORIES = ("laptop", "keyboard", "mouse", "tv", "cell phone", "book", "cup", "bottle", "remote", "scissors")


def cmd_desk(args: argparse.Namespace) -> int:
    """Detection on desk scenes (the regression set behind docs/SCORING.md): COCO val2017 images annotated with a
    laptop or a keyboard and at least three categories of DESK_CATEGORIES. For each deployed detector at the display
    threshold: recall and precision on COCO-80 objects (same class, IoU >= 0.5, greedy by score), recall by COCO object
    size, detections and annotated objects per image, and how many LVIS objects of categories outside COCO-80 any
    detection overlaps at IoU >= 0.5 (LVIS is federated: those counts are lower bounds)."""
    import collections
    import statistics

    import numpy as np

    from app.evaluation import records
    from app.evaluation.detection import iou_matrix
    from app.evaluation.vocabulary import classify_lvis_categories

    data = paths.data_root()
    coco = adapters.CocoAdapter()
    lvis = adapters.LvisAdapter()
    cats = classify_lvis_categories(lvis.gt["categories"], args.wordnet_dir)
    gt = collections.defaultdict(list)
    for a in coco.gt["annotations"]:
        if not a.get("iscrowd"):
            gt[a["image_id"]].append(a)
    outside = collections.defaultdict(list)
    for a in lvis.gt["annotations"]:
        if cats[a["category_id"]]["relation"] == "outside":
            outside[a["image_id"]].append(a)
    desk = sorted(desk_images(coco.gt))
    xyxy = lambda b: [b[0], b[1], b[0] + b[2], b[1] + b[3]]  # noqa: E731
    result: dict[str, dict] = {}
    for model in ("efficientdet_lite0", "yolox_s"):
        f = data / "processed" / "predictions" / f"{model}__coco_val2017__val2017__deployed.json"
        if not f.exists():
            raise SystemExit(f"run first: python tools/evaluate.py detection --model {model} --dataset coco_val2017")
        preds = collections.defaultdict(list)
        for p in json.loads(f.read_text()):
            if p["score"] >= args.threshold:
                preds[p["image_id"]].append(p)
        tp = fp = n_gt = out_total = out_hit = 0
        size_n: collections.Counter = collections.Counter()
        size_hit: collections.Counter = collections.Counter()
        dets_per_image, annotated_per_image = [], []
        for iid in desk:
            g, o = gt[iid], outside[iid]
            p = sorted(preds[iid], key=lambda x: -x["score"])
            dets_per_image.append(len(p))
            annotated_per_image.append(len(g) + len(o))
            used: set[int] = set()
            for d in p:
                best, bj = 0.5, -1
                for j, a in enumerate(g):
                    if j in used or a["category_id"] != d["category_id"]:
                        continue
                    v = float(iou_matrix(np.array([xyxy(d["bbox"])]), np.array([xyxy(a["bbox"])]))[0, 0])
                    if v >= best:
                        best, bj = v, j
                if bj >= 0:
                    used.add(bj)
                    tp += 1
                else:
                    fp += 1
            for j, a in enumerate(g):
                size = "small" if a["area"] < 32**2 else "medium" if a["area"] < 96**2 else "large"
                size_n[size] += 1
                size_hit[size] += j in used
            n_gt += len(g)
            out_total += len(o)
            if p and o:
                out_hit += int((iou_matrix(np.array([xyxy(a["bbox"]) for a in o]), np.array([xyxy(d["bbox"]) for d in p])).max(1) >= 0.5).sum())
        result[model] = {
            "coco_objects": n_gt, "detections": tp + fp, "true_positives": tp,
            "recall": round(tp / n_gt, 3) if n_gt else None, "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall_by_size": {k: {"objects": size_n[k], "recall": round(size_hit[k] / size_n[k], 3) if size_n[k] else None} for k in ("small", "medium", "large")},
            "median_detections_per_image": statistics.median(dets_per_image) if dets_per_image else None,
            "median_annotated_objects_per_image": statistics.median(annotated_per_image) if annotated_per_image else None,
            "lvis_objects_outside_coco80": out_total, "lvis_outside_overlapped_by_a_detection": out_hit,
        }
    record = {
        "task": "desk-scenes", "date": records.now(),
        "model": {"id": "efficientdet_lite0,yolox_s", "name": "deployed fast and deep detectors (each on its own)", "version": "deployed"},
        "dataset": {"id": "coco_val2017+lvis_v1_minival", "split": f"val2017-desk-{len(desk)}"},
        "config": {"selection": "COCO annotations include laptop or keyboard, and >= 3 categories of " + ", ".join(DESK_CATEGORIES),
                   "detection_threshold": args.threshold, "match": "same class, IoU >= 0.5, greedy by score; COCO size buckets 32^2 / 96^2 px"},
        "config_name": "default", "environment": records.environment(), "metrics": {"images": len(desk), "image_ids": desk, **result},
        "notes": ["Predictions are the deployed-settings runs in data/processed/predictions (detection subcommand).",
                  "LVIS is federated: objects outside COCO-80 are lower bounds."],
    }
    path = records.write(record)
    print(json.dumps({k: v for k, v in record["metrics"].items() if k != "image_ids"}, indent=1))
    print("  wrote", path.relative_to(ROOT))
    return 0


def desk_images(coco_gt: dict, lvis_outside: dict | None = None) -> list[int]:
    """COCO val2017 desk scenes (see cmd_desk), most LVIS objects outside COCO-80 first (then most COCO objects, id)."""
    import collections

    names = {c["id"]: c["name"] for c in coco_gt["categories"]}
    per_image: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for a in coco_gt["annotations"]:
        per_image[a["image_id"]][names[a["category_id"]]] += 1
    rows = [(len((lvis_outside or {}).get(iid, [])), sum(c.values()), iid) for iid, c in per_image.items()
            if {"laptop", "keyboard"} & set(c) and len(set(c) & set(DESK_CATEGORIES)) >= 3]
    return [iid for _, _, iid in sorted(rows, reverse=True)]


def cmd_composite(args: argparse.Namespace) -> int:
    """Compose desk scenes into one large photo with ground truth, for browser regression runs of Image Debug
    (frontend/scripts/still-eval.mjs). Each cell keeps its photo's pixels (scaled down to fit, never up), so the
    composite behaves like a high-resolution photo of a cluttered scene: many objects, each small in the frame."""
    import collections

    import cv2
    import numpy as np

    from app.evaluation.vocabulary import classify_lvis_categories

    coco = adapters.CocoAdapter()
    lvis = adapters.LvisAdapter()
    cats = classify_lvis_categories(lvis.gt["categories"], args.wordnet_dir)
    outside = collections.defaultdict(list)
    for a in lvis.gt["annotations"]:
        if cats[a["category_id"]]["relation"] == "outside":
            outside[a["image_id"]].append(a)
    ids = desk_images(coco.gt, outside)[args.offset: args.offset + args.cols * args.rows]
    names = {c["id"]: c["name"] for c in coco.gt["categories"]}
    images = {im["id"]: im for im in coco.gt["images"]}
    by_image = collections.defaultdict(list)
    for a in coco.gt["annotations"]:
        by_image[a["image_id"]].append(a)
    canvas = np.full((args.rows * args.cell_h, args.cols * args.cell_w, 3), 114, np.uint8)
    objects = []
    for k, iid in enumerate(ids):
        img = cv2.imread(str(coco.image_path(images[iid])))
        h, w = img.shape[:2]
        s = min(1.0, args.cell_w / w, args.cell_h / h)
        nw, nh = round(w * s), round(h * s)
        if s < 1.0:
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        ox, oy = (k % args.cols) * args.cell_w + (args.cell_w - nw) // 2, (k // args.cols) * args.cell_h + (args.cell_h - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = img
        for a in by_image[iid]:
            x, y, bw, bh = a["bbox"]
            objects.append({"label": names[a["category_id"]], "box": [round(ox + x * s, 2), round(oy + y * s, 2), round(ox + (x + bw) * s, 2), round(oy + (y + bh) * s, 2)],
                            "iscrowd": a["iscrowd"], "area_px": round(bw * bh * s * s, 1), "source_image": iid})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    out.with_suffix(".gt.json").write_text(json.dumps({"width": canvas.shape[1], "height": canvas.shape[0], "sources": ids, "objects": objects}))
    print(out, f"{canvas.shape[1]}x{canvas.shape[0]}", len(objects), "objects from", len(ids), "photos")
    return 0


def cmd_confidence(args: argparse.Namespace) -> int:
    """Precision by confidence band on COCO val2017 (deployed settings), per detector and for objects both detectors
    agree on (same class, IoU >= --agree-iou, as Image Debug's fusion marks them "confirmed by both"). Grounds the
    evidence gate for findings (config/diagnostics.json confidence.minEvidence)."""
    import collections

    import numpy as np

    from app.evaluation import records
    from app.evaluation.detection import iou_matrix

    data = paths.data_root()
    coco = adapters.CocoAdapter()
    gt = collections.defaultdict(list)
    for a in coco.gt["annotations"]:
        if not a.get("iscrowd"):
            gt[a["image_id"]].append(a)
    preds: dict[str, dict[int, list[dict]]] = {}
    for model in ("efficientdet_lite0", "yolox_s"):
        f = data / "processed" / "predictions" / f"{model}__coco_val2017__val2017__deployed.json"
        if not f.exists():
            raise SystemExit(f"run first: python tools/evaluate.py detection --model {model} --dataset coco_val2017")
        by_image = collections.defaultdict(list)
        for p in json.loads(f.read_text()):
            if p["score"] >= args.threshold:
                by_image[p["image_id"]].append(p)
        preds[model] = by_image
    xyxy = lambda b: [b[0], b[1], b[0] + b[2], b[1] + b[3]]  # noqa: E731
    bands = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]

    def band_of(score: float) -> tuple[float, float] | None:
        return next((b for b in bands if b[0] <= score < b[1]), None)

    def correct(image_id: int, dets: list[dict]) -> list[bool]:
        """Greedy matching by score (same class, IoU >= 0.5): which detections are true positives."""
        g = gt[image_id]
        used: set[int] = set()
        out = []
        for d in dets:
            best, bj = 0.5, -1
            for j, a in enumerate(g):
                if j in used or a["category_id"] != d["category_id"]:
                    continue
                v = float(iou_matrix(np.array([xyxy(d["bbox"])]), np.array([xyxy(a["bbox"])]))[0, 0])
                if v >= best:
                    best, bj = v, j
            if bj >= 0:
                used.add(bj)
            out.append(bj >= 0)
        return out

    tallies = {k: {b: [0, 0] for b in bands} for k in ("efficientdet_lite0", "yolox_s", "yolox_s_confirmed_by_both", "yolox_s_deep_only")}
    for img in coco.gt["images"]:
        iid = img["id"]
        fast = preds["efficientdet_lite0"][iid]
        for model in ("efficientdet_lite0", "yolox_s"):
            dets = sorted(preds[model][iid], key=lambda x: -x["score"])
            ok = correct(iid, dets)
            for d, c in zip(dets, ok, strict=True):
                b = band_of(d["score"])
                if b is None:
                    continue
                tallies[model][b][0] += c
                tallies[model][b][1] += 1
                if model == "yolox_s":
                    agree = any(f["category_id"] == d["category_id"] and
                                float(iou_matrix(np.array([xyxy(f["bbox"])]), np.array([xyxy(d["bbox"])]))[0, 0]) >= args.agree_iou for f in fast)
                    key = "yolox_s_confirmed_by_both" if agree else "yolox_s_deep_only"
                    tallies[key][b][0] += c
                    tallies[key][b][1] += 1
    result = {k: {f"{lo:.1f}-{min(hi, 1.0):.1f}": {"detections": n, "precision": round(tp / n, 3) if n else None} for (lo, hi), (tp, n) in t.items()}
              for k, t in tallies.items()}
    record = {
        "task": "detection-confidence-bands", "date": records.now(),
        "model": {"id": "efficientdet_lite0,yolox_s", "name": "deployed fast and deep detectors", "version": "deployed"},
        "dataset": {"id": "coco_val2017", "split": "val2017"},
        "config": {"detection_threshold": args.threshold, "agreement": f"same class, IoU >= {args.agree_iou} with a fast-detector box",
                   "match": "same class, IoU >= 0.5, greedy by score"},
        "config_name": "default", "environment": records.environment(), "metrics": result,
        "notes": ["Precision is a lower bound where COCO leaves objects unannotated."],
    }
    path = records.write(record)
    print(json.dumps(result, indent=1))
    print("  wrote", path.relative_to(ROOT))
    return 0


def cmd_mosaic(args: argparse.Namespace) -> int:
    """Large images with many small objects: 3x3 mosaics of held-out COCO val2017 images (1920x1920, ground truth
    shifted with the images). Compares one downscaled whole-image pass (what the app did) with a bounded tiled pass
    (tiles sized so that at most grid x grid tiles cover the image, plus the whole image), merged by NMM or NMS."""
    import math

    import cv2
    import numpy as np

    from app.evaluation import detection as det_eval
    from app.evaluation import records
    from app.vision.detectors.yolox import YoloxConfig, YoloxDetector
    from app.vision.tiling import TilingConfig, sliced_detect

    registry = ModelRegistry()
    entry = registry.get(args.model)
    avail = registry.availability(args.model, verify=True)
    if not avail.available:
        raise SystemExit(f"{args.model}: {avail.reason}")
    det = YoloxDetector(entry, Path(avail.path), list(adapters.COCO80))
    deployed = YoloxConfig(score_floor=_v(DETECTION["deep"]["preNmsScore"]), nms_iou=_v(DETECTION["deep"]["nmsIou"]),
                           class_agnostic=_v(DETECTION["deep"]["classAgnosticNms"]))
    coco = adapters.CocoAdapter()
    images = sorted(coco.gt["images"], key=lambda i: i["id"])[args.offset: args.offset + args.mosaics * 9]
    cell_w = cell_h = 640  # COCO images are at most 640 px on either side
    anns: dict[int, list[dict]] = {}
    for a in coco.gt["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    gt_images, gt_anns, preds = [], [], {"full": [], "tiled-nmm": [], "tiled-nms": []}
    timings: dict[str, list[float]] = {k: [] for k in preds}
    side = 3 * cell_w
    tile = max(det.native_size, math.ceil(side / (args.grid - (args.grid - 1) * 0.2)))
    configs = {"tiled-nmm": TilingConfig(tile=tile, overlap=0.2, full_image=True, merge="nmm", match_metric="ios", match_threshold=0.5),
               "tiled-nms": TilingConfig(tile=tile, overlap=0.2, full_image=True, merge="nms", match_metric="iou", match_threshold=0.5)}
    for m in range(args.mosaics):
        canvas = np.full((3 * cell_h, 3 * cell_w, 3), 40, np.uint8)
        for k, img in enumerate(images[m * 9:(m + 1) * 9]):
            bgr = cv2.imread(str(coco.image_path(img)))
            h, w = bgr.shape[:2]
            ox, oy = (k % 3) * cell_w, (k // 3) * cell_h
            canvas[oy:oy + h, ox:ox + w] = bgr
            for a in anns.get(img["id"], []):
                x, y, bw, bh = a["bbox"]
                gt_anns.append({**a, "id": len(gt_anns) + 1, "image_id": m, "bbox": [x + ox, y + oy, bw, bh]})
        gt_images.append({"id": m, "width": canvas.shape[1], "height": canvas.shape[0], "file_name": f"mosaic{m}"})
        t0 = time.perf_counter()
        full = det.detect(canvas, config=deployed)
        timings["full"].append((time.perf_counter() - t0) * 1000)
        results = {"full": full}
        for name, cfg in configs.items():
            t0 = time.perf_counter()
            results[name] = sliced_detect(lambda im: det.detect(im, config=deployed), canvas, cfg, model_id=det.model_id, model_version=det.model_version)
            timings[name].append((time.perf_counter() - t0) * 1000)
        name_to_cat = {c["name"]: c["id"] for c in coco.gt["categories"]}
        for name, res in results.items():
            for d in res.detections:
                x1, y1, x2, y2 = d.box
                preds[name].append({"image_id": m, "category_id": name_to_cat[d.class_name], "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(d.confidence)})
        print(f"  mosaic {m + 1}/{args.mosaics}", flush=True)
    gt = {"images": gt_images, "annotations": gt_anns, "categories": coco.gt["categories"]}
    cat_names = {c["id"]: c["name"] for c in coco.gt["categories"]}
    thr = _v(DETECTION["deep"]["scoreThreshold"])
    metrics = {}
    for name, res in preds.items():
        cocom = det_eval.coco_metrics(gt, res, cat_ids=None, category_names=cat_names)
        ops = det_eval.operating_points(det_eval.build_image_evals(gt, res, None, None))
        at = next((r for r in ops if abs(r["threshold"] - thr) < 1e-6), None)
        metrics[name] = {"coco": cocom["summary"], "at_operating_threshold": at, "detections_at_threshold": sum(1 for p in res if p["score"] >= thr),
                         "ms_per_image_median": round(float(np.median(timings[name])), 1),
                         "passes": 1 if name == "full" else None}
        s = cocom["summary"]
        print(f"  {name:10} AP {s['AP']:5.2f} APs {s['APs']:5.2f} | @{thr}: P {at['precision']:.3f} R {at['recall']:.3f} "
              f"| {metrics[name]['ms_per_image_median']} ms/image", flush=True)
    record = {
        "task": "detection-mosaic-tiling", "date": records.now(),
        "model": {"id": args.model, "name": entry.get("name"), "version": entry.get("version"), "weights_sha256": entry["weights"].get("sha256")},
        "dataset": {"id": "coco_val2017", "split": f"images {args.offset}-{args.offset + args.mosaics * 9 - 1} by id as {args.mosaics} 3x3 mosaics (1920x1920)"
                                                    + (" [tune]" if args.offset else " [test]"),
                    "images": args.mosaics, "instances": sum(1 for a in gt_anns if not a.get("iscrowd"))},
        "config": {"detector": "deployed (config/detection.json deep.*)", "grid": args.grid, "tile_px": tile, "overlap": 0.2,
                   "operating_threshold": thr},
        "config_name": f"grid{args.grid}", "environment": records.environment(), "metrics": metrics,
        "notes": ["Mosaics put held-out images at native resolution side by side, so a single whole-image pass sees every object at 1/3 scale;",
                  "tiles of the planned browser tiling see them at tile_px/640 of native scale."],
    }
    path = records.write(record)
    print("  wrote", path.relative_to(ROOT))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("detection")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, choices=["coco_val2017", "visdrone_det_val"])
    p.add_argument("--split", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=cmd_detection)
    p = sub.add_parser("tiling")
    p.add_argument("--model", default="yolox_s")
    p.add_argument("--dataset", required=True, choices=["coco_val2017", "visdrone_det_val"])
    p.add_argument("--split", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tiles", default="640,512,384")
    p.add_argument("--overlaps", default="0.2")
    p.add_argument("--merges", default="nmm")
    p.add_argument("--no-full", action="store_true", help="tiles only, no full-image pass")
    p.set_defaults(fn=cmd_tiling)
    p = sub.add_parser("robustness")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="coco_val2017", choices=["coco_val2017"])
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--corruptions", default="", help="comma-separated subset of app.evaluation.corruptions.CORRUPTIONS")
    p.set_defaults(fn=cmd_robustness)
    p = sub.add_parser("tracking")
    p.add_argument("--inputs", default="oracle,efficientdet_lite0,yolox_s")
    p.add_argument("--variants", default="")
    p.set_defaults(fn=cmd_tracking)
    p = sub.add_parser("vocabulary")
    p.add_argument("--model", default="yolox_s")
    p.add_argument("--wordnet-dir", default=None, help="directory containing NLTK corpora/wordnet (else NLTK's default search path)")
    p.set_defaults(fn=cmd_vocabulary)
    p = sub.add_parser("mosaic")
    p.add_argument("--model", default="yolox_s")
    p.add_argument("--mosaics", type=int, default=40)
    p.add_argument("--grid", type=int, default=3)
    p.add_argument("--offset", type=int, default=0, help="first image (by id order); 0 = test mosaics, 360 = tune mosaics")
    p.set_defaults(fn=cmd_mosaic)
    p = sub.add_parser("coverage")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--wordnet-dir", default=None)
    p.set_defaults(fn=cmd_coverage)
    p = sub.add_parser("desk")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--wordnet-dir", default=None)
    p.set_defaults(fn=cmd_desk)
    p = sub.add_parser("composite")
    p.add_argument("--out", required=True, help="output JPEG (ground truth is written next to it as .gt.json)")
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cell-w", type=int, default=640)
    p.add_argument("--cell-h", type=int, default=480)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--wordnet-dir", default=None)
    p.set_defaults(fn=cmd_composite)
    p = sub.add_parser("confidence")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--agree-iou", type=float, default=0.3, help="fusion matchIou (config/detection.json)")
    p.set_defaults(fn=cmd_confidence)
    p = sub.add_parser("quality")
    p.add_argument("--model", default="yolox_s")
    p.add_argument("--limit", type=int, default=500)
    p.set_defaults(fn=cmd_quality)
    p = sub.add_parser("anomaly")
    p.add_argument("--categories", default="", help="comma-separated VisA categories (default: all 12)")
    p.add_argument("--configs", default="", help="comma-separated subset of the configurations")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_anomaly)
    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
