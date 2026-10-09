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
    p = sub.add_parser("anomaly")
    p.add_argument("--categories", default="", help="comma-separated VisA categories (default: all 12)")
    p.add_argument("--configs", default="", help="comma-separated subset of the configurations")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_anomaly)
    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
