#!/usr/bin/env python3
"""Data engine CLI: frames -> model pre-annotations -> review queue -> labels -> versioned dataset.

Annotation happens in CVAT or Label Studio (both import COCO and YOLO); this
tool prepares their input and versions their output. Run it with the
backend's Python environment (it uses the same models and code as the app).

    python tools/annotate.py frames clip.mp4 --out data/interim/mydata/images --every 1.0
    python tools/annotate.py predict data/interim/mydata/images --out data/interim/mydata/predictions.json [--mode precision]
    python tools/annotate.py export data/interim/mydata/predictions.json --format coco --out data/interim/mydata/preannotations.json
    python tools/annotate.py export data/interim/mydata/predictions.json --format yolo --out data/interim/mydata/labels
    python tools/annotate.py queue data/interim/mydata/predictions.json --out data/interim/mydata/review_queue.json --limit 200
    python tools/annotate.py version data/processed/mydata --name mydata --version 1 [--coco annotations.json] [--splits splits.json]

Images stay local: nothing is uploaded anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.datasets import annotation as ann  # noqa: E402
from app.datasets import paths  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def cmd_frames(args: argparse.Namespace) -> int:
    kept = ann.extract_frames(Path(args.video), Path(args.out), every_s=args.every, max_frames=args.max)
    (Path(args.out) / "frames.json").write_text(json.dumps(kept, indent=1))
    print(f"kept {len(kept)} frames (near-duplicates dropped) -> {args.out}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    import cv2

    from app.config import Settings
    from app.runtime_config import load_config
    from app.services.inference_service import VisionService
    from app.services.ontology import load_ontology

    service = VisionService(Settings(), load_config(), load_ontology())
    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    out = []
    for i, path in enumerate(files, 1):
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"  skipped unreadable {path.name}")
            continue
        result = service.detect(bgr, mode=args.mode, profile=args.profile)
        out.append({"file": path.name, "dhash": f"{ann.dhash_file(path):016x}", "response": result})
        if i % 25 == 0:
            print(f"  {i}/{len(files)}")
    Path(args.out).write_text(json.dumps({"model": out[0]["response"]["model"] if out else None, "images": out}))
    print(f"predictions for {len(out)} images -> {args.out}")
    return 0


def _load_predictions(path: str, include_tentative: bool) -> tuple[list[ann.ImageDetections], list[str], dict]:
    data = json.loads(Path(path).read_text())
    images = [ann.from_detect_response(it["file"], it["response"], include_tentative=include_tentative) for it in data["images"]]
    from app.vision.registry import ModelRegistry

    model_id = (data.get("model") or {}).get("id", "yolox_s")
    categories = [c for c in ModelRegistry().labels(model_id) if c]
    return images, categories, data


def cmd_export(args: argparse.Namespace) -> int:
    images, categories, _ = _load_predictions(args.predictions, args.include_tentative)
    if args.format == "coco":
        Path(args.out).write_text(json.dumps(ann.to_coco(images, categories), indent=1))
    else:
        ann.to_yolo(images, categories, Path(args.out))
    print(f"{args.format} pre-annotations for {len(images)} images -> {args.out} (review every box before training)")
    return 0


def cmd_queue(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.predictions).read_text())
    items = [{"file": it["file"], "dhash": it.get("dhash"),
              "uncertainty": ann.uncertainty(it["response"]["objects"], it["response"]["model"]["operating_threshold"]),
              "objects": len(it["response"]["objects"])} for it in data["images"]]
    queue = ann.review_queue(items, limit=args.limit)
    Path(args.out).write_text(json.dumps(queue, indent=1))
    print(f"review queue of {len(queue)} images (most uncertain first) -> {args.out}")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    splits = json.loads(Path(args.splits).read_text()) if args.splits else None
    manifest = ann.version_manifest(Path(args.root), name=args.name, version=args.version,
                                    coco_json=Path(args.coco) if args.coco else None, splits=splits)
    out = paths.manifests_dir() / f"{args.name}-v{args.version}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1))
    print(f"{args.name} v{args.version}: {manifest['files']} files, content {manifest['content_sha256'][:16]}... -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("frames")
    p.add_argument("video")
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=float, default=1.0, help="seconds between sampled frames")
    p.add_argument("--max", type=int, default=500)
    p.set_defaults(fn=cmd_frames)
    p = sub.add_parser("predict")
    p.add_argument("images")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", default="standard", choices=["standard", "precision"])
    p.add_argument("--profile", default="general")
    p.set_defaults(fn=cmd_predict)
    p = sub.add_parser("export")
    p.add_argument("predictions")
    p.add_argument("--format", choices=["coco", "yolo"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--include-tentative", action="store_true", help="also export boxes below the operating threshold")
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("queue")
    p.add_argument("predictions")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=100)
    p.set_defaults(fn=cmd_queue)
    p = sub.add_parser("version")
    p.add_argument("root")
    p.add_argument("--name", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--coco", default=None)
    p.add_argument("--splits", default=None, help='JSON {"train": [files], "val": [...], "test": [...]}')
    p.set_defaults(fn=cmd_version)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
