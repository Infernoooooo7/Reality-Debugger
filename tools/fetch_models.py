#!/usr/bin/env python3
"""Download the local detectors, verify them and write their manifests.

Every manifest's label list is read from the model's own metadata (the
labels file packed inside the TFLite model, or the class list in the
official repository the ONNX model was exported from) - never typed in by
hand. The frontend loads these manifests at runtime and
``build_ontology.py`` derives the semantic ontology from them.

Usage:  python tools/fetch_models.py            (download what is missing, rebuild manifests)
        python tools/fetch_models.py --fetch    (download what is missing, keep the committed manifests)
        python tools/fetch_models.py --check    (verify files + manifests, no network)

Only the small fast model is committed; the deep model (YOLOX-S, 36 MB) is
fetched with ``--fetch`` (the Docker build does this) and verified by SHA-256.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "frontend" / "public" / "models"

MODELS = [
    {
        "id": "efficientdet_lite0",
        "name": "EfficientDet-Lite0 (int8)",
        "role": "fast",
        "file": "efficientdet_lite0.tflite",
        "url": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
        "sha256": "0720bf247bd76e6594ea28fa9c6f7c5242be774818997dbbeffc4da460c723bb",
        "runtime": "mediapipe-tasks-vision",
        "license": "Apache-2.0",
        "labels_from": "tflite-metadata",
        "dataset": "COCO 2017 (80 thing categories)",
        "paper": "M. Tan, R. Pang, Q. V. Le, EfficientDet: Scalable and Efficient Object Detection, CVPR 2020, arXiv:1911.09070",
        "reported": {
            "coco_val2017_ap": 25.69,
            "size_mb_int8": 4.4,
            "latency_ms_pixel4_cpu_4threads": 37,
            "source": "TensorFlow Lite Model Maker object detection guide (EfficientDet-Lite table)",
        },
        "input": {
            "width": 320,
            "height": 320,
            "layout": "NHWC",
            "source": "TFLite input tensor shape [1, 320, 320, 3] (read with the tflite flatbuffer schema)",
        },
        "postprocess": {
            "nms_iou": 0.3,
            "nms_class_agnostic": True,
            "source": "MediaPipe object_detector_options.proto defaults (min_suppression_threshold=0.3, multiclass_nms=false); the model has no in-model NMS",
        },
    },
    {
        "id": "yolox_s",
        "name": "YOLOX-S",
        "role": "deep",
        "file": "yolox_s.onnx",
        "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
        "sha256": "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
        "runtime": "onnxruntime-web",
        "license": "Apache-2.0",
        "labels_from": "https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/0.1.1rc0/yolox/data/datasets/coco_classes.py",
        "dataset": "COCO 2017 (80 thing categories)",
        "paper": "Z. Ge, S. Liu, F. Wang, Z. Li, J. Sun, YOLOX: Exceeding YOLO Series in 2021, arXiv:2107.08430",
        "reported": {
            "coco_val2017_map": 40.5,
            "params_m": 9.0,
            "gflops": 26.8,
            "source": "YOLOX README (Standard Models table)",
        },
        "input": {
            "width": 640,
            "height": 640,
            "layout": "NCHW",
            "channel_order": "BGR",
            "pad_value": 114,
            "placement": "top-left",
            "normalize": "none (0-255 floats)",
            "source": "YOLOX demo/ONNXRuntime/onnx_inference.py + yolox/data/data_augment.py preproc()",
        },
        "decode": {"strides": [8, 16, 32], "format": "cx,cy,w,h,objectness,80 class scores", "source": "yolox/utils/demo_utils.py demo_postprocess()"},
        "postprocess": {
            "pre_nms_score": 0.1,
            "nms_iou": 0.45,
            "nms_class_agnostic": True,
            "display_score": 0.3,
            "source": "YOLOX ONNX demo: multiclass_nms(nms_thr=0.45, score_thr=0.1), class_agnostic=True; visualisation conf 0.3",
        },
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as out:  # noqa: S310 - fixed https URLs
        while chunk := resp.read(1 << 20):
            out.write(chunk)


def tflite_labels(path: Path) -> list[str | None]:
    """TFLite metadata packs associated files (labels.txt) as a zip appended to the model."""
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".txt"))
        lines = zf.read(name).decode("utf-8").splitlines()
    # '???' marks unused slots of the original 91-id COCO label space; keep them as
    # None so indices still line up with the model's output columns.
    return [None if line.strip() in ("", "???") else line.strip() for line in lines]


def repo_class_list(url: str) -> list[str | None]:
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https URL
        source = resp.read().decode("utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "COCO_CLASSES":
            return list(ast.literal_eval(node.value))
    raise RuntimeError(f"COCO_CLASSES not found in {url}")


def manifest_path(model: dict) -> Path:
    return MODEL_DIR / f"{model['id']}.manifest.json"


def build(check_only: bool, keep_manifests: bool = False) -> int:
    problems = 0
    for model in MODELS:
        path = MODEL_DIR / model["file"]
        print(f"{model['id']}: {path.relative_to(ROOT)}")
        if not path.exists():
            if check_only:
                print("  MISSING"); problems += 1; continue
            tmp = path.with_suffix(path.suffix + ".part")
            download(model["url"], tmp)
            tmp.replace(path)
        digest = sha256(path)
        if digest != model["sha256"]:
            print(f"  SHA-256 mismatch: {digest}"); problems += 1
            if not check_only:
                path.unlink()  # never keep (or serve) a file that is not the published model
            continue
        if check_only or keep_manifests:
            labels = json.loads(manifest_path(model).read_text())["labels"]
        elif model["labels_from"] == "tflite-metadata":
            labels = tflite_labels(path)
        else:
            labels = repo_class_list(model["labels_from"])
        manifest = {
            **{k: v for k, v in model.items() if k not in ("labels_from",)},
            "bytes": path.stat().st_size,
            "labels_source": "labels.txt packed in the TFLite model metadata" if model["labels_from"] == "tflite-metadata" else model["labels_from"],
            "num_classes": sum(1 for label in labels if label),
            "labels": labels,
        }
        if not (check_only or keep_manifests):
            manifest_path(model).write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  ok - {manifest['num_classes']} classes, {manifest['bytes'] / 1e6:.1f} MB")
    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify only; no downloads or writes")
    parser.add_argument("--fetch", action="store_true", help="download missing models; keep the committed manifests")
    args = parser.parse_args()
    sys.exit(1 if build(args.check, keep_manifests=args.fetch) else 0)
