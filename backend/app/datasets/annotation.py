"""Data engine: from raw images or video to reviewed, versioned training data.

Annotation itself is done in an existing tool (CVAT or Label Studio both
import COCO and YOLO); this module produces what those tools need and what
comes back from them:

* ``extract_frames`` - sample a video at a fixed interval, dropping
  near-duplicate frames (difference hash), so a dataset is not dominated by
  one static shot.
* ``to_coco`` / ``to_yolo`` - write model detections (``/api/vision/detect``
  responses or evaluation predictions) as pre-annotations for review.
  Pre-annotations are suggestions: every box must be checked by a person.
* ``review_queue`` - rank unlabelled images for annotation (active learning):
  most uncertain first (ambiguous and tentative detections, low class
  margins), with near-duplicates spread out so one burst of frames cannot
  fill the queue.
* ``version_manifest`` - freeze a dataset folder as a version: every file's
  SHA-256, label statistics and the split it belongs to.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from app.datasets.validate import dhash

NEAR_DUPLICATE_BITS = 6  # dHash Hamming distance treated as "the same shot" (64-bit hash)


# -- frames ---------------------------------------------------------------------------------------


def extract_frames(video: Path, out_dir: Path, *, every_s: float = 1.0, max_frames: int = 500,
                   dedupe_bits: int = NEAR_DUPLICATE_BITS) -> list[dict[str, Any]]:
    """Save one JPEG every ``every_s`` seconds, skipping frames that look like the previous kept one."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * every_s))
    out_dir.mkdir(parents=True, exist_ok=True)
    kept: list[dict[str, Any]] = []
    last_hash: int | None = None
    index = 0
    try:
        while len(kept) < max_frames:
            ok = cap.grab()
            if not ok:
                break
            if index % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                h = dhash(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if last_hash is None or (h ^ last_hash).bit_count() > dedupe_bits:
                    name = f"{video.stem}_{index:07d}.jpg"
                    cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    kept.append({"file": name, "frame": index, "time_s": round(index / fps, 3), "dhash": f"{h:016x}"})
                    last_hash = h
            index += 1
    finally:
        cap.release()
    return kept


# -- pre-annotations ------------------------------------------------------------------------------


@dataclass(slots=True)
class ImageDetections:
    file: str
    width: int
    height: int
    objects: list[dict[str, Any]]  # {"label", "box": [x1, y1, x2, y2] px, "confidence", "state"?}


def from_detect_response(file: str, response: dict[str, Any], *, include_tentative: bool = False) -> ImageDetections:
    """Read a /api/vision/detect response (or the same shape saved to disk)."""
    objs = [o for o in response["objects"] if include_tentative or o.get("state") != "tentative"]
    return ImageDetections(file, int(response["image"]["width"]), int(response["image"]["height"]), objs)


def to_coco(images: Sequence[ImageDetections], categories: Sequence[str]) -> dict[str, Any]:
    """COCO detection JSON; boxes become [x, y, w, h]; the model score is kept as an extra field."""
    cat_id = {name: i + 1 for i, name in enumerate(categories)}
    out_images, annotations = [], []
    for image_id, img in enumerate(images, 1):
        out_images.append({"id": image_id, "file_name": img.file, "width": img.width, "height": img.height})
        for o in img.objects:
            if o["label"] not in cat_id:
                continue
            x1, y1, x2, y2 = (float(v) for v in o["box"])
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": cat_id[o["label"]],
                                "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)], "area": round(w * h, 2),
                                "iscrowd": 0, "score": o.get("confidence"), "review": "pending"})
    return {
        "info": {"description": "Reality Debugger pre-annotations - model suggestions, every box needs human review"},
        "images": out_images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n, "supercategory": ""} for n, i in cat_id.items()],
    }


def to_yolo(images: Sequence[ImageDetections], categories: Sequence[str], out_dir: Path) -> list[Path]:
    """One .txt per image: ``class cx cy w h`` normalised to 0..1 (Ultralytics/Darknet layout), plus classes.txt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {name: i for i, name in enumerate(categories)}
    written = []
    for img in images:
        lines = []
        for o in img.objects:
            if o["label"] not in index:
                continue
            x1, y1, x2, y2 = (float(v) for v in o["box"])
            x1, x2 = max(0.0, min(x1, img.width)), max(0.0, min(x2, img.width))
            y1, y2 = max(0.0, min(y1, img.height)), max(0.0, min(y2, img.height))
            if x2 <= x1 or y2 <= y1:
                continue
            cx, cy = (x1 + x2) / 2 / img.width, (y1 + y2) / 2 / img.height
            lines.append(f"{index[o['label']]} {cx:.6f} {cy:.6f} {(x2 - x1) / img.width:.6f} {(y2 - y1) / img.height:.6f}")
        path = out_dir / f"{Path(img.file).stem}.txt"
        path.write_text("\n".join(lines) + ("\n" if lines else ""))
        written.append(path)
    (out_dir / "classes.txt").write_text("\n".join(categories) + "\n")
    return written


def from_yolo(label_file: Path, categories: Sequence[str], width: int, height: int) -> list[dict[str, Any]]:
    """Read YOLO labels back to pixel boxes (round trip check and import of reviewed labels)."""
    objs = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, w, h = int(parts[0]), *(float(v) for v in parts[1:])
        objs.append({"label": categories[c], "box": [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height]})
    return objs


# -- active learning ------------------------------------------------------------------------------


def uncertainty(objects: Iterable[dict[str, Any]], operating_threshold: float = 0.3) -> float:
    """How much a person's label would teach the model about this image (0 = nothing to learn).

    Sum over detections of: ambiguous -> 1, tentative -> closeness of the score to the threshold,
    plus 1 - class margin ratio when the second-best class is known. Images without any detection
    get a small base value, so they are reviewed eventually (they may hold missed objects).
    """
    total = 0.0
    n = 0
    for o in objects:
        n += 1
        state = o.get("state")
        if state == "ambiguous":
            total += 1.0
        elif state == "tentative":
            total += max(0.0, 1.0 - abs(float(o.get("confidence", 0)) - operating_threshold) / operating_threshold)
        second = (o.get("uncertainty") or {}).get("second_class_score")
        conf = float(o.get("confidence", 0) or 0)
        if second is not None and conf > 0:
            total += min(1.0, float(second) / conf) * 0.5
    return round(total if n else 0.25, 4)


def review_queue(items: Sequence[dict[str, Any]], *, limit: int = 100, dedupe_bits: int = NEAR_DUPLICATE_BITS) -> list[dict[str, Any]]:
    """Order images for annotation: most uncertain first, near-duplicates of a picked image pushed back.

    items: {"file", "uncertainty", "dhash" (int or hex str, optional)}.
    """
    pool = sorted(items, key=lambda it: -float(it["uncertainty"]))
    picked: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    hashes: list[int] = []
    for it in pool:
        h = it.get("dhash")
        h = int(h, 16) if isinstance(h, str) else h
        if h is not None and any((h ^ p).bit_count() <= dedupe_bits for p in hashes):
            deferred.append({**it, "near_duplicate_of_queued": True})
            continue
        picked.append(it)
        if h is not None:
            hashes.append(h)
        if len(picked) >= limit:
            break
    return (picked + deferred)[:limit]


# -- versioning -----------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def version_manifest(root: Path, *, name: str, version: str, coco_json: Path | None = None,
                     splits: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """Freeze a dataset folder: per-file SHA-256, label counts per split, and a content hash of the whole version."""
    files = sorted(p for p in root.rglob("*") if p.is_file())
    entries = [{"path": str(p.relative_to(root)), "sha256": _sha256(p), "bytes": p.stat().st_size} for p in files]
    content = hashlib.sha256("".join(e["path"] + e["sha256"] for e in entries).encode()).hexdigest()
    labels: dict[str, Any] = {}
    if coco_json is not None:
        coco = json.loads(coco_json.read_text())
        names = {c["id"]: c["name"] for c in coco["categories"]}
        split_of = {f: s for s, fs in (splits or {}).items() for f in fs}
        file_of = {im["id"]: im["file_name"] for im in coco["images"]}
        per_split: dict[str, Counter[str]] = {}
        for a in coco["annotations"]:
            split = split_of.get(file_of[a["image_id"]], "unsplit")
            per_split.setdefault(split, Counter())[names[a["category_id"]]] += 1
        labels = {s: dict(c) for s, c in per_split.items()}
    return {"name": name, "version": version, "content_sha256": content, "files": len(entries),
            "bytes": sum(e["bytes"] for e in entries), "labels_per_split": labels, "splits": {k: len(v) for k, v in (splits or {}).items()},
            "entries": entries}


def image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"unreadable image {path}")
    return int(img.shape[1]), int(img.shape[0])


def dhash_file(path: Path) -> int:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"unreadable image {path}")
    return dhash(img)
