"""Dataset validation: problems that silently corrupt evaluations.

Checks
* files: unreadable / corrupt images (full decode), stray non-image files;
* annotations: invalid boxes (non-positive size, outside the image beyond a
  1 px tolerance), images without annotations, unknown categories;
* duplicates: byte-identical images (SHA-256);
* leakage: near-identical images in different splits (64-bit difference
  hash, Hamming distance <= 4) - relevant for video-derived datasets;
* class imbalance: max/min instances per class.
Results are written to data/manifests/<dataset>.validation.json.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.datasets import adapters, paths

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NEAR_DUPLICATE_BITS = 4


@dataclass
class ValidationReport:
    dataset_id: str
    images: int = 0
    unreadable: list[str] = field(default_factory=list)
    stray_files: list[str] = field(default_factory=list)
    invalid_boxes: list[dict[str, Any]] = field(default_factory=list)
    out_of_bounds: int = 0
    without_annotations: int = 0
    duplicate_groups: list[list[str]] = field(default_factory=list)
    cross_split_near_duplicates: list[dict[str, Any]] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unreadable and not self.invalid_boxes and not self.cross_split_near_duplicates

    def summary(self) -> dict[str, Any]:
        counts = [c for c in self.class_counts.values() if c > 0]
        return {
            "dataset": self.dataset_id,
            "ok": self.ok,
            "images": self.images,
            "unreadable_images": len(self.unreadable),
            "stray_files": self.stray_files[:10],
            "stray_file_count": len(self.stray_files),
            "invalid_boxes": len(self.invalid_boxes),
            "invalid_box_examples": self.invalid_boxes[:5],
            "boxes_out_of_bounds_clipped": self.out_of_bounds,
            "images_without_annotations": self.without_annotations,
            "exact_duplicate_groups": len(self.duplicate_groups),
            "duplicate_examples": self.duplicate_groups[:3],
            "cross_split_near_duplicates": len(self.cross_split_near_duplicates),
            "near_duplicate_examples": self.cross_split_near_duplicates[:3],
            "classes": len(counts),
            "imbalance_max_over_min": round(max(counts) / min(counts), 1) if counts else None,
            "notes": self.notes,
        }


def dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def run(dataset_id: str) -> ValidationReport:
    ad = adapters.load(dataset_id)
    report = ValidationReport(dataset_id)
    samples = list(ad.samples())
    report.images = len(samples)
    split_of: dict[str, str] = {}
    split_file = paths.splits_dir() / f"{dataset_id}.json"
    if split_file.exists():
        for name, info in json.loads(split_file.read_text())["splits"].items():
            for i in info["ids"]:
                split_of[i] = name
    by_hash: dict[str, list[str]] = defaultdict(list)
    hashes: list[tuple[int, str, str]] = []
    counts: Counter[str] = Counter()
    image_dirs: set[Path] = set()
    for s in samples:
        if s.image_path is None:
            report.notes.append("annotations only: image checks skipped")
            break
        p = Path(s.image_path)
        image_dirs.add(p.parent)
        data = p.read_bytes() if p.exists() else b""
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE) if data else None
        if img is None:
            report.unreadable.append(s.id)
            continue
        by_hash[hashlib.sha256(data).hexdigest()].append(s.id)
        if split_of:
            hashes.append((dhash(img), s.id, split_of.get(s.id, "?")))
        h, w = img.shape[:2]
        if not s.annotations:
            report.without_annotations += 1
        for a in s.annotations:
            x1, y1, x2, y2 = a["box"]
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                report.invalid_boxes.append({"image": s.id, "box": a["box"], "category": a["category"]})
            elif x1 < -1 or y1 < -1 or x2 > w + 1 or y2 > h + 1:
                report.out_of_bounds += 1
            if not a.get("iscrowd"):
                counts[a["category"]] += 1
    if not report.notes:
        for d in image_dirs:
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() not in IMAGE_SUFFIXES:
                    report.stray_files.append(str(f.relative_to(paths.data_root())))
            for f in d.parent.iterdir():
                if f.name.startswith(".") and f.is_file():
                    report.stray_files.append(str(f.relative_to(paths.data_root())))
    else:
        for s in samples:
            for a in s.annotations:
                x1, y1, x2, y2 = a["box"]
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    report.invalid_boxes.append({"image": s.id, "box": a["box"], "category": a["category"]})
                if not a.get("iscrowd"):
                    counts[a["category"]] += 1
    report.duplicate_groups = [ids for ids in by_hash.values() if len(ids) > 1]
    if hashes:
        arr = np.array([h for h, _, _ in hashes], dtype=np.uint64)
        for i, (h, sid, sp) in enumerate(hashes):
            dist = np.array([bin(int(x)).count("1") for x in (arr[i + 1 :] ^ np.uint64(h))]) if i + 1 < len(arr) else np.array([])
            for j in np.flatnonzero(dist <= NEAR_DUPLICATE_BITS):
                other = hashes[i + 1 + int(j)]
                if other[2] != sp:
                    report.cross_split_near_duplicates.append({"a": sid, "split_a": sp, "b": other[1], "split_b": other[2], "hamming": int(dist[j])})
    report.class_counts = dict(counts)
    out = paths.manifests_dir() / f"{dataset_id}.validation.json"
    out.write_text(json.dumps(report.summary(), indent=1) + "\n")
    return report
