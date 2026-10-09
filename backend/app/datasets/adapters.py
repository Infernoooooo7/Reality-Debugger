"""Dataset adapters: read each dataset's native format into normalised samples.

Boxes are ``[x1, y1, x2, y2]`` in pixels (origin top-left). Every adapter can
also produce COCO-format ground truth (``coco_gt``) so one evaluation path
(pycocotools) serves all detection datasets; crowd / ignore regions are kept
as ``iscrowd=1`` so they are neither true nor false positives.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import Any

from app.datasets import paths
from app.vision.interfaces import Sample

# COCO 2017 thing categories in contiguous model order (index = YOLOX/EfficientDet class id).
COCO80: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


class DatasetMissing(RuntimeError):
    pass


def _require(path: Path, dataset_id: str) -> Path:
    if not path.exists():
        raise DatasetMissing(f"{dataset_id}: {path} not found - run: python tools/datasets.py fetch {dataset_id}")
    return path


# -- COCO -----------------------------------------------------------------------------------------


class CocoAdapter:
    """COCO 2017 val with the official instances_val2017.json."""

    dataset_id = "coco_val2017"

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or paths.raw_dir(self.dataset_id) / "coco"
        self.annotation_file = _require(self.root / "annotations" / "instances_val2017.json", self.dataset_id)
        self.image_dir = self.root / "images" / "val2017"

    @cached_property
    def gt(self) -> dict[str, Any]:
        return json.loads(self.annotation_file.read_text())

    @property
    def categories(self) -> list[str]:
        return [c["name"] for c in sorted(self.gt["categories"], key=lambda c: c["id"])]

    @cached_property
    def name_to_category_id(self) -> dict[str, int]:
        return {c["name"]: c["id"] for c in self.gt["categories"]}

    def image_path(self, image: dict[str, Any]) -> Path:
        return self.image_dir / image["file_name"]

    def coco_gt(self, image_ids: set[int] | None = None) -> dict[str, Any]:
        if image_ids is None:
            return self.gt
        return {
            "images": [i for i in self.gt["images"] if i["id"] in image_ids],
            "annotations": [a for a in self.gt["annotations"] if a["image_id"] in image_ids],
            "categories": self.gt["categories"],
        }

    def samples(self, split: str | None = None) -> Iterator[Sample]:
        by_image: dict[int, list[dict[str, Any]]] = {}
        cats = {c["id"]: c["name"] for c in self.gt["categories"]}
        for a in self.gt["annotations"]:
            x, y, w, h = a["bbox"]
            by_image.setdefault(a["image_id"], []).append(
                {"category": cats[a["category_id"]], "box": [x, y, x + w, y + h], "iscrowd": a["iscrowd"], "area": a["area"]}
            )
        for img in sorted(self.gt["images"], key=lambda i: i["id"]):
            yield Sample(str(img["id"]), str(self.image_path(img)), img["width"], img["height"], by_image.get(img["id"], []))

    def stats(self) -> dict[str, Any]:
        anns = self.gt["annotations"]
        cats = {c["id"]: c["name"] for c in self.gt["categories"]}
        per_class = Counter(cats[a["category_id"]] for a in anns if not a["iscrowd"])
        sizes = Counter("small" if a["area"] < 32**2 else "medium" if a["area"] < 96**2 else "large" for a in anns if not a["iscrowd"])
        return {
            "images": len(self.gt["images"]),
            "instances": len(anns),
            "crowd_regions": sum(1 for a in anns if a["iscrowd"]),
            "categories": len(self.gt["categories"]),
            "images_without_annotations": len({i["id"] for i in self.gt["images"]} - {a["image_id"] for a in anns}),
            "size_distribution": dict(sizes),
            "most_common": per_class.most_common(5),
            "least_common": per_class.most_common()[-5:],
            "imbalance_ratio_max_min": round(per_class.most_common(1)[0][1] / per_class.most_common()[-1][1], 1),
        }


# -- LVIS (official json on COCO images) ----------------------------------------------------------


class LvisAdapter:
    """LVIS v1 minival (the 5000 COCO val2017 images) with the official json."""

    dataset_id = "lvis_v1_labels"

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or paths.raw_dir(self.dataset_id) / "lvis"
        self.annotation_file = _require(self.root / "annotations" / "lvis_v1_minival.json", self.dataset_id)

    @cached_property
    def gt(self) -> dict[str, Any]:
        return json.loads(self.annotation_file.read_text())

    @property
    def categories(self) -> list[str]:
        return [c["name"] for c in sorted(self.gt["categories"], key=lambda c: c["id"])]

    def samples(self, split: str | None = None) -> Iterator[Sample]:
        by_image: dict[int, list[dict[str, Any]]] = {}
        cats = {c["id"]: c for c in self.gt["categories"]}
        for a in self.gt["annotations"]:
            x, y, w, h = a["bbox"]
            c = cats[a["category_id"]]
            by_image.setdefault(a["image_id"], []).append(
                {"category": c["name"], "synset": c["synset"], "frequency": c["frequency"], "box": [x, y, x + w, y + h], "area": a["area"], "iscrowd": 0}
            )
        for img in sorted(self.gt["images"], key=lambda i: i["id"]):
            yield Sample(str(img["id"]), None, img["width"], img["height"], by_image.get(img["id"], []),
                         meta={"coco_url": img.get("coco_url"), "neg_category_ids": img.get("neg_category_ids", []),
                               "not_exhaustive_category_ids": img.get("not_exhaustive_category_ids", [])})

    def stats(self) -> dict[str, Any]:
        freq = Counter(c["frequency"] for c in self.gt["categories"])
        return {"images": len(self.gt["images"]), "instances": len(self.gt["annotations"]), "categories": len(self.gt["categories"]),
                "category_frequency_groups": dict(freq), "annotation": "federated (not exhaustive per image)"}


# -- VisDrone -------------------------------------------------------------------------------------

VISDRONE_CLASSES = ("ignored", "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor", "others")
#: VisDrone category -> COCO-80 name for evaluating COCO detectors. None = no COCO equivalent (counted as unsupported).
VISDRONE_TO_COCO: dict[str, str | None] = {
    "pedestrian": "person", "people": "person", "bicycle": "bicycle", "car": "car", "van": "car", "truck": "truck",
    "tricycle": None, "awning-tricycle": None, "bus": "bus", "motor": "motorcycle",
}
VISDRONE_EVAL_CATEGORIES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")


class VisDroneAdapter:
    """VisDrone2019-DET val in the original format.

    Annotation line: bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion.
    Category 0 ("ignored regions") and 11 ("others") become ignore regions (iscrowd=1).
    """

    dataset_id = "visdrone_det_val"

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or paths.raw_dir(self.dataset_id) / "VisDrone2019-DET-val"
        _require(self.root / "annotations", self.dataset_id)

    @property
    def categories(self) -> list[str]:
        return list(VISDRONE_CLASSES[1:11])

    @staticmethod
    def group_of(stem: str) -> str:
        """Images come from video clips: '0000001_02999_d_0000005' -> clip '0000001'."""
        return stem.split("_")[0]

    def _image_size(self, path: Path) -> tuple[int, int]:
        from PIL import Image

        with Image.open(path) as im:
            return im.size

    @cached_property
    def _samples(self) -> list[Sample]:
        out = []
        for ann_path in sorted((self.root / "annotations").glob("*.txt")):
            img_path = self.root / "images" / (ann_path.stem + ".jpg")
            if not img_path.exists():
                continue
            w, h = self._image_size(img_path)
            anns = []
            for line in ann_path.read_text().splitlines():
                parts = [int(p) for p in line.strip().rstrip(",").split(",")[:8]]
                if len(parts) < 8:
                    continue
                x, y, bw, bh, _score, cat, trunc, occ = parts
                name = VISDRONE_CLASSES[cat] if 0 <= cat < len(VISDRONE_CLASSES) else "others"
                ignore = name in ("ignored", "others")
                anns.append({"category": name, "box": [x, y, x + bw, y + bh], "iscrowd": int(ignore), "area": bw * bh,
                             "truncation": trunc, "occlusion": occ})
            out.append(Sample(ann_path.stem, str(img_path), w, h, anns, group=self.group_of(ann_path.stem)))
        return out

    def samples(self, split: str | None = None) -> Iterator[Sample]:
        ids = load_split(self.dataset_id, split) if split else None
        for s in self._samples:
            if ids is None or s.id in ids:
                yield s

    def coco_gt(self, image_ids: set[str] | None = None) -> dict[str, Any]:
        """COCO-format GT for evaluating COCO-80 detectors on the VisDrone classes they can name.

        Categories are the COCO names in VISDRONE_EVAL_CATEGORIES (with COCO-80 order ids). Ignored regions,
        "others" and classes without a COCO equivalent (tricycles) become ignore regions for every evaluated
        category, so a detector is neither rewarded nor penalised there (COCO crowd semantics are per class).
        """
        categories = [{"id": COCO80.index(n) + 1, "name": n} for n in VISDRONE_EVAL_CATEGORIES]
        name_to_id = {c["name"]: c["id"] for c in categories}
        images, annotations = [], []
        for idx, s in enumerate(self._samples):
            if image_ids is not None and s.id not in image_ids:
                continue
            images.append({"id": idx, "file_name": s.image_path, "width": s.width, "height": s.height, "visdrone_id": s.id, "group": s.group})
            for a in s.annotations:
                x1, y1, x2, y2 = a["box"]
                bbox, area = [x1, y1, x2 - x1, y2 - y1], (x2 - x1) * (y2 - y1)
                target = None if a["iscrowd"] else VISDRONE_TO_COCO.get(a["category"])
                if target is None:
                    for cat_id in name_to_id.values():
                        annotations.append({"id": len(annotations) + 1, "image_id": idx, "category_id": cat_id, "bbox": bbox, "area": area,
                                            "iscrowd": 1, "source_category": a["category"]})
                else:
                    annotations.append({"id": len(annotations) + 1, "image_id": idx, "category_id": name_to_id[target], "bbox": bbox,
                                        "area": area, "iscrowd": 0, "source_category": a["category"]})
        return {"images": images, "annotations": annotations, "categories": categories}

    def stats(self) -> dict[str, Any]:
        per_class = Counter(a["category"] for s in self._samples for a in s.annotations)
        sizes = Counter("small" if a["area"] < 32**2 else "medium" if a["area"] < 96**2 else "large"
                        for s in self._samples for a in s.annotations if not a["iscrowd"])
        resolutions = Counter(f"{s.width}x{s.height}" for s in self._samples)
        return {"images": len(self._samples), "clips": len({s.group for s in self._samples}), "instances": sum(per_class.values()),
                "per_class": dict(per_class), "size_distribution": dict(sizes), "resolutions": dict(resolutions.most_common(5))}


# -- KITTI tracking -------------------------------------------------------------------------------

KITTI_TRACKING_CLASSES = ("Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare")


class KittiTrackingAdapter:
    """KITTI tracking training labels (label_02) and the downloaded image sequences.

    Label line: frame track_id type truncated occluded alpha left top right bottom h w l x y z ry
    """

    dataset_id = "kitti_tracking"
    fps = 10.0  # KITTI raw sequences are recorded at 10 Hz

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or paths.raw_dir(self.dataset_id) / "training"
        _require(self.root / "label_02", self.dataset_id)

    @property
    def categories(self) -> list[str]:
        return list(KITTI_TRACKING_CLASSES)

    def sequences(self) -> list[str]:
        return sorted(p.name for p in (self.root / "image_02").iterdir() if p.is_dir() and any(p.iterdir()))

    def frames(self, sequence: str) -> list[Path]:
        return sorted((self.root / "image_02" / sequence).glob("*.png"))

    def labels(self, sequence: str) -> dict[int, list[dict[str, Any]]]:
        out: dict[int, list[dict[str, Any]]] = {}
        for line in (self.root / "label_02" / f"{sequence}.txt").read_text().splitlines():
            f = line.split()
            frame, track, cls = int(f[0]), int(f[1]), f[2]
            box = [float(v) for v in f[6:10]]
            out.setdefault(frame, []).append({"track_id": track, "category": cls, "truncated": float(f[3]), "occluded": int(f[4]), "box": box,
                                              "iscrowd": int(cls == "DontCare")})
        return out

    def samples(self, split: str | None = None) -> Iterator[Sample]:
        for seq in self.sequences():
            labels = self.labels(seq)
            for frame_path in self.frames(seq):
                frame = int(frame_path.stem)
                yield Sample(f"{seq}/{frame:06d}", str(frame_path), 1242, 375, labels.get(frame, []), group=seq, meta={"frame": frame})

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"sequences": {}}
        for seq in self.sequences():
            labels = self.labels(seq)
            per_class = Counter(a["category"] for anns in labels.values() for a in anns)
            tracks = {(a["category"], a["track_id"]) for anns in labels.values() for a in anns if a["track_id"] >= 0}
            out["sequences"][seq] = {"frames": len(self.frames(seq)), "boxes": dict(per_class),
                                     "tracks": dict(Counter(c for c, _ in tracks))}
        return out


# -- Open Images (annotations only) ---------------------------------------------------------------


class OpenImagesAdapter:
    dataset_id = "openimages_v5_val_boxes"

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or paths.raw_dir(self.dataset_id)
        self.boxes_csv = _require(self.root / "validation-annotations-bbox.csv", self.dataset_id)
        self.classes_csv = _require(self.root / "oidv7-class-descriptions-boxable.csv", self.dataset_id)

    @cached_property
    def class_names(self) -> dict[str, str]:
        with self.classes_csv.open(newline="") as fh:
            rows = list(csv.reader(fh))
        if rows and rows[0][0].lower() in ("labelname", "label_name"):
            rows = rows[1:]
        return {r[0]: r[1] for r in rows if len(r) >= 2}

    @property
    def categories(self) -> list[str]:
        return sorted(self.class_names.values())

    def boxes(self) -> Iterator[dict[str, Any]]:
        with self.boxes_csv.open(newline="") as fh:
            for row in csv.DictReader(fh):
                yield {"image_id": row["ImageID"], "label": self.class_names.get(row["LabelName"], row["LabelName"]),
                       "group_of": row.get("IsGroupOf") == "1", "depiction": row.get("IsDepiction") == "1",
                       "area_rel": (float(row["XMax"]) - float(row["XMin"])) * (float(row["YMax"]) - float(row["YMin"]))}

    def samples(self, split: str | None = None) -> Iterator[Sample]:  # pragma: no cover - annotations only
        raise NotImplementedError("Open Images is used here for annotation statistics only (no images downloaded)")

    def stats(self) -> dict[str, Any]:
        per_class = Counter(b["label"] for b in self.boxes())
        return {"boxes": sum(per_class.values()), "classes_with_boxes": len(per_class), "boxable_classes": len(self.class_names),
                "images": len({b["image_id"] for b in self.boxes()})}


# -- splits ---------------------------------------------------------------------------------------


def load_split(dataset_id: str, split: str) -> set[str]:
    path = paths.splits_dir() / f"{dataset_id}.json"
    if not path.exists():
        raise DatasetMissing(f"no split definition for {dataset_id} (run: python tools/datasets.py split {dataset_id})")
    data = json.loads(path.read_text())
    return set(data["splits"][split]["ids"])


ADAPTERS = {
    "coco_val2017": CocoAdapter,
    "lvis_v1_labels": LvisAdapter,
    "visdrone_det_val": VisDroneAdapter,
    "kitti_tracking": KittiTrackingAdapter,
    "openimages_v5_val_boxes": OpenImagesAdapter,
}


def load(dataset_id: str) -> Any:
    try:
        return ADAPTERS[dataset_id]()
    except KeyError:
        raise DatasetMissing(f"no adapter for {dataset_id}") from None
