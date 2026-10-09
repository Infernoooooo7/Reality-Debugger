"""How much of what is in real images can the bundled detectors name at all?

Two measurements, both on held-out annotations:

1. LVIS v1 minival (1203 categories, the 5000 COCO val2017 images). Each LVIS
   category is classified against the COCO-80 vocabulary through WordNet:
   ``same`` (its synset is a COCO class synset), ``subtype`` (a hyponym of a
   COCO class synset, e.g. a sports car of car) or ``outside``. Then, using a
   detector's real predictions on the same images, each LVIS instance counts
   as "localised" when any prediction >= the display threshold overlaps it
   with IoU >= 0.5 - whatever label it carried. LVIS is federated (not every
   category is annotated in every image), so counts are lower bounds.

2. Open Images V5 validation boxes (600 boxable classes): the share of boxes
   whose class maps onto a COCO-80 label through the project's generated
   lexicon (WordNet hyponyms, config/lexicon.generated.json).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from app.datasets.adapters import COCO80, CocoAdapter, LvisAdapter, OpenImagesAdapter
from app.evaluation.detection import iou_matrix

ROOT_DIR = Path(__file__).resolve().parents[3]


def coco_synsets() -> dict[str, str]:
    data = json.loads((ROOT_DIR / "tools" / "data" / "lvis_coco_to_synset.json").read_text())
    return {v["synset"]: name for name, v in data.items()}


def classify_lvis_categories(categories: list[dict[str, Any]], wordnet_dir: str | None = None) -> dict[int, dict[str, Any]]:
    import nltk
    from nltk.corpus import wordnet as wn

    if wordnet_dir:
        nltk.data.path.insert(0, wordnet_dir)
    targets = coco_synsets()
    out = {}
    for c in categories:
        name = c["synset"]
        relation, coco = "outside", None
        if name in targets:
            relation, coco = "same", targets[name]
        else:
            try:
                syn = wn.synset(name)
                ancestors = {s.name() for s in syn.closure(lambda s: s.hypernyms() + s.instance_hypernyms())}
                hit = sorted(ancestors & set(targets))
                if hit:
                    relation, coco = "subtype", targets[hit[0]]
            except Exception:  # noqa: BLE001 - synsets missing from WordNet 3.0 stay "outside"
                pass
        out[c["id"]] = {"name": c["name"], "frequency": c["frequency"], "relation": relation, "coco": coco}
    return out


def lvis_coverage(predictions_file: Path, threshold: float, wordnet_dir: str | None = None) -> dict[str, Any]:
    lvis = LvisAdapter()
    coco = CocoAdapter()
    cats = classify_lvis_categories(lvis.gt["categories"], wordnet_dir)
    coco_names = {c["id"]: c["name"] for c in coco.gt["categories"]}
    preds_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in json.loads(predictions_file.read_text()):
        if r["score"] >= threshold:
            preds_by_image[r["image_id"]].append(r)
    instances = Counter()
    localized = Counter()
    label_used: dict[str, Counter[str]] = defaultdict(Counter)
    per_category_n = Counter()
    per_category_hit = Counter()
    for a in lvis.gt["annotations"]:
        cat = cats[a["category_id"]]
        key = (cat["relation"], cat["frequency"])
        instances[key] += 1
        per_category_n[a["category_id"]] += 1
        preds = preds_by_image.get(a["image_id"], [])
        if not preds:
            continue
        x, y, w, h = a["bbox"]
        pb = np.array([[p["bbox"][0], p["bbox"][1], p["bbox"][0] + p["bbox"][2], p["bbox"][1] + p["bbox"][3]] for p in preds])
        ious = iou_matrix(np.array([[x, y, x + w, y + h]]), pb)[0]
        j = int(np.argmax(ious))
        if ious[j] >= 0.5:
            localized[key] += 1
            per_category_hit[a["category_id"]] += 1
            label_used[cat["name"]][coco_names.get(preds[j]["category_id"], "?")] += 1
    total = sum(instances.values())

    def share(rel: str) -> float:
        return round(sum(v for (r, _), v in instances.items() if r == rel) / total, 4)

    def loc_rate(rel: str) -> float | None:
        n = sum(v for (r, _), v in instances.items() if r == rel)
        return round(sum(v for (r, _), v in localized.items() if r == rel) / n, 4) if n else None

    outside = [(cid, n) for cid, n in per_category_n.items() if cats[cid]["relation"] == "outside"]
    outside.sort(key=lambda x: -x[1])
    top_outside = [{"category": cats[cid]["name"], "instances": n, "localized_by_coco_detector": round(per_category_hit[cid] / n, 3),
                    "labels_used": dict(label_used[cats[cid]["name"]].most_common(3))} for cid, n in outside[:25]]
    relation_counts = Counter(c["relation"] for c in cats.values())
    return {
        "dataset": "LVIS v1 minival (COCO val2017 images), official json",
        "categories": {"total": len(cats), **dict(relation_counts)},
        "instances_total": total,
        "instance_share": {r: share(r) for r in ("same", "subtype", "outside")},
        "localized_rate_by_relation": {r: loc_rate(r) for r in ("same", "subtype", "outside")},
        "by_relation_and_frequency": {f"{r}/{f}": {"instances": n, "localized": localized[(r, f)], "rate": round(localized[(r, f)] / n, 4)}
                                       for (r, f), n in sorted(instances.items())},
        "most_frequent_outside_categories": top_outside,
        "threshold": threshold,
        "predictions": str(predictions_file.relative_to(ROOT_DIR)) if predictions_file.is_relative_to(ROOT_DIR) else str(predictions_file),
        "caveat": "LVIS annotation is federated: counts are lower bounds and 'localized' ignores which COCO label was used.",
    }


def _oi_parents(node: dict[str, Any], parent: str | None, out: dict[str, set[str]]) -> None:
    mid = node.get("LabelName")
    if mid and parent:
        out.setdefault(mid, set()).add(parent)
    for child in node.get("Subcategory", []):
        _oi_parents(child, mid, out)


def openimages_coverage() -> dict[str, Any]:
    """Map Open Images classes onto COCO-80 through Open Images' own class hierarchy.

    A class maps when its name, or the name of one of its ancestors in
    bbox_labels_600_hierarchy.json, equals a COCO-80 name (case-insensitive):
    e.g. Man -> Person -> person, Eagle -> Bird -> bird. No hand-written aliases.
    """
    oi = OpenImagesAdapter()
    hierarchy = json.loads((oi.root / "bbox_labels_600_hierarchy.json").read_text())
    parents: dict[str, set[str]] = {}
    _oi_parents(hierarchy, None, parents)
    names = oi.class_names  # MID -> name
    mid_of = {v: k for k, v in names.items()}
    coco = set(COCO80)

    def mapped(name: str) -> tuple[str, str] | None:
        if name.lower() in coco:
            return ("same", name.lower())
        seen, frontier = set(), set(parents.get(mid_of.get(name, ""), set()))
        while frontier:
            mid = frontier.pop()
            if mid in seen:
                continue
            seen.add(mid)
            ancestor = names.get(mid, "")
            if ancestor.lower() in coco:
                return ("subtype", ancestor.lower())
            frontier |= parents.get(mid, set())
        return None

    per_class = Counter(b["label"] for b in oi.boxes())
    mapping = {c: mapped(c) for c in per_class}
    total = sum(per_class.values())
    same = sum(n for c, n in per_class.items() if mapping[c] and mapping[c][0] == "same")
    sub = sum(n for c, n in per_class.items() if mapping[c] and mapping[c][0] == "subtype")
    unmapped = sorted(((c, n) for c, n in per_class.items() if not mapping[c]), key=lambda x: -x[1])
    return {
        "dataset": "Open Images V5 validation boxes",
        "classes_with_boxes": len(per_class),
        "classes_same_as_coco80": sum(1 for c in per_class if mapping[c] and mapping[c][0] == "same"),
        "classes_subtype_of_coco80": sum(1 for c in per_class if mapping[c] and mapping[c][0] == "subtype"),
        "boxes_total": total,
        "box_share_same": round(same / total, 4),
        "box_share_subtype": round(sub / total, 4),
        "box_share_mapped": round((same + sub) / total, 4),
        "subtype_examples": sorted(f"{c} -> {mapping[c][1]}" for c in per_class if mapping[c] and mapping[c][0] == "subtype"),
        "most_frequent_unmapped_classes": [{"class": c, "boxes": n} for c, n in unmapped[:25]],
        "mapping_method": "exact COCO name, else an ancestor in Open Images' official bbox_labels_600_hierarchy.json with a COCO name",
        "caveat": "Open Images labels are hierarchical (e.g. 'Clothing', 'Person' and 'Man' on the same object) and include parts ('Human arm'); shares count boxes, not distinct objects.",
    }
