"""Leakage-free splits.

Video-derived datasets (VisDrone-DET images come from clips, KITTI from
sequences) must be split by clip/sequence, never by frame: near-identical
frames on both sides of a split would inflate results. Splits are
deterministic (hash of the group id with a fixed salt), written to
data/splits/<dataset>.json and versioned with the code. Parameters are tuned
on "tune" only; "test" is used once per reported configuration.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from app.datasets import paths

SALT = "reality-debugger-splits-v1"
#: dataset -> (fraction for "tune", description)
PLANS: dict[str, tuple[float, str]] = {
    "visdrone_det_val": (0.5, "VisDrone2019-DET val split in two by video clip: 'tune' for choosing tiling parameters, 'test' for reporting."),
}


def _bucket(group: str) -> float:
    digest = hashlib.sha256(f"{SALT}:{group}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _near_duplicate_pairs(samples: list[Any]) -> list[tuple[str, str, int]]:
    """Pairs of images whose 64-bit difference hashes differ by <= NEAR_DUPLICATE_BITS."""
    import cv2
    import numpy as np

    from app.datasets.validate import NEAR_DUPLICATE_BITS, dhash

    ids, hashes = [], []
    for s in samples:
        img = cv2.imread(s.image_path, cv2.IMREAD_GRAYSCALE) if s.image_path else None
        if img is not None:
            ids.append(s.id)
            hashes.append(dhash(img))
    arr = np.array(hashes, dtype=np.uint64)
    pairs = []
    for i in range(len(arr) - 1):
        x = arr[i + 1 :] ^ arr[i]
        dist = np.array([bin(int(v)).count("1") for v in x])
        for j in np.flatnonzero(dist <= NEAR_DUPLICATE_BITS):
            pairs.append((ids[i], ids[i + 1 + int(j)], int(dist[j])))
    return pairs


def make(dataset_id: str) -> dict[str, Any]:
    from app.datasets import adapters

    if dataset_id not in PLANS:
        raise ValueError(f"no split plan for {dataset_id}")
    fraction, description = PLANS[dataset_id]
    adapter = adapters.load(dataset_id)
    samples = list(adapter.samples())
    group_of = {s.id: s.group or s.id for s in samples}
    # Union groups that share near-identical images (VisDrone repeats some frames under different clip ids),
    # so no near-duplicate can end up on both sides of the split.
    parent: dict[str, str] = {g: g for g in group_of.values()}

    def find(g: str) -> str:
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    pairs = _near_duplicate_pairs(samples)
    for a, b, _ in pairs:
        ra, rb = find(group_of[a]), find(group_of[b])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    component = {g: find(g) for g in parent}
    roots = sorted(set(component.values()))
    tune_roots = {r for r in roots if _bucket(r) < fraction}
    splits: dict[str, dict[str, Any]] = {"tune": {"ids": [], "groups": []}, "test": {"ids": [], "groups": []}}
    for s in samples:
        name = "tune" if component[group_of[s.id]] in tune_roots else "test"
        splits[name]["ids"].append(s.id)
    for name in splits:
        ids = set(splits[name]["ids"])
        splits[name]["groups"] = sorted({group_of[i] for i in ids})
        splits[name]["images"] = len(ids)
        splits[name]["instances"] = sum(len([a for a in s.annotations if not a.get("iscrowd")]) for s in samples if s.id in ids)
    overlap = set(splits["tune"]["groups"]) & set(splits["test"]["groups"])
    assert not overlap, f"group leakage: {overlap}"
    split_of = {i: n for n, v in splits.items() for i in v["ids"]}
    cross = [p for p in pairs if split_of[p[0]] != split_of[p[1]]]
    assert not cross, f"near-duplicate leakage: {cross}"
    merged = sorted({(min(group_of[a], group_of[b]), max(group_of[a], group_of[b])) for a, b, _ in pairs if group_of[a] != group_of[b]})
    result = {
        "dataset": dataset_id,
        "method": f"group-aware: groups linked by near-duplicate images (dHash Hamming <= 4) are merged; sha256('{SALT}:<root group>') < {fraction} -> tune, else test",
        "description": description,
        "group_definition": "video clip id (first token of the file name), merged across near-duplicate frames",
        "near_duplicate_pairs": len(pairs),
        "groups_merged_because_of_duplicates": [list(m) for m in merged],
        "splits": splits,
        "group_overlap": 0,
        "cross_split_near_duplicates": 0,
        "category_counts": {name: dict(Counter(a["category"] for s in samples if s.id in set(splits[name]["ids"]) for a in s.annotations)) for name in splits},
    }
    out = paths.splits_dir() / f"{dataset_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    summary = {k: v for k, v in result.items() if k != "splits"}
    summary["splits"] = {k: {"images": v["images"], "instances": v["instances"], "groups": len(v["groups"])} for k, v in splits.items()}
    return summary
