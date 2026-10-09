#!/usr/bin/env python3
"""Generate config/ontology.generated.json from model and dataset metadata.

Inputs (nothing is typed in by hand):
  * the label lists of every detector manifest in frontend/public/models
    (written by fetch_models.py from the models' own metadata);
  * COCO category metadata (supercategories) - tools/data/coco_panoptic_categories.json;
  * the LVIS COCO->WordNet synset alignment - tools/data/lvis_coco_to_synset.json;
  * WordNet 3.0 (via NLTK) for definitions, lemmas, hypernym closure and a
    hyponym lexicon;
  * config/ontology_roots.json - which WordNet roots / supercategories define
    each semantic attribute the diagnostic engine reasons about.

Output: for every detector label - its COCO id and supercategory, WordNet
synset, gloss, attributes, and the models that can detect it - plus a
lexicon mapping free-text nouns (e.g. "mug", "burger", "laptop computer")
to detector labels and/or attributes, used to anchor AI-written labels to
tracked objects without hand-written synonym tables.

Usage:  pip install -r tools/requirements.txt
        python tools/build_ontology.py
"""

from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import nltk
from nltk.corpus import wordnet as wn

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "frontend" / "public" / "models"
DATA = ROOT / "tools" / "data"
OUT = ROOT / "config" / "ontology.generated.json"
LEXICON_OUT = ROOT / "config" / "lexicon.generated.json"

# Lexicon depth limits keep the generated file small. Hyponyms further down
# are rare words a vision model is unlikely to use for an everyday object.
LABEL_HYPONYM_DEPTH = 3
ATTRIBUTE_HYPONYM_DEPTH = 2
MAX_LEMMAS_PER_ROOT = 250
# Roots whose hyponym trees are enormous (thousands of species / roles). They
# keep their own lemmas but contribute no hyponym lexicon.
NO_LEXICON_ROOTS = {"animal.n.01", "person.n.01", "food.n.01", "food.n.02"}
# Labels with these attributes keep only their own lemmas: their hyponym trees
# are full of figurative senses ("mug" and "monitor" are kinds of person in
# WordNet, "charger" is a kind of horse) that would mis-anchor everyday words.
NO_HYPONYM_ATTRIBUTES = {"person", "animal"}


def ensure_wordnet() -> None:
    try:
        wn.synset("entity.n.01")
    except LookupError:
        nltk.download("wordnet", quiet=True)


def lemma_names(synset) -> list[str]:
    return sorted({lemma.name().replace("_", " ").lower() for lemma in synset.lemmas()})


def closure(synset, relation, depth: int) -> list:
    out, frontier = [], [synset]
    for _ in range(depth):
        nxt = []
        for s in frontier:
            for t in relation(s):
                if t not in out:
                    out.append(t)
                    nxt.append(t)
        frontier = nxt
    return out


def resolve_synset(label: str, lvis: dict) -> tuple[object | None, str | None]:
    entry = lvis.get(label)
    if entry:
        try:
            return wn.synset(entry["synset"]), "lvis-api coco_to_synset.json"
        except Exception:  # noqa: BLE001 - e.g. stop_sign.n.01 is not in WordNet 3.0
            pass
    candidates = wn.synsets(label.replace(" ", "_"), pos=wn.NOUN)
    if candidates:
        return candidates[0], "WordNet first noun sense of the label"
    return None, None


def main() -> None:
    ensure_wordnet()
    manifests = [json.loads(p.read_text()) for p in sorted(MODEL_DIR.glob("*.manifest.json"))]
    coco = {c["name"]: c for c in json.loads((DATA / "coco_panoptic_categories.json").read_text()) if c["isthing"]}
    lvis = json.loads((DATA / "lvis_coco_to_synset.json").read_text())
    roots_cfg = json.loads((ROOT / "config" / "ontology_roots.json").read_text())["attributes"]

    root_synsets = {attr: [wn.synset(name) for name in spec["synsets"]] for attr, spec in roots_cfg.items()}

    label_models: dict[str, list[str]] = defaultdict(list)
    for m in manifests:
        for label in m["labels"]:
            if label:
                label_models[label].append(m["id"])

    labels: dict[str, dict] = {}
    lexicon: dict[str, dict[str, set]] = defaultdict(lambda: {"labels": set(), "attributes": set()})

    for label in sorted(label_models):
        synset, synset_source = resolve_synset(label, lvis)
        hypernyms = sorted({h for path in synset.hypernym_paths() for h in path}, key=lambda s: s.name()) if synset else []
        supercategory = coco.get(label, {}).get("supercategory")
        attributes = sorted(
            attr
            for attr, spec in roots_cfg.items()
            if (supercategory and supercategory in spec["supercategories"])
            or any(root in hypernyms for root in root_synsets[attr])
        )
        labels[label] = {
            "models": label_models[label],
            "coco_id": coco.get(label, {}).get("id"),
            "supercategory": supercategory,
            "synset": synset.name() if synset else None,
            "synset_source": synset_source,
            "definition": synset.definition() if synset else None,
            "lemmas": lemma_names(synset) if synset else [label],
            "attributes": attributes,
        }
        # Lexicon: the label itself, its lemmas, and hyponym lemmas map to the label.
        words = {label.lower(), *labels[label]["lemmas"]}
        if synset and not NO_HYPONYM_ATTRIBUTES & set(attributes):
            for hypo in closure(synset, lambda s: s.hyponyms(), LABEL_HYPONYM_DEPTH):
                words.update(lemma_names(hypo))
        for word in words:
            lexicon[word]["labels"].add(label)
            lexicon[word]["attributes"].update(attributes)

    # Attribute lexicon: hyponyms of each attribute root map to the attribute
    # (e.g. "mug" -> liquid_container, "extension cord" -> cable).
    for attr, synsets in root_synsets.items():
        for root in synsets:
            words = set(lemma_names(root))
            if root.name() not in NO_LEXICON_ROOTS:
                for hypo in closure(root, lambda s: s.hyponyms(), ATTRIBUTE_HYPONYM_DEPTH):
                    words.update(lemma_names(hypo))
                    if len(words) >= MAX_LEMMAS_PER_ROOT:
                        break
            for word in words:
                lexicon[word]["attributes"].add(attr)

    groups: dict[str, list[str]] = defaultdict(list)
    for label, info in labels.items():
        if info["supercategory"]:
            groups[info["supercategory"]].append(label)

    out = {
        "generated_by": "tools/build_ontology.py",
        "generated_at": dt.date.today().isoformat(),
        "wordnet_version": wn.get_version(),
        "sources": {
            "models": {m["id"]: {"labels_source": m["labels_source"], "num_classes": m["num_classes"]} for m in manifests},
            "coco_categories": "tools/data/coco_panoptic_categories.json (cocodataset/panopticapi)",
            "synsets": "tools/data/lvis_coco_to_synset.json (lvis-dataset/lvis-api), WordNet first noun sense otherwise",
            "attribute_roots": "config/ontology_roots.json",
        },
        "attributes": {attr: {"roots": spec["synsets"], "supercategories": spec["supercategories"], "used_for": spec["used_for"]} for attr, spec in roots_cfg.items()},
        "supercategory_groups": {k: sorted(v) for k, v in sorted(groups.items())},
        "labels": labels,
        "lexicon_file": "config/lexicon.generated.json",
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    lexicon_out = {word: {k: sorted(v) for k, v in entry.items() if v} for word, entry in sorted(lexicon.items())}
    LEXICON_OUT.write_text(json.dumps(lexicon_out, separators=(",", ":"), ensure_ascii=False) + "\n")
    with_attr = sum(1 for v in labels.values() if v["attributes"])
    print(f"wrote {OUT.relative_to(ROOT)}: {len(labels)} labels ({with_attr} with attributes), {OUT.stat().st_size / 1024:.0f} KB")
    print(f"wrote {LEXICON_OUT.relative_to(ROOT)}: {len(lexicon_out)} entries, {LEXICON_OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
