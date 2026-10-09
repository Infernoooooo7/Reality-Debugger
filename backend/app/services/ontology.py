"""Object ontology generated from model and dataset metadata.

``config/ontology.generated.json`` (written by ``tools/build_ontology.py``)
holds, for every label the bundled detectors can output, its COCO
supercategory, WordNet synset and the semantic attributes the diagnostic
engine reasons about (``electronic``, ``liquid_container`` ...).
``config/lexicon.generated.json`` maps free-text nouns to labels and
attributes through WordNet lemmas and hyponyms. Diagnostics never test label
strings: they ask for attributes, so a new detector with a new vocabulary
only needs the ontology to be regenerated.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from app.config import ROOT_DIR

DEFAULT_DIR = ROOT_DIR / "config"
_WORD = re.compile(r"[a-z][a-z' -]*[a-z]|[a-z]")


class Ontology:
    def __init__(self, data: dict, lexicon: dict) -> None:
        self._labels: dict[str, dict] = data.get("labels", {})
        self._attributes: dict[str, dict] = data.get("attributes", {})
        self._lexicon: dict[str, dict] = lexicon
        self.models = sorted(data.get("sources", {}).get("models", {}))

    # -- labels ---------------------------------------------------------------

    @property
    def labels(self) -> list[str]:
        return sorted(self._labels)

    @property
    def attribute_names(self) -> list[str]:
        return sorted(self._attributes)

    def known(self, label: str) -> bool:
        return label.lower() in self._labels

    def attributes(self, label: str) -> frozenset[str]:
        info = self._labels.get(label.lower())
        if info is not None:
            return frozenset(info.get("attributes", ()))
        # Unknown label (e.g. from a newer model or an AI answer): fall back to the lexicon.
        _, attrs = self.match_text(label)
        return frozenset(attrs)

    def has(self, label: str, attribute: str) -> bool:
        return attribute in self.attributes(label)

    def supercategory(self, label: str) -> str | None:
        info = self._labels.get(label.lower())
        return info.get("supercategory") if info else None

    def compatible(self, a: str, b: str) -> bool:
        """Same label, or same COCO supercategory (detectors mostly confuse similar classes)."""
        if a.lower() == b.lower():
            return True
        sa, sb = self.supercategory(a), self.supercategory(b)
        return sa is not None and sa == sb

    def definition(self, label: str) -> str | None:
        info = self._labels.get(label.lower())
        return info.get("definition") if info else None

    # -- free text --------------------------------------------------------------

    def match_text(self, text: str) -> tuple[set[str], set[str]]:
        """Map free text ("a coffee mug", "laptops") to detector labels and attributes.

        Tries the whole phrase, then each word, with a naive plural strip. Returns
        (labels, attributes); both may be empty for words WordNet does not cover.
        """
        phrase = " ".join(_WORD.findall(text.lower()))
        if not phrase:
            return set(), set()
        candidates = [phrase]
        words = phrase.split()
        if len(words) > 1:
            candidates += [" ".join(words[i:]) for i in range(1, len(words))]  # "red coffee mug" -> "coffee mug", "mug"
        labels: set[str] = set()
        attributes: set[str] = set()
        for cand in candidates:
            for form in _singulars(cand):
                entry = self._lexicon.get(form)
                if entry:
                    labels.update(entry.get("labels", ()))
                    attributes.update(entry.get("attributes", ()))
            if labels or attributes:
                break
        return labels, attributes


def _singulars(word: str) -> list[str]:
    forms = [word]
    if word.endswith("ies") and len(word) > 4:
        forms.append(word[:-3] + "y")
    if word.endswith("es") and len(word) > 3:
        forms.append(word[:-2])
    if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
        forms.append(word[:-1])
    return forms


def load_ontology(directory: Path | None = None) -> Ontology:
    directory = directory or DEFAULT_DIR
    data = json.loads((directory / "ontology.generated.json").read_text(encoding="utf-8"))
    lexicon = json.loads((directory / "lexicon.generated.json").read_text(encoding="utf-8"))
    return Ontology(data, lexicon)


@lru_cache
def get_ontology() -> Ontology:
    return load_ontology()
