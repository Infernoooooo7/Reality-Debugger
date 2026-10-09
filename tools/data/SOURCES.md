# Vendored dataset metadata

These small files are inputs to `tools/build_ontology.py`. They are copied verbatim
from their official sources so the ontology can be regenerated offline.

| File | Source | License | SHA-256 |
| --- | --- | --- | --- |
| `coco_panoptic_categories.json` | [cocodataset/panopticapi `panoptic_coco_categories.json`](https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json) — the official COCO category list with `supercategory` and `isthing` (80 "thing" categories used by detection + 53 "stuff" categories). | COCO annotations: CC BY 4.0 (COCO Consortium) | `0365beb2…de83b2` |
| `lvis_coco_to_synset.json` | [lvis-dataset/lvis-api `data/coco_to_synset.json`](https://github.com/lvis-dataset/lvis-api/blob/master/data/coco_to_synset.json) — maps each of the 80 COCO category names to a WordNet synset (used by LVIS to align COCO and LVIS vocabularies). | BSD 2-Clause, © 2019 Agrim Gupta and Ross Girshick | `e6d4a9ae…62f9b40` |

WordNet itself is not vendored: `build_ontology.py` uses the NLTK WordNet 3.0
corpus (`nltk.download("wordnet")`). Retrieved for this project in October 2026.

References:

- T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014, arXiv:1405.0312.
- A. Gupta, P. Dollár, R. Girshick, "LVIS: A Dataset for Large Vocabulary Instance Segmentation", CVPR 2019, arXiv:1908.03195.
- G. A. Miller, "WordNet: A Lexical Database for English", Communications of the ACM 38(11), 1995.
