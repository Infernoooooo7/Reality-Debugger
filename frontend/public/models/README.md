# On-device models

Both detectors run in the browser. Each one has a manifest
(`<id>.manifest.json`) with its source URL, SHA-256, licence, paper,
preprocessing, post-processing defaults and its **label list read from the
model's own metadata**. The frontend and the ontology generator read the labels
from these manifests; they are never typed by hand.

| File | Model | Role | Size | Licence | In git |
| --- | --- | --- | --- | --- | --- |
| `efficientdet_lite0.tflite` | EfficientDet-Lite0 int8, 320×320, 80 COCO classes (MediaPipe model zoo) | fast, every frame | 4.6 MB | Apache-2.0 | yes |
| `yolox_s.onnx` | YOLOX-S, 640×640, 80 COCO classes (official 0.1.1rc0 ONNX export) | deep, selective | 35.9 MB | Apache-2.0 | no: fetched |

```bash
python tools/fetch_models.py --fetch   # download what is missing, verify SHA-256 (the Docker build runs this)
python tools/fetch_models.py --check   # verify files and manifests, no network
python tools/fetch_models.py           # download what is missing and rebuild the manifests from model metadata
```

A file whose SHA-256 does not match its manifest is deleted, never served.
The browser also checks the deep model's SHA-256 before caching it.

To add a detector with a supported output format (a MediaPipe object detector,
or a YOLOX-style ONNX model), add an entry to `MODELS` in
`tools/fetch_models.py`. Then run that script and `python tools/build_ontology.py`.
The diagnostic rules work on attributes (liquid container, electronic,
sharp…), so they apply to new classes automatically.
