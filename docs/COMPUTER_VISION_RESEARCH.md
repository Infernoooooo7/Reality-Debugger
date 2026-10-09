# Computer vision research notes

This document records what was researched for the local-first computer-vision upgrade, what was chosen, why, and what was measured. Every number either cites a source or names the measurement that produced it. Measurements made for this project live in [`docs/benchmarks/`](benchmarks/).

The goal: object detection, tracking, scene understanding and diagnostics must work on the device with no API key. An LLM (Gemini or Claude) is an optional reasoning layer on top.

---

## 1. Architecture in one picture

```
camera / image / video frame
  │
  ├─ fast detector  EfficientDet-Lite0 int8 (MediaPipe Tasks, WASM/WebGL, worker)   every frame, 6–15 fps
  │     └─ ByteTrack tracker + Kalman filter → tracks with id, velocity, movement, persistence, occlusion
  ├─ pixel signals  motion, brightness, sharpness (variance of Laplacian), scene fingerprint → view changes
  ├─ deep detector  YOLOX-S (ONNX Runtime Web, WebGPU or multi-threaded WASM, worker)  selective:
  │                 Deep Scan, Image Debug, video keyframes, live verification when fast enough
  │     └─ late fusion: verified (both agree) / relabelled / deep-only objects
  ▼
scene model (JSON: objects + geometry + temporal attributes + signals + events)   ← no image
  ▼
backend local diagnostic engine (16 attribute-based rules) → finding lifecycle
  DISCOVERED → CONFIRMED → TRACKING → RESOLVED (+ out-of-view, reopened)
  ▼
optional AI reasoning layer (Gemini / Claude): explains local findings, adds context
  called only on triggers, with cooldown, budget, de-duplication and caching
```

This follows the idea in *Looking Fast and Slow* (Liu et al., 2019): run a cheap model continuously and a more accurate model selectively, keeping state across frames. Here the "memory" is the tracker and the scene model, not a learned recurrent unit.

---

## 2. Object detection

### 2.1 Candidates

| Model | COCO AP (reported) | Size / cost | Licence | Browser path | Notes |
|---|---|---|---|---|---|
| EfficientDet-Lite0 int8 | 25.69 (val2017) | 4.4 MB, 37 ms Pixel 4 CPU | Apache-2.0 | MediaPipe Tasks (WASM, WebGL) | [TFLite Model Maker table](#refs); already in the app |
| EfficientDet-Lite1 … Lite4 | 30.55 … 41.96 | 5.8 … 19.9 MB, 49 … 260 ms Pixel 4 | Apache-2.0 | MediaPipe Tasks | Lite4 = 7× the latency of Lite0 |
| YOLOX-Nano / Tiny / S | 25.8 / 32.8 / 40.5 | 0.91 / 5.06 / 9.0 M params; ONNX 3.7 / 20.2 / 35.9 MB | Apache-2.0 | ONNX Runtime Web | official ONNX exports on GitHub releases |
| RT-DETR / RT-DETRv2 | 46.5–54.8 (CVPR 2024) | 20–76 M params | Apache-2.0 | ONNX (heavy) | transformer decoder, no NMS |
| D-FINE N … X | 42.8 … 55.8; Obj365+COCO S 50.7 … X 59.3 | 4–62 M params | Apache-2.0 code; Objects365 weights carry dataset terms | needs PyTorch to export | weights are `.pth` only |
| OWL-ViT / OWLv2 | open-vocabulary (LVIS rare AP 31.2 → 44.6 with OWL-ST) | B/32 ≈ 613 MB fp32 | Apache-2.0 | too large for live use | text-prompted |
| Grounding DINO | 52.5 AP zero-shot COCO | Swin backbone, very heavy | Apache-2.0 | not practical in a browser | text-prompted |
| YOLO-World | 35.4 AP LVIS at 52 FPS (V100) | YOLOv8-based | **GPL-3.0** | ONNX possible | licence incompatible with this project's terms |

Sources: EfficientDet-Lite table from the TensorFlow Lite Model Maker object-detection guide; YOLOX README; RT-DETR and D-FINE READMEs; OWLv2 paper/README; Grounding DINO abstract; YOLO-World CVPR 2024 paper and its `LICENSE` file. Full references are in [§10](#refs).

### 2.2 Choice and measured evidence

**Fast path: EfficientDet-Lite0 int8 (kept).** It is the only candidate small enough to run at 10 fps on a phone CPU through a mature runtime with a verified GPU delegate. Its class names come from `labels.txt`, which is packed inside the model's own metadata. The labels have 90 slots: 80 COCO names plus 10 `???` placeholders, which are kept as `null` so indices line up.

**Deep path: YOLOX-S (new).** It was chosen for accuracy per byte, an Apache-2.0 licence, an official ONNX export with documented preprocessing, and a CNN graph that ONNX Runtime Web runs on both WebGPU and WASM. The class list comes from the official repository at the release tag (`coco_classes.py`), read with Python's `ast` and never typed by hand.

Measured on **coco128**: the first 128 COCO train2017 images, 929 boxes, scored with `pycocotools` by `tools/eval_detectors.py`. Both models were trained on COCO train2017, so absolute AP is optimistic; use it for relative comparison only.

| Detector (as used in the app) | AP | AP50 | AP_small | AR100 | CPU median |
|---|---|---|---|---|---|
| EfficientDet-Lite0, letterboxed | 37.0 | 56.9 | 5.0 | 40.8 | 29.4 ms (MediaPipe Python) |
| EfficientDet-Lite0, stretched (MediaPipe default) | 36.3 | 54.6 | 6.1 | 40.5 | 28.0 ms |
| YOLOX-S, official preprocessing | **48.3** | **68.0** | **21.3** | **53.2** | 81.7 ms (onnxruntime, 4 threads) |

Two conclusions:
- Letterboxing camera frames before the square detector is slightly better than MediaPipe's default stretch (+0.7 AP), so the app letterboxes.
- YOLOX-S recovers roughly 4× the small-object AP. That is why it verifies keyframes and deep scans.

The **colour order matters**: the same YOLOX model fed RGB instead of BGR turned four oranges into one vase (measured). The deep worker follows the official `preproc()`: BGR order, 114 padding, top-left placement, raw 0–255 values.

**In the browser** (`frontend/vision-lab.html`, headless Chromium 141, 4-core x86 container CPU, cross-origin isolated):

| Stage | Runtime | Median |
|---|---|---|
| EfficientDet-Lite0 (worker, XNNPACK CPU) | MediaPipe Tasks WASM | 70–94 ms per frame (3-run medians); 63–113 ms across all repeat runs |
| YOLOX-S total (pre + inference + decode/NMS) | ONNX Runtime Web WASM, 4 threads | 446–528 ms |
| YOLOX-S via WebGPU on a **software** adapter (SwiftShader) | ONNX Runtime Web WebGPU | 27.5–29.3 s |

The last row is why the deep worker ignores software WebGPU adapters (SwiftShader, llvmpipe) and benchmarks WebAssembly when a WebGPU warm-up is slow. On hardware GPUs, WebGPU is expected to be much faster, but no hardware GPU was available to measure here. That remains unmeasured.

### 2.3 Operating points (`config/detection.json`)

Thresholds were chosen from the precision/recall curve measured on coco128 (operating points are in `docs/benchmarks/detector_eval_coco128.json`):

- Lite0's F1-optimal threshold is **0.30** (precision 0.771, recall 0.412). It is the fast detector's display threshold and ByteTrack's "high" band.
- **0.15** is the low band for ByteTrack's second association (precision 0.427, roughly where ByteTrack's own 0.1 floor sits for its detector).
- **0.35** for starting a new track (precision 0.856).
- YOLOX-S: pre-NMS score 0.1 and NMS IoU 0.45, as in the official demo; display threshold 0.30 (its F1-optimal point, precision 0.772, recall 0.598).
- Fusion: a deep-only object needs ≥ 0.40 (YOLOX precision 0.825). A conflicting label is replaced only when the deep score beats the fast score by 0.15.

### 2.4 Open-vocabulary detection

Investigated, not shipped in this release:

- **Size.** OWL-ViT B/32 is about 613 MB in fp32, and Grounding DINO is larger. That is not viable for a live browser loop or a phone download. Quantised OWLv2 variants exist, but were not available from a reachable host.
- **Availability.** The usable exports (OWL-ViT/OWLv2, Grounding DINO, quantised variants) are hosted on Hugging Face, which this build environment could not reach. D-FINE's Objects365 weights are PyTorch-only, and its README warns that they may be subject to the Objects365 terms.
- **Licence.** YOLO-World is GPL-3.0.

The architecture is ready for one. Detectors are described by manifests with labels taken from model metadata. `tools/build_ontology.py` regenerates `config/ontology.generated.json` for any new vocabulary, and the diagnostic rules test **attributes** (`liquid_container`, `electronic`, `cable`, `outlet` …), never label strings. Two rules, `cable_congestion` and `liquid_near_outlet`, are inactive today because no bundled detector has cable or outlet classes. They activate automatically when a model with such classes is added; `/api/health` lists the active rules.

---

## 3. Vocabulary and ontology (no hand-written object lists)

- **Labels** come from the detectors' own metadata (§2.2).
- **Supercategories** come from COCO's `panoptic_coco_categories.json` (cocodataset/panopticapi): 12 groups for the 80 thing classes.
- **WordNet synsets** come from LVIS's `coco_to_synset.json` (lvis-api, BSD-2): 79 of 80 resolve in WordNet 3.0. `stop_sign.n.01` is an LVIS addition.
- **Attributes** are defined once in `config/ontology_roots.json` as WordNet root synsets (e.g. `electronic_equipment.n.01`, `vessel.n.03`, `edge_tool.n.01`) plus COCO supercategories. A label gets an attribute when its synset is a hyponym of a root. Result: `cup`, `bottle`, `wine glass` and `vase` → `liquid_container`; `laptop`, `tv`, `cell phone`, `keyboard`, `mouse`, `remote` → `electronic`; `knife` and `scissors` → `sharp`.
- **Lexicon**: hyponym lemmas give a free-text → label/attribute map (`config/lexicon.generated.json`), so words from an AI answer ("mug", "notebook computer") map onto the same attributes. Person and animal roots are not expanded with hyponyms: WordNet's `person` subtree turns everyday nouns (e.g. "monitor", "charger") into people.

---

## 4. Tracking (`config/tracking.json`)

| Tracker | Idea | Fit here |
|---|---|---|
| SORT (Bewley et al., ICIP 2016) | Kalman + Hungarian on IoU | simple, but drops occluded objects |
| DeepSORT (Wojke et al., ICIP 2017) | + appearance embedding (re-ID CNN) | needs a second network per box; too heavy for phones |
| **ByteTrack** (Zhang et al., ECCV 2022) | + second association with **low-score** boxes | **chosen**: no appearance model, keeps occluded/blurred objects alive |
| OC-SORT (Cao et al., CVPR 2023) | observation-centric re-update for non-linear motion | marginal for mostly static desks |

The implementation (`frontend/src/vision/tracker.ts`, `kalman.ts`, `assignment.ts`) follows the official `byte_tracker.py`, `matching.py` and `kalman_filter.py`:

- 8-dimensional state (cx, cy, aspect, height + velocities), `std_weight_position` = 1/20 and `std_weight_velocity` = 1/160.
- First association (tracked + lost vs high-score boxes) uses IoU × score ≥ 0.1, from ByteTrack's `match_thresh` 0.9 on the fused cost. The second (remaining tracks vs low-score boxes) uses IoU ≥ 0.5. Unconfirmed tracks use IoU × score ≥ 0.3. Duplicates are removed at IoU > 0.85.
- Optimal assignment matches `lap.lapjv` with a cost limit: the cost matrix is extended with "unmatched" entries at cost thresh/2 and solved with the Hungarian algorithm.

Adaptations, each documented in the config with its source:
1. **Score bands re-derived for our detector.** ByteTrack's 0.6/0.1 bands are tuned to YOLOX on MOT17. Lite0's scores are lower, so the bands are 0.30/0.15/0.35 from the coco128 precision curve.
2. **Variable frame rate.** The reference assumes a fixed rate. Here the Kalman transition uses dt in nominal frames, and the lost-track buffer is 1 s rather than 30 frames.
3. **Multi-class gating.** ByteTrack is single-class. Here a detection can only extend a track of the same label or the same COCO supercategory, following Hoiem et al. (ECCV 2012): detectors mostly confuse similar categories.

**Temporal attributes per track:** Kalman speed; moving/static with hysteresis; time static; horizontal direction reversals within a 10 s window; occlusion (share of the box overlapped by other boxes); persistence; and truncation by the frame edge.

**Movement thresholds were measured, not guessed** (`docs/benchmarks/browser_bench.json`, *tracker* section). Three real photos were run through the real detector and tracker at 10 fps:

| Scenario | Kalman speed (frame widths/s) |
|---|---|
| identical frames | 0 |
| static camera, ±0.5 % random jitter per frame | p95 0.007–0.020 |
| static camera, ±1 % jitter | p95 0.014–0.035, max 0.085 |
| pan at 0.05 fw/s | median 0.044–0.048 |
| pan at 0.10 fw/s | median 0.092–0.097 |
| pan at 0.20 fw/s | median 0.187–0.191 |

From this: `stillSpeed` = 0.04, above all jitter p95 values, and `movingSpeed` = 0.09, above the largest jitter speed measured (0.085). The Kalman medians run 3–13 % below the true pan speeds, so a slow deliberate movement of about 0.1 fw/s is borderline (estimated 0.092–0.097), while 0.2 fw/s is always 'moving'. The trade-off is deliberate: hand tremor must never read as movement.

---

## 5. Scene model, relations and diagnostics

The browser sends a **scene model** (`backend/app/schemas/scene.py`), not pixels. Each object carries: id, label, confidence, normalised box, source (fast / deep / fused), whether both detectors verified it, age, persistence, movement, speed, time static, reversals, occlusion and truncation. The model also carries frame signals (motion, brightness, sharpness, scene change), track events (entered, left, scene change) and stats (fps, tentative tracks, mean confidence, detectors used).

**Relations** are computed from boxes, never asked from an LLM:
- *touching / overlaps*: edge gap ≤ 1 % of the frame, or IoU > 0;
- *near*: edge gap ≤ 0.35 × the size of the larger box (scale-invariant);
- *on*: ≥ 50 % of the object's box lies inside a surface's box and its base point is inside that box.

All of this is 2D: without depth, "overlaps" can mean in front of, behind or on top of. Reports say so.

**The local diagnostic engine** (`backend/app/services/local_diagnostics.py`) has 16 rules over attributes and measurements:
- hazard pairs: spill risk, food near electronics, liquid near outlet, animal near equipment;
- single objects: sharp tool left out, container near a surface edge;
- density: clutter, surface congestion, stacked items, cable congestion;
- temporal: persistent occlusion, left-behind item, repeated movement, view obstruction;
- signals: lighting, blur.

Every finding stores its measurements: gaps, IoU, occupancy, durations, counts.

The clutter rule counts loose objects and their union area. That is a deliberately simple proxy. Perceptual clutter measures such as Feature Congestion (Rosenholtz, Li and Nakano, J. Vision 2007) model local colour, orientation and luminance variability, which this pipeline does not compute. The rule says "N loose items", not "visually cluttered".

**Lifecycle** (`config/temporal.json`):
- A finding is CONFIRMED after 2 observations spanning ≥ 1.5 s. A single noisy frame never confirms a bug.
- In a deep scan where both detectors agree on every object involved, it is confirmed immediately.
- TRACKING from the 3rd observation.
- RESOLVED only after the condition has been absent for ≥ 3 s and ≥ 2 observations **in the same view**.
- If the camera moved to another view, or an involved object left through the frame edge, the finding is *out of view*, not resolved.
- Track-id changes (lost and re-acquired) are bridged by the same rule + same labels + overlapping box.

---

## 6. Frame signals and video

- **Sharpness** is the variance of the Laplacian on a 128×96 grey thumbnail, on a log scale. The measure is commonly attributed to Pech-Pacheco et al. (ICPR 2000), a comparative study of autofocus measures. The log mapping (offset 4, span 2.6) was chosen after the raw value saturated on test images.
- **Scene fingerprint** is a 4×4×4 RGB histogram plus a 16×12 luminance grid. The distance is 0.55 × histogram L1/2 + 0.45 × min(1, 3 × grid L1), a classic histogram-difference shot-boundary cue.
- Measured with the fake-camera feed: same scene 0.00–0.09, hard cuts 0.85–0.87. The cut threshold is therefore 0.32.
- A **view** changes only on a *settled* frame (motion < 0.07) that differs from the view's reference frame by more than that threshold. This covers cuts and pans while ignoring motion blur.
- **Video** runs every sample (about 2 per second, at most 72) through the same detector and tracker, with video time as the clock. The backend replays the sampled scene models through the same lifecycle as Live Scan, so a video gets DISCOVERED / CONFIRMED / RESOLVED events without any AI. Keyframes are verified by the deep detector within a 15 s budget.
- **Small objects.** SAHI (Akyon et al., ICIP 2022) slices images to raise small-object recall. It was not adopted for the live loop because of cost; it is listed as a future upgrade for deep scans.

---

## 7. Browser inference runtimes

| Runtime | Result | Used for |
|---|---|---|
| **MediaPipe Tasks Vision** (WASM + WebGL delegate) | Mature and small. The GPU delegate is checked with a self-test image and used only if it detects the probe and is faster than CPU. | fast detector |
| **ONNX Runtime Web** 1.30 (WASM SIMD + threads, WebGPU) | Runs YOLOX-S exactly as the official demo. The WASM build is 14.2 MB and the WebGPU build 26.8 MB; only the chosen one is downloaded. | deep detector |
| **WebGPU** | Functionally verified (identical detections to WASM). Software adapters were about 55× slower than WASM here (27.5 s vs 0.49 s per image) and are skipped. "Experimental" per the ORT README; Chromium 113+ only. | deep detector, when a hardware adapter exists |
| **WASM multi-threading** | Needs `crossOriginIsolated`. The backend and dev server send COOP/COEP (same-origin resources only). Measured with 4 threads. | deep detector |
| **Web Workers** | Fast and deep detectors run in separate workers; the UI thread only draws. | both |
| **TensorFlow.js** | Not chosen: no maintained YOLOX/EfficientDet-Lite path that beats MediaPipe/ORT, and it adds another runtime. | — |
| **OpenCV.js** | Not chosen: ~8 MB for operations (Laplacian, histograms, frame differencing) the app computes directly on small thumbnails. The *server-side* video fallback uses OpenCV (Python). | — |

---

## 8. The optional AI layer

- **Providers:** Gemini through the official `google-genai` SDK (2.29.0, `generate_content` with `response_mime_type="application/json"` + `response_json_schema`), and Claude through the official `anthropic` SDK.
- **Default model:** `gemini-flash-latest`, the alias the SDK README uses throughout, which tracks the current Flash model. It can be overridden with `GEMINI_MODEL`.
- **What the AI receives:** the scene model, computed relations, local findings with their measurements, and (only when a provider is configured) one compressed frame. Its job is to explain local findings, add issues outside the detector vocabulary, and name the scene. It does not re-detect objects.
- **When it is called:** a confirmed finding of severity ≥ MEDIUM, a confirmed relation, a new view, an ambiguous scene (mean confidence < 0.40, the fast detector's 0.89-precision point), Deep Scan, or Explain.
- **Limits:** 15 s cooldown, at most 6 calls per minute, average-hash de-duplication with a 2-minute cache, and back-off after failures.
- **Fallback:** never implicit (`AI_FALLBACK_PROVIDER=none` by default).

---

## 9. Limitations

- coco128 is a training subset, so AP is optimistic. No evaluation on real indoor desk footage with ground truth was possible here.
- Only COCO's 80 classes are detected, so cables, sockets, papers and stains are invisible to the local engine (the AI layer can add them).
- 2D geometry only: no depth, so "near" and "on" are image-plane relations.
- Browser timings were measured on one x86 container CPU. Phone CPUs will be slower, and real GPUs (WebGPU) faster. Neither was available to measure.
- The movement thresholds come from synthetic camera motion on real photos, not from hand-held phone footage.

<a id="refs"></a>
## 10. References

- M. Tan, R. Pang, Q. V. Le. *EfficientDet: Scalable and Efficient Object Detection.* CVPR 2020. arXiv:1911.09070.
- TensorFlow Lite Model Maker, object detection guide: EfficientDet-Lite0…4 table (AP, size, Pixel 4 latency).
- C. Lugaresi et al. *MediaPipe: A Framework for Building Perception Pipelines.* arXiv:1906.08172, 2019.
- Z. Ge, S. Liu, F. Wang, Z. Li, J. Sun. *YOLOX: Exceeding YOLO Series in 2021.* arXiv:2107.08430. (Megvii-BaseDetection/YOLOX, Apache-2.0.)
- W. Lv et al. *DETRs Beat YOLOs on Real-time Object Detection* (RT-DETR). CVPR 2024. arXiv:2304.08069; RT-DETRv2: arXiv:2407.17140.
- Y. Peng et al. *D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement.* arXiv:2410.13842.
- M. Minderer et al. *Simple Open-Vocabulary Object Detection with Vision Transformers* (OWL-ViT). ECCV 2022. arXiv:2205.06230.
- M. Minderer, A. Gritsenko, N. Houlsby. *Scaling Open-Vocabulary Object Detection* (OWLv2). NeurIPS 2023. arXiv:2306.09683.
- S. Liu et al. *Grounding DINO.* arXiv:2303.05499.
- T. Cheng et al. *YOLO-World: Real-Time Open-Vocabulary Object Detection.* CVPR 2024, pp. 16901–16911. arXiv:2401.17270.
- A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). arXiv:2103.00020.
- A. Kirillov et al. *Segment Anything.* arXiv:2304.02643.
- A. Bewley et al. *Simple Online and Realtime Tracking* (SORT). ICIP 2016. arXiv:1602.00763.
- N. Wojke, A. Bewley, D. Paulus. *Simple Online and Realtime Tracking with a Deep Association Metric* (DeepSORT). ICIP 2017. arXiv:1703.07402.
- Y. Zhang et al. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022. arXiv:2110.06864.
- J. Cao et al. *Observation-Centric SORT* (OC-SORT). CVPR 2023. arXiv:2203.14360.
- M. Liu, M. Zhu, M. White, Y. Li, D. Kalenichenko. *Looking Fast and Slow: Memory-Guided Mobile Video Object Detection.* arXiv:1903.10172, 2019.
- F. C. Akyon, S. O. Altinuc, A. Temizel. *Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection* (SAHI). ICIP 2022, pp. 966–970. doi:10.1109/ICIP46576.2022.9897990.
- D. Hoiem, Y. Chodpathumwan, Q. Dai. *Diagnosing Error in Object Detectors.* ECCV 2012, pp. 340–353. doi:10.1007/978-3-642-33712-3_25.
- J. L. Pech-Pacheco, G. Cristóbal, J. Chamorro-Martínez, J. Fernández-Valdivia. *Diatom autofocusing in brightfield microscopy: a comparative study.* ICPR 2000, vol. 3, pp. 314–317.
- R. Rosenholtz, Y. Li, L. Nakano. *Measuring Visual Clutter.* Journal of Vision 7(2), 2007, pp. 1–22.
- T.-Y. Lin et al. *Microsoft COCO: Common Objects in Context.* ECCV 2014. arXiv:1405.0312. Categories: cocodataset/panopticapi `panoptic_coco_categories.json`.
- A. Gupta, P. Dollár, R. Girshick. *LVIS: A Dataset for Large Vocabulary Instance Segmentation.* CVPR 2019, pp. 5356–5364. arXiv:1908.03195. (lvis-api `coco_to_synset.json`, BSD-2.)
- S. Shao et al. *Objects365: A Large-Scale, High-Quality Dataset for Object Detection.* ICCV 2019.
- A. Kuznetsova et al. *The Open Images Dataset V4.* IJCV 2020. arXiv:1811.00982.
- G. A. Miller. *WordNet: A Lexical Database for English.* Communications of the ACM 38(11), 1995.
- ONNX Runtime Web 1.30.0 (npm) README and build entry points; Gemini API v1beta discovery document (revision 20261008); `google-genai` 2.29.0 README.
