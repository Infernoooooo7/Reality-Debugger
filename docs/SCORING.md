# Score, status and inspection coverage

This document defines what a Reality Debugger report claims and how its numbers
are computed. It also records the "false 100/100" failure and how it was fixed.

**In one sentence:** an empty findings list means only that no check fired.
Whether that says anything about the scene depends on what the scan could
examine, and the report now states that separately.

## 1. The failure

A user photographed a visibly cluttered computer desk. Only four objects were
recognised (laptop, keyboard, mouse, bottle). The report said:

- System score 100/100;
- Status STABLE;
- 0 bugs;
- "No measurable issues found by the local checks."

### Reproduction

The original photo is not available, so real COCO val2017 desk photos stand in
for it. They were run through the unchanged app (Image Debug in headless
Chromium, real detectors, real backend).

- **A tidy-looking but busy desk (COCO 37740, 640×480):** 9 objects
  recognised, no findings, 100/100 STABLE, "No measurable issues found by the
  local checks".
- **High-resolution photo of a cluttered desk:** stood in for by a composite of
  36 desk photos (3840×2880, 437 annotated objects). The old pipeline returned
  22 objects (recall 0.041).
- **No usable detection:** with no scene (or no detector output), the backend
  returned STABLE 100, "No objects detected - nothing to measure".

### Root cause, stage by stage (code at commit 177afa7)

| Stage | What happened | Effect on the desk photo |
|---|---|---|
| Decode | Full-resolution bitmap, EXIF orientation applied. | Fine. |
| Fast detector | EfficientDet-Lite0 resizes the whole image to 320×320; at most 30 results ≥ 0.3. | A 4032×3024 photo is seen at 8% of its resolution. |
| Deep detector | YOLOX-S letterboxes the whole image into 640×640. | 16% of the resolution: small objects lose almost all their pixels. |
| Deep failure | A failed deep pass was shown as "skipped". | A model failure looked like a choice. |
| Fast failure | `scene = null`; the image was uploaded to `/api/analyze/image` even without AI. | The backend built a scene with no objects, which was then scored 100. |
| Fusion | Deep-only objects need ≥ 0.4. | Measured trade-off, kept (section 5). |
| Payload | At most 60 objects, silently truncated. | Not reached with 4 objects. |
| Rules | `clutter` fired only on object count (≥ 8 loose objects), `overlap_stack` needed two pairs at IoU ≥ 0.3, and `spill_risk` needs proximity. | With 4 objects, nothing fires. |
| Score | `score = max(5, 100 − Σ penalties)`. | No findings, so the score is **100**. |
| Status | STABLE when score ≥ 80 and no HIGH/CRITICAL finding. | **STABLE**. |
| Diagnosis | "No measurable issues found by the local checks." | Reads as an all-clear. |
| UI | Always a number "/ 100"; "No findings. Either this place is immaculate, or the frame was too ambiguous." | No uncertainty shown. |

Two things went wrong:

1. **Too few objects reached the rules.** Downscaling, a closed vocabulary of
   80 categories and count-only rules meant that very little was examined.
2. **The score treated "nothing examined" as "nothing wrong".**

## 2. What a report claims now

Every report (Image Debug, Video Debug, Live Scan, Deep Scan) carries an
`inspection` record. It is computed in `backend/app/services/inspection.py` and
shown in the "Inspection coverage" panel. It separates:

| Field | Values | Meaning |
|---|---|---|
| processing | completed | The request was processed (otherwise you get an error, not a report). |
| `detection` | `ok` · `partial` · `failed` · `not_run` | Whether the detectors produced results. Each detector reports its own status: ok / failed / skipped / unavailable. |
| `coverage` | `sufficient` · `limited` · `insufficient` | Whether what was examined supports a conclusion. `reasons` say why. |
| `findings` | `findings` · `no_findings` | Whether any check fired. |
| `analysis_status` | `complete` · `limited` · `inconclusive` · `detection_failed` | Summary of the above. |

The record also lists:

- each detector run (model, status, boxes, input size, passes, tile size,
  effective resolution, time);
- object and category counts;
- confidence statistics;
- image size;
- the structure measure;
- image quality;
- the detectors' vocabulary;
- the checks that ran, and the analyses that did not run, with the reason.

### Coverage reasons

The reasons and their thresholds live in `config/diagnostics.json` under
`inspection` and `confidence`. Each reason is a measurement or a reported
failure, never a guess about how many objects "should" be there. Any
`insufficient` reason makes coverage `insufficient`; otherwise any `limited`
reason makes it `limited`.

| Code | Level | When | Basis |
|---|---|---|---|
| `detection_not_run` | insufficient | No scene, or no detector reported running. | — |
| `detection_failed` | insufficient | Every detector failed, was skipped or was unavailable. | Errors are quoted. |
| `detector_not_used` | limited | One detector failed or was skipped; only the other's output was used. | — |
| `fast_detector_only` | limited | Still image (Image Debug, video sample) and no deep or server detector produced a result. | 118 COCO desk scenes: recall 0.344 (fast) vs 0.524 (deep) [desk record]. |
| `runs_unreported` | limited | Older client: detectors listed by name only. | — |
| `downscaled` | limited | Best effective resolution < 0.5 network px per source px. Tiled runs count their tile size. | Deep detector recall at full detail: 0.272 for small objects vs 0.763 for large [desk record]. |
| `tiling_incomplete` | limited | The tiled pass hit its time budget. | — |
| `structure_unmeasured` | limited | The client sent no structure measure. | — |
| `nothing_recognised` | insufficient | Detection ran, nothing was recognised, and edge density ≥ 0.02. | The 27 COCO images in which neither detector found anything had a median edge density of 0.29 and 3 annotated objects [coverage record]. |
| `unexplained_structure` | limited | More than half of the visible detail lies outside every recognised box. | A descriptive majority threshold, not a recall estimate: its Spearman correlation with missed objects is only 0.10 [coverage record]. |
| `objects_capped` | limited | More objects were recognised than the 80 sent to the rules. | — |
| `closed_vocabulary` | limited | Every detection scan. | Median share of annotated objects (COCO + LVIS) that the detectors cannot recognise: 0.50 over 4978 COCO images [coverage record]. |
| `low_confidence` | limited | More than half of the objects score below 0.5. | YOLOX-S precision 0.32 / 0.42 for scores 0.3–0.4 / 0.4–0.5 [confidence record]. |
| `weak_evidence` | limited | Image Debug only: objects below the evidence threshold were not used for findings (section 5). | [confidence record] |
| `too_dark`, `too_blurred` | limited | Image-quality thresholds from `config/inference.json`. | — |

**The structure measure.** It is defined identically in
`backend/app/vision/coverage.py` and `frontend/src/vision/coverage.ts` (a
parity test covers both). It is computed as follows:

1. Take the luminance of the image reduced to a long side of at most 256 px.
2. Compute central-difference gradients and mark a pixel as "structure" when
   its gradient is ≥ 0.05.
3. Rasterise the boxes of the recognised objects.
4. `unexplained_share` is the share of structure pixels that lie outside every
   box.

It describes how much visible detail no recognised object accounts for. It
does **not** estimate how many objects were missed, and it is never reported
as a percentage of objects detected.

**Every detection scan is `limited`**, because the closed vocabulary always
applies: on held-out COCO images, half of the annotated objects in the median
image are of kinds the detectors cannot recognise. An absent finding therefore
never means "nothing is wrong". Coverage can become `sufficient` only with a
detector or segmenter whose vocabulary covers the scene, for example an
open-vocabulary model. None is loaded today.

## 3. Status and scores

```
penalty(f)    = weight(severity) × (0.5 + 0.5 × confidence)
                weight: CRITICAL 30 · HIGH 16 · MEDIUM 8 · LOW 3 · INFO 0
issue_score   = max(5, 100 − Σ penalty(f) over open findings)
                blended with the AI's holistic score when present: round(0.5 × ai + 0.5 × computed)
                null when there is no open finding (an empty list is not evidence of a perfect scene)
scene score   = the same formula, but only when coverage is sufficient AND at least one object was
(system_score)  recognised; otherwise null, shown as UNRATED
status        = CRITICAL      an open CRITICAL finding, or issue_score < 45
                DEGRADED      an open HIGH finding, or issue_score < 80
                INCONCLUSIVE  coverage insufficient (incl. detection failed)
                LIMITED       coverage limited   (shown as "LIMITED INSPECTION")
                STABLE        otherwise: no serious finding AND sufficient coverage
```

Measured problems are reported whatever the coverage: a HIGH spill risk is
DEGRADED even when the inspection is limited. An all-clear (STABLE) needs
evidence that the scan could have found problems.

What each number means:

| Shown | Meaning | Not meant |
|---|---|---|
| **Scene score** N/100 | Condition of the whole scene, rated only when the inspection could cover it. | Never shown when coverage is limited. |
| **UNRATED** | The inspection cannot justify a scene score. | Not "0", not "unknown error". |
| **Issue score** N/100 | Severity-weighted burden of the findings that were measured. | Not a statement about anything that was not examined. |
| **LIMITED INSPECTION** | No serious finding, but at least one measured limitation (reasons listed). | Not "stable". |
| **INCONCLUSIVE** | The evidence cannot support any conclusion (detection failed, or nothing recognised in a detailed image). | Not "no issues". |

The final diagnosis follows the same rules:

- **Detection failed:** "Detection failed. … Nothing in this scene was
  examined, so no conclusion is possible."
- **No findings, limited coverage:** "No issues found among the recognised
  objects - but this is not an all-clear. N object(s) in K categories were
  recognised. <main limitations>"
- **Findings, limited coverage:** the top finding, followed by "The inspection
  is limited, so other issues may exist: …".

## 4. Seeing more of a large photo: tiled deep pass

Image Debug now runs the deep detector on the whole image and also on
overlapping tiles of the original pixels. Boxes are mapped back and merged by
class-aware NMS (`frontend/src/vision/deep/tiling.ts`, configured in
`config/vision.json` under `deep.tiling`):

- Images are tiled only when the long side exceeds 960 px.
- Tiles are sized so that at most 3 cover the long side with 20% overlap:
  `tile = max(640, ceil(long / (3 − 2 × 0.2)))`.
- That gives about 10 network passes.
- Tiling stops after 20 s and reports `tiling_incomplete`.
- Images are never upscaled.

### Choosing the merge

Merging was chosen on separate tune mosaics and reported on test mosaics. Both
sets are 40 COCO 3×3 mosaics at 1920×1920, run with the deployed YOLOX-S in
Python (`tools/evaluate.py mosaic`):

| Merge | Tune AP | Test AP | Test AP small | Test precision @0.3 | Test recall @0.3 | ms / image |
|---|---|---|---|---|---|---|
| whole image only (before) | 22.3 | 18.8 | 1.9 | 0.743 | 0.261 | 111 |
| tiles + whole, greedy NMM (SAHI) | 34.3 | 31.2 | 15.5 | 0.659 | 0.486 | 1121 |
| **tiles + whole, NMS 0.5 (chosen)** | **38.9** | **36.9** | **18.9** | 0.630 | **0.553** | 1165 |

### The app pipeline in the browser

These are the Image Debug detection stages in headless Chromium (WebAssembly,
4 cores), scored at the display threshold of 0.3. The composites are built from
real COCO desk photos (`tools/evaluate.py composite`). Scoring is done by
`frontend/scripts/still-eval.mjs` [still-pipeline record].

| Image | Annotated | Before: fused, whole image | Now: fused, tiled | Tiled pass time |
|---|---|---|---|---|
| desk photo, 640×480 | 15 | R 0.600 · P 0.750 | identical (not tiled) | 0.76 s (one pass) |
| 2×2 composite, 1280×960 | 35 | R 0.114 · P 0.571 (7 objects) | R 0.429 · P 0.556 (27 objects) | 4.4 s (7 passes) |
| 3×3 composite, 1920×1440 | 101 | R 0.109 · P 0.733 (15) | R 0.337 · P 0.531 (64) | 6.1 s (10 passes) |
| 6×6 composite, 3840×2880 | 437 | R 0.041 · P 0.818 (22) | R 0.275 · P 0.732 (164) | 6.1 s (10 passes) |

Recall rises 3–7× on large images. Precision falls by up to 0.20.

**Where the false positives come from.** On the 3×3 composite, the deep tiled
pass makes 49 false detections:

- 30 have no annotated object at all. COCO leaves many objects unannotated, so
  some of these are real.
- 14 are partial or misplaced boxes of an annotated object, 5 of them at a tile
  edge.
- 5 have the wrong label.

**Small objects are still mostly missed.** At 3840×2880, 3 tiles of 1477 px see
the image at 43% of its resolution, and small-object recall is 0.008. The
report says so (`downscaled`) rather than hiding it.

## 5. Findings rest on measurable evidence

**Evidence gate (Image Debug).** A single photo has no temporal confirmation.
Its objects support a finding only when such detections are more likely right
than wrong on held-out data:

- fast-detector objects need a score ≥ 0.4;
- deep or fused objects need a score ≥ 0.5.

Precision on COCO val2017 [confidence record]:

| Score band | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | ≥ 0.7 |
|---|---|---|---|---|---|
| EfficientDet-Lite0 | 0.386 | **0.606** | 0.765 | 0.886 | 0.970 |
| YOLOX-S | 0.321 | 0.420 | **0.554** | 0.657 | 0.906 |
| YOLOX-S confirmed by both detectors | 0.408 | 0.465 | **0.625** | 0.698 | 0.930 |
| YOLOX-S, deep only | 0.302 | 0.405 | **0.518** | 0.626 | 0.799 |

Objects below the gate:

- stay in the scene graph;
- are counted in the inspection record (`weak_evidence`);
- are not used to claim a hazard.

Live Scan, Deep Scan and Video keep every tracked object, because persistence
over time is their evidence (`diagnostics.confidence.persistenceFullMs`).

**Not lowered blindly.** Display threshold (0.3) and deep-only fusion floor
(0.4) are unchanged. Removing the 0.4 floor would bring back the deep-only
detections scored 0.3–0.4 that fusion drops:

| Composite | True positives gained | False positives gained |
|---|---|---|
| 2×2 | 1 | 7 |
| 3×3 | 3 | 20 |
| 6×6 | 12 | 19 |

Measured on the composites of section 4.

### Spatial rules

The spatial rules measure arrangement; they do not count objects.

| Rule | Measures | Reported as |
|---|---|---|
| `dense_region` | ≥ 5 recognised loose objects within 0.12 of the frame diagonal of one object's centre. | LOW, "N recognised items concentrated in one area", with the radius and count. |
| `overlap_cluster` | ≥ 3 objects connected by box overlap ≥ 0.3 of the smaller box (union-find). | LOW, "N recognised items overlap each other". |
| `surface_congestion` | Union of object boxes covers ≥ 60% of a detected surface. | LOW. |
| `keep_clear_zone` | Objects with ≥ 50% of their box inside an area the user marked to stay clear (scene `zones`). | MEDIUM. |
| `spill_risk` etc. | Proximity between attribute classes (unchanged). | As before, subject to the evidence gate. |

These parameters are heuristics and are marked as such in the config.

The count-only `clutter` rule was removed: an object count says nothing about
arrangement. The pairwise `overlap_stack` rule was removed too.

Not measured:

- **Cables and power strips:** no loaded model has these categories.
- **Pixel areas:** no segmentation model, so the rules use boxes only.

Both appear in the report as analyses that did not run.

## 6. Regression tests

- **Backend** (`backend/tests/test_inspection.py`):
  - no scene;
  - all detectors failed (quoted errors);
  - one detector failed;
  - a 12-megapixel photo with and without tiles;
  - nothing recognised in a detailed image vs a blank wall;
  - unexplained structure;
  - low confidence;
  - closed vocabulary;
  - capped objects;
  - incomplete tiling;
  - an old client;
  - a dark image;
  - checks run and skipped;
  - **the reported case** (four objects on a 4032×3024 photo: UNRATED, LIMITED, "not an all-clear", never "No measurable issues");
  - a detector failure;
  - an image without detections;
  - findings under limited coverage;
  - the status rules;
  - an empty live scan;
  - video samples;
  - the evidence gate in photo and live modes.
- **Frontend:**
  - `vision/__tests__/coverage.test.ts` (parity with Python);
  - `tiling.test.ts` (window layout identical to the backend, merging, time budget);
  - `scene.test.ts` (run reports, payload cap, scene built after a failure).
- **Browser:**
  - `frontend/scripts/vision-lab.mjs` with `?still=1`, scored by `still-eval.mjs` (section 4);
  - Image Debug end to end on desk photos and composites, with the deep model blocked and with both models blocked:

| Case | Before | Now |
|---|---|---|
| Busy desk, no finding (COCO 37740) | 100/100 STABLE, "No measurable issues found" | UNRATED, LIMITED INSPECTION, "not an all-clear", reasons listed |
| 3840×2880 composite | 22 objects | 165 objects (80 most confident analysed, `objects_capped`), DEGRADED, issue score 54, `downscaled` reported |
| Deep model blocked | deep shown as "skipped" | deep run `failed` ("Failed to fetch"), LIMITED, `detector_not_used`, `fast_detector_only` |
| Both models blocked | (fast failure) STABLE 100, "No objects detected - nothing to measure" | INCONCLUSIVE, UNRATED, "Detection failed … no conclusion is possible" |

## 7. Limitations

- **Closed vocabulary.** Coverage is never `sufficient` for a detection scan,
  so the scene score is UNRATED in practice. This follows from the evidence: no
  available model can show that a scene has no problems outside its
  categories.
- **Tiled results are partly unverified.** Tiling raises recall but lowers
  precision (section 4). The browser numbers come from one 4-core machine; your
  device's time is shown in the Image Debug stage notes.
- **Composites only.** The composites stand in for a cluttered high-resolution
  photo. They are real photos, but not one real scene.
- **Spatial thresholds are heuristics.** They are documented in the config,
  not tuned on labelled clutter data, because no such dataset is used here.
- **No calibrated confidence.** The structure measure and the confidence gate
  are evidence indicators, not calibrated probabilities that the scene is
  fine.

Records cited (`docs/benchmarks/records/`):

- [desk record] `desk-scenes-efficientdet-lite0-yolox-s-coco-val2017-lvis-v1-minival-val2017-desk-118-default.json`
- [coverage record] `inspection-coverage-efficientdet-lite0-yolox-s-coco-val2017-lvis-v1-minival-val2017-default.json`
- [confidence record] `detection-confidence-bands-efficientdet-lite0-yolox-s-coco-val2017-val2017-default.json`
- [still-pipeline record] `still-pipeline-efficientdet-lite0-yolox-s-coco-val2017-desk-composites-browser.json`
- mosaic records `detection-mosaic-tiling-yolox-s-coco-val2017-*`
