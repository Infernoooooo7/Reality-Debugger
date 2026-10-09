# Architecture notes

These notes describe how Reality Debugger is put together and the exact rules
the code implements. Paths are relative to the repository root. The research
behind the model, tracker and threshold choices, with measurements and
citations, is in [`COMPUTER_VISION_RESEARCH.md`](COMPUTER_VISION_RESEARCH.md).
Every tunable number lives in [`config/`](../config/README.md), with its unit,
purpose and source.

## 1. Pipeline

```
 ┌─────────────────────────────── browser (on device) ───────────────────────────────┐
 │ camera / image / video frame                                                       │
 │   ├─ fast worker (every frame, ≤ 10 fps): letterbox 320 → EfficientDet-Lite0        │
 │   │      → ByteTrack (Kalman + Hungarian) → temporal attributes                     │
 │   │      → pixel signals (motion, brightness, sharpness, scene fingerprint, view)   │
 │   ├─ deep worker (selective): YOLOX-S on ONNX Runtime Web (WebGPU / WASM threads)   │
 │   │      → fusion: verify / relabel / add objects                                  │
 │   └─ scene model: objects + attributes + signals + events  (numbers only)           │
 └──────────────────────────────────────────┬─────────────────────────────────────────┘
                                            │ POST /api/scan/observe (JSON, no image)
 ┌──────────────────────────────────── FastAPI backend ───────────────────────────────┐
 │ validate scene → relations (geometry) → local diagnostic engine (attribute rules)   │
 │   → finding lifecycle DISCOVERED → CONFIRMED → TRACKING → RESOLVED → score/status   │
 │   → AI policy: is reasoning worth a call now?  ── no ──▶ local report (always)      │
 │                                       └─ yes, and a provider is configured:          │
 │        frame + scene + local findings → Gemini / Claude → validated JSON            │
 │        → explanations attached to local findings, extra AI findings (source "ai")   │
 └─────────────────────────────────────────────────────────────────────────────────────┘
```

The data flows Detector → `Detection` → tracked object → `SceneModel` →
diagnostic engine. No stage hard-codes object labels:
- the label lists come from each model's own metadata (manifests in `frontend/public/models/`);
- what a label *is* (liquid container, electronic, sharp, food…) comes from the generated ontology (`config/ontology.generated.json`).

Folders:
- `frontend/src/vision/`: detectors, tracker, fusion and scene model.
- `frontend/src/live/`: the Live Scan loop.
- `backend/app/services/local_diagnostics.py`: rules.
- `backend/app/services/diagnostic_service.py`: lifecycle and scoring.
- `backend/app/services/ai_*.py`, `pipeline.py`: the optional AI layer.
- `backend/app/api/`: HTTP.

## 2. Browser vision (`frontend/src/vision`)

- **Fast detector** (`detector.ts`, `engine.ts`, `vision.worker.ts`): MediaPipe
  Tasks `ObjectDetector` with EfficientDet-Lite0 int8, in a module Web Worker,
  falling back to the main thread.
  - Frames are letterboxed to the model's input size (320, read from the manifest).
  - Detections down to the low band (0.15) are kept for the tracker. Display uses 0.30.
  - GPU vs CPU is decided by a self-test on a procedurally drawn stop sign. GPU must find the probe and be more than 20 % faster.
  - The choice is cached for 14 days. Override it with `?vision=cpu`.
- **Tracker** (`tracker.ts`, `kalman.ts`, `assignment.ts`): ByteTrack with an 8-dim Kalman filter and a variable time step.
  - Three association stages: high-score boxes, low-score boxes, unconfirmed tracks.
  - Optimal assignment uses a cost limit, as in `lap.lapjv`.
  - Detections only extend tracks of a compatible class (same COCO supercategory).
  - Each track carries `speed`, `movement` (static/moving with hysteresis), `static_ms`, direction `reversals`, `occlusion`/`occluded_ms`, `truncated` and `persistent`.
  - Track events: `entered`, `left`, `scene_change`.
- **Signals and views** (`signals.ts`): computed on a 128×96 thumbnail.
  - Motion: luminance difference plus the changed-pixel box.
  - Brightness, and log Laplacian sharpness.
  - Scene fingerprint: RGB histogram plus luminance grid.
  - A new **view** starts when a settled frame (motion < 0.07) differs from the current view's reference frame by > 0.32.
- **Deep detector** (`deep/`): YOLOX-S ONNX in its own worker, with
  preprocessing exactly as the official `preproc()`.
  - Runtime: ONNX Runtime Web on WebGPU when a hardware adapter exists. Software adapters are refused.
  - Otherwise WASM, multi-threaded when the page is cross-origin isolated.
  - If a WebGPU warm-up exceeds 3 s, WASM is measured too and the faster one is kept.
  - The 36 MB model is fetched once, SHA-256-checked and stored in Cache Storage.
- **Fusion** (`fusion.ts`, `Tracker.verify`):
  - A deep box overlapping a fast box (IoU ≥ 0.3) with a compatible class verifies it.
  - A conflicting class is replaced only when the deep score is ≥ 0.15 higher.
  - Deep-only objects need a score ≥ 0.40.
- **Scene model** (`scene.ts`): the JSON sent to the backend (`backend/app/schemas/scene.py`).
- **Relations** (`relations.ts`): the same 2D geometry as the backend, for the overlay.

## 3. Live Scan (`frontend/src/live/session.ts`)

1. Every frame goes through the fast detector and tracker in the worker. The
   overlay interpolates boxes at display rate, and React reads a throttled
   snapshot.
2. Every **1.5 s** when the scene model changed, or every **6 s** as a
   heartbeat, the scene model is posted to `/api/scan/observe`. Only numbers
   are sent; there is no image.
3. When the fast and deep detectors are both available and a deep pass takes
   ≤ 600 ms on this device, a deep check runs at most every 12 s. It verifies
   tracks in the tracker.
4. **Deep Scan** freezes the frame and runs the deep detector and fusion, then
   calls `/api/scan/deep`. The frame is attached only if AI is enabled.
5. If the backend's response carries an `ai_suggestion` (AI enabled only), the
   next *settled* frame is captured (≤ 1280 px JPEG) and sent to
   `/api/analyze/frame`. **Explain** on a finding does the same with
   `trigger=user_explain`.
6. **Clear session** deletes the server session (`DELETE /api/scan/{id}`) and
   all local state.

## 4. Local diagnostic engine (`backend/app/services/local_diagnostics.py`)

The engine has 16 rules over **attributes** and measurements, never over
label strings. Thresholds and severities are in `config/diagnostics.json`.

| Group | Rules |
| --- | --- |
| hazard pairs | `spill_risk` (liquid container near/touching electronics), `food_near_electronics`, `liquid_near_outlet`, `animal_near_equipment` |
| single objects | `sharp_exposed`, `edge_placement` |
| density | `clutter`, `surface_congestion`, `overlap_stack`, `cable_congestion` |
| temporal | `persistent_occlusion`, `long_presence`, `repeated_movement`, `view_obstruction` |
| signals | `lighting`, `blur` |

- A rule is **armed** only if the loaded vocabulary has a label with the
  attributes it needs.
- With the COCO detectors, 14 rules are armed. `cable_congestion` and
  `liquid_near_outlet` wait for a model with cable/outlet classes.
- `/api/health` lists the armed rules.

Relations (`geometry.py`) use the edge gap relative to the frame and to the
larger box, IoU, containment and the base point. Each finding stores its
measurements (gaps, IoU, occupancy, durations, counts) and the ids of the
objects involved.

## 5. Finding lifecycle (`backend/app/services/diagnostic_service.py`)

Rules from `config/temporal.json`:

- **DISCOVERED**: when a rule first fires.
- **CONFIRMED**: after 2 observations spanning ≥ 1.5 s. In a Deep Scan where both detectors verify every involved object, confirmation is immediate.
- **TRACKING**: from the 3rd observation.
- **RESOLVED**: only after the condition has been absent for ≥ 3 s **and** ≥ 2 observations in the same view.
- **Out of view**: if the camera moved to another view, or an involved object left through the frame edge (margin 4 %), the finding is flagged `out_of_view`, not resolved.
- **REOPENED**: a resolved finding that reappears emits this event.
- **Re-identification**: a finding is matched across track-id changes by its key, or by the same rule + labels + box IoU ≥ 0.3.

Images and videos use the same engine:
- An image is a single observation, so its findings report as CONFIRMED.
- A video replays its samples through the lifecycle with video time as the clock.

## 6. Score and status

```
penalty   = Σ over open findings  weight(severity) × (0.5 + 0.5 × confidence)
            weight: CRITICAL 30 · HIGH 16 · MEDIUM 8 · LOW 3 · INFO 0
computed  = max(5, 100 − penalty)
score     = computed                                  (local only)
          = round(0.5 × ai_score + 0.5 × computed)    (when the AI gave a holistic score)
status    = CRITICAL if score < 45 or a CRITICAL is open
            DEGRADED if score < 80 or a HIGH is open
            STABLE   otherwise
```

## 7. Optional AI layer (`backend/app/services`)

- **Provider selection** (`ai_service.py`): `AI_PROVIDER=auto` uses Gemini when
  `GEMINI_API_KEY` is set and is local-only otherwise. Claude/OpenAI must be named
  explicitly. A fallback provider is called only if `AI_FALLBACK_PROVIDER` names
  one (default `none`). A failed Gemini call never silently becomes a paid
  Claude call.
- **Providers** (`ai_providers.py`):
  - Gemini: official `google-genai` SDK, `response_json_schema`, key sent in a header.
  - Claude: official `anthropic` SDK, JSON-schema output.
  - OpenAI-compatible: for local servers such as Ollama.
  - Each provider maps its errors to typed codes (`AI_AUTH_FAILED`, `AI_RATE_LIMITED`, …).
- **When** (`ai_policy.py`, `config/ai.json`):
  - Automatic triggers: a confirmed finding of severity ≥ MEDIUM or a confirmed relation not yet explained, a new view, or an ambiguous scene (mean confidence < 0.40).
  - User triggers: Explain, Deep Scan, Image/Video Debug with AI on.
  - Automatic calls have a 15 s cooldown and a cap of 6 per minute per scan. User actions bypass the cooldown.
  - A 64-bit average hash of the frame de-duplicates near-identical frames, with a 2-minute cache.
  - After a failure, calls back off for 60 s (transient) or 10 min (configuration problems).
- **What** (`prompts.py`): the scene model, the computed relations and the local
  findings with their measurements, plus one compressed frame.
  - The AI explains the local findings through `local_notes {finding_id, note, agrees}`, adds issues outside the detector vocabulary and names the scene.
  - It does not re-detect objects. Its output is validated item by item.
- **Failure isolation** (`pipeline.py`): the local report is computed first. An AI
  failure only sets `report.ai.status = "unavailable"` with a typed error. The
  request still succeeds, and the UI shows *LOCAL CV ACTIVE · AI REASONING
  UNAVAILABLE*.
- **Metrics** (`metrics.py`, `GET /api/metrics`): local evaluation timings,
  findings per rule, lifecycle counts, AI calls/skips/cache hits/errors/tokens
  and latency. The *Dev panel* in Live Scan shows them next to the on-device
  timings.

## 8. Image and Video Debug

- **Image** (`screens/ImageDebug.tsx`): decode → fast detector → deep detector →
  fusion → `/api/analyze/scene`. That route is local and uploads no image. With
  AI on, `/api/analyze/image` receives the image and the scene instead. The
  stages are shown as they run.
- **Video** (`video/sampler.ts`, `screens/VideoDebug.tsx`):
  - The file never leaves the device. It is sampled every 0.5 s (8–72 samples), and every sample goes through the fast detector and tracker with video time as the clock.
  - Scene boundaries come from the fingerprint distance (step > 0.30, drift > 0.48). Near-duplicates are dropped (< 0.06).
  - Up to 8 keyframes in total: the sharpest frame in the middle 60 % of each scene, then object enter/leave moments, then the frames least like their scene's representative. The deep detector verifies keyframes within a 15 s budget.
  - The manifest of samples goes to `/api/analyze/video`. Keyframe JPEGs are added only when AI is on.
  - The backend replays the samples through the lifecycle and returns a timeline.
  - Videos the browser cannot decode can be sampled by OpenCV on the server. That path gives signal-only diagnostics (lighting, sharpness, scene changes) because no detector runs server-side. The temporary file is deleted immediately.

## 9. Privacy and security

- API keys exist only in the backend environment. The frontend never sees them,
  and the log filter masks anything key-shaped (`sk-…`, `AIza…`).
- Without AI, nothing but scene models (numbers) leaves the device. With AI,
  only frames the policy selects (or the user sends) are uploaded.
- Uploads are checked by magic bytes, size and pixel count, decoded in memory,
  re-encoded without EXIF/GPS, and never stored.
- Sessions hold findings text in memory only and expire after 2 h. *Clear
  session* deletes one immediately.
- AI output is validated against a schema item by item. Malformed items are
  dropped with a warning.
- COOP/COEP headers isolate the page (needed for multi-threaded WASM). All
  assets, including the model files, are same-origin.

## 10. Why REST and no WebSocket

Every exchange is a small request/response initiated by the client, and the
realtime part (detection, tracking, overlay) runs on the device. Observations
are tiny JSON posts every 1.5 s. A WebSocket would add reconnection logic
without carrying anything a POST response doesn't already contain.
