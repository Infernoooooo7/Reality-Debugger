# Architecture notes

These notes document the decisions behind Reality Debugger and the exact rules
the code implements. File references are relative to the repository root.

## 1. Pipeline

```
 ┌──────────────────────────── browser (on device) ────────────────────────────┐
 │ camera (getUserMedia)                                                        │
 │   └─ FrameGrabber: 480-px ImageBitmap, transferred (zero-copy) ──┐            │
 │                                                                  ▼            │
 │   vision worker: letterbox → EfficientDet-Lite0 → Tracker → SignalAnalyzer    │
 │                                                                  │            │
 │   ◀── tracks · motion · scene Δ · sharpness · brightness ────────┘            │
 │   spatial heuristics → ScanDirector ──(decision)──▶ JpegCapturer (1280 px)    │
 │   OverlayRenderer (rAF, interpolated) · HUD (5 Hz) · findings panel           │
 └───────────────────────────────────────┬──────────────────────────────────────┘
                                         │ POST /api/analyze/frame (+ local context)
 ┌────────────────────────────── FastAPI backend ──────────────────────────────┐
 │ validate upload → normalise (≤1024 px, EXIF stripped) → prompt + schema      │
 │ → provider (Claude / OpenAI-compatible / demo) → extract + validate JSON     │
 │ → merge into scan session (lifecycle) → score → ScanState + report + events │
 └──────────────────────────────────────────────────────────────────────────────┘
```

Responsibilities are split by folder: `frontend/src/live/camera.ts` (camera),
`frontend/src/vision/*` (vision + tracking), `backend/app/services/ai_*`
(AI), `backend/app/services/diagnostic_service.py` (diagnostics),
`backend/app/api/*` (API) and `frontend/src/screens|components` (UI).

## 2. Local vision engine (`frontend/src/vision`)

- **Runtime:** a module Web Worker (`vision.worker.ts`) owns a `VisionEngine`
  (`engine.ts`). If workers or `OffscreenCanvas` are unavailable the same engine
  runs on the main thread (`client.ts`).
- **Detector** (`detector.ts`): MediaPipe Tasks `ObjectDetector`, model
  `public/models/efficientdet_lite0.tflite`, `VIDEO` running mode, score
  threshold 0.3, up to 25 results. WASM loader/binary are bundled through
  Vite `?url` imports, so nothing is fetched from a CDN.
- **Delegate selection:** `auto` probes CPU and GPU with a procedurally drawn
  stop sign (detected at ~0.96 by EfficientDet-Lite0). GPU is used only if it
  detects the probe and is at least 20 % faster than CPU. In testing,
  a software WebGL GPU returned *no detections at all* while CPU was correct —
  hence verification rather than assumption. The result is cached in
  `localStorage` for 14 days.
- **Letterboxing:** frames are drawn centred on a 384×384 grey canvas before
  detection and boxes are mapped back. MediaPipe stretches non-square inputs;
  on test frames this lifted laptop detection from *not detected* to 0.52–0.79.
- **Tracker** (`tracker.ts`): greedy IoU matching on constant-velocity
  predictions (centre-distance fallback for fast motion), compatible-class
  matching (cup↔wine glass, tv↔laptop…) with class voting; tracks are
  `tentative` until 3 hits, `lost` while coasting, dropped after 1.1 s unseen.
- **Signals** (`signals.ts`, 128×96 thumbnail, buffers reused): motion = mean
  absolute luminance difference (×6, clipped) plus the bounding box of changed
  pixels; fingerprint = 4×4×4 RGB histogram + 16×12 luminance grid;
  `distance = 0.55·½‖h₁−h₂‖₁ + 0.45·min(1, 3·mean|g₁−g₂|)`; sharpness =
  log-scaled Laplacian variance.
- **Spatial heuristics** (`relations.ts`): container next to electronics,
  food next to electronics, sharp tools, ≥3 drink containers, ≥9 objects.

## 3. Scan director (`frontend/src/live/director.ts`)

Local ML runs on every frame (capped at 6/10/15 fps); the vision model is
called only when one of these fires, in priority order:

| Trigger | Condition |
| --- | --- |
| `first_look` | no analysis yet and 1.4 s since start |
| `new_object` | a confirmed track of a class not present at the last analysis, area ≥ 0.6 % of frame, visible > 300 ms |
| `relationship` | a spatial heuristic not seen before |
| `confirmation` (resolution) | all objects a finding depends on have been missing > 2.5 s while the scene is similar (Δ < 0.3); at most every 15 s per finding |
| `scene_change` | fingerprint distance to the last analysed frame > 0.32 |
| `confirmation` | a DISCOVERED finding older than 4.5 s (≤ every 12 s per finding) |
| `interval` | periodic re-check of a stable scene (10/20/40 s or off) |
| `deep_scan` / `freeze` | user request (bypasses the director) |

Gates: never more than one request in flight, at least 5 s between analyses,
and the frame must be settled (motion < 0.07 and no track coasting on a missed
detection) unless it has waited 2.2 s since sending became allowed. Only tracks
matched in the sent frame go into its `context`, so objects that just left the
view (for example after a hard cut) are never reported. Errors back off (4 s doubling to 60 s); rate limits wait 15 s;
authentication/configuration errors stop automatic analysis and offer demo
mode. The backend additionally enforces `SCAN_AI_CALLS_PER_MINUTE` (12).

Finding anchors: a finding's `related_objects` labels are mapped to COCO
classes (`live/labels.ts`, e.g. *mug → cup*, *monitor → tv*); if matching
tracks exist, the overlay anchors the finding to their live boxes, otherwise
to the analysed box while the scene is similar (Δ < 0.22).

## 4. Finding lifecycle (`backend/app/services/diagnostic_service.py`)

- The model receives the scan's active findings with canonical ids
  (`BUG-001`…) and must return one `status_updates` entry per finding:
  `PRESENT`, `RESOLVED` (only if the area is clearly visible and the issue is
  gone) or `NOT_VISIBLE`. New issues get `new_1`, `new_2`….
- Matching: canonical id → demo rule id → fuzzy match (same category and
  title similarity ≥ 0.45, or box IoU ≥ 0.4 with some title overlap, or shared
  objects with overlapping boxes).
- Transitions:
  - new finding → **DISCOVERED** (Deep Scan with confidence ≥ 0.8 → straight to
    **CONFIRMED**);
  - second sighting → **CONFIRMED**; third → **TRACKING**;
  - `RESOLVED` update → **RESOLVED** (with the model's observation as the note);
  - a resolved finding that reappears → **REOPENED** event, back to DISCOVERED;
  - `NOT_VISIBLE` → stays open, flagged `out_of_view`.
- Confidence is smoothed (0.6 new + 0.4 old). Every transition is recorded in
  the finding's `history` and emitted as a lifecycle event the UI animates.

## 5. Score and status

```
penalty   = Σ over open findings  weight(severity) × (0.5 + 0.5 × confidence)
            weight: CRITICAL 30 · HIGH 16 · MEDIUM 8 · LOW 3 · INFO 0
computed  = max(5, 100 − penalty)
score     = round(0.5 × model_score + 0.5 × computed)     (computed only, if no model score)
status    = CRITICAL if score < 45 or a CRITICAL is open
            DEGRADED if score < 80 or a HIGH is open
            STABLE   otherwise
```

Blending keeps the model's holistic judgement but makes the score move when a
bug is resolved, and stops a generous model score from hiding open HIGH bugs.

## 6. AI integration (`backend/app/services`)

- `prompts.py` – one stable system prompt (cached with `cache_control`), a
  per-request user message (mode, personality, trigger, local detector hints,
  active findings) and `DIAGNOSIS_SCHEMA`, which follows structured-output
  constraints (every object `additionalProperties: false`, all keys required,
  nullable via `anyOf`, no numeric/string limits).
- `ai_providers.py` – `AnthropicProvider` (official SDK, `output_config.format`
  JSON schema, `effort`, server-side refusal fallback `fallbacks: "default"`
  under beta `server-side-fallback-2026-07-01`, typed error mapping) and
  `OpenAICompatibleProvider` (`/chat/completions`, falls back from
  `json_schema` → `json_object` → prompt-only on servers that reject it).
- `ai_service.py` – provider selection, JSON extraction (fences/prose tolerant),
  one repair retry on malformed output, concurrency limit.
- `schemas/analysis.py::parse_diagnosis` – validates every list item on its
  own, coerces percentages, unknown enums, pixel boxes and long text; drops
  what cannot be saved and reports warnings instead of failing.
- Default model `claude-opus-5-5`; live frames use effort `low`, everything
  else `medium` (both configurable).

## 7. Video pipeline (`frontend/src/video/sampler.ts`)

1. Metadata via a `<video>` element on an object URL (the file is streamed,
   never read into memory); `Infinity` durations (MediaRecorder WebM) are
   resolved by seeking to the end.
2. `N = clamp(duration / 0.5 s, 8, 72)` evenly spaced samples; each is seeked,
   presented (`requestVideoFrameCallback`), fingerprinted and run through the
   detector.
3. Scene boundaries where the step distance > 0.30 or drift from the scene's
   first frame > 0.48.
4. Redundant frames: distance < 0.06 from the last kept frame of the scene.
5. Representatives: sharpest frame in the middle 60 % of each scene; extra
   budget goes to object enter/leave moments, then to the frames least like
   their scene's representative; at most 8 keyframes.
6. Keyframes (1024 px JPEG) + manifest (timestamps, scenes, local detections,
   local events) → `POST /api/analyze/video`; the model narrates a timeline
   referencing frame numbers which the backend maps back to timestamps and
   merges with the local events.

Videos the browser can't decode can be uploaded whole; the backend streams
them to a temporary file, runs the same selection with OpenCV and deletes the
file immediately.

## 8. Why REST and no WebSocket

Every vision-model analysis is a request/response pair initiated by the
client, and the realtime part (detection, tracking, overlay) runs on the
device. A WebSocket would add reconnection logic without carrying anything the
POST response doesn't already contain. It becomes worthwhile for streamed
partial findings or multi-device mirroring (see *Next steps* in the README).

## 9. Demo mode (`backend/app/services/demo_reasoner.py`)

Used when no provider is configured (or forced per request with `demo=true`).
It applies fixed rules to the browser's real detections and pixel statistics —
liquid container next to electronics, several drink containers, sharp tools,
food next to electronics, phone in the focus zone, object density, book pile,
animals, plants, a person in frame, low/harsh light — in three personalities,
and produces the same wire format, so it goes through the same validation and
lifecycle code. Every result carries `simulated: true`, `provider: "demo"`,
and the UI labels it with a hazard-stripe *DEMO MODE · SIMULATED* badge.
