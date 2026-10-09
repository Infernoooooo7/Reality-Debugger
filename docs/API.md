# HTTP API

Base URL: `http://localhost:8000`, or same-origin `/api` through the Vite dev
server. Interactive OpenAPI docs are at **`/api/docs`** (Swagger UI) and
`/api/redoc`. The schema is at `/api/openapi.json`.

**Local first.** The browser runs the detectors and sends a **scene model**, which
is measured data and not pixels. The backend's local diagnostic engine turns that
scene model into findings. An image or keyframe is uploaded only when the
optional AI layer is enabled (`GEMINI_API_KEY`, or an explicit `AI_PROVIDER`) and
the frontend chooses to send one. Every endpoint works with no API key.

JSON endpoints take `application/json`. Endpoints that accept images take
`multipart/form-data`. All errors use one envelope:

```json
{ "error": { "code": "UNSUPPORTED_MEDIA_TYPE", "message": "…", "hint": "…", "retryable": false } }
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Status, local engine (active rules, vocabulary, detectors), AI layer state, limits, features |
| `GET` | `/api/health/ai` | Verifies the AI credentials/model with a cheap call (cached 60 s). Returns `AI_OFF` in local-only mode |
| `GET` | `/api/metrics` | Developer counters: local evaluations, timings, findings per rule, lifecycle, AI calls, cache hits, errors, tokens |
| `POST` | `/api/scan` | Start a live-scan session (`{"personality": "brutal"}`) → `ScanState` |
| `POST` | `/api/scan/observe` | Live Scan: one scene model (JSON, **no image**) → `ScanAnalysisResponse` (updates the lifecycle) |
| `POST` | `/api/analyze/frame` | Live Scan with optional AI: scene model + optional frame, `trigger`, `focus` → `ScanAnalysisResponse` |
| `POST` | `/api/scan/deep` | Deep Scan: fast + deep detector scene model, optional frame → `ScanAnalysisResponse` |
| `GET` | `/api/scan/{scan_id}` | Current `ScanState` |
| `DELETE` | `/api/scan/{scan_id}` | Clear a session (privacy) → 204 |
| `POST` | `/api/analyze/scene` | Image Debug without AI: scene model (JSON) → `DiagnosticReport` |
| `POST` | `/api/analyze/image` | Image Debug with AI: image + scene model → `DiagnosticReport` |
| `POST` | `/api/analyze/video` | Video Debug: sampled scene models (+ keyframes if AI is on), or a whole video (server fallback) → `VideoReport` |

`personality` is `serious` (the default), `brutal` or `unhinged`. It changes
wording only, never measurements.

## The scene model

The scene model is produced by `frontend/src/vision/scene.ts` and validated by
`backend/app/schemas/scene.py`. Boxes are normalised `{x, y, w, h}` in 0..1,
with the origin at the top left. Invalid objects are dropped one at a time, so
one bad box does not reject the scene.

```json
{
  "version": 2, "at_ms": 4200, "view_id": 0, "width": 1280, "height": 720,
  "objects": [
    { "id": "t1", "label": "laptop", "confidence": 0.86, "box": { "x": 0.30, "y": 0.35, "w": 0.38, "h": 0.40 },
      "source": "fused", "verified": true, "age_ms": 4000, "persistent": true,
      "movement": "static", "speed": 0.004, "static_ms": 4000,
      "reversals": 0, "occlusion": 0.0, "occluded_ms": 0, "truncated": false }
  ],
  "signals": { "motion": 0.01, "brightness": 0.52, "sharpness": 0.61, "scene_change": 0.02 },
  "events": [ { "kind": "entered", "at_ms": 1200, "object_id": "t2", "label": "cup" } ],
  "stats": { "fps": 9.6, "tentative_tracks": 0, "mean_confidence": 0.78,
             "detectors": ["efficientdet_lite0", "yolox_s"] }
}
```

- `source`: `fast` (EfficientDet-Lite0 track), `deep` (YOLOX-S only) or `fused` (both).
- `verified`: both detectors agree.
- `speed`: Kalman centre speed in frame widths per second.
- `view_id`: increments when the camera settles on a different view.

At most 80 objects and 40 events are accepted. The serialised scene must be at
most `MAX_SCENE_CHARS` (96 000) characters.

## Live Scan

### `POST /api/scan/observe` (JSON)

```json
{ "scan_id": "scan_…", "personality": "serious", "scene": { /* scene model */ } }
```

If `scan_id` is missing or expired, a new session is created. The frontend
sends an observation every 1.5 s while the scene changes, plus a 6 s heartbeat.

### `POST /api/analyze/frame` (multipart)

Fields:
- `scene`: JSON scene model.
- `image`: optional JPG/PNG/WEBP, ≤ 5 MB. Only used for AI reasoning.
- `scan_id`.
- `personality`.
- `trigger`: why AI reasoning is requested. `first_look`, `scene_change`, `confirmed_finding`, `relationship` and `ambiguous` come from the backend's `ai_suggestion`. The others are `user_explain`, `manual`, `deep_scan` and `freeze`. `observe`, `new_object`, `confirmation` and `interval` are also accepted.
- `focus`: the finding id to explain, used with `user_explain`.

### `POST /api/scan/deep` (multipart)

Fields: `scene` (fast and deep detectors fused on the device), an optional
`image`, `scan_id`, `personality` and `trigger` (default `deep_scan`). When both
detectors verify every object involved in a finding, the finding is confirmed
immediately.

### Response: `ScanAnalysisResponse`

```json
{
  "scan": {
    "scan_id": "scan_…", "personality": "brutal", "observations": 2, "analyses": 0,
    "status": "STABLE", "system_score": 93, "engine": "local-diagnostics/2",
    "findings": [ /* Finding, see below */ ], "counts": { … }, "events": [ … ], "ai": null
  },
  "report": { /* DiagnosticReport for this observation */ },
  "events": [
    { "id": "EVT-0002", "type": "CONFIRMED", "finding_id": "BUG-001", "severity": "MEDIUM",
      "title": "The cup is one elbow away from a hardware incident",
      "note": "Measured in 2 observations over 1.7 s.", "at": "…" }
  ],
  "ai_suggestion": null
}
```

`ai_suggestion` is `{trigger, reason, finding_ids}` when AI is enabled and the
backend decides that a frame is worth reasoning about. Examples: a newly
confirmed finding of severity MEDIUM or higher, a new view, or an ambiguous
scene. The frontend then sends one frame to `/api/analyze/frame`. In local-only
mode it is always `null`.

**Lifecycle** (`config/temporal.json`):
- `DISCOVERED` → `CONFIRMED` after 2 observations spanning ≥ 1.5 s.
- `TRACKING` from the 3rd observation.
- `RESOLVED` after the condition has been absent for ≥ 3 s and ≥ 2 observations in the same view.

A finding whose objects left the view is marked `out_of_view` and is not
resolved. A resolved finding that comes back is `REOPENED`.

## Image Debug

- `POST /api/analyze/scene` (JSON): `{"personality": "serious", "scene": {…}}`.
  Purely local; no image is sent.
- `POST /api/analyze/image` (multipart): `image` (≤ 15 MB, ≤ 50 MP),
  `personality` and `scene`. The image is decoded in memory, EXIF/GPS is
  stripped, and it is forwarded to the AI provider only if one is configured.

### Response: `DiagnosticReport`

This is a real local-only response, abridged:

```json
{
  "report_id": "rpt_…", "mode": "image", "personality": "serious",
  "provider": "local", "model": "local-diagnostics/2", "engine": "local-diagnostics/2",
  "detectors": ["efficientdet_lite0", "yolox_s"],
  "ai": { "status": "off", "provider": null, "model": null, "trigger": "manual",
          "reason": "Local-only mode: no GEMINI_API_KEY is configured. AI reasoning is optional.", "error": null },
  "status": "STABLE", "system_score": 93,
  "counts": { "active_bugs": 1, "high_priority": 0, "optimizations": 0, "resolved": 0 },
  "objects": [ { "id": "t1", "label": "laptop", "confidence": 0.86, "box": { … }, "source": "fused", "verified": true } ],
  "relationships": [ { "subject": "t2", "relation": "near", "object": "t1", "source": "local",
                       "observation": "cup is right of the laptop; edge gap 2.0% of the frame, 0.05x the larger box, IoU 0.00" } ],
  "findings": [ {
      "id": "BUG-001", "severity": "MEDIUM", "category": "SAFETY", "status": "CONFIRMED",
      "title": "Cup within spill range of the laptop",
      "evidence": "The local detector found a cup (71%) 0.05x its size away (edge gap 2.0% of the frame) next to a laptop (86%).",
      "inference": "A container that can hold liquid sits within knock-over distance of an electronic device.",
      "impact": "A spill could damage the laptop.",
      "recommendation": "Move the cup at least an arm's length from the laptop, or use a lidded cup.",
      "confidence": 0.785, "source": "local", "rule": "spill_risk",
      "measurements": { "edge_gap": 0.02, "relative_gap": 0.051, "overlap_iou": 0.0,
                        "cup_confidence": 0.71, "laptop_confidence": 0.86 },
      "related_objects": ["cup", "laptop"], "object_ids": ["t2", "t1"], "box": { … },
      "ai_note": null, "ai_agrees": null, "sightings": 1, "out_of_view": false, "history": [ … ]
  } ],
  "final_diagnosis": "…", "warnings": []
}
```

Field values:
- `source`: `local` (rule engine) or `ai` (added by the reasoning layer).
- `ai_note` / `ai_agrees`: the AI's comment on a local finding.
- `ai.status`: `off`, `ok`, `cached`, `skipped` or `unavailable`. With `unavailable`, the local results are still valid and `ai.error` explains the failure.
- Severities: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `INFO`.
- Statuses: `DISCOVERED`, `CONFIRMED`, `TRACKING`, `RESOLVED`.

## Video Debug

### `POST /api/analyze/video` (multipart)

**Manifest form** (what the frontend sends):
- `manifest`: JSON `VideoManifest`. Its `samples` (≤ 240) are the scene models
  measured on the device about every 0.5 s.
- `frames`: repeated JPEG keyframes (≤ 12). Sent **only when AI is enabled**.

```json
{ "duration_s": 84.2, "width": 1920, "height": 1080, "name": "desk.mp4", "size_bytes": 1234567,
  "detector": "efficientdet_lite0",
  "samples": [ { "t": 0.5, "scene": 0, "objects": [ /* scene objects (<= 40) */ ],
                  "brightness": 0.52, "sharpness": 0.61, "motion": 0.02, "detectors": ["efficientdet_lite0"] } ],
  "scenes": [ { "index": 0, "start_t": 0.0, "end_t": 12.4 } ],
  "frames": [ { "index": 0, "t": 4.1, "scene": 0, "reason": "scene representative", "objects": [ … ] } ],
  "events": [ { "t": 12.9, "kind": "SCENE_CHANGE", "text": "…" } ] }
```

The backend replays the samples through the same lifecycle as Live Scan, with
video time as the clock. A video therefore gets DISCOVERED, CONFIRMED and
RESOLVED events with no AI.

**Server fallback**: a single `video` file (≤ 300 MB), for codecs the browser
cannot decode. OpenCV samples it on the server and computes frame signals only
(blur, lighting, scene changes). Object findings need the browser detectors.

The response is a `VideoReport`:
- `report`: a `DiagnosticReport` whose findings carry `first_seen_s` and `last_seen_s`.
- `timeline`: entries with `t`, `kind`, `severity`, `finding_id`, `text` and `source` (`local` or `ai`).
- `video`, `sampling` and `keyframes`.

## Error codes

| HTTP | Code | Meaning |
| --- | --- | --- |
| 413 | `PAYLOAD_TOO_LARGE`, `INVALID_SCENE` | Upload or request body over the limit; scene model over `MAX_SCENE_CHARS` |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | Not JPG/PNG/WEBP (checked by magic bytes), or not a video |
| 422 | `INVALID_REQUEST`, `INVALID_UPLOAD`, `INVALID_SCENE`, `INVALID_MANIFEST`, `INVALID_PERSONALITY`, `INVALID_FOCUS`, `MISSING_INPUT`, `MISSING_FILE`, `TOO_MANY_FRAMES`, `VIDEO_UNREADABLE`, `VIDEO_DECODER_MISSING` | Bad input |
| 404 | `SCAN_NOT_FOUND`, `NOT_FOUND` | Unknown or expired scan, or unknown route |
| 405 | `METHOD_NOT_ALLOWED` | Wrong HTTP method |
| 500 | `INTERNAL_ERROR` | Unexpected; details only in the server log |

AI failures never fail a request. They are reported in the report's `ai` block:

```json
"ai": { "status": "unavailable", "provider": "gemini", "error": { "code": "AI_RATE_LIMITED", "message": "…", "retryable": true } }
```

The AI error codes are:
- `AI_AUTH_FAILED`: bad key.
- `AI_MODEL_NOT_FOUND`.
- `AI_RATE_LIMITED`: also used for an exhausted quota.
- `AI_UNAVAILABLE`: provider down or unreachable.
- `AI_TIMEOUT`.
- `AI_REFUSED`: the model declined or blocked the frame.
- `AI_REQUEST_REJECTED`.
- `AI_MALFORMED_RESPONSE`.
- `AI_NOT_CONFIGURED`: a provider was selected without its key.
- `AI_ERROR`.

Configuration failures (auth, model, rejected request, missing key) pause
automatic AI calls for 10 minutes. Transient failures pause them for 1 minute.
`GET /api/health/ai` returns the same codes, and `AI_OFF` when no provider is
configured.

## Examples

```bash
curl http://localhost:8000/api/health

# Local diagnosis of a scene model (no image, no key)
curl -H 'content-type: application/json' \
     -d '{"scene":{"objects":[{"id":"a","label":"cup","confidence":0.8,"box":{"x":0.62,"y":0.5,"w":0.08,"h":0.15}},
                              {"id":"b","label":"laptop","confidence":0.9,"box":{"x":0.25,"y":0.35,"w":0.36,"h":0.4}}]}}' \
     http://localhost:8000/api/analyze/scene

# Live scan: start a session, then send observations
curl -X POST -H 'content-type: application/json' -d '{"personality":"brutal"}' http://localhost:8000/api/scan
curl -H 'content-type: application/json' -d '{"scan_id":"scan_…","scene":{…}}' http://localhost:8000/api/scan/observe

# Server-side video fallback
curl -F video=@clip.mp4 http://localhost:8000/api/analyze/video
```
