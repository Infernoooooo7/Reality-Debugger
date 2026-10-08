# HTTP API

Base URL: `http://localhost:8000` (or same-origin `/api` through the Vite dev
server). Interactive OpenAPI docs: **`/api/docs`** (Swagger UI) and
`/api/redoc`; schema at `/api/openapi.json`.

All uploads are `multipart/form-data`. All errors use one envelope:

```json
{ "error": { "code": "UNSUPPORTED_MEDIA_TYPE", "message": "…", "hint": "…", "retryable": false } }
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Status, AI provider/model, limits, features |
| `GET` | `/api/health/ai` | Verifies the AI credentials/model with a cheap call (cached 60 s) |
| `POST` | `/api/analyze/image` | Image Debug: full diagnostic of one image → `DiagnosticReport` |
| `POST` | `/api/analyze/frame` | Live Scan frame → `ScanAnalysisResponse` (updates the lifecycle) |
| `POST` | `/api/analyze/video` | Video Debug: keyframes + manifest, or a whole video (fallback) → `VideoReport` |
| `POST` | `/api/scan` | Start a scan session (`{"personality": "brutal"}`) → `ScanState` |
| `POST` | `/api/scan/deep` | Deep Scan of a frozen frame → `ScanAnalysisResponse` |
| `GET` | `/api/scan/{scan_id}` | Current `ScanState` |
| `DELETE` | `/api/scan/{scan_id}` | Clear a session (privacy) → 204 |

### Common form fields

| Field | Values |
| --- | --- |
| `personality` | `serious` (default) · `brutal` · `unhinged` |
| `context` | JSON `LocalContext` from the on-device engine (optional): `{detector, objects:[{track_id,label,confidence,box}], motion, scene_change, brightness, sharpness, relationships:[…], trigger_detail}` |
| `demo` | `true` forces DEMO MODE for this request |

Boxes are normalised `{x, y, w, h}` in 0..1 with the origin top-left.

### `POST /api/analyze/frame` and `POST /api/scan/deep`

Fields: `image` (JPG/PNG/WEBP, ≤ 5 MB), `scan_id` (optional — a new session is
created if missing or expired), `trigger` (`first_look`, `new_object`,
`scene_change`, `relationship`, `confirmation`, `interval`, `deep_scan`,
`freeze`, `manual`), `personality`, `context`, `demo`.

Response:

```json
{
  "scan":   { "scan_id": "scan_…", "status": "DEGRADED", "system_score": 72,
              "findings": [ /* Finding with lifecycle */ ], "counts": {…}, "events": [ … ] },
  "report": { /* DiagnosticReport for this frame */ },
  "events": [ { "type": "CONFIRMED", "finding_id": "BUG-001", "title": "…", "at": "…" } ]
}
```

### `POST /api/analyze/image`

Fields: `image` (≤ 15 MB, ≤ 50 MP), `personality`, `context`, `demo`.
Returns a `DiagnosticReport`:

```json
{
  "report_id": "rpt_…", "mode": "image", "provider": "anthropic", "model": "claude-opus-5-5",
  "simulated": false, "latency_ms": 8123,
  "system_name": "WORKSPACE_v3.1",
  "scene": { "name": "WORKSPACE", "version": "3.1", "summary": "…", "confidence": 0.91 },
  "status": "DEGRADED", "system_score": 72, "ai_score": 73,
  "counts": { "active_bugs": 3, "high_priority": 1, "optimizations": 4, "resolved": 0 },
  "objects": [ { "id": "obj_01", "label": "laptop", "confidence": 0.97, "box": {…} } ],
  "relationships": [ { "subject": "obj_03", "relation": "overlaps", "object": "obj_01", "observation": "…" } ],
  "findings": [ {
      "id": "BUG-001", "severity": "HIGH", "category": "EFFICIENCY",
      "title": "Cable crossing mouse path",
      "evidence": "what is visible", "inference": "what it probably means",
      "impact": "…", "recommendation": "…", "confidence": 0.89, "quip": "…",
      "status": "CONFIRMED", "box": {…}, "related_objects": ["cable", "mouse"], "history": [ … ]
  } ],
  "optimizations": [ { "id": "opt_01", "title": "…", "effort": "LOW", "impact": "…" } ],
  "final_diagnosis": "Functional, but suffering from technical debt.",
  "warnings": []
}
```

Severities: `CRITICAL HIGH MEDIUM LOW INFO`. Categories: `EFFICIENCY
ORGANIZATION CONSISTENCY WORKFLOW SAFETY ERGONOMICS AESTHETICS SPATIAL
TECH_DEBT ABSURD`. Statuses: `DISCOVERED CONFIRMED TRACKING RESOLVED`.

### `POST /api/analyze/video`

Keyframe form (what the frontend sends): repeated `frames` files (≤ 12) plus
`manifest`:

```json
{ "duration_s": 84.2, "width": 1920, "height": 1080, "name": "desk.mp4", "size_bytes": 1234567,
  "sampled_frames": 72, "redundant_removed": 31,
  "scenes": [ { "index": 0, "start_t": 0.6, "end_t": 12.4 } ],
  "frames": [ { "index": 0, "t": 4.1, "scene": 0, "reason": "scene representative",
                "objects": [ { "label": "laptop", "confidence": 0.8, "box": {…} } ] } ],
  "events": [ { "t": 12.9, "kind": "SCENE_CHANGE", "text": "…" } ] }
```

Fallback form: a single `video` file (≤ 300 MB), sampled on the server.

Response `VideoReport`: `report` (a `DiagnosticReport`, findings carry
`first_seen_s`/`last_seen_s`), `timeline` (`t`, `kind`, `severity`,
`finding_id`, `text`, `source`: `ai` · `demo` · `local`), `video`,
`sampling` (`processed_on`, `sampled_frames`, `scenes`, `redundant_removed`,
`keyframes`) and `keyframes`.

## Error codes

| HTTP | Code | Meaning |
| --- | --- | --- |
| 413 | `PAYLOAD_TOO_LARGE` | Upload or request body over the limit |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | Not JPG/PNG/WEBP (checked by magic bytes) or not a video |
| 422 | `INVALID_REQUEST`, `INVALID_UPLOAD`, `INVALID_CONTEXT`, `INVALID_MANIFEST`, `INVALID_PERSONALITY`, `MISSING_FILE`, `TOO_MANY_FRAMES`, `VIDEO_UNREADABLE` | Bad input |
| 404 | `SCAN_NOT_FOUND`, `NOT_FOUND` | Unknown or expired scan / route |
| 429 | `SCAN_RATE_LIMITED`, `AI_RATE_LIMITED` | Cost guard or provider rate limit (`Retry-After` set) |
| 422 | `AI_REFUSED` | The model declined the frame |
| 502 | `AI_AUTH_FAILED`, `AI_MODEL_NOT_FOUND`, `AI_REQUEST_REJECTED`, `AI_MALFORMED_RESPONSE` | Provider problems (key, model, unusable output) |
| 503 | `AI_NOT_CONFIGURED`, `AI_UNAVAILABLE` | No provider / provider down |
| 504 | `AI_TIMEOUT` | Provider too slow |
| 500 | `INTERNAL_ERROR` | Unexpected; details only in the server log |

## Examples

```bash
curl http://localhost:8000/api/health

curl -F image=@desk.jpg -F personality=brutal http://localhost:8000/api/analyze/image

curl -F image=@frame.jpg -F trigger=new_object \
     -F 'context={"objects":[{"label":"cup","confidence":0.8,"box":{"x":0.6,"y":0.4,"w":0.1,"h":0.15}}]}' \
     http://localhost:8000/api/analyze/frame

curl -F video=@clip.mp4 http://localhost:8000/api/analyze/video
```
