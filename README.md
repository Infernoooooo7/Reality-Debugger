# Reality Debugger

> **Your world has bugs.**

Reality Debugger treats the physical world like software. Point your phone at a
desk, a kitchen or a gaming setup — or upload a photo or a video — and get a
debugging report for it: bugs with severity, evidence, impact and a fix, a
score, and a final diagnosis. Every report also says what the scan could and
could not examine: which detectors ran, how much detail they saw, and what
they cannot recognise. "No findings" is never presented as an all-clear when
the inspection was limited, and no score is invented when it cannot be
justified (UNRATED, see [`docs/SCORING.md`](docs/SCORING.md)). In Live Scan the findings have a lifecycle
— **DISCOVERED → CONFIRMED → TRACKING → RESOLVED** — so when you move the mug
away from the laptop, the bug closes in front of you.

It is a normal standalone project with four parts:
- a **React + TypeScript + Vite** frontend;
- a **Python + FastAPI** backend;
- a **local computer-vision pipeline**: two detectors in Web Workers, a ByteTrack tracker, and a rule-based diagnostic engine;
- an **optional** AI reasoning layer (Gemini, or Claude if you choose it), called by the backend only when it adds something.

**No API key is needed.** Detection, tracking, relations, diagnostics and the
finding lifecycle all run locally.

```
camera ─▶ fast detector (EfficientDet-Lite0) + ByteTrack, every frame, in a worker
       ─▶ deep detector (YOLOX-S, ONNX Runtime Web) verifies selectively
       ─▶ scene model: objects, attributes, motion, occlusion, views (numbers only)
       ─▶ backend: geometric relations → local diagnostic rules → finding lifecycle
       ─▶ optional: Gemini explains confirmed findings (only when worth a call)
       ─▶ live overlay + HUD + report
```

[`docs/COMPUTER_VISION_RESEARCH.md`](docs/COMPUTER_VISION_RESEARCH.md)
explains which models, trackers and thresholds were chosen, what was measured,
and the papers behind them.

---

## Quick start

Requirements: **Python 3.10+** (tested with 3.13), **Node.js 20.19+** (tested
with 22), a recent Chrome, Edge, Safari or Firefox.

### 1. Backend (terminal 1)

```bash
cd backend
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or use the helper that does all of that: `./run.sh` (macOS/Linux) or
`.\run.ps1` (Windows PowerShell).

Check it: <http://localhost:8000/api/health> · interactive API docs:
<http://localhost:8000/api/docs>

### 2. Frontend (terminal 2)

```bash
python tools/fetch_models.py --fetch   # once: downloads the deep detector (YOLOX-S, 36 MB, SHA-256 checked)
cd frontend
npm install
npm run dev
```

Open **<http://localhost:5173>**. The home screen's self-test shows
`LOCAL CV ENGINE [ ACTIVE · 14 RULES ]` and `AI REASONING [ OFF · LOCAL ONLY ]`. Everything
works in this state. If the deep model was not fetched, the app still runs on the
fast detector alone and says so.

### 3. Add AI reasoning (optional)

```bash
cp .env.example .env               # in the repository root (or backend/.env)
# edit .env and set:  GEMINI_API_KEY=...
```

Restart the backend. With a key, the backend sends one compressed frame plus the
measured scene to Gemini only when it is useful:
- a newly confirmed finding of severity MEDIUM or higher;
- a new view;
- an ambiguous scene;
- you press *Explain* or *Deep Scan*.

Calls have a cooldown, a per-minute cap and de-duplication. If Gemini fails or
is rate limited, the local results stay on screen and the UI shows *LOCAL CV
ACTIVE · AI REASONING UNAVAILABLE*.

To use Claude instead, set `AI_PROVIDER=claude` and `ANTHROPIC_API_KEY`. A
second provider is **never** called automatically: `AI_FALLBACK_PROVIDER`
defaults to `none`.

---

## Use it on your phone (same Wi-Fi)

Browsers only allow camera access on **HTTPS** or on `localhost`. A phone
opening `http://192.168.x.x:5173` is neither, so **Live Scan needs the HTTPS
dev server** (Image Debug and Video Debug work over plain HTTP too).

1. Start the backend as above.
2. Start the frontend with HTTPS:

   ```bash
   cd frontend
   npm run dev:https
   ```

   Vite prints the addresses, e.g. `Network: https://192.168.1.23:5173/`.
3. Find your computer's LAN IP if needed:
   - **macOS:** `ipconfig getifaddr en0` (or System Settings → Wi-Fi → Details)
   - **Windows:** `ipconfig` → *IPv4 Address* of the Wi-Fi adapter
   - **Linux:** `hostname -I` or `ip -4 addr`
4. On the phone open **`https://<LAN-IP>:5173`**. The certificate is
   self-signed, so the browser warns once:
   - **Android / Chrome:** *Advanced* → *Proceed to 192.168.x.x*
   - **iPhone / Safari:** *Show Details* → *visit this website* → *Visit Website*
5. Tap **Enter live scan** and allow camera access.

Notes:

- The phone only talks to port **5173**. Vite proxies `/api` to the backend on
  your computer, so there are no CORS or mixed-content problems.
- **Firewall:** allow incoming TCP **5173** on your *private* network.
  Windows shows a prompt the first time Node listens — choose *Private
  networks*. macOS: System Settings → Network → Firewall → allow `node`.
  Linux/ufw: `sudo ufw allow 5173/tcp`. (Port 8000 only needs to be reachable
  from the phone if you call the API directly.)
- Prefer a certificate without warnings? Create one with
  [mkcert](https://github.com/FiloSottile/mkcert) (`mkcert -install`,
  `mkcert 192.168.1.23 localhost`, install mkcert's root CA on the phone), then
  `HTTPS_CERT=./192.168.1.23+1.pem HTTPS_KEY=./192.168.1.23+1-key.pem npm run dev:https`.
- If the camera prompt never appears, check the site permissions in the
  browser (address-bar icon) and that no other app is using the camera.

---

## Deploy (one public HTTPS URL)

The root `Dockerfile` builds the frontend and serves it from the backend
(`FRONTEND_DIST`), so the whole app is one service on one origin: no CORS, and
the camera works because the host provides HTTPS.

```bash
docker build -t reality-debugger .
docker run -p 8000:8000 --env-file .env reality-debugger   # http://localhost:8000
```

The image build downloads the deep detector and verifies its SHA-256. The
backend sends COOP/COEP headers, so the page is cross-origin isolated and the
deep detector can use multi-threaded WebAssembly.

On **Render**: New → Web Service → this repository → runtime **Docker**. No
environment variable is required. Optionally add `GEMINI_API_KEY` to enable AI
reasoning. Free instances sleep when idle, so the first visit can take about a
minute. Anyone with the URL can use the app. Once a key is set, keep the URL
private: AI calls are billed to that key, within the per-scan limits in
`config/ai.json`.

---

## What you can do

| Mode | What happens |
| --- | --- |
| **Live Scan** | Full-screen camera. The fast detector and tracker run on every frame. Every 1.5 s the scene model (numbers, no image) goes to the backend, whose rule engine finds issues. Findings move through **DISCOVERED → CONFIRMED → TRACKING → RESOLVED**, and a finding is resolved only when the condition is measured gone in the same view. Also: **Deep Scan** (freeze, YOLOX-S + fusion, verified objects ✓), **Explain** on any finding (with AI on), pause/resume, switch camera, an optional dev panel with live metrics, an end-scan summary, and clear session. |
| **Image Debug** | Take a photo, pick from the gallery or drag a file in. JPG, PNG and WEBP work, plus HEIC/AVIF wherever the browser can decode them. Visible stages: decode → fast detector → deep detector → fusion → local diagnostics → AI reasoning (if enabled) → report. Without AI the image never leaves the device. |
| **Video Debug** | The video stays on your device. It is sampled every ~0.5 s, and every sample runs through the detector and tracker with video time as the clock. Samples are segmented into scenes, keyframes are verified by the deep detector, and the backend replays the samples through the same lifecycle to build a timeline. Keyframes are uploaded only with AI on. Click any event to jump there. Videos the browser can't decode can be processed on the backend instead (OpenCV, signal-only). |
| **Personalities** | **Serious** (incident report), **Brutal** (*"This desk technically functions, but the cable management is committing crimes."*), **Unhinged** (*"ERROR 418: Desk has achieved maximum mug density."*). Humour lives in a separate `quip`; evidence and fixes stay factual. |
| **Share** | Copy the report as text, or download/share a 1080×1350 diagnostic card (optionally with the photo and bug boxes). |

---

## Configuration

Backend settings come from environment variables or `.env` (repository root or
`backend/`). See [`.env.example`](.env.example) for all of them.

| Variable | Default | Purpose |
| --- | --- | --- |
| `AI_PROVIDER` | `auto` | `auto` (Gemini if `GEMINI_API_KEY` is set, else local-only) · `none` · `gemini` · `claude` · `openai` |
| `AI_FALLBACK_PROVIDER` | `none` | Second provider after a failure. Only used if set explicitly |
| `GEMINI_API_KEY` | – | Enables the optional Gemini reasoning layer |
| `GEMINI_MODEL` / `GEMINI_BASE_URL` | `gemini-flash-latest` / SDK default | Gemini model and endpoint |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | – / `claude-opus-5-5` | Claude (with `AI_PROVIDER=claude`) |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` | – | Any OpenAI-compatible vision endpoint, e.g. Ollama (with `AI_PROVIDER=openai`) |
| `SCAN_AI_CALLS_PER_MINUTE` | `6` (from `config/ai.json`) | AI cost guard per live scan |
| `CONFIG_DIR` | `<repo>/config` | Shared CV/tracking/diagnostics/AI parameter files |
| `CROSS_ORIGIN_ISOLATION` | `true` | COOP/COEP headers (multi-threaded WASM for the deep detector) |
| `MAX_IMAGE_BYTES`, `MAX_FRAME_BYTES`, `MAX_VIDEO_BYTES`, `MAX_SCENE_CHARS` | 15 / 5 / 300 MB, 96 000 | Upload and payload limits |
| `CORS_ORIGINS`, `CORS_ALLOW_LAN` | localhost + private LAN | Allowed browser origins |

Every computer-vision, tracking, temporal, diagnostic and AI-usage parameter
is in [`config/`](config/README.md). Each one records its value, unit, purpose
and **source**: model documentation, a paper, a measurement, a library
default, or a heuristic.

Frontend (optional, `frontend/.env` or shell): `BACKEND_URL` (proxy target,
default `http://127.0.0.1:8000`), `PORT` (5173), `HTTPS_CERT`/`HTTPS_KEY`,
`VITE_API_BASE` (call a backend directly instead of the proxy).

**Cost:** without a key there is none. With AI on, one live-scan call is one
frame (downscaled to ≤ 1024 px for the model) plus the scene model as text. Automatic calls are limited in
four ways:
- a 15 s cooldown;
- at most 6 per minute per scan;
- near-identical frames reuse the cached answer for 2 min;
- after a failure, calls back off for 1 min, or 10 min for a bad key or model.

Explain, Deep Scan, Image and Video Debug are single calls that you trigger.

---

## How it works

### On-device vs. backend

| Runs in the browser (on device) | Runs in your backend |
| --- | --- |
| Camera capture, frame throttling | Scene-model and upload validation (schema, magic bytes, size, pixels) |
| Fast detector: EfficientDet-Lite0 int8 (MediaPipe, worker), every frame | Geometric relations: touching, near, on, overlaps (no LLM) |
| Deep detector: YOLOX-S (ONNX Runtime Web, WebGPU or multi-threaded WASM), selective | Local diagnostic engine: 16 attribute-based rules with measurements |
| ByteTrack tracking (Kalman + Hungarian), movement, occlusion, persistence | Finding lifecycle, scoring, scan sessions (in memory) |
| Fusion of both detectors (verify / relabel / add) | AI policy: triggers, cooldown, budget, de-duplication, back-off |
| Motion, brightness, sharpness, scene fingerprints, view changes | Optional Gemini/Claude call, schema-validated (key never leaves the backend) |
| Video sampling, tracking over samples, keyframe selection | Video lifecycle replay and timeline; OpenCV fallback sampling |
| Overlay, HUD, dev metrics, reports, share card | Metrics (`/api/metrics`) |

### Local computer vision

- **Two detectors.**
  - EfficientDet-Lite0 (4.6 MB, committed) runs on every frame at ≤ 10 fps. It measured 63–113 ms per frame on a 4-core CPU in headless Chromium.
  - YOLOX-S (36 MB, fetched) verifies selectively: Deep Scan, image and video keyframes, and live checks on devices fast enough for them. It measured ~0.45–0.53 s per frame with multi-threaded WASM.
  - On coco128, YOLOX-S reaches 48.3 AP vs 37.0 for Lite0, and recovers about 4× more small objects.
- **No hand-written label lists.** Class names come from each model's own
  metadata. Semantic attributes (liquid container, electronic, sharp, food,
  surface…) come from a generated ontology built from COCO supercategories,
  LVIS synsets and WordNet. The rules test attributes, so a new model with new
  classes works without code changes.
- **ByteTrack** keeps objects alive through occlusion and blur, and measures
  speed, static time, direction reversals, occlusion and truncation. The
  movement thresholds were measured, not guessed.
- **Views.** A camera cut or pan to a new view is detected on settled frames.
  That way a finding that leaves the frame is *out of view*, not falsely
  *resolved*.

### Optional AI reasoning

The AI is the reasoning layer on top of measured data, not the detector. It
receives:
- the scene model;
- the computed relations;
- the local findings with their measurements;
- one compressed frame.

It returns structured JSON (a strict schema, validated item by item). The JSON
explains the local findings (agree/disagree with a note), adds issues the
detectors cannot see, and names the scene. The prompt enforces these rules:
- only visible evidence;
- observation separated from inference;
- no invented objects;
- no identification of people;
- no sensitive attributes;
- no medical or legal conclusions;
- no dangerous instructions;
- honest confidence;
- 3–7 findings;
- actionable fixes;
- humour grounded in the scene.

For details, see:
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the pipeline, lifecycle rules, scoring and AI policy;
- [`docs/API.md`](docs/API.md) for the HTTP API;
- [`docs/COMPUTER_VISION_RESEARCH.md`](docs/COMPUTER_VISION_RESEARCH.md) for the research and measurements.

---

## Privacy & security

- API keys live only in the backend's environment; the frontend never sees them.
- Without an AI key, **no image or video frame ever leaves the device**: the
  backend receives only scene models (labels, boxes, numbers). With AI on,
  only the frames the policy selects (or you send) are uploaded.
- Images are processed in memory, re-encoded (stripping EXIF/GPS) and never
  stored. Live-scan sessions keep findings text only, in memory, and expire
  after 2 h of inactivity — *Clear session* deletes them immediately.
- Logs contain method, path, status and timing — never images, bodies or keys
  (a log filter also masks anything that looks like a key).
- Uploads are checked by magic bytes, size and pixel count; request bodies are
  capped; CORS allows only localhost and private-LAN origins.

---

## Testing

```bash
# Backend: local engine, lifecycle, AI provider selection and policy, Gemini /
# Claude / OpenAI request paths against mock HTTP transports, API, security
cd backend && source venv/bin/activate
pip install -r requirements-dev.txt
pytest

# Frontend: unit tests (assignment, Kalman filter, ByteTrack tracker, YOLOX
# decoding, detector fusion), type check, lint, production build
cd frontend
npm test
npm run typecheck
npm run lint
npm run build

# End-to-end (real browser, fake camera, no API key needed; backend must be running)
npx playwright install chromium   # once
npm run test:e2e

# Models: verify the detector files and manifests (no network)
python tools/fetch_models.py --check
```

**Benchmarks.** `npm run dev`, then open
`/vision-lab.html?images=a.jpg,b.jpg&runs=5` with images placed in
`frontend/public/test-media/` (git-ignored). It times both detectors and the
tracker in your browser and exposes the results as `window.__lab`;
`node scripts/vision-lab.mjs "<that URL>" out.json` (in `frontend/`) runs it in
headless Chromium and saves them. Recorded results are in
[`docs/benchmarks/`](docs/benchmarks/). The backend's local engine is
timed by `tools/bench_local_engine.py`, and the detector accuracy evaluation is
`tools/eval_detectors.py`; see their docstrings.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Home self-test: **BACKEND LINK [ OFFLINE ]** | Start the backend (step 1). The frontend proxies `/api` to `127.0.0.1:8000`. |
| **Camera needs HTTPS** on the phone | Use `npm run dev:https` and the `https://` LAN address. |
| **Camera permission denied** | Re-enable camera access in the browser's site settings, then *Retry*. |
| Vision engine slow / **FAULT** | Set *Vision backend* to *CPU* in Instrument settings (or open `?vision=cpu`), reload. |
| **AI REASONING UNAVAILABLE** | Local results are still valid. Check `GEMINI_API_KEY` / `GEMINI_MODEL` (or your `AI_PROVIDER` settings) and `GET /api/health/ai`, then restart the backend. Bad keys pause automatic AI calls for 10 min. |
| **Deep detector not installed** | Run `python tools/fetch_models.py --fetch` and reload. The fast detector keeps working without it. |
| Deep detector slow | It picks WebGPU only on a hardware GPU and otherwise multi-threaded WASM. Threads need cross-origin isolation (`CROSS_ORIGIN_ISOLATION=true`, the default). In *Instrument settings*, *Deep detector* can be switched off and *Deep detector runtime* can force WebGPU or WASM. |
| **Cannot read this video** | The browser can't decode the codec (e.g. HEVC on some desktops). Use *Process on the backend instead*, or convert to MP4 (H.264). |
| Phone can't load the page | Same Wi-Fi? Firewall allows 5173? Some guest/office networks isolate devices. |

---

## Project structure

```
.
├── .env.example                 backend configuration template
├── backend/
│   ├── app/
│   │   ├── main.py              FastAPI app, middleware, CORS, routers
│   │   ├── config.py            environment-based settings
│   │   ├── errors.py            error envelope (no stack traces to clients)
│   │   ├── api/                 health/metrics, scan (observe, deep), analyze (frame, scene), image, video
│   │   ├── services/            local_diagnostics (rules), geometry, diagnostic_service (lifecycle),
│   │   │                        pipeline, ai_policy, ai_service, ai_providers, prompts, metrics,
│   │   │                        scan_store, video_service, vision_service
│   │   ├── schemas/             scene model, request/AI wire models, response models
│   │   └── utils/               lenient coercion helpers
│   ├── tests/                   pytest suite
│   ├── requirements.txt         pinned runtime dependencies
│   └── run.sh / run.ps1         one-step start scripts
├── config/                      all CV / tracking / temporal / diagnostics / AI parameters + generated ontology
├── tools/                       fetch_models.py, build_ontology.py, eval_detectors.py
├── frontend/
│   ├── public/models/           detector manifests + EfficientDet-Lite0 (YOLOX-S is fetched)
│   ├── vision-lab.html          in-browser benchmark harness (dev only)
│   ├── src/
│   │   ├── vision/              fast worker, deep/ (YOLOX-S worker), tracker (ByteTrack), kalman,
│   │   │                        assignment, fusion, scene model, ontology, signals, relations
│   │   ├── config/              typed access to ../config
│   │   ├── live/                camera, capture, observe loop, overlay, session
│   │   ├── video/               on-device sampling + keyframe selection
│   │   ├── screens/             Home, LiveScan, ImageDebug, VideoDebug
│   │   ├── components/          report, finding cards, meters, error panel…
│   │   ├── lib/                 API client (+ Zod schemas), share card, formatting
│   │   └── styles/              design tokens and screen styles
│   ├── e2e/                     Playwright end-to-end tests
│   └── vite.config.ts           dev server (0.0.0.0, /api proxy, HTTPS mode)
└── docs/                        architecture, API reference, CV research, benchmarks/
```

## Limitations

- Both detectors know the 80 COCO classes. Cables, sockets, papers and stains
  are invisible to the local engine. With AI on they can appear as AI findings,
  but they cannot be tracked frame by frame. The `cable_congestion` and
  `liquid_near_outlet` rules activate automatically once a model with those
  classes is added.
- Geometry is 2D, with no depth: "near" and "on" are image-plane relations.
- Timings were measured on one 4-core x86 container CPU in headless Chromium.
  Phones will be slower. WebGPU on a real GPU was not available to measure.
- Detector accuracy was measured on coco128, a COCO training subset, so the
  absolute AP is optimistic.
- Live-scan sessions live in backend memory: restarting the backend forgets them.
- The self-signed HTTPS certificate triggers a browser warning on the phone
  (use mkcert to avoid it).

## Next steps

1. **Open-vocabulary deep detector.** A quantised OWLv2 or YOLO-World-class
   model would add cables, sockets and stains to the local vocabulary. The
   manifest and ontology pipeline is ready for one.
2. **Depth.** Monocular depth (e.g. Depth Anything V2 small) would turn 2D
   "near"/"on" into 3D relations.
3. **Fine-tuning on desk scenes.** Collect labelled indoor desk/kitchen frames
   (opt-in) to measure real accuracy and tune the operating points.
