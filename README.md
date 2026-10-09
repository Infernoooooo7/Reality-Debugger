# Reality Debugger

> **Your world has bugs.**

Reality Debugger treats the physical world like software. Point your phone at a
desk, a kitchen or a gaming setup — or upload a photo or a video — and get a
debugging report for it: bugs with severity, evidence, impact and a fix, a
system score, and a final diagnosis. In Live Scan the findings have a lifecycle
— **DISCOVERED → CONFIRMED → TRACKING → RESOLVED** — so when you move the mug
away from the laptop, the bug closes in front of you.

It is a normal standalone project: a **React + TypeScript + Vite** frontend, a
**Python + FastAPI** backend, an **on-device vision engine** (MediaPipe
EfficientDet-Lite0 in a Web Worker) and a **vision-capable LLM** (Claude by
default) called only by the backend, only on frames that are worth it.

```
camera ─▶ on-device detection + tracking (every frame, 6-15 fps)
       ─▶ motion · scene change · spatial heuristics
       ─▶ scan director picks a frame when something meaningful happens
       ─▶ your backend validates it, calls the vision model, validates the JSON
       ─▶ diagnostic engine updates the finding lifecycle ─▶ live overlay + HUD
```

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
cd frontend
npm install
npm run dev
```

Open **<http://localhost:5173>**.

### 3. Add an AI key (optional)

Without a key the app runs in **DEMO MODE**: everything works, the on-device ML
is real, but diagnoses are produced by fixed heuristics and are clearly labelled
*simulated* everywhere (yellow hazard-stripe badge, on reports and share cards).

For real vision-model diagnostics:

```bash
cp .env.example .env               # in the repository root (or backend/.env)
# edit .env and set:  ANTHROPIC_API_KEY=sk-ant-...
```

Restart the backend. The home screen's self-test should show
`AI REASONER [ ONLINE · claude-opus-5-5 ]`.

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

On **Render**: New → Web Service → this repository → runtime **Docker**. Add
`ANTHROPIC_API_KEY` under Environment for real diagnostics (otherwise DEMO
MODE). Free instances sleep when idle, so the first visit can take about a
minute. Anyone with the URL can use the app, so keep it private once a key is
set: every analysis is billed to that key.

---

## What you can do

| Mode | What happens |
| --- | --- |
| **Live Scan** | Full-screen camera. Local ML detects and tracks objects every frame; the scan director sends a frame to the vision model only on a first look, a new object, an interesting spatial relationship, a substantial scene change, a finding that needs confirmation or a possible resolution, or a periodic re-check. Findings animate through their lifecycle. Pause/resume, **Deep Scan** (freeze + most thorough analysis + *Return to live*), switch camera, end-scan summary, clear session. |
| **Image Debug** | Take a photo, pick from the gallery or drag a file in (JPG, PNG, WEBP — and HEIC/AVIF wherever the browser can decode them; images are re-encoded to JPEG before upload). Visible stages: local preprocessing → on-device detection → relationship analysis → EXIF-stripped upload → vision reasoning → validated report. Findings are drawn on the image. |
| **Video Debug** | The video stays on your device: sampled every ~0.5 s, fingerprinted, run through the detector, segmented into scenes, de-duplicated; ≤ 8 keyframes go to the model, which narrates a chronological timeline. Click any event, keyframe or finding time to jump there. Videos the browser can't decode can be processed on the backend instead (OpenCV). |
| **Personalities** | **Serious** (incident report), **Brutal** (*"This desk technically functions, but the cable management is committing crimes."*), **Unhinged** (*"ERROR 418: Desk has achieved maximum mug density."*). Humour lives in a separate `quip`; evidence and fixes stay factual. |
| **Share** | Copy the report as text, or download/share a 1080×1350 diagnostic card (optionally with the photo and bug boxes). |

---

## Configuration

Backend settings come from environment variables or `.env` (repository root or
`backend/`). See [`.env.example`](.env.example) for all of them.

| Variable | Default | Purpose |
| --- | --- | --- |
| `AI_PROVIDER` | `auto` | `auto` · `anthropic` · `openai` · `demo` |
| `ANTHROPIC_API_KEY` | – | Enables Claude |
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | Any vision-capable Claude model |
| `ANTHROPIC_REFUSAL_FALLBACK` | `true` | Server-side refusal fallback (beta); auto-disabled if rejected |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` | – | Any OpenAI-compatible vision endpoint (OpenAI, Ollama, LM Studio, vLLM…) |
| `AI_EFFORT_LIVE` / `AI_EFFORT_DEEP` | `low` / `medium` | Model effort for live frames vs. deep/image/video |
| `SCAN_AI_CALLS_PER_MINUTE` | `12` | Cost guard per live scan |
| `MAX_IMAGE_BYTES`, `MAX_FRAME_BYTES`, `MAX_VIDEO_BYTES` | 15 / 5 / 300 MB | Upload limits |
| `CORS_ORIGINS`, `CORS_ALLOW_LAN` | localhost + private LAN | Allowed browser origins |

Frontend (optional, `frontend/.env` or shell): `BACKEND_URL` (proxy target,
default `http://127.0.0.1:8000`), `PORT` (5173), `HTTPS_CERT`/`HTTPS_KEY`,
`VITE_API_BASE` (call a backend directly instead of the proxy).

**Cost:** a live analysis sends one ~1024 px frame plus a cached system
prompt. The scan director only calls the model on meaningful changes, at most
once every 5 s, and the backend caps a scan at `SCAN_AI_CALLS_PER_MINUTE` (12);
lengthen or switch off the periodic re-check in *Instrument settings* to save
more. Deep scans, images and videos (≤ 8 keyframes) are single calls. To trade
quality for cost and latency, set `ANTHROPIC_MODEL` to a smaller Claude model or
lower `AI_EFFORT_DEEP`.

**Fully local option:** run [Ollama](https://ollama.com) with a vision model and
set `OPENAI_BASE_URL=http://localhost:11434/v1`, `OPENAI_MODEL=llama3.2-vision`
— no cloud calls at all.

---

## How it works

### On-device vs. backend

| Runs in the browser (on device) | Runs in your backend |
| --- | --- |
| Camera capture, frame throttling | Upload validation (magic bytes, size, pixel limits) |
| Object detection (EfficientDet-Lite0, 80 COCO classes) | Image normalisation, EXIF/GPS stripping |
| Multi-object tracking with stable IDs | Vision-model calls (API key never leaves the backend) |
| Motion detection, scene-change fingerprints, sharpness | Strict schema validation + repair of model output |
| Spatial heuristics (e.g. cup next to laptop) | Finding lifecycle, scoring, scan sessions (in memory) |
| Deciding *which* frame deserves the AI | Video timeline assembly; OpenCV fallback sampling |
| Video sampling, scene detection, keyframe selection | Demo-mode heuristics |
| Overlay, HUD, reports, share card | |

### Local ML

- **Model:** MediaPipe Tasks **ObjectDetector + EfficientDet-Lite0 (int8)**,
  bundled in `frontend/public/models/` (4.6 MB, Apache-2.0), runs on
  WebAssembly/XNNPACK or the WebGL GPU delegate inside a **module Web Worker**
  (main-thread fallback). Chosen over TF.js COCO-SSD (lower COCO accuracy,
  framework in maintenance mode) and ONNX Runtime + YOLO (larger runtime and
  models, AGPL-licensed weights) because it is small, accurate for its size,
  runs on phones without a GPU and works cleanly inside a worker.
- **Verified delegate:** at start-up a procedurally drawn stop sign is run
  through CPU and GPU; GPU is kept only if it actually detects it *and* is
  faster (some GPU drivers silently return nothing). The choice is cached per
  device; override it in *Instrument settings* or with `?vision=cpu`.
- **Letterboxing:** frames are padded to a square before detection —
  stretching 4:3/16:9 frames measurably hurt recall in testing.
- **Tracking:** IoU association with constant-velocity prediction, class
  voting, tentative/confirmed/lost states.
- **Signals:** motion by frame differencing, scene change by colour-histogram +
  luminance-grid fingerprints, Laplacian sharpness, brightness.
- **Replaceable:** everything depends on the `Detector` interface in
  `frontend/src/vision/detector.ts`.

### Vision AI

The model is used for the parts that need reasoning: what the scene is, how
objects relate, what is wrong, why it matters, how to fix it, severity,
confidence, score and the final diagnosis. Requests use **structured JSON
output** (a strict JSON schema), images are downscaled (1024 px live, 1600 px
deep/image), the system prompt is cached, and **every response is validated**
field by field — malformed items are dropped with a warning, unusable output is
retried once and otherwise reported as a clean error. The prompt enforces the
safety rules: only visible evidence, observation separated from inference, no
invented objects, no identification of people, no sensitive attributes, no
medical or legal conclusions, no dangerous instructions, honest confidence,
3–7 findings, actionable fixes, humour grounded in the scene.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the scan director,
lifecycle rules, scoring and video pipeline, and [`docs/API.md`](docs/API.md)
for the HTTP API.

---

## Privacy & security

- API keys live only in the backend's environment; the frontend never sees them.
- Frames leave the device only when the local engine selects them; videos are
  sampled on the device and only keyframes are uploaded.
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
# Backend: 46 tests (endpoints, validation, lifecycle, video, CORS, and the
# Anthropic/OpenAI request paths against mock HTTP transports)
cd backend && source venv/bin/activate
pip install -r requirements-dev.txt
pytest

# Frontend: type check + production build
cd frontend
npm run typecheck
npm run build

# End-to-end (real browser, fake camera; backend must be running, demo is fine)
npx playwright install chromium   # once
npm run test:e2e
```

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Home self-test: **BACKEND LINK [ OFFLINE ]** | Start the backend (step 1). The frontend proxies `/api` to `127.0.0.1:8000`. |
| **Camera needs HTTPS** on the phone | Use `npm run dev:https` and the `https://` LAN address. |
| **Camera permission denied** | Re-enable camera access in the browser's site settings, then *Retry*. |
| Vision engine slow / **FAULT** | Set *Vision backend* to *CPU* in Instrument settings (or open `?vision=cpu`), reload. |
| **AI key rejected** / **model not found** | Check `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` in `.env`, restart the backend. Live Scan offers *Switch to demo mode*. |
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
│   │   ├── api/                 health, analyze (frame), image, video, scan
│   │   ├── services/            ai_service, ai_providers, prompts, demo_reasoner,
│   │   │                        diagnostic_service, scan_store, video_service,
│   │   │                        vision_service
│   │   ├── schemas/             request/AI wire models + response models
│   │   └── utils/               lenient coercion helpers
│   ├── tests/                   pytest suite
│   ├── requirements.txt         pinned runtime dependencies
│   └── run.sh / run.ps1         one-step start scripts
├── frontend/
│   ├── public/models/           EfficientDet-Lite0 model (bundled)
│   ├── src/
│   │   ├── vision/              worker, engine, detector, tracker, signals, relations
│   │   ├── live/                camera, capture, scan director, overlay, session
│   │   ├── video/               on-device sampling + keyframe selection
│   │   ├── screens/             Home, LiveScan, ImageDebug, VideoDebug
│   │   ├── components/          report, finding cards, meters, error panel…
│   │   ├── lib/                 API client (+ Zod schemas), share card, formatting
│   │   └── styles/              design tokens and screen styles
│   ├── e2e/                     Playwright end-to-end tests
│   └── vite.config.ts           dev server (0.0.0.0, /api proxy, HTTPS mode)
└── docs/                        architecture notes and API reference
```

## Limitations

- The on-device detector knows the 80 COCO classes; things like cables or
  sockets are only understood by the vision model, so their findings can't be
  tracked frame-by-frame (they show at their analysed position while the view
  is similar, and are re-checked on the next analysis).
- Finding positions come from the vision model and are approximate.
- Live-scan sessions live in backend memory: restarting the backend forgets them.
- The self-signed HTTPS certificate triggers a browser warning on the phone
  (use mkcert to avoid it).
- Demo mode is deliberately simple: a handful of fixed rules over detections and
  pixel statistics.

## Next steps

1. **Region tracking for non-COCO findings** — track the analysed region with
   optical flow / feature matching so cable or socket bugs follow the camera.
2. **Streamed reasoning** — stream partial findings over a WebSocket so the
   first bug appears while the model is still thinking.
3. **Scan history & regressions** — persist scans (opt-in) and diff reality
   over time: *"BUG-003 regressed since Tuesday."*
