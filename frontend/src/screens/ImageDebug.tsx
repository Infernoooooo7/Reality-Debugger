import { useCallback, useEffect, useRef, useState, type DragEvent } from 'react'
import { AnnotatedImage } from '../components/AnnotatedImage'
import { AIBadge, LocalBadge, PersonalitySwitch } from '../components/Bits'
import { ErrorPanel } from '../components/ErrorPanel'
import { Icon } from '../components/Icon'
import { ReportView } from '../components/ReportView'
import { DETECTION, VISION } from '../config'
import { analyzeImage, analyzeScene, ApiError, isAbort, toApiError } from '../lib/api'
import { formatBytes, formatMs } from '../lib/format'
import { navigate } from '../lib/router'
import type { Report } from '../lib/schemas'
import { useSettings } from '../state/settings'
import { aiEnabled, aiMode, providerLabel, useSystem } from '../state/system'
import { JpegCapturer } from '../live/capture'
import { vision } from '../vision/client'
import { measureCoverage } from '../vision/coverage'
import { deepDetector } from '../vision/deep/client'
import { detectTiled, planTile, tileWindows, TILING_DEFAULTS, type TiledResult } from '../vision/deep/tiling'
import { fuseStill, type FusedObject } from '../vision/fusion'
import { detectorRun, modelFacts } from '../vision/runs'
import { coveragePayload, stillScene, type CoveragePayload, type DetectorRunPayload } from '../vision/scene'
import type { StillResult } from '../vision/types'
import '../styles/debug.css'

// JPG/PNG/WEBP everywhere; HEIC/HEIF/AVIF etc. work wherever the browser can
// decode them, because the image is re-encoded to JPEG before upload.
const ACCEPT = 'image/jpeg,image/png,image/webp,image/heic,image/heif,image/avif'
const IMAGE_NAME = /\.(jpe?g|png|webp|heic|heif|avif)$/i

type StageId = 'decode' | 'fast' | 'deep' | 'fuse' | 'local' | 'reason' | 'report'
type StageState = 'pending' | 'active' | 'done' | 'skipped' | 'failed'

const STAGES: { id: StageId; label: string }[] = [
  { id: 'decode', label: 'Local preprocessing' },
  { id: 'fast', label: 'Fast detector (on device)' },
  { id: 'deep', label: 'Deep detector (on device)' },
  { id: 'fuse', label: 'Detector fusion → scene model' },
  { id: 'local', label: 'Local diagnostic engine' },
  { id: 'reason', label: 'AI reasoning (optional)' },
  { id: 'report', label: 'Validated report' },
]

interface Picked {
  file: File
  url: string
  width: number
  height: number
}

export default function ImageDebug() {
  const settings = useSettings()
  const health = useSystem((s) => s.health)
  const record = useSystem((s) => s.record)
  const [picked, setPicked] = useState<Picked | null>(null)
  const [stages, setStages] = useState<Record<StageId, { state: StageState; note?: string }>>(() => resetStages())
  const [error, setError] = useState<ApiError | null>(null)
  const [report, setReport] = useState<Report | null>(null)
  const [detections, setDetections] = useState<FusedObject[]>([])
  const [focusId, setFocusId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const capturer = useRef<JpegCapturer | null>(null)
  const [imageEl, setImageEl] = useState<HTMLImageElement | null>(null)
  const limit = health?.limits.max_image_bytes ?? 15 * 1024 * 1024
  const withAI = aiEnabled(health)

  useEffect(() => () => abortRef.current?.abort(), [])
  useEffect(() => () => (picked ? URL.revokeObjectURL(picked.url) : undefined), [picked])

  const setStage = (id: StageId, state: StageState, note?: string) =>
    setStages((prev) => ({ ...prev, [id]: { state, note } }))

  const run = useCallback(
    async (file: File) => {
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller
      setError(null)
      setReport(null)
      setDetections([])
      setFocusId(null)
      setStages(resetStages())

      if (!file.type.startsWith('image/') && !IMAGE_NAME.test(file.name)) {
        setError(new ApiError('UNSUPPORTED_MEDIA_TYPE', `“${file.name}” is not a supported image.`, { hint: 'Use a JPG, PNG or WEBP image.' }))
        return
      }
      if (file.size > limit) {
        setError(new ApiError('PAYLOAD_TOO_LARGE', `The image is ${formatBytes(file.size)}; the limit is ${formatBytes(limit)}.`, { hint: 'Export a smaller version and try again.' }))
        return
      }

      setBusy(true)
      try {
        // 1. Local preprocessing: decode (respecting EXIF orientation) + size checks.
        setStage('decode', 'active')
        const started = performance.now()
        let bitmap: ImageBitmap
        try {
          bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' })
        } catch {
          throw new ApiError('INVALID_UPLOAD', 'This image could not be decoded.', {
            hint: 'The file may be corrupted, or use a format this browser cannot read (e.g. HEIC). Export it as JPG or PNG.',
          })
        }
        const url = URL.createObjectURL(file)
        setPicked({ file, url, width: bitmap.width, height: bitmap.height })
        setStage('decode', 'done', `${bitmap.width}×${bitmap.height} · ${formatMs(performance.now() - started)}`)

        // 2. Fast detector (EfficientDet-Lite0). Every detector reports what it actually did
        //    (status, boxes, input size, tiling): the backend uses this to tell "nothing found"
        //    apart from "nothing examined" (backend/app/services/inspection.py).
        const runs: DetectorRunPayload[] = []
        const [fastFacts, deepFacts] = await Promise.all([modelFacts(VISION.fast.model), modelFacts(VISION.deep.model)])
        let fast: StillResult | null = null
        setStage('fast', 'active', vision.ready ? undefined : 'loading model…')
        try {
          fast = await vision.analyzeStill(await createImageBitmap(bitmap))
          setDetections(fuseStill(fast.detections, null))
          runs.push(detectorRun(VISION.fast.model, 'fast', 'ok', fastFacts, { boxes: fast.detections.length, ms: fast.inferenceMs }))
          setStage('fast', 'done', `${fast.detections.length} boxes · ${fastFacts.inputSize ?? '?'} px input · ${formatMs(fast.inferenceMs)}`)
        } catch (e) {
          const message = toApiError(e).message
          runs.push(detectorRun(VISION.fast.model, 'fast', 'failed', fastFacts, { note: message }))
          setStage('fast', 'failed', message.slice(0, 60))
        }

        // 3. Deep detector (YOLOX-S), loaded on first use. Large photos are also scanned in
        //    overlapping tiles so that small objects keep their pixels (vision/deep/tiling.ts).
        let deep: TiledResult | null = null
        if (deepDetector.enabled) {
          setStage('deep', 'active', deepDetector.ready ? undefined : 'loading model (one-time download)…')
          const inputSize = deepFacts.inputSize ?? DETECTION.deep.inputSize
          const tile = TILING_DEFAULTS.enabled === false ? null : planTile(bitmap.width, bitmap.height, inputSize, TILING_DEFAULTS)
          const planned = tile === null ? 1 : 1 + tileWindows(bitmap.width, bitmap.height, tile, TILING_DEFAULTS.overlap).length
          let pass = 0
          try {
            // Load first (one-time download) so that the reported time is detection time only.
            if (!deepDetector.ready) await deepDetector.init()
            deep = await detectTiled(
              bitmap,
              (b) => {
                pass++
                if (planned > 1) setStage('deep', 'active', `pass ${pass} of ${planned} (${tile} px tiles)…`)
                return deepDetector.detect(b)
              },
              inputSize,
              TILING_DEFAULTS,
            )
            const info = deepDetector.getSnapshot().info
            const tiled = deep.tilePx !== null ? ` · ${deep.passes} passes (${deep.tilePx} px tiles${deep.incomplete ? ', stopped early' : ''})` : ''
            runs.push(
              detectorRun(VISION.deep.model, 'deep', 'ok', deepFacts, {
                boxes: deep.detections.length,
                ms: deep.totalMs,
                passes: deep.passes,
                tile_px: deep.tilePx,
                incomplete: deep.incomplete,
                note: deep.incomplete ? `time budget reached after ${deep.passes - 1} of ${deep.planned} tiles` : null,
              }),
            )
            setStage('deep', 'done', `${deep.detections.length} boxes${tiled} · ${formatMs(deep.totalMs)} · ${info?.backend ?? ''}`)
          } catch (e) {
            const message = toApiError(e).message
            runs.push(detectorRun(VISION.deep.model, 'deep', 'failed', deepFacts, { note: message }))
            setStage('deep', 'failed', message.slice(0, 70))
          }
        } else {
          runs.push(detectorRun(VISION.deep.model, 'deep', 'skipped', deepFacts, { note: 'switched off in settings' }))
          setStage('deep', 'skipped', 'switched off in settings')
        }

        // 4. Fusion -> scene model (measured on this device). The scene is always built, even
        //    when a detector failed: the report must say so rather than look clean.
        const fused = fuseStill(fast?.detections ?? [], deep?.detections ?? null)
        setDetections(fused)
        const detectors = runs.filter((r) => r.status === 'ok').map((r) => r.model)
        let coverage: CoveragePayload | null = null
        try {
          coverage = coveragePayload(measureCoverage(bitmap, bitmap.width, bitmap.height, fused.map((o) => o.box)))
        } catch {
          coverage = null // reported by the backend as "structure not measured"
        }
        const scene = stillScene(fused, fast?.signals ?? null, { width: bitmap.width, height: bitmap.height, detectors, runs, coverage })
        const verified = fused.filter((o) => o.verified).length
        setStage(
          'fuse',
          detectors.length ? 'done' : 'failed',
          detectors.length
            ? `${fused.length} objects · ${verified} confirmed by both${coverage?.unexplained_share != null ? ` · ${Math.round(coverage.unexplained_share * 100)}% of detail unexplained` : ''}`
            : 'no detector produced a result',
        )
        setImageEl(await loadImage(url))

        // 5. Local diagnostics, plus AI reasoning when a provider is configured
        //    (only then does the image - resized, EXIF stripped - leave the device).
        let result: Report
        setStage('local', 'active')
        const reasonStarted = performance.now()
        if (withAI) {
          capturer.current ??= new JpegCapturer()
          const shot = await capturer.current.capture(bitmap, VISION.capture.deepScanMaxEdge, VISION.capture.deepScanQuality)
          setStage('reason', 'active', `${providerLabel(health?.ai.provider)} · ${formatBytes(shot.blob.size)} JPEG`)
          result = await analyzeImage(shot.blob, { personality: settings.personality, scene, signal: controller.signal })
          const run = result.ai
          setStage(
            'reason',
            run.status === 'ok' || run.status === 'cached' ? 'done' : run.status === 'unavailable' ? 'failed' : 'skipped',
            run.status === 'ok' ? formatMs(run.latency_ms ?? performance.now() - reasonStarted) : (run.error?.code ?? run.reason ?? run.status),
          )
        } else {
          result = await analyzeScene(scene, { personality: settings.personality, signal: controller.signal })
          setStage('reason', 'skipped', 'off - no image uploaded')
        }
        bitmap.close()
        setStage('local', 'done', `${result.findings.filter((f) => f.source === 'local').length} measured findings`)
        setStage('report', 'done', `${result.findings.length} findings${result.warnings.length ? ` · ${result.warnings.length} warning(s)` : ''}`)
        setReport(result)
        record(result)
      } catch (e) {
        if (isAbort(e)) return
        const err = toApiError(e)
        setError(err)
        setStages((prev) => {
          const next = { ...prev }
          for (const s of STAGES) if (next[s.id].state === 'active') next[s.id] = { state: 'failed', note: err.code }
          return next
        })
      } finally {
        setBusy(false)
      }
    },
    [health, limit, record, settings.personality, withAI],
  )

  const onFiles = (files: FileList | null) => {
    const file = files?.[0]
    if (file) void run(file)
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    onFiles(e.dataTransfer.files)
  }

  return (
    <div className="debug" onDragOver={(e) => (e.preventDefault(), setDragging(true))} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
      <header className="debug__bar">
        <button type="button" className="key key--small key--ghost" onClick={() => navigate('home')}>
          <Icon name="back" size={16} /> Home
        </button>
        <span className="debug__title">
          <span className="t-data t-muted">IN·B</span> Image debug
        </span>
        <LocalBadge />
        <AIBadge mode={aiMode(health)} provider={providerLabel(health?.ai.provider)} />
      </header>

      <div className="debug__grid">
        <section className="debug__input">
          {picked ? (
            <div className="debug__preview">
              {report ? (
                <AnnotatedImage src={picked.url} findings={report.findings} focusedId={focusId} onFocus={setFocusId} alt="Analysed image" />
              ) : (
                <figure className="annotated" data-scanning={busy || undefined}>
                  <img src={picked.url} alt="Selected image" />
                  {detections.map((d) => (
                    <span
                      key={d.id}
                      className="annotated__local"
                      data-source={d.source}
                      style={{ left: `${d.box.x * 100}%`, top: `${d.box.y * 100}%`, width: `${d.box.w * 100}%`, height: `${d.box.h * 100}%` }}
                    >
                      <em>
                        {d.label} {Math.round(d.confidence * 100)}%{d.verified ? ' ✓' : d.source === 'deep' ? ' ◆' : ''}
                      </em>
                    </span>
                  ))}
                </figure>
              )}
              <dl className="debug__meta t-data">
                <div>
                  <dt>File</dt>
                  <dd>{picked.file.name}</dd>
                </div>
                <div>
                  <dt>Dimensions</dt>
                  <dd>
                    {picked.width}×{picked.height}
                  </dd>
                </div>
                <div>
                  <dt>Size</dt>
                  <dd>{formatBytes(picked.file.size)}</dd>
                </div>
                <div>
                  <dt>Type</dt>
                  <dd>{picked.file.type || 'unknown'}</dd>
                </div>
              </dl>
            </div>
          ) : (
            <div className={`dropzone${dragging ? ' dropzone--over' : ''}`}>
              <div className="dropzone__mark" aria-hidden="true">
                <Icon name="image" size={40} strokeWidth={1.6} />
              </div>
              <p className="dropzone__title t-display">Drop an image</p>
              <p className="t-muted">JPG · PNG · WEBP · up to {formatBytes(limit)}</p>
            </div>
          )}

          <div className="debug__pickers">
            <label className="key key--signal debug__pick">
              <Icon name="upload" size={18} /> {picked ? 'Another image' : 'Choose image'}
              <input type="file" accept={ACCEPT} onChange={(e) => onFiles(e.target.files)} disabled={busy} hidden />
            </label>
            <label className="key debug__pick">
              <Icon name="scan" size={18} /> Take photo
              <input type="file" accept="image/*" capture="environment" onChange={(e) => onFiles(e.target.files)} disabled={busy} hidden />
            </label>
          </div>

          <div className="module">
            <div className="module__head">
              <span className="t-label">Debug personality</span>
            </div>
            <PersonalitySwitch value={settings.personality} onChange={(personality) => settings.update({ personality })} />
          </div>
        </section>

        <section className="debug__output">
          {picked || error ? (
            <ol className="stages" aria-label="Analysis progress">
              {STAGES.map((s) => (
                <li key={s.id} data-state={stages[s.id].state}>
                  <span className="stages__dot" aria-hidden="true" />
                  <span className="stages__label">{s.label}</span>
                  <span className="stages__note t-data">{stages[s.id].note ?? ''}</span>
                </li>
              ))}
            </ol>
          ) : (
            <div className="debug__explain">
              <p className="t-label">What happens</p>
              <p>
                Your image is decoded and scanned on this device by two detectors (a fast one and a deeper one); their
                boxes are fused into a scene model that the local diagnostic engine turns into measured findings. Only
                those numbers go to your backend. If an AI provider is configured, a resized copy with its metadata
                (including GPS) stripped is sent for optional reasoning. Nothing is stored.
              </p>
            </div>
          )}

          {error ? (
            <ErrorPanel
              error={error}
              actions={
                <>
                  {picked && error.retryable ? (
                    <button type="button" className="key key--small" onClick={() => void run(picked.file)}>
                      <Icon name="retry" size={16} /> Retry
                    </button>
                  ) : null}

                </>
              }
            />
          ) : null}

          {busy && !report ? <div className="spinner-bar" /> : null}

          {report && picked ? (
            <ReportView
              report={report}
              image={imageEl ? { source: imageEl, width: picked.width, height: picked.height } : null}
              focusedId={focusId}
              onFocus={setFocusId}
            />
          ) : null}
        </section>
      </div>
    </div>
  )
}

function resetStages(): Record<StageId, { state: StageState; note?: string }> {
  return {
    decode: { state: 'pending' },
    fast: { state: 'pending' },
    deep: { state: 'pending' },
    fuse: { state: 'pending' },
    local: { state: 'pending' },
    reason: { state: 'pending' },
    report: { state: 'pending' },
  }
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('image load failed'))
    img.src = url
  })
}
