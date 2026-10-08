import { useCallback, useEffect, useRef, useState, type DragEvent } from 'react'
import { AnnotatedImage } from '../components/AnnotatedImage'
import { DemoBadge, PersonalitySwitch } from '../components/Bits'
import { ErrorPanel } from '../components/ErrorPanel'
import { Icon } from '../components/Icon'
import { ReportView } from '../components/ReportView'
import { analyzeImage, ApiError, isAbort, toApiError } from '../lib/api'
import { formatBytes, formatMs } from '../lib/format'
import { navigate } from '../lib/router'
import type { Report } from '../lib/schemas'
import { useSettings } from '../state/settings'
import { isDemo, useSystem } from '../state/system'
import { JpegCapturer } from '../live/capture'
import { vision } from '../vision/client'
import { spatialRelations } from '../vision/relations'
import type { Detection, Track } from '../vision/types'
import '../styles/debug.css'

const ACCEPT = 'image/jpeg,image/png,image/webp'
const ACCEPTED = new Set(['image/jpeg', 'image/jpg', 'image/png', 'image/webp'])

type StageId = 'decode' | 'vision' | 'relations' | 'upload' | 'reason' | 'report'
type StageState = 'pending' | 'active' | 'done' | 'skipped' | 'failed'

const STAGES: { id: StageId; label: string }[] = [
  { id: 'decode', label: 'Local preprocessing' },
  { id: 'vision', label: 'On-device detection' },
  { id: 'relations', label: 'Relationship analysis' },
  { id: 'upload', label: 'Secure upload (EXIF stripped)' },
  { id: 'reason', label: 'Vision reasoning' },
  { id: 'report', label: 'Validated report' },
]

interface Picked {
  file: File
  url: string
  width: number
  height: number
}

function detectionsToTracks(detections: Detection[]): Track[] {
  return detections.map((d, i) => ({
    id: i + 1,
    key: `${d.label}·${String(i + 1).padStart(2, '0')}`,
    label: d.label,
    score: d.score,
    box: d.box,
    vx: 0,
    vy: 0,
    hits: 3,
    misses: 0,
    state: 'confirmed' as const,
    firstSeen: 0,
    lastSeen: 0,
  }))
}

export default function ImageDebug() {
  const settings = useSettings()
  const health = useSystem((s) => s.health)
  const record = useSystem((s) => s.record)
  const [picked, setPicked] = useState<Picked | null>(null)
  const [stages, setStages] = useState<Record<StageId, { state: StageState; note?: string }>>(() => resetStages())
  const [error, setError] = useState<ApiError | null>(null)
  const [report, setReport] = useState<Report | null>(null)
  const [detections, setDetections] = useState<Detection[]>([])
  const [focusId, setFocusId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const capturer = useRef<JpegCapturer | null>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const limit = health?.limits.max_image_bytes ?? 15 * 1024 * 1024
  const demo = isDemo(health, settings.forceDemo)

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

      if (!ACCEPTED.has(file.type) && !/\.(jpe?g|png|webp)$/i.test(file.name)) {
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

        // 2. On-device detection (optional: continue without it on failure).
        let localDetections: Detection[] = []
        setStage('vision', 'active', vision.ready ? undefined : 'loading model…')
        try {
          const forVision = await createImageBitmap(bitmap)
          const still = await vision.analyzeStill(forVision)
          localDetections = still.detections
          setDetections(still.detections)
          setStage('vision', 'done', `${still.detections.length} objects · ${formatMs(still.inferenceMs)}`)
        } catch (e) {
          setStage('vision', 'skipped', `engine unavailable: ${toApiError(e).message.slice(0, 60)}`)
        }

        // 3. Local relationship heuristics.
        const relations = spatialRelations(detectionsToTracks(localDetections))
        setStage('relations', 'done', relations.length ? `${relations.length} flagged` : 'none flagged')

        // 4. Re-encode (resize + strip metadata) and upload.
        setStage('upload', 'active')
        const imgEl = await loadImage(url)
        imageRef.current = imgEl
        capturer.current ??= new JpegCapturer()
        const shot = await capturer.current.capture(bitmap, 1600, 0.88)
        bitmap.close()
        setStage('upload', 'done', `${formatBytes(shot.blob.size)} JPEG`)

        // 5. Vision reasoning on the backend.
        setStage('reason', 'active', demo ? 'demo heuristics (simulated)' : health?.ai.model ?? undefined)
        const reasonStarted = performance.now()
        const result = await analyzeImage(shot.blob, {
          personality: settings.personality,
          demo: settings.forceDemo,
          signal: controller.signal,
          context: {
            detector: vision.getSnapshot().info?.model ?? null,
            objects: localDetections.slice(0, 30).map((d, i) => ({
              track_id: `${d.label}·${String(i + 1).padStart(2, '0')}`,
              label: d.label,
              confidence: Math.round(d.score * 100) / 100,
              box: d.box,
            })),
            relationships: relations.map((r) => r.text),
          },
        })
        setStage('reason', 'done', formatMs(performance.now() - reasonStarted))
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
    [demo, health, limit, record, settings.forceDemo, settings.personality],
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
        {demo ? <DemoBadge text="Demo" /> : null}
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
                  {detections.map((d, i) => (
                    <span
                      key={i}
                      className="annotated__local"
                      style={{ left: `${d.box.x * 100}%`, top: `${d.box.y * 100}%`, width: `${d.box.w * 100}%`, height: `${d.box.h * 100}%` }}
                    >
                      <em>
                        {d.label} {Math.round(d.score * 100)}%
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
                Your image is decoded and scanned by the on-device detector first. Then a resized copy, with its metadata
                (including GPS) stripped, goes to your own backend, which asks the vision model for a structured
                diagnostic. Nothing is stored.
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
                  {error.code.startsWith('AI_') ? (
                    <button
                      type="button"
                      className="key key--small"
                      onClick={() => {
                        settings.update({ forceDemo: true })
                        if (picked) void run(picked.file)
                      }}
                    >
                      Run in demo mode
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
              image={imageRef.current ? { source: imageRef.current, width: picked.width, height: picked.height } : null}
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
    vision: { state: 'pending' },
    relations: { state: 'pending' },
    upload: { state: 'pending' },
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
