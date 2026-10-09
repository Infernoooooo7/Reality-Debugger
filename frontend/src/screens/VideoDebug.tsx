import { useEffect, useMemo, useRef, useState } from 'react'
import { AIBadge, LocalBadge, PersonalitySwitch } from '../components/Bits'
import { ErrorPanel } from '../components/ErrorPanel'
import { Icon } from '../components/Icon'
import { Meter } from '../components/Meter'
import { ReportView } from '../components/ReportView'
import { VISION } from '../config'
import { analyzeVideoFile, analyzeVideoManifest, ApiError, isAbort, toApiError } from '../lib/api'
import { formatBytes, formatClock, formatMs, pad2 } from '../lib/format'
import { navigate } from '../lib/router'
import type { TimelineEvent, VideoReport } from '../lib/schemas'
import type { CardImage } from '../lib/share'
import { useSettings } from '../state/settings'
import { aiEnabled, aiMode, providerLabel, useSystem } from '../state/system'
import {
  buildManifest,
  captureKeyframes,
  loadVideo,
  sampleVideo,
  selectKeyframes,
  type Progress,
  type VideoInfo,
} from '../video/sampler'
import { useVision, vision } from '../vision/client'
import { deepDetector } from '../vision/deep/client'
import { detectorRun, modelFacts } from '../vision/runs'
import { verifyObjects } from '../vision/fusion'
import '../styles/debug.css'

type Phase = 'idle' | 'loading' | 'ready' | 'scanning' | 'selecting' | 'capturing' | 'verifying' | 'reasoning' | 'uploading' | 'done' | 'error'

const PHASE_TEXT: Record<Phase, string> = {
  idle: 'Awaiting input',
  loading: 'Reading metadata',
  ready: 'Ready to scan',
  scanning: 'Scanning frames…',
  selecting: 'Selecting keyframes…',
  capturing: 'Capturing keyframes…',
  verifying: 'Deep detector on keyframes…',
  uploading: 'Uploading to backend…',
  reasoning: 'Local diagnostics…',
  done: 'Diagnostic complete',
  error: 'Fault',
}

/** Wall-clock helper for the (event-handler only) timing below. */
const clock = () => performance.now()

const KIND_LABEL: Record<TimelineEvent['kind'], string> = {
  DISCOVERED: 'Discovered',
  CONFIRMED: 'Confirmed',
  ESCALATED: 'Escalated',
  RESOLVED: 'Resolved ✓',
  OBSERVATION: 'Observation',
  SCENE_CHANGE: 'Scene cut',
  OBJECT_ENTERED: 'Object in',
  OBJECT_LEFT: 'Object out',
}

export default function VideoDebug() {
  const settings = useSettings()
  const health = useSystem((s) => s.health)
  const record = useSystem((s) => s.record)
  const videoRef = useRef<HTMLVideoElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [url, setUrl] = useState<string | null>(null)
  const [info, setInfo] = useState<VideoInfo | null>(null)
  const [phase, setPhase] = useState<Phase>('idle')
  const [progress, setProgress] = useState<Progress | null>(null)
  const [stats, setStats] = useState<{ scenes: number; keyframes: number; redundant: number; deep?: string } | null>(null)
  const [uploadPct, setUploadPct] = useState<number | null>(null)
  const [result, setResult] = useState<VideoReport | null>(null)
  const [thumbs, setThumbs] = useState<string[]>([])
  const [error, setError] = useState<ApiError | null>(null)
  const [activeT, setActiveT] = useState<number | null>(null)
  const [currentT, setCurrentT] = useState(0)
  const [cardImage, setCardImage] = useState<CardImage | null>(null)
  const [showLocal, setShowLocal] = useState(false)
  const visionState = useVision()
  const withAI = aiEnabled(health)
  const serverMax = health?.limits.max_video_bytes ?? 300 * 1024 * 1024
  const maxKeyframes = Math.min(8, health?.limits.max_video_keyframes ?? 8)

  useEffect(() => () => abortRef.current?.abort(), [])
  useEffect(() => () => (url ? URL.revokeObjectURL(url) : undefined), [url])
  useEffect(() => () => thumbs.forEach((t) => URL.revokeObjectURL(t)), [thumbs])

  const busy = ['loading', 'scanning', 'selecting', 'capturing', 'verifying', 'uploading', 'reasoning'].includes(phase)

  const pick = async (picked: File | undefined) => {
    if (!picked) return
    abortRef.current?.abort()
    setError(null)
    setResult(null)
    setStats(null)
    setProgress(null)
    setThumbs([])
    setCardImage(null)
    setUploadPct(null)
    if (picked.type && !picked.type.startsWith('video/')) {
      setFile(null)
      setPhase('error')
      setError(new ApiError('UNSUPPORTED_MEDIA_TYPE', `“${picked.name}” is not a video file.`, { hint: 'Use MP4, MOV or WEBM.' }))
      return
    }
    const objectUrl = URL.createObjectURL(picked)
    setFile(picked)
    setUrl(objectUrl)
    setPhase('loading')
    try {
      const meta = await loadVideo(videoRef.current!, objectUrl)
      setInfo(meta)
      setPhase('ready')
      vision.init().catch(() => undefined)
    } catch (e) {
      setInfo(null)
      setPhase('error')
      setError(toApiError(e))
    }
  }

  const scan = async () => {
    const video = videoRef.current
    if (!video || !info || !file) return
    const controller = new AbortController()
    abortRef.current = controller
    setError(null)
    setResult(null)
    try {
      video.pause()
      setPhase('scanning')
      const samples = await sampleVideo(video, info, { signal: controller.signal, onProgress: setProgress })
      setPhase('selecting')
      const selection = selectKeyframes(samples, maxKeyframes)
      setStats({ scenes: selection.scenes.length, keyframes: selection.keyframes.length, redundant: selection.redundant })
      setPhase('capturing')
      const blobs = await captureKeyframes(video, selection.keyframes)
      const thumbUrls = blobs.map((b) => URL.createObjectURL(b))
      setThumbs(thumbUrls)
      const first = new Image()
      first.src = thumbUrls[0]!
      setCardImage({ source: first, width: info.width, height: info.height })

      // Deep detector verifies the keyframes (within the time budget). Each keyframe's sample
      // records whether the deep pass ran on it, so the report knows what was examined.
      const detectors = [vision.getSnapshot().info?.modelId ?? VISION.fast.model]
      if (deepDetector.enabled) {
        setPhase('verifying')
        const deepFacts = await modelFacts(VISION.deep.model)
        const started = clock()
        let done = 0
        let current = -1
        try {
          for (const [i, k] of selection.keyframes.entries()) {
            const sample = samples[k.sample]!
            if (clock() - started > VISION.deep.videoBudgetMs) {
              sample.runs.push(detectorRun(VISION.deep.model, 'deep', 'skipped', deepFacts, { note: 'video time budget reached' }))
              continue
            }
            current = k.sample
            const deep = await deepDetector.detect(await createImageBitmap(blobs[i]!))
            sample.objects = verifyObjects(sample.objects, deep.detections, `k${i + 1}d`)
            sample.runs.push(detectorRun(VISION.deep.model, 'deep', 'ok', deepFacts, { boxes: deep.detections.length, ms: deep.totalMs, note: 'keyframe verification' }))
            done++
          }
          if (done) detectors.push(VISION.deep.model)
          const note = `${done}/${selection.keyframes.length} keyframes · ${formatMs(clock() - started)}`
          setStats((prev) => (prev ? { ...prev, deep: note } : prev))
        } catch (e) {
          const message = toApiError(e).message
          if (current >= 0) samples[current]!.runs.push(detectorRun(VISION.deep.model, 'deep', 'failed', deepFacts, { note: message }))
          const note = `unavailable: ${message.slice(0, 60)}`
          setStats((prev) => (prev ? { ...prev, deep: note } : prev))
        }
      }

      setPhase('reasoning')
      const manifest = buildManifest(file, info, samples, selection, detectors)
      // Keyframe images are uploaded only when an AI provider is configured.
      const report = await analyzeVideoManifest(manifest, withAI ? blobs : [], {
        personality: settings.personality,
        signal: controller.signal,
      })
      setResult(report)
      record(report.report)
      setPhase('done')
      video.currentTime = 0
    } catch (e) {
      if (isAbort(e)) return
      setPhase('error')
      setError(toApiError(e))
    }
  }

  const serverFallback = async () => {
    if (!file) return
    const controller = new AbortController()
    abortRef.current = controller
    setError(null)
    setPhase('uploading')
    setUploadPct(0)
    try {
      const report = await analyzeVideoFile(file, {
        personality: settings.personality,
        signal: controller.signal,
        onUploadProgress: (f) => {
          setUploadPct(f)
          if (f >= 1) setPhase('reasoning')
        },
      })
      setResult(report)
      setStats({ scenes: report.sampling.scenes, keyframes: report.sampling.keyframes, redundant: report.sampling.redundant_removed })
      record(report.report)
      setPhase('done')
    } catch (e) {
      if (isAbort(e)) return
      setPhase('error')
      setError(toApiError(e))
    }
  }

  const jump = (t: number) => {
    const video = videoRef.current
    if (!video) return
    video.currentTime = t
    video.pause()
    setActiveT(t)
  }

  const findingsCount = result?.report.findings.length ?? 0
  const duration = info?.duration ?? result?.video.duration_s ?? 0
  const scanFraction = progress ? progress.done / progress.total : phase === 'done' ? 1 : 0
  const timeline = useMemo(() => result?.timeline ?? [], [result])
  const sceneCuts = useMemo(() => timeline.filter((e) => e.kind === 'SCENE_CHANGE'), [timeline])
  const objectEvents = timeline.filter((e) => e.kind === 'OBJECT_ENTERED' || e.kind === 'OBJECT_LEFT').length
  const listed = showLocal ? timeline : timeline.filter((e) => e.kind !== 'OBJECT_ENTERED' && e.kind !== 'OBJECT_LEFT')

  return (
    <div className="debug">
      <header className="debug__bar">
        <button type="button" className="key key--small key--ghost" onClick={() => navigate('home')}>
          <Icon name="back" size={16} /> Home
        </button>
        <span className="debug__title">
          <span className="t-data t-muted">IN·C</span> Video debug
        </span>
        <LocalBadge />
        <AIBadge mode={aiMode(health)} provider={providerLabel(health?.ai.provider)} />
      </header>

      <div className="debug__grid">
        <section className="debug__input">
          <div className="vplayer" hidden={!url || phase === 'error' && !info}>
            <video
              ref={videoRef}
              controls={phase === 'done' || phase === 'ready'}
              playsInline
              muted
              onTimeUpdate={(e) => setCurrentT(e.currentTarget.currentTime)}
            />
          </div>
          {!url ? (
            <div className="dropzone">
              <div className="dropzone__mark" aria-hidden="true">
                <Icon name="video" size={40} strokeWidth={1.6} />
              </div>
              <p className="dropzone__title t-display">Select a video</p>
              <p className="t-muted">Processed on this device · only measurements are uploaded{withAI ? ' (+ keyframes for AI)' : ''}</p>
            </div>
          ) : null}

          {file ? (
            <dl className="debug__meta t-data">
              <div>
                <dt>File</dt>
                <dd>{file.name}</dd>
              </div>
              <div>
                <dt>Duration</dt>
                <dd>{info ? formatClock(info.duration) : '—'}</dd>
              </div>
              <div>
                <dt>Resolution</dt>
                <dd>{info ? `${info.width}×${info.height}` : '—'}</dd>
              </div>
              <div>
                <dt>Size</dt>
                <dd>{formatBytes(file.size)}</dd>
              </div>
            </dl>
          ) : null}

          <div className="debug__pickers">
            <label className={`key ${phase === 'ready' ? '' : 'key--signal'} debug__pick`}>
              <Icon name="upload" size={18} /> {file ? 'Another video' : 'Choose video'}
              <input type="file" accept="video/*" hidden disabled={busy} onChange={(e) => void pick(e.target.files?.[0])} />
            </label>
            <button type="button" className="key key--signal debug__pick" disabled={phase !== 'ready'} onClick={() => void scan()}>
              <Icon name="scan" size={18} /> Scan video
            </button>
          </div>

          <div className="module">
            <div className="module__head">
              <span className="t-label">Debug personality</span>
            </div>
            <PersonalitySwitch value={settings.personality} onChange={(personality) => settings.update({ personality })} />
          </div>
        </section>

        <section className="debug__output">
          <div className="scanstats">
            <div className="scanstats__phase">
              <span className="t-label">Video {duration ? formatClock(duration) : ''}</span>
              <b>{PHASE_TEXT[phase]}</b>
              <Meter
                value={phase === 'uploading' ? uploadPct : scanFraction}
                color={phase === 'error' ? 'var(--critical)' : 'var(--signal)'}
                height={10}
                segments={30}
                label="Progress"
              />
              <span className="t-data t-muted">
                {phase === 'scanning' && progress
                  ? `frame ${progress.done}/${progress.total} · t=${formatClock(progress.t)} · ${Math.round(scanFraction * 100)}%`
                  : phase === 'scanning'
                    ? visionState.status === 'loading'
                      ? 'loading the on-device vision model…'
                      : 'seeking first frame…'
                    : phase === 'uploading' && uploadPct != null
                    ? `uploading whole file for server-side sampling · ${Math.round(uploadPct * 100)}%`
                    : phase === 'verifying'
                      ? 'YOLOX-S checking the keyframes (loads once, then cached)'
                      : phase === 'reasoning'
                        ? withAI
                          ? `local engine + ${providerLabel(health?.ai.provider)} reading ${stats?.keyframes ?? ''} keyframes`
                          : `local engine replaying ${progress?.done ?? ''} tracked samples`
                        : stats?.deep
                          ? `deep detector: ${stats.deep}`
                          : ' '}
              </span>
            </div>
            <div>
              <span className="t-label">Frames sampled</span>
              <b>{pad2(progress?.done ?? result?.sampling.sampled_frames ?? 0)}</b>
            </div>
            <div>
              <span className="t-label">Scenes detected</span>
              <b>{pad2(stats?.scenes ?? progress?.scenes ?? 0)}</b>
            </div>
            <div>
              <span className="t-label">{withAI ? 'Keyframes sent' : 'Keyframes'}</span>
              <b>{pad2(stats?.keyframes ?? 0)}</b>
            </div>
            <div>
              <span className="t-label">Objects tracked</span>
              <b>{pad2(result?.sampling.tracked_objects ?? progress?.objects ?? 0)}</b>
            </div>
            <div>
              <span className="t-label">Findings</span>
              <b>{pad2(findingsCount)}</b>
            </div>
          </div>

          {error ? (
            <ErrorPanel
              error={error}
              actions={
                <>
                  {error.code === 'VIDEO_UNREADABLE' && file && file.size <= serverMax && health?.features.server_video_fallback ? (
                    <button type="button" className="key key--small" onClick={() => void serverFallback()}>
                      <Icon name="upload" size={16} /> Process on the backend instead
                    </button>
                  ) : null}
                  {info && error.retryable ? (
                    <button type="button" className="key key--small" onClick={() => void scan()}>
                      <Icon name="retry" size={16} /> Retry
                    </button>
                  ) : null}

                </>
              }
            />
          ) : null}

          {result ? (
            <>
              <div className="module">
                <div className="module__head">
                  <span className="t-label">Diagnostic timeline</span>
                  <span className="t-data">
                    {result.sampling.processed_on === 'browser' ? 'sampled on device' : 'sampled on backend'} ·{' '}
                    {result.sampling.redundant_removed} redundant frames dropped
                  </span>
                </div>
                <div className="track" role="list" aria-label="Timeline track">
                  {sceneCuts.map((e, i) => (
                    <span key={`s${i}`} className="track__scene" style={{ left: `${(e.t / Math.max(0.01, duration)) * 100}%` }} />
                  ))}
                  {timeline
                    .filter((e) => e.kind !== 'SCENE_CHANGE')
                    .map((e, i) => (
                      <button
                        key={i}
                        type="button"
                        role="listitem"
                        className="track__marker"
                        data-kind={e.kind}
                        data-sev={e.severity}
                        data-active={activeT === e.t || undefined}
                        style={{ left: `${(e.t / Math.max(0.01, duration)) * 100}%` }}
                        title={`${formatClock(e.t)} ${e.text}`}
                        onClick={() => jump(e.t)}
                      />
                    ))}
                  <span className="track__head" style={{ left: `${(currentT / Math.max(0.01, duration)) * 100}%` }} />
                </div>
                <div className="track__labels t-data">
                  <span>00:00</span>
                  <span>{formatClock(duration)}</span>
                </div>
                {objectEvents ? (
                  <label className="timeline__filter t-data">
                    <input type="checkbox" checked={showLocal} onChange={(e) => setShowLocal(e.target.checked)} /> show{' '}
                    {objectEvents} local object events
                  </label>
                ) : null}
                <ol className="timeline">
                  {listed.map((e, i) => (
                    <li key={i} data-kind={e.kind} data-sev={e.severity} data-source={e.source} data-active={activeT === e.t || undefined}>
                      <button type="button" onClick={() => jump(e.t)}>
                        <span className="timeline__t t-data">{formatClock(e.t)}</span>
                        <span className="timeline__kind">
                          {KIND_LABEL[e.kind]}
                          {e.kind === 'DISCOVERED' || e.kind === 'CONFIRMED' || e.kind === 'ESCALATED' ? ` · ${e.severity}` : ''}
                        </span>
                        <span className="timeline__text">
                          {e.text}
                          {e.finding_id ? <small>{e.finding_id}</small> : null}
                          {e.source === 'local' ? <small>local</small> : null}
                        </span>
                      </button>
                    </li>
                  ))}
                </ol>
              </div>

              {thumbs.length ? (
                <div className="keyframes" aria-label="Keyframes">
                  {result.keyframes.map((k, i) =>
                    thumbs[i] ? (
                      <button key={k.index} type="button" onClick={() => jump(k.t)} title={k.reason}>
                        <img src={thumbs[i]} alt={`Keyframe ${k.index} at ${formatClock(k.t)}`} />
                        <span>
                          #{k.index} · {formatClock(k.t)}
                        </span>
                      </button>
                    ) : null,
                  )}
                </div>
              ) : null}

              <ReportView
                report={result.report}
                image={cardImage}
                showTimes
                heading="Video diagnostic"
                onJump={(t) => {
                  jump(t)
                  videoRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
                }}
              />
            </>
          ) : !error ? (
            <div className="debug__explain">
              <p className="t-label">How video debug works</p>
              <p>
                The video stays on this device. It is sampled every ~0.5 s; every sample runs through the fast detector
                and the tracker, so objects keep their identity over time. Scene cuts are detected, redundant frames
                dropped, and up to {maxKeyframes} keyframes are checked by the deep detector. The local engine then
                replays the tracked samples to build a timeline of measured findings. With an AI provider configured,
                the keyframes can also be sent for optional reasoning.
              </p>
            </div>
          ) : null}
        </section>
      </div>
    </div>
  )
}
