import { useEffect, useMemo, useRef, useState } from 'react'
import { AnnotatedImage } from '../components/AnnotatedImage'
import { DemoBadge, StatusChip } from '../components/Bits'
import { DecodeText } from '../components/DecodeText'
import { ErrorPanel } from '../components/ErrorPanel'
import { FindingCard } from '../components/FindingCard'
import { Icon } from '../components/Icon'
import { scoreColor } from '../components/Meter'
import { ReportView } from '../components/ReportView'
import { formatMs, pad2 } from '../lib/format'
import { navigate } from '../lib/router'
import type { Finding } from '../lib/schemas'
import { LiveSession } from '../live/session'
import { LIVING_TEXT, useLive } from '../live/store'
import { scanToReport } from '../live/summary'
import { useSettings } from '../state/settings'
import { isDemo, useSystem } from '../state/system'
import { toast } from '../state/toasts'
import { useVision } from '../vision/client'
import '../styles/live.css'

const SEVERITY_ORDER = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3, INFO: 4 } as const

function sortFindings(findings: Finding[]): Finding[] {
  return [...findings].sort((a, b) => {
    const resolved = Number(a.status === 'RESOLVED') - Number(b.status === 'RESOLVED')
    return resolved || SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity] || b.confidence - a.confidence
  })
}

function Sparkline({ values }: { values: number[] }) {
  const points = values
    .map((v, i) => `${(i / Math.max(1, values.length - 1)) * 100},${22 - Math.min(1, v * 2.5) * 20}`)
    .join(' ')
  return (
    <svg className="spark" viewBox="0 0 100 24" preserveAspectRatio="none" aria-hidden="true">
      <polyline points={points} fill="none" stroke="currentColor" strokeWidth="1.4" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 100)
    return () => window.clearInterval(id)
  }, [])
  return <span className="t-data">{((now - since) / 1000).toFixed(1)}s</span>
}

export default function LiveScan() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sessionRef = useRef<LiveSession | null>(null)
  const { phase, bootStep, fatal, hud, ai, living, scan, deep, events } = useLive()
  const settings = useSettings()
  const health = useSystem((s) => s.health)
  const visionState = useVision()
  const [panelOpen, setPanelOpen] = useState(true)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [exitOpen, setExitOpen] = useState(false)
  const [focusId, setFocusId] = useState<string | null>(null)
  const [bootKey, setBootKey] = useState(0)

  useEffect(() => {
    useLive.getState().reset()
    const session = new LiveSession(videoRef.current!, canvasRef.current!)
    sessionRef.current = session
    void session.boot()
    return () => {
      session.dispose()
      sessionRef.current = null
    }
  }, [bootKey])

  useEffect(() => sessionRef.current?.configure(settings.interval), [settings.interval])
  useEffect(() => sessionRef.current?.setForceDemo(settings.forceDemo), [settings.forceDemo])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return
      if (e.code === 'Space') {
        e.preventDefault()
        const s = useLive.getState()
        if (s.phase === 'running') sessionRef.current?.pause()
        else if (s.phase === 'paused') void sessionRef.current?.resume()
      } else if (e.key === 'd' || e.key === 'D') void sessionRef.current?.deepScan()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const demo = isDemo(health, settings.forceDemo)
  const findings = useMemo(() => sortFindings(scan?.findings ?? []), [scan])
  const open = findings.filter((f) => f.status !== 'RESOLVED')
  const resolved = findings.filter((f) => f.status === 'RESOLVED')
  const info = visionState.info
  const tone =
    living.state === 'CONFIRMED' || living.state === 'DISCOVERED'
      ? 'signal'
      : living.state === 'RESOLVED' || living.state === 'STABLE'
        ? 'ok'
        : living.state === 'ANOMALY' || living.state === 'INVESTIGATING'
          ? 'amber'
          : 'neutral'
  const livingText =
    living.state === 'MONITORING' && scan
      ? `Monitoring ${pad2(scan.counts.active_bugs)} bug${scan.counts.active_bugs === 1 ? '' : 's'}`
      : LIVING_TEXT[living.state]

  const exit = async (clear: boolean) => {
    if (clear) {
      await sessionRef.current?.clearSession()
      toast('Scan session cleared from this device and the backend', 'ok')
    } else if (scan && scan.analyses > 0) {
      useSystem.getState().record(scanToReport(scan))
    }
    navigate('home')
  }

  return (
    <div className="live" data-phase={phase}>
      <div className="live__stage">
        <video ref={videoRef} className="live__video" playsInline muted autoPlay />
        <canvas ref={canvasRef} className="live__overlay" />
        <div className="live__viewport" aria-hidden="true" />

        {phase === 'booting' ? (
          <div className="live__boot">
            <span className="t-label">Live scan</span>
            <DecodeText className="live__boot-step" text={bootStep} />
            <div className="spinner-bar" />
            {bootStep.startsWith('Requesting') ? (
              <p className="t-data t-muted">Allow camera access when the browser asks.</p>
            ) : null}
          </div>
        ) : null}

        {phase === 'error' && fatal ? (
          <div className="live__fatal">
            <ErrorPanel
              error={fatal}
              actions={
                <>
                  <button type="button" className="key key--small" onClick={() => setBootKey((k) => k + 1)}>
                    <Icon name="retry" size={16} /> Retry
                  </button>
                  <button type="button" className="key key--small" onClick={() => navigate('image')}>
                    <Icon name="image" size={16} /> Image debug
                  </button>
                  <button type="button" className="key key--small key--ghost" onClick={() => navigate('home')}>
                    Home
                  </button>
                </>
              }
            />
          </div>
        ) : null}
      </div>

      <header className="live__hud">
        <div className="live__hud-row">
          <button type="button" className="hud-btn" onClick={() => setExitOpen(true)} aria-label="End scan">
            <Icon name="back" size={18} />
            <span>End</span>
          </button>
          <div className="plate" data-tone={tone}>
            <DecodeText className="plate__text" text={livingText} />
            {living.detail ? <span className="plate__detail t-data">{living.detail}</span> : null}
          </div>
          <div className="hud-fps">
            <span className="hud-fps__value">{hud.fps ? hud.fps.toFixed(1) : '--'}</span>
            <span className="t-label">fps</span>
          </div>
        </div>
        <div className="live__hud-row live__hud-row--chips t-data">
          <span className="hud-chip">
            VISION {info ? `${info.delegate} · ${Math.round(hud.inferenceMs)}ms` : visionState.status.toUpperCase()}
          </span>
          <span className="hud-chip">
            OBJ {pad2(hud.objects)}
            {hud.tentative ? <em> +{hud.tentative}</em> : null}
          </span>
          <span className="hud-chip" data-ai={ai.state}>
            AI {ai.state === 'analyzing' ? <>ANALYZING <Elapsed since={ai.startedAt} /></> : ai.state.toUpperCase()}
            {ai.state === 'idle' && ai.lastLatencyMs ? <em> · last {formatMs(ai.lastLatencyMs)}</em> : null}
          </span>
          <span className="hud-chip hud-chip--trace" title="Motion (frame differencing)">
            MOTION <Sparkline values={hud.motionTrace} />
          </span>
          <span className="hud-chip" title="Scene change since the last analysed frame">
            SCENE Δ {hud.sceneDelta.toFixed(2)}
          </span>
          {demo ? <DemoBadge text="Demo · simulated" /> : null}
        </div>
        {ai.error && (ai.state === 'blocked' || ai.state === 'error' || ai.state === 'budget') ? (
          <div className="live__ai-alert" role="status">
            <span>
              <b>{ai.error.code}</b> · {ai.error.message} {ai.state !== 'blocked' ? 'Retrying automatically - local tracking continues.' : ''}
            </span>
            {ai.state === 'blocked' ? (
              <button type="button" className="key key--small" onClick={() => settings.update({ forceDemo: true })}>
                Switch to demo mode
              </button>
            ) : null}
          </div>
        ) : null}
      </header>

      <section className="live__panel" data-open={panelOpen}>
        <button type="button" className="live__panel-head" onClick={() => setPanelOpen((v) => !v)} aria-expanded={panelOpen}>
          <span className="live__panel-stat">
            <span className="t-label">Active bugs</span>
            <b>{pad2(scan?.counts.active_bugs ?? 0)}</b>
          </span>
          <span className="live__panel-stat">
            <span className="t-label">High</span>
            <b>{pad2(scan?.counts.high_priority ?? 0)}</b>
          </span>
          <span className="live__panel-stat">
            <span className="t-label">Resolved</span>
            <b className="ok">{pad2(scan?.counts.resolved ?? 0)}</b>
          </span>
          <span className="live__panel-stat live__panel-stat--score">
            <span className="t-label">Score</span>
            <b style={{ color: scoreColor(scan?.system_score) }}>{scan?.system_score ?? '--'}</b>
          </span>
          {scan ? <StatusChip status={scan.status} /> : null}
          <span className="live__panel-toggle t-data">{panelOpen ? '▾' : '▴'}</span>
        </button>
        {panelOpen ? (
          <div className="live__findings">
            {!scan ? (
              <p className="live__empty t-data">
                {phase === 'running'
                  ? 'Watching. The first analysis runs once the view is steady.'
                  : 'Findings appear here as the scan runs.'}
              </p>
            ) : open.length === 0 && resolved.length === 0 ? (
              <p className="live__empty t-data">No bugs filed for this view (yet).</p>
            ) : null}
            {open.map((f) => (
              <FindingCard
                key={f.id}
                finding={f}
                expanded={expanded === f.id}
                onToggle={() => setExpanded((cur) => (cur === f.id ? null : f.id))}
              />
            ))}
            {resolved.length ? <div className="live__divider t-label">Resolved</div> : null}
            {resolved.map((f) => (
              <FindingCard
                key={f.id}
                finding={f}
                expanded={expanded === f.id}
                onToggle={() => setExpanded((cur) => (cur === f.id ? null : f.id))}
              />
            ))}
            {events.length ? (
              <>
                <div className="live__divider t-label">Event log</div>
                <ol className="eventlog t-data">
                  {events.slice(0, 14).map((e) => (
                    <li key={e.id} data-type={e.type}>
                      <time>{new Date(e.at).toLocaleTimeString([], { hour12: false })}</time>
                      <b>{e.type}</b>
                      <span>
                        {e.finding_id} · {e.title}
                      </span>
                    </li>
                  ))}
                </ol>
              </>
            ) : null}
          </div>
        ) : null}
      </section>

      <nav className="live__controls" aria-label="Scan controls">
        {phase === 'paused' ? (
          <button type="button" className="ctl" onClick={() => void sessionRef.current?.resume()}>
            <Icon name="play" size={22} />
            <span>Resume</span>
          </button>
        ) : (
          <button type="button" className="ctl" onClick={() => sessionRef.current?.pause()} disabled={phase !== 'running'}>
            <Icon name="pause" size={22} />
            <span>Pause</span>
          </button>
        )}
        <button
          type="button"
          className="ctl ctl--deep key--signal"
          onClick={() => void sessionRef.current?.deepScan()}
          disabled={phase !== 'running' && phase !== 'paused'}
        >
          <Icon name="scan" size={24} />
          <span>{phase === 'paused' ? 'Deep scan frame' : 'Deep scan'}</span>
        </button>
        <button
          type="button"
          className="ctl"
          onClick={() => void sessionRef.current?.switchCamera()}
          disabled={phase !== 'running' && phase !== 'paused'}
        >
          <Icon name="flip" size={22} />
          <span>Flip</span>
        </button>
      </nav>

      {deep ? (
        <div className="deep" role="dialog" aria-modal="true" aria-label="Deep scan">
          <div className="deep__inner">
            <header className="deep__head">
              <span className="t-label">Deep scan · frozen frame</span>
              {deep.status === 'analyzing' ? (
                <span className="deep__timer">
                  INVESTIGATING <Elapsed since={deep.startedAt} />
                </span>
              ) : null}
            </header>
            <div className="deep__image" data-scanning={deep.status === 'analyzing'}>
              {deep.report ? (
                <AnnotatedImage
                  src={deep.imageUrl}
                  findings={deep.report.findings}
                  focusedId={focusId}
                  onFocus={setFocusId}
                  alt="Frozen frame with findings"
                />
              ) : (
                <figure className="annotated">
                  <img src={deep.imageUrl} alt="Frozen frame being analysed" />
                </figure>
              )}
            </div>
            {deep.status === 'analyzing' ? (
              <ol className="deep__steps t-data">
                <li data-done="true">Frame frozen · {deep.width}×{deep.height}</li>
                <li data-done="true">Local context attached · {pad2(hud.objects)} tracked objects</li>
                <li data-active="true">{demo ? 'Demo heuristics running (simulated)' : 'Vision model reasoning'}</li>
                <li>Schema validation</li>
                <li>Lifecycle merge</li>
              </ol>
            ) : null}
            {deep.status === 'error' && deep.error ? (
              <ErrorPanel
                error={deep.error}
                actions={
                  <button type="button" className="key key--small" onClick={() => void sessionRef.current?.resumeAfterDeep()}>
                    Return to live
                  </button>
                }
              />
            ) : null}
            {deep.report ? (
              <ReportView
                report={deep.report}
                image={{ source: imageElement(deep.imageUrl), width: deep.width, height: deep.height }}
                focusedId={focusId}
                onFocus={setFocusId}
                heading="Deep scan diagnostic"
              />
            ) : null}
          </div>
          <div className="deep__return">
            <button type="button" className="key key--signal deep__return-btn" onClick={() => void sessionRef.current?.resumeAfterDeep()}>
              <Icon name="play" size={18} /> Return to live
            </button>
          </div>
        </div>
      ) : null}

      {exitOpen ? (
        <div className="exit" role="dialog" aria-modal="true" aria-label="End scan">
          <div className="exit__card module">
            <div className="module__head">
              <span className="t-label">End scan</span>
              <button type="button" className="t-data" onClick={() => setExitOpen(false)}>
                close
              </button>
            </div>
            {scan && scan.analyses > 0 ? (
              <div className="exit__summary">
                <ReportView report={scanToReport(scan)} heading="Live scan summary" />
              </div>
            ) : (
              <p className="t-muted">No analysis has completed in this scan yet.</p>
            )}
            <div className="exit__actions">
              <button type="button" className="key" onClick={() => void exit(false)}>
                Exit
              </button>
              <button type="button" className="key" onClick={() => void exit(true)}>
                <Icon name="trash" size={16} /> Clear session &amp; exit
              </button>
              <button type="button" className="key key--ghost" onClick={() => setExitOpen(false)}>
                Keep scanning
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}

const imageCache = new Map<string, HTMLImageElement>()

function imageElement(url: string): HTMLImageElement {
  let img = imageCache.get(url)
  if (!img) {
    img = new Image()
    img.src = url
    imageCache.clear()
    imageCache.set(url, img)
  }
  return img
}
