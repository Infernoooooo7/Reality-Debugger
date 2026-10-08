import { useEffect, useState, type ReactNode } from 'react'
import { DemoBadge, PersonalitySwitch } from '../components/Bits'
import { Icon } from '../components/Icon'
import { Meter, scoreColor } from '../components/Meter'
import { BACKEND_START_HINT } from '../lib/api'
import { cameraSupport } from '../lib/env'
import { formatMs, pad2, timeOfDay } from '../lib/format'
import { navigate } from '../lib/router'
import type { Personality } from '../lib/schemas'
import { useSettings, type AnalysisInterval } from '../state/settings'
import { isDemo, useSystem } from '../state/system'
import { delegatePreference, setDelegatePreference, useVision, vision } from '../vision/client'
import '../styles/home.css'

const VOICE: Record<Personality, { line: string; sample: string }> = {
  serious: {
    line: 'Professional and practical. Incident-report tone.',
    sample: 'Cable crosses the mouse path; route it behind the monitor arm.',
  },
  brutal: {
    line: 'Blunt and funny. Still tells you how to fix it.',
    sample: 'This desk technically functions, but the cable management is committing crimes.',
  },
  unhinged: {
    line: 'Unreasonably creative, strictly evidence-based.',
    sample: 'ERROR 418: Desk has achieved maximum mug density.',
  },
}

type LineState = 'ok' | 'warn' | 'fault' | 'busy' | 'idle'

function PostLine({ label, state, value, detail }: { label: string; state: LineState; value: string; detail?: ReactNode }) {
  return (
    <li className="post__line" data-state={state}>
      <span className="post__label">{label}</span>
      <span className="post__dots" aria-hidden="true" />
      <span className="post__value t-data">[ {value} ]</span>
      {detail ? <span className="post__detail t-data">{detail}</span> : null}
    </li>
  )
}

function Clock() {
  const [now, setNow] = useState(() => timeOfDay())
  useEffect(() => {
    const id = window.setInterval(() => setNow(timeOfDay()), 1000)
    return () => window.clearInterval(id)
  }, [])
  return <span className="t-data">{now}</span>
}

export function Home() {
  const { health, healthError, healthLatencyMs, aiCheck, history, refreshHealth, refreshAICheck, clearHistory } =
    useSystem()
  const settings = useSettings()
  const visionState = useVision()
  const camera = cameraSupport()
  const [showSettings, setShowSettings] = useState(false)
  const demo = isDemo(health, settings.forceDemo)

  // Warm up the vision engine shortly after the instrument "boots" so that
  // Live Scan starts instantly. This is real model loading, shown live below.
  useEffect(() => {
    const id = window.setTimeout(() => {
      vision.init().catch(() => undefined)
    }, 900)
    return () => window.clearTimeout(id)
  }, [])

  useEffect(() => {
    if (health?.ai.configured && !aiCheck) void refreshAICheck()
  }, [health, aiCheck, refreshAICheck])

  const latest = history[0] ?? null
  const critical = history.filter((h) => h.status === 'CRITICAL').length

  // ---- self-test lines (all reflect real state) ----
  const backendLine = healthError
    ? { state: 'fault' as const, value: 'OFFLINE', detail: BACKEND_START_HINT }
    : health
      ? { state: 'ok' as const, value: `OK · ${formatMs(healthLatencyMs)}`, detail: undefined }
      : { state: 'busy' as const, value: 'PROBING', detail: undefined }

  let aiLine: { state: LineState; value: string; detail?: string }
  if (!health) aiLine = { state: 'idle', value: 'UNKNOWN' }
  else if (settings.forceDemo) aiLine = { state: 'warn', value: 'DEMO · FORCED', detail: 'Simulated diagnostics (toggle in settings).' }
  else if (!health.ai.configured)
    aiLine = { state: 'warn', value: 'DEMO MODE', detail: 'No API key in backend .env - diagnostics are simulated.' }
  else if (!aiCheck) aiLine = { state: 'busy', value: `CHECKING ${health.ai.model ?? ''}` }
  else if (aiCheck.ok) aiLine = { state: 'ok', value: `ONLINE · ${aiCheck.model ?? health.ai.provider}` }
  else aiLine = { state: 'fault', value: aiCheck.error?.code ?? 'FAULT', detail: aiCheck.error?.hint ?? aiCheck.error?.message }

  const info = visionState.info
  const visionLine: { state: LineState; value: string; detail?: string } =
    visionState.status === 'ready' && info
      ? {
          state: 'ok',
          value: `ONLINE · ${info.delegate} · ${info.runtime === 'worker' ? 'WORKER' : 'MAIN'}`,
          detail: `${info.model} · loaded in ${formatMs(info.loadMs)}`,
        }
      : visionState.status === 'loading'
        ? { state: 'busy', value: 'LOADING MODEL', detail: 'EfficientDet-Lite0 · 4.6 MB · WebAssembly' }
        : visionState.status === 'error'
          ? { state: 'fault', value: 'FAULT', detail: visionState.error ?? undefined }
          : { state: 'idle', value: 'STANDBY' }

  const selfTest = info?.selfTest
  const selfTestLine: { state: LineState; value: string; detail?: string } = selfTest
    ? selfTest.label
      ? { state: 'ok', value: `PASS · ${formatMs(selfTest.ms)}`, detail: `synthetic stop sign detected at ${Math.round(selfTest.score * 100)}%` }
      : { state: 'warn', value: 'DEGRADED', detail: 'probe object not detected; results may be unreliable' }
    : { state: visionState.status === 'loading' ? 'busy' : 'idle', value: visionState.status === 'loading' ? 'RUNNING' : 'PENDING' }

  const cameraLine: { state: LineState; value: string; detail?: string } = camera.available
    ? { state: 'ok', value: 'READY', detail: 'permission is requested when Live Scan starts' }
    : { state: 'warn', value: camera.reason === 'insecure' ? 'NEEDS HTTPS' : 'UNSUPPORTED', detail: camera.detail }

  return (
    <div className="home">
      <header className="home__bar">
        <span className="home__os">
          <span className="home__mark" aria-hidden="true" />
          Reality OS <span className="t-data t-muted">v{health?.version ?? '1.0.0'}</span>
        </span>
        <span className="home__bar-right">
          {demo ? <DemoBadge text="Demo mode" /> : <span className="home__live t-data">● AI LINKED</span>}
          <Clock />
        </span>
      </header>
      <div className="ticks" />

      <main className="home__grid">
        <section className="home__hero">
          <h1 className="home__title t-display">
            Your world
            <br />
            has <span className="squiggle home__bugs">bugs</span>.
          </h1>
          <p className="home__lede">
            Point a camera at a desk, a kitchen, a gaming setup. An on-device vision engine watches continuously. When
            something changes, a vision model files the bug report: severity, evidence, impact and a fix.
          </p>

          <nav className="modes" aria-label="Input modes">
            <button type="button" className="mode mode--primary key--signal" onClick={() => navigate('live')}>
              <span className="mode__ch t-data">IN·A</span>
              <span className="mode__name">Enter live scan</span>
              <span className="mode__desc">Continuous camera diagnostics · local ML every frame · AI on change</span>
              <span className="mode__go" aria-hidden="true">
                <Icon name="arrow" size={28} strokeWidth={2.4} />
              </span>
            </button>
            <button type="button" className="mode" onClick={() => navigate('image')}>
              <span className="mode__ch t-data">IN·B</span>
              <span className="mode__name">Image debug</span>
              <span className="mode__desc">Photo, gallery or upload → deep diagnostic</span>
              <span className="mode__go" aria-hidden="true">
                <Icon name="image" size={22} />
              </span>
            </button>
            <button type="button" className="mode" onClick={() => navigate('video')}>
              <span className="mode__ch t-data">IN·C</span>
              <span className="mode__name">Video debug</span>
              <span className="mode__desc">Scene detection · keyframes · diagnostic timeline</span>
              <span className="mode__go" aria-hidden="true">
                <Icon name="video" size={22} />
              </span>
            </button>
          </nav>

          <div className="signal-path" aria-label="Processing pipeline">
            <div className="signal-path__zone">
              <span className="t-label">On device</span>
              <ol>
                <li>Camera</li>
                <li>Detect</li>
                <li>Track</li>
                <li>Scene Δ</li>
                <li>Select frame</li>
              </ol>
            </div>
            <div className="signal-path__zone signal-path__zone--server">
              <span className="t-label">Your backend</span>
              <ol>
                <li>Validate</li>
                <li>Vision AI</li>
                <li>Diagnose</li>
                <li>Lifecycle</li>
              </ol>
            </div>
          </div>
        </section>

        <aside className="home__side">
          <section className="module post">
            <div className="module__head">
              <span className="t-label">Power-on self test</span>
              <button type="button" className="post__retry t-data" onClick={() => void refreshHealth()}>
                re-run
              </button>
            </div>
            <ul className="post__lines">
              <PostLine label="Backend link" {...backendLine} />
              <PostLine label="AI reasoner" {...aiLine} />
              <PostLine label="Vision engine" {...visionLine} />
              <PostLine label="Detector self-test" {...selfTestLine} />
              <PostLine label="Camera" {...cameraLine} />
            </ul>
            {visionState.status === 'error' ? (
              <button type="button" className="key key--small" onClick={() => vision.init().catch(() => undefined)}>
                <Icon name="retry" size={16} /> Retry vision engine
              </button>
            ) : null}
          </section>

          <section className="module world">
            <div className="module__head">
              <span className="t-label">World status</span>
              <span className="t-data">{latest ? latest.systemName : 'uncalibrated'}</span>
            </div>
            <div className="world__row">
              <span className="world__score" style={{ color: scoreColor(latest?.score) }}>
                {latest ? pad2(latest.score) : '—'}
              </span>
              <div className="world__meter">
                <Meter value={latest ? latest.score / 100 : null} color={scoreColor(latest?.score)} label="World status" />
                <span className="t-data">
                  {latest
                    ? critical
                      ? `${critical} CRITICAL FAILURE${critical > 1 ? 'S' : ''} ON RECORD`
                      : 'NO CRITICAL FAILURES'
                    : 'NO SCANS YET · RUN ONE TO CALIBRATE'}
                </span>
              </div>
            </div>
            {history.length ? (
              <ol className="history">
                {history.slice(0, 5).map((h) => (
                  <li key={h.id}>
                    <span className="t-data history__score" style={{ color: scoreColor(h.score) }}>
                      {pad2(h.score)}
                    </span>
                    <span className="history__name">{h.systemName}</span>
                    <span className="t-data t-muted">
                      {h.mode}
                      {h.simulated ? ' · demo' : ''}
                    </span>
                  </li>
                ))}
              </ol>
            ) : null}
            {history.length ? (
              <button type="button" className="history__clear t-data" onClick={clearHistory}>
                clear local history
              </button>
            ) : null}
          </section>

          <section className="module voice">
            <div className="module__head">
              <span className="t-label">Debug personality</span>
            </div>
            <PersonalitySwitch value={settings.personality} onChange={(personality) => settings.update({ personality })} />
            <p className="voice__line">{VOICE[settings.personality].line}</p>
            <p className="voice__sample">“{VOICE[settings.personality].sample}”</p>
          </section>

          <section className="module settings">
            <button
              type="button"
              className="module__head settings__toggle"
              aria-expanded={showSettings}
              onClick={() => setShowSettings((v) => !v)}
            >
              <span className="t-label">Instrument settings</span>
              <span className="t-data">{showSettings ? '−' : '+'}</span>
            </button>
            {showSettings ? (
              <div className="settings__body">
                <label className="settings__row">
                  <span>
                    Force demo mode
                    <small>Simulated diagnostics even if an API key is configured.</small>
                  </span>
                  <input
                    type="checkbox"
                    checked={settings.forceDemo}
                    onChange={(e) => settings.update({ forceDemo: e.target.checked })}
                  />
                </label>
                <label className="settings__row">
                  <span>
                    Periodic re-check
                    <small>How often a stable scene is re-analysed by the AI.</small>
                  </span>
                  <select
                    value={settings.interval}
                    onChange={(e) => settings.update({ interval: Number(e.target.value) as AnalysisInterval })}
                  >
                    <option value={10}>10 s</option>
                    <option value={20}>20 s</option>
                    <option value={40}>40 s</option>
                    <option value={0}>off</option>
                  </select>
                </label>
                <label className="settings__row">
                  <span>
                    Local vision rate
                    <small>Lower saves battery on phones.</small>
                  </span>
                  <select
                    value={settings.maxFps}
                    onChange={(e) => settings.update({ maxFps: Number(e.target.value) as 6 | 10 | 15 })}
                  >
                    <option value={6}>6 fps</option>
                    <option value={10}>10 fps</option>
                    <option value={15}>15 fps</option>
                  </select>
                </label>
                <label className="settings__row">
                  <span>
                    Vision backend
                    <small>Auto benchmarks GPU vs CPU with a self-test. Applies after reload.</small>
                  </span>
                  <select
                    defaultValue={delegatePreference()}
                    onChange={(e) => setDelegatePreference(e.target.value as 'auto' | 'CPU' | 'GPU')}
                  >
                    <option value="auto">auto</option>
                    <option value="CPU">CPU (WASM)</option>
                    <option value="GPU">GPU (WebGL)</option>
                  </select>
                </label>
              </div>
            ) : null}
          </section>
        </aside>
      </main>

      <footer className="home__foot t-data">
        Frames leave this device only when the local engine selects them for analysis · images are never stored ·
        <a href="/api/docs" target="_blank" rel="noreferrer">
          API docs
        </a>
      </footer>
    </div>
  )
}
