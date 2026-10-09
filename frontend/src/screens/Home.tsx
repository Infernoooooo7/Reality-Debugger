import { useEffect, useState, type ReactNode } from 'react'
import { AIBadge, LocalBadge, PersonalitySwitch } from '../components/Bits'
import { Icon } from '../components/Icon'
import { Meter, scoreColor } from '../components/Meter'
import { BACKEND_START_HINT } from '../lib/api'
import { cameraSupport } from '../lib/env'
import { formatMs, pad2, timeOfDay } from '../lib/format'
import { navigate } from '../lib/router'
import type { Personality } from '../lib/schemas'
import { useSettings } from '../state/settings'
import { aiMode, providerLabel, useSystem } from '../state/system'
import { delegatePreference, setDelegatePreference, useVision, vision } from '../vision/client'
import { useDeep } from '../vision/deep/client'
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
  const deep = useDeep()
  const mode = aiMode(health)

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

  const localLine: { state: LineState; value: string; detail?: string } = health
    ? {
        state: 'ok',
        value: `ACTIVE · ${health.local.rules.length} RULES`,
        detail: `${health.local.engine} · ${health.local.labels} classes from model metadata · ${health.local.attributes.length} attributes`,
      }
    : { state: healthError ? 'fault' : 'busy', value: healthError ? 'OFFLINE' : 'PROBING' }

  // The AI layer is optional: "off" is a normal state, never a fault.
  let aiLine: { state: LineState; value: string; detail?: string }
  if (!health) aiLine = { state: 'idle', value: 'UNKNOWN' }
  else if (mode === 'off') aiLine = { state: 'idle', value: 'OFF · LOCAL ONLY', detail: health.ai.detail }
  else if (!aiCheck) aiLine = { state: 'busy', value: `CHECKING ${providerLabel(health.ai.provider).toUpperCase()}` }
  else if (aiCheck.ok) aiLine = { state: 'ok', value: `ONLINE · ${providerLabel(aiCheck.provider)} · ${aiCheck.model ?? ''}` }
  else aiLine = { state: 'warn', value: 'UNAVAILABLE', detail: `${aiCheck.error?.message ?? 'Provider check failed'} Local CV keeps working.` }

  const deepLine: { state: LineState; value: string; detail?: string } =
    settings.deepMode === 'off'
      ? { state: 'idle', value: 'OFF', detail: 'Switched off in settings.' }
      : deep.status === 'ready' && deep.info
        ? {
            state: 'ok',
            value: `READY · ${deep.info.backend.toUpperCase()}${deep.emaMs ? ` · ${formatMs(deep.emaMs)}` : ''}`,
            detail: `${deep.info.model} · ${deep.info.threads} thread(s)${deep.info.fallbackReason ? ` · ${deep.info.fallbackReason}` : ''}`,
          }
        : deep.status === 'loading'
          ? {
              state: 'busy',
              value: deep.progress?.total ? `LOADING ${Math.round((100 * deep.progress.loaded) / deep.progress.total)}%` : 'LOADING',
              detail: 'YOLOX-S · one-time download, cached afterwards',
            }
          : deep.status === 'error'
            ? { state: 'warn', value: 'UNAVAILABLE', detail: deep.error ?? undefined }
            : { state: 'idle', value: 'ON DEMAND', detail: 'Loads for Deep Scan, Image and Video Debug.' }

  const info = visionState.info
  const visionLine: { state: LineState; value: string; detail?: string } =
    visionState.status === 'ready' && info
      ? {
          state: 'ok',
          value: `ONLINE · ${info.delegate} · ${info.runtime === 'worker' ? 'WORKER' : 'MAIN'}`,
          detail: `${info.model} · loaded in ${formatMs(info.loadMs)}`,
        }
      : visionState.status === 'loading'
        ? { state: 'busy', value: 'LOADING MODEL', detail: 'fast detector · WebAssembly' }
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
          <LocalBadge />
          <AIBadge mode={mode} provider={providerLabel(health?.ai.provider)} />
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
            Point a camera at a desk, a kitchen, a gaming setup. On-device computer vision detects, tracks and measures
            every object continuously, and a local diagnostic engine files the bug reports - severity, measured evidence,
            impact and a fix. No API key needed; an optional AI layer can add explanations.
          </p>

          <nav className="modes" aria-label="Input modes">
            <button type="button" className="mode mode--primary key--signal" onClick={() => navigate('live')}>
              <span className="mode__ch t-data">IN·A</span>
              <span className="mode__name">Enter live scan</span>
              <span className="mode__desc">Continuous camera diagnostics · detection + tracking every frame · AI optional</span>
              <span className="mode__go" aria-hidden="true">
                <Icon name="arrow" size={28} strokeWidth={2.4} />
              </span>
            </button>
            <button type="button" className="mode" onClick={() => navigate('image')}>
              <span className="mode__ch t-data">IN·B</span>
              <span className="mode__name">Image debug</span>
              <span className="mode__desc">Photo, gallery or upload → fast + deep detectors → diagnostic</span>
              <span className="mode__go" aria-hidden="true">
                <Icon name="image" size={22} />
              </span>
            </button>
            <button type="button" className="mode" onClick={() => navigate('video')}>
              <span className="mode__ch t-data">IN·C</span>
              <span className="mode__name">Video debug</span>
              <span className="mode__desc">Tracking across samples · scene cuts · diagnostic timeline</span>
              <span className="mode__go" aria-hidden="true">
                <Icon name="video" size={22} />
              </span>
            </button>
            <button type="button" className="mode" onClick={() => navigate('inspect')}>
              <span className="mode__ch t-data">IN·D</span>
              <span className="mode__name">Inspect</span>
              <span className="mode__desc">Server models · small-object precision scan · compare a part with known-good references</span>
              <span className="mode__go" aria-hidden="true">
                <Icon name="scan" size={22} />
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
                <li>Measure</li>
                <li>Scene model</li>
              </ol>
            </div>
            <div className="signal-path__zone signal-path__zone--server">
              <span className="t-label">Your backend</span>
              <ol>
                <li>Rules</li>
                <li>Lifecycle</li>
                <li>Score</li>
                <li>AI (optional)</li>
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
              <PostLine label="Local CV engine" {...localLine} />
              <PostLine label="AI reasoning" {...aiLine} />
              <PostLine label="Fast detector" {...visionLine} />
              <PostLine label="Deep detector" {...deepLine} />
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
                {latest?.score != null ? pad2(latest.score) : '—'}
              </span>
              <div className="world__meter">
                <Meter value={latest?.score != null ? latest.score / 100 : null} color={scoreColor(latest?.score)} label="World status" />
                <span className="t-data">
                  {latest
                    ? latest.score === null
                      ? `LAST SCAN UNRATED · ${latest.status === 'LIMITED' ? 'LIMITED INSPECTION' : latest.status}`
                      : critical
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
                    <span className="t-data history__score" style={{ color: scoreColor(h.score) }} title={h.score === null ? `unrated (${h.status})` : undefined}>
                      {h.score === null ? '--' : pad2(h.score)}
                    </span>
                    <span className="history__name">{h.systemName}</span>
                    <span className="t-data t-muted">
                      {h.mode}
                      {h.ai ? ' · AI' : ' · local'}
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
                    Deep detector
                    <small>YOLOX-S (36 MB, cached): Deep Scan, Image/Video Debug, live checks when fast enough.</small>
                  </span>
                  <select value={settings.deepMode} onChange={(e) => settings.update({ deepMode: e.target.value as 'auto' | 'off' })}>
                    <option value="auto">auto</option>
                    <option value="off">off</option>
                  </select>
                </label>
                <label className="settings__row">
                  <span>
                    Deep detector runtime
                    <small>WebGPU when available, else multi-threaded WebAssembly. Applies after reload.</small>
                  </span>
                  <select
                    value={settings.deepBackend}
                    onChange={(e) => settings.update({ deepBackend: e.target.value as 'auto' | 'webgpu' | 'wasm' })}
                  >
                    <option value="auto">auto</option>
                    <option value="webgpu">WebGPU</option>
                    <option value="wasm">WebAssembly</option>
                  </select>
                </label>
                <label className="settings__row">
                  <span>
                    Developer metrics
                    <small>Latency, tracking and AI-usage panel in Live Scan.</small>
                  </span>
                  <input type="checkbox" checked={settings.devPanel} onChange={(e) => settings.update({ devPanel: e.target.checked })} />
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
        Only measurements leave this device unless AI reasoning is configured · images are never stored ·
        <a href="/api/docs" target="_blank" rel="noreferrer">
          API docs
        </a>
      </footer>
    </div>
  )
}
