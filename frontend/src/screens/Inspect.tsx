import { useEffect, useMemo, useRef, useState } from 'react'
import { ErrorPanel } from '../components/ErrorPanel'
import { Icon } from '../components/Icon'
import { ApiError, getProfiles, getVisionStatus, isAbort, toApiError, visionCompare, visionDetect, visionWorkspaceDiff } from '../lib/api'
import { formatMs } from '../lib/format'
import { navigate } from '../lib/router'
import type { Profile, Profiles, QueryAnswer, VisionCompare, VisionDetect, VisionObject, VisionStatus, VisionWorkspace, WorkspaceObject } from '../lib/schemas'
import '../styles/debug.css'
import '../styles/inspect.css'

const ACCEPT = 'image/jpeg,image/png,image/webp'
const MAX_REFERENCES = 10

const ANALYSIS_TEXT: Record<VisionDetect['state'], string> = {
  complete: 'Analysis complete',
  analysis_incomplete: 'Analysis incomplete - part of the image was not analysed at full resolution',
  model_unavailable: 'Model unavailable',
  insufficient_image_quality: 'Insufficient image quality - absent objects cannot be ruled out',
}

const QUERY_TEXT: Record<QueryAnswer['state'], string> = {
  detected: 'Detected',
  tentative: 'Tentative',
  ambiguous: 'Ambiguous',
  not_detected: 'Not detected',
  unsupported_category: 'Unsupported category',
  insufficient_image_quality: 'Image too poor to tell',
  analysis_incomplete: 'Analysis incomplete',
}

interface Picked {
  file: File
  url: string
}

function pick(file: File): Picked {
  return { file, url: URL.createObjectURL(file) }
}

export default function Inspect() {
  const [status, setStatus] = useState<VisionStatus | null>(null)
  const [profiles, setProfiles] = useState<Profiles | null>(null)
  const [profileId, setProfileId] = useState('general')
  const [loadError, setLoadError] = useState<ApiError | null>(null)
  const [tab, setTab] = useState<'detect' | 'compare' | 'workspace'>('detect')

  useEffect(() => {
    const controller = new AbortController()
    Promise.all([getVisionStatus({ signal: controller.signal }), getProfiles({ signal: controller.signal })])
      .then(([s, p]) => {
        setStatus(s)
        setProfiles(p)
      })
      .catch((e) => {
        if (!isAbort(e)) setLoadError(toApiError(e))
      })
    return () => controller.abort()
  }, [])

  const profile = profiles?.[profileId] ?? null

  const chooseProfile = (id: string) => {
    setProfileId(id)
    const p = profiles?.[id]
    if (p) setTab(p.default_mode === 'compare' ? 'compare' : 'detect')
  }

  return (
    <div className="debug inspect">
      <header className="debug__bar">
        <button type="button" className="key key--small key--ghost" onClick={() => navigate('home')}>
          <Icon name="back" size={16} /> Home
        </button>
        <span className="debug__title">
          <span className="t-data t-muted">IN·D</span> Inspect
        </span>
        <span className="badge t-data" data-tone="server">
          SERVER MODELS
        </span>
      </header>

      {loadError ? <ErrorPanel error={loadError} /> : null}

      <div className="debug__grid">
        <section className="debug__input">
          <div className="module">
            <div className="module__head">
              <span className="t-label">Domain profile</span>
            </div>
            <div className="profiles" role="radiogroup" aria-label="Domain profile">
              {profiles
                ? Object.entries(profiles).map(([id, p]) => (
                    <button
                      key={id}
                      type="button"
                      role="radio"
                      aria-checked={id === profileId}
                      className="profiles__item"
                      data-active={id === profileId || undefined}
                      onClick={() => chooseProfile(id)}
                    >
                      <span className="profiles__name">{p.name}</span>
                      <span className="state-chip t-data" data-status={p.status}>
                        {p.status}
                      </span>
                    </button>
                  ))
                : null}
            </div>
            {profile ? <ProfileCard profile={profile} /> : null}
          </div>

          <ModelStatus status={status} />
        </section>

        <section className="debug__output">
          {profile?.status === 'unavailable' ? (
            <div className="notice" data-tone="block">
              <p className="t-label">Not available</p>
              <p>{profile.summary}</p>
              {profile.safety ? <p className="notice__safety">{profile.safety}</p> : null}
            </div>
          ) : (
            <>
              <div className="tabs" role="tablist">
                <button type="button" role="tab" aria-selected={tab === 'detect'} onClick={() => setTab('detect')}>
                  Detect objects
                </button>
                <button type="button" role="tab" aria-selected={tab === 'compare'} onClick={() => setTab('compare')}>
                  Compare with references
                </button>
                <button type="button" role="tab" aria-selected={tab === 'workspace'} onClick={() => setTab('workspace')}>
                  Workspace vs reference
                </button>
              </div>
              {tab === 'detect' ? (
                <DetectPanel key={profileId} profileId={profileId} profile={profile} />
              ) : tab === 'compare' ? (
                <ComparePanel profile={profile} />
              ) : (
                <WorkspacePanel />
              )}
              <p className="t-muted inspect__privacy">
                Images go to this app's own backend, are analysed in memory and discarded; nothing is stored or logged.
              </p>
            </>
          )}
        </section>
      </div>
    </div>
  )
}

function ProfileCard({ profile }: { profile: Profile }) {
  return (
    <div className="profile-card">
      <p>{profile.summary}</p>
      {profile.safety ? <p className="notice__safety">{profile.safety}</p> : null}
      {profile.unsupported.length ? (
        <details>
          <summary className="t-label">Not supported ({profile.unsupported.length})</summary>
          <ul>
            {profile.unsupported.map((u) => (
              <li key={u}>{u}</li>
            ))}
          </ul>
        </details>
      ) : null}
      {Array.isArray(profile.categories) && profile.categories.length ? (
        <details>
          <summary className="t-label">Categories in this profile ({profile.categories.length})</summary>
          <p className="t-data">{profile.categories.join(' · ')}</p>
        </details>
      ) : null}
    </div>
  )
}

function ModelStatus({ status }: { status: VisionStatus | null }) {
  if (!status) return null
  return (
    <div className="module">
      <div className="module__head">
        <span className="t-label">Server models</span>
        <span className="t-data t-muted">
          budget {status.memory_budget_mb} MB · {status.threads} thread{status.threads === 1 ? '' : 's'}
        </span>
      </div>
      <ul className="model-list">
        {status.models.map((m) => (
          <li key={m.id}>
            <span className="state-chip t-data" data-status={m.available ? 'available' : 'unavailable'}>
              {m.available ? (m.loaded ? 'loaded' : 'ready') : 'unavailable'}
            </span>
            <span>{m.name ?? m.id}</span>
            {!m.available || m.load_error ? <span className="t-muted model-list__why">{m.load_error ?? m.reason}</span> : null}
          </li>
        ))}
      </ul>
    </div>
  )
}

function ImagePicker({ label, picked, onPick, disabled }: { label: string; picked: Picked | null; onPick: (f: File) => void; disabled?: boolean }) {
  return (
    <label className="key key--signal debug__pick">
      <Icon name="upload" size={18} /> {picked ? `Another ${label}` : `Choose ${label}`}
      <input type="file" accept={ACCEPT} hidden disabled={disabled} onChange={(e) => e.target.files?.[0] && onPick(e.target.files[0])} />
    </label>
  )
}

function DetectPanel({ profileId, profile }: { profileId: string; profile: Profile | null }) {
  const [picked, setPicked] = useState<Picked | null>(null)
  const [mode, setMode] = useState<'standard' | 'precision'>(profile?.default_mode === 'precision' ? 'precision' : 'standard')
  const [queries, setQueries] = useState('')
  const [result, setResult] = useState<VisionDetect | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [busy, setBusy] = useState(false)
  const [focus, setFocus] = useState<number | null>(null)
  const [showTentative, setShowTentative] = useState(false)
  const abort = useRef<AbortController | null>(null)

  useEffect(() => () => abort.current?.abort(), [])
  useEffect(() => () => (picked ? URL.revokeObjectURL(picked.url) : undefined), [picked])

  const run = async () => {
    if (!picked) return
    abort.current?.abort()
    const controller = new AbortController()
    abort.current = controller
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      setResult(await visionDetect(picked.file, { mode, profile: profileId, queries, signal: controller.signal }))
    } catch (e) {
      if (!isAbort(e)) setError(toApiError(e))
    } finally {
      setBusy(false)
    }
  }

  const suggestions = Array.isArray(profile?.categories) ? profile.categories.slice(0, 4).join(', ') : 'person, bottle, forklift'

  return (
    <div className="inspect__panel">
      <div className="debug__pickers">
        <ImagePicker label="image" picked={picked} disabled={busy} onPick={(f) => (setPicked(pick(f)), setResult(null))} />
        <div className="segmented" role="radiogroup" aria-label="Detection mode">
          {(['standard', 'precision'] as const).map((m) => (
            <button key={m} type="button" role="radio" aria-checked={mode === m} onClick={() => setMode(m)} disabled={busy}>
              {m === 'standard' ? 'Standard' : 'Precision (tiles)'}
            </button>
          ))}
        </div>
      </div>
      <label className="inspect__query">
        <span className="t-label">Look for (optional, comma-separated)</span>
        <input type="text" value={queries} placeholder={suggestions} onChange={(e) => setQueries(e.target.value)} maxLength={300} />
      </label>
      <button type="button" className="key key--signal" disabled={!picked || busy} onClick={() => void run()}>
        <Icon name="scan" size={18} /> {busy ? 'Analysing on the server…' : 'Analyse'}
      </button>
      {busy ? <div className="spinner-bar" /> : null}
      {error ? <ErrorPanel error={error} /> : null}

      {picked ? (
        <figure className="annotated inspect__figure">
          <img src={picked.url} alt="Image to analyse" />
          {result?.objects
            .filter((o) => showTentative || o.state !== 'tentative')
            .map((o) => (
              <ObjectBox key={o.id} o={o} result={result} focused={focus === o.id} onFocus={setFocus} />
            ))}
        </figure>
      ) : null}
      {result?.objects.some((o) => o.state === 'tentative') ? (
        <label className="inspect__toggle">
          <input type="checkbox" checked={showTentative} onChange={(e) => setShowTentative(e.target.checked)} /> Show tentative boxes (
          {result.objects.filter((o) => o.state === 'tentative').length}, below the operating threshold)
        </label>
      ) : null}

      {result ? <DetectSummary result={result} /> : null}
    </div>
  )
}

function ObjectBox({ o, result, focused, onFocus }: { o: VisionObject; result: VisionDetect; focused: boolean; onFocus: (id: number | null) => void }) {
  const [x1, y1, x2, y2] = o.box
  const { width: w, height: h } = result.image
  return (
    <span
      className="vbox"
      data-state={o.state}
      data-focused={focused || undefined}
      onMouseEnter={() => onFocus(o.id)}
      onMouseLeave={() => onFocus(null)}
      style={{ left: `${(x1 / w) * 100}%`, top: `${(y1 / h) * 100}%`, width: `${((x2 - x1) / w) * 100}%`, height: `${((y2 - y1) / h) * 100}%` }}
    >
      <em>
        {o.label} {Math.round(o.confidence * 100)}%{o.state === 'ambiguous' && o.alternative_label ? ` / ${o.alternative_label}?` : ''}
      </em>
    </span>
  )
}

function DetectSummary({ result }: { result: VisionDetect }) {
  const counts = useMemo(() => {
    const c = { detected: 0, tentative: 0, ambiguous: 0 }
    for (const o of result.objects) c[o.state] += 1
    return c
  }, [result])
  return (
    <div className="inspect__result">
      <div className="notice" data-tone={result.state === 'complete' ? 'ok' : 'warn'}>
        <p className="t-label">{ANALYSIS_TEXT[result.state]}</p>
        <p className="t-data">
          {counts.detected} detected · {counts.ambiguous} ambiguous · {counts.tentative} tentative · {result.mode} mode ·{' '}
          {result.model.id} ({result.model.vocabulary}) · {formatMs(result.timings_ms.request ?? result.timings_ms.total ?? 0)}
        </p>
        {result.quality.issues.map((i) => (
          <p key={i.code}>{i.message}</p>
        ))}
        {result.notes.map((n) => (
          <p key={n} className="t-muted">
            {n}
          </p>
        ))}
      </div>

      {result.queries.length ? (
        <ul className="answers">
          {result.queries.map((q) => (
            <li key={q.query}>
              <span className="state-chip t-data" data-query={q.state}>
                {QUERY_TEXT[q.state]}
              </span>
              <strong>{q.query}</strong>
              <span className="t-muted">{q.explanation}</span>
            </li>
          ))}
        </ul>
      ) : null}

      <details className="legend">
        <summary className="t-label">What the box styles mean</summary>
        <ul>
          <li>
            <span className="vbox-key" data-state="detected" /> detected - at or above the operating threshold ({result.model.operating_threshold})
          </li>
          <li>
            <span className="vbox-key" data-state="ambiguous" /> ambiguous - a second class scores at least half as high
          </li>
          <li>
            <span className="vbox-key" data-state="tentative" /> tentative - below the threshold; may be wrong
          </li>
        </ul>
      </details>
    </div>
  )
}

function ComparePanel({ profile }: { profile: Profile | null }) {
  const [refs, setRefs] = useState<Picked[]>([])
  const [target, setTarget] = useState<Picked | null>(null)
  const [result, setResult] = useState<VisionCompare | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [busy, setBusy] = useState(false)
  const abort = useRef<AbortController | null>(null)

  const urls = useRef<string[]>([])
  useEffect(() => () => abort.current?.abort(), [])
  useEffect(() => () => urls.current.forEach((u) => URL.revokeObjectURL(u)), [])

  const addRefs = (files: FileList | null) => {
    if (!files) return
    const added = Array.from(files).slice(0, MAX_REFERENCES - refs.length).map(pick)
    urls.current.push(...added.map((r) => r.url))
    setRefs([...refs, ...added])
    setResult(null)
  }

  const run = async () => {
    if (!target || refs.length < 2) return
    abort.current?.abort()
    const controller = new AbortController()
    abort.current = controller
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      setResult(await visionCompare(target.file, refs.map((r) => r.file), { signal: controller.signal }))
    } catch (e) {
      if (!isAbort(e)) setError(toApiError(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="inspect__panel">
      <p className="t-muted">
        Photograph 2-{MAX_REFERENCES} known-good parts and the part to check the same way (same pose, background and light).
        {profile?.status === 'experimental' ? ' Experimental: a screening aid, not a pass/fail decision.' : ''}
      </p>
      <div className="debug__pickers">
        <label className="key debug__pick">
          <Icon name="upload" size={18} /> Add references ({refs.length}/{MAX_REFERENCES})
          <input type="file" accept={ACCEPT} multiple hidden disabled={busy} onChange={(e) => addRefs(e.target.files)} />
        </label>
        <ImagePicker label="part to check" picked={target} disabled={busy} onPick={(f) => (target && URL.revokeObjectURL(target.url), setTarget(pick(f)), setResult(null))} />
      </div>
      {refs.length ? (
        <div className="refs">
          {refs.map((r, i) => (
            <figure key={r.url}>
              <img src={r.url} alt={`Reference ${i + 1}`} />
              <button type="button" className="key key--small key--ghost" onClick={() => (URL.revokeObjectURL(r.url), setRefs(refs.filter((x) => x !== r)))} aria-label={`Remove reference ${i + 1}`}>
                ×
              </button>
            </figure>
          ))}
        </div>
      ) : null}
      <button type="button" className="key key--signal" disabled={!target || refs.length < 2 || busy} onClick={() => void run()}>
        <Icon name="scan" size={18} /> {busy ? 'Comparing on the server…' : 'Compare'}
      </button>
      {busy ? <div className="spinner-bar" /> : null}
      {error ? <ErrorPanel error={error} /> : null}

      {target ? (
        <figure className="annotated inspect__figure">
          <img src={target.url} alt="Part to check" />
          {result?.heatmap.data ? <Heatmap data={result.heatmap.data} /> : null}
          {result?.regions.map((r, i) => {
            const [x1, y1, x2, y2] = r.box
            const { width: w, height: h } = result.image
            return (
              <span
                key={i}
                className="vbox"
                data-state="anomaly"
                style={{ left: `${(x1 / w) * 100}%`, top: `${(y1 / h) * 100}%`, width: `${((x2 - x1) / w) * 100}%`, height: `${((y2 - y1) / h) * 100}%` }}
              >
                <em>region {i + 1}</em>
              </span>
            )
          })}
        </figure>
      ) : null}

      {result ? (
        <div className="notice" data-tone={result.verdict === 'anomalous' ? 'warn' : 'ok'}>
          <p className="t-label">{result.verdict === 'anomalous' ? 'Differs from every reference' : 'Within the references’ variation'}</p>
          <p className="t-data">
            score {result.score.toFixed(3)} vs threshold {result.threshold.toFixed(3)} ({result.score_to_threshold.toFixed(2)}×) · {result.regions.length}{' '}
            region{result.regions.length === 1 ? '' : 's'} · {result.references} references · {formatMs(result.timings_ms.total ?? 0)}
          </p>
          <p className="t-muted">
            {result.method.name}; threshold = {result.method.threshold_rule}.
          </p>
          {result.quality.issues.map((i) => (
            <p key={i.code}>{i.message}</p>
          ))}
          {result.notes.map((n) => (
            <p key={n}>{n}</p>
          ))}
        </div>
      ) : null}
    </div>
  )
}

/** A box (pixel xyxy in an image of the given size) as a percentage-positioned overlay. */
function Overlay({ box, size, state, label }: { box: readonly number[]; size: { width: number; height: number }; state: string; label: string }) {
  const [x1, y1, x2, y2] = box as [number, number, number, number]
  return (
    <span
      className="vbox"
      data-state={state}
      style={{
        left: `${(x1 / size.width) * 100}%`,
        top: `${(y1 / size.height) * 100}%`,
        width: `${((x2 - x1) / size.width) * 100}%`,
        height: `${((y2 - y1) / size.height) * 100}%`,
      }}
    >
      <em>{label}</em>
    </span>
  )
}

const pct = (v: number | null | undefined) => (v == null ? '—' : `${Math.round(v * 100)}%`)

function ChangeList({ title, items, describe }: { title: string; items: WorkspaceObject[]; describe: (o: WorkspaceObject) => string }) {
  if (!items.length) return null
  return (
    <div className="ws-list">
      <p className="t-label">
        {title} ({items.length})
      </p>
      <ul>
        {items.map((o, i) => (
          <li key={`${o.label}-${i}`}>
            <strong>{o.label}</strong> <span className="t-muted">{describe(o)}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/**
 * Workspace vs reference: the user's own "as it should be" photo against a photo taken now. Reports measured
 * differences only (app/vision/workspace.py); it never judges what a good workspace looks like.
 */
function WorkspacePanel() {
  const [reference, setReference] = useState<Picked | null>(null)
  const [current, setCurrent] = useState<Picked | null>(null)
  const [mode, setMode] = useState<'standard' | 'precision'>('standard')
  const [result, setResult] = useState<VisionWorkspace | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [busy, setBusy] = useState(false)
  const abort = useRef<AbortController | null>(null)

  useEffect(() => () => abort.current?.abort(), [])
  useEffect(() => () => (reference ? URL.revokeObjectURL(reference.url) : undefined), [reference])
  useEffect(() => () => (current ? URL.revokeObjectURL(current.url) : undefined), [current])

  const run = async () => {
    if (!reference || !current) return
    abort.current?.abort()
    const controller = new AbortController()
    abort.current = controller
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      setResult(await visionWorkspaceDiff(reference.file, current.file, { mode, signal: controller.signal }))
    } catch (e) {
      if (!isAbort(e)) setError(toApiError(e))
    } finally {
      setBusy(false)
    }
  }

  const objects = result?.objects ?? null
  const reg = result?.registration ?? null
  const refSize = result?.reference ?? null
  const curSize = result?.current ?? null

  return (
    <div className="inspect__panel">
      <p className="t-muted">
        Photograph your workspace the way you want it (the reference), then photograph it now from about the same place. The comparison reports
        measured differences: recognised objects added, removed or moved, and areas whose pixels changed. It does not judge what a good
        workspace looks like.
      </p>
      <div className="debug__pickers">
        <ImagePicker label="reference" picked={reference} disabled={busy} onPick={(f) => (setReference(pick(f)), setResult(null))} />
        <ImagePicker label="current photo" picked={current} disabled={busy} onPick={(f) => (setCurrent(pick(f)), setResult(null))} />
        <div className="segmented" role="radiogroup" aria-label="Detection mode">
          {(['standard', 'precision'] as const).map((m) => (
            <button key={m} type="button" role="radio" aria-checked={mode === m} onClick={() => setMode(m)} disabled={busy}>
              {m === 'standard' ? 'Standard' : 'Precision (tiles)'}
            </button>
          ))}
        </div>
      </div>
      <button type="button" className="key key--signal" disabled={!reference || !current || busy} onClick={() => void run()}>
        <Icon name="scan" size={18} /> {busy ? 'Comparing on the server…' : 'Compare with reference'}
      </button>
      {busy ? <div className="spinner-bar" /> : null}
      {error ? <ErrorPanel error={error} /> : null}

      {reference || current ? (
        <div className="ws-pair">
          {reference ? (
            <figure className="annotated inspect__figure">
              <img src={reference.url} alt="Reference photo" />
              <figcaption className="t-label">Reference</figcaption>
              {objects && refSize
                ? [
                    ...objects.removed.map((o, i) => <Overlay key={`r${i}`} box={o.reference_box!} size={refSize} state="removed" label={`${o.label} removed`} />),
                    ...objects.moved.map((o, i) => <Overlay key={`m${i}`} box={o.reference_box!} size={refSize} state="moved-from" label={`${o.label} was here`} />),
                  ]
                : null}
            </figure>
          ) : null}
          {current ? (
            <figure className="annotated inspect__figure">
              <img src={current.url} alt="Current photo" />
              <figcaption className="t-label">Now</figcaption>
              {objects && curSize
                ? [
                    ...objects.added.map((o, i) => <Overlay key={`a${i}`} box={o.current_box!} size={curSize} state="added" label={`${o.label} added`} />),
                    ...objects.moved.map((o, i) => <Overlay key={`m${i}`} box={o.current_box!} size={curSize} state="moved" label={`${o.label} moved`} />),
                    ...(result?.changes?.regions ?? []).map((g, i) => (
                      <Overlay
                        key={`c${i}`}
                        box={g.current_box}
                        size={curSize}
                        state="changed"
                        label={g.explained_by.length ? `changed: ${g.explained_by.join(', ')}` : g.possibly.length ? `changed: possibly ${g.possibly.join(', ')}` : 'changed: not recognised'}
                      />
                    )),
                  ]
                : null}
            </figure>
          ) : null}
        </div>
      ) : null}

      {result && reg ? (
        <div className="inspect__result">
          <div className="notice" data-tone={reg.ok ? 'ok' : 'warn'}>
            <p className="t-label">{reg.ok ? 'Photos aligned' : 'Photos could not be aligned'}</p>
            <p>{result.summary}</p>
            <p className="t-data">
              {reg.method}: {reg.inliers} of {reg.matches} matches agree ({pct(reg.inlier_ratio)})
              {reg.overlap != null ? ` · ${pct(reg.overlap)} of the reference overlaps` : ''}
              {result.changes ? ` · ${(result.changes.changed_fraction * 100).toFixed(1)}% of the overlap changed` : ''} · {result.model.id} (
              {result.model.vocabulary}) · {formatMs(result.timings_ms.total ?? 0)}
            </p>
            {[...result.quality.reference.issues.map((i) => `Reference: ${i.message}`), ...result.quality.current.issues.map((i) => `Now: ${i.message}`)].map(
              (m) => (
                <p key={m}>{m}</p>
              ),
            )}
          </div>

          {objects ? (
            <div className="ws-lists">
              <ChangeList title="Added" items={objects.added} describe={(o) => `${pct(o.confidence)} · ${pct(o.pixel_change)} of its box changed`} />
              <ChangeList title="Removed" items={objects.removed} describe={(o) => `${pct(o.confidence)} · ${pct(o.pixel_change)} of its box changed`} />
              <ChangeList title="Moved" items={objects.moved} describe={(o) => `by ${pct(o.moved_by)} of the frame diagonal`} />
              <ChangeList title="Uncertain" items={objects.uncertain} describe={(o) => `${o.change ?? ''} · ${pct(o.confidence)} · ${o.why ?? 'low detection confidence'}`} />
              <p className="t-muted t-data">{objects.unchanged.length} recognised object(s) unchanged.</p>
            </div>
          ) : Object.keys(result.counts).length ? (
            <div className="ws-list">
              <p className="t-label">Category counts (confident detections)</p>
              <ul>
                {Object.entries(result.counts).map(([label, c]) => (
                  <li key={label}>
                    <strong>{label}</strong> <span className="t-muted">reference {c.reference} · now {c.current}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          <details className="legend">
            <summary className="t-label">Limits of this comparison</summary>
            <ul>
              {[...result.limitations, ...result.notes].map((l) => (
                <li key={l}>{l}</li>
              ))}
            </ul>
          </details>
        </div>
      ) : null}
    </div>
  )
}

/** Colourise the server's anomaly map (128 = decision threshold) and lay it over the image. */
function Heatmap({ data }: { data: string }) {
  const canvas = useRef<HTMLCanvasElement | null>(null)
  useEffect(() => {
    const img = new Image()
    img.onload = () => {
      const c = canvas.current
      if (!c) return
      c.width = img.width
      c.height = img.height
      const ctx = c.getContext('2d', { willReadFrequently: true })
      if (!ctx) return
      ctx.drawImage(img, 0, 0)
      const pixels = ctx.getImageData(0, 0, c.width, c.height)
      const d = pixels.data
      for (let i = 0; i < d.length; i += 4) {
        const v = d[i] ?? 0
        const above = v >= 128
        d[i] = 255
        d[i + 1] = above ? 40 : 190
        d[i + 2] = 0
        d[i + 3] = Math.round(Math.min(1, Math.max(0, (v - 64) / 128)) * 170)
      }
      ctx.putImageData(pixels, 0, 0)
    }
    img.src = `data:image/png;base64,${data}`
  }, [data])
  return <canvas ref={canvas} className="heatmap" aria-label="Anomaly heat map" />
}
