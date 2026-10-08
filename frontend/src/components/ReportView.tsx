import { useMemo, useState, type ReactNode } from 'react'
import { canShareFiles } from '../lib/env'
import { formatMs, formatPercent, pad2 } from '../lib/format'
import type { Report } from '../lib/schemas'
import { copyText, downloadBlob, renderShareCard, reportText, type CardImage } from '../lib/share'
import { toast } from '../state/toasts'
import { DemoBadge, StatusChip } from './Bits'
import { FindingCard } from './FindingCard'
import { Icon } from './Icon'
import { Meter, scoreColor } from './Meter'

export function ScoreReadout({ report }: { report: Report }) {
  const rows: [string, ReactNode][] = [
    ['System', <span className="t-data report__sys">{report.system_name}</span>],
    ['Status', <StatusChip status={report.status} />],
    ['Active bugs', <span className="report__num">{pad2(report.counts.active_bugs)}</span>],
    ['High priority', <span className="report__num">{pad2(report.counts.high_priority)}</span>],
    ['Optimizations', <span className="report__num">{pad2(report.counts.optimizations)}</span>],
  ]
  return (
    <div className="readout">
      <div className="readout__score">
        <span className="t-label">System score</span>
        <span className="readout__value" style={{ color: scoreColor(report.system_score) }}>
          {Math.round(report.system_score)}
          <small>/ 100</small>
        </span>
        <Meter value={report.system_score / 100} color={scoreColor(report.system_score)} height={10} label="System score" />
      </div>
      <dl className="readout__rows">
        {rows.map(([k, v]) => (
          <div key={k}>
            <dt className="t-label">{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

export function ReportActions({ report, image }: { report: Report; image: CardImage | null }) {
  const [busy, setBusy] = useState(false)
  const [includePhoto, setIncludePhoto] = useState(true)
  const filename = `reality-debugger-${report.system_name.toLowerCase().replace(/[^a-z0-9.]+/g, '-')}.png`

  const onCopy = async () => {
    const ok = await copyText(reportText(report))
    toast(ok ? 'Report copied to clipboard' : 'Copy failed - select the text manually', ok ? 'ok' : 'error')
  }

  const makeCard = async () => renderShareCard(report, includePhoto ? image : null)

  const onDownload = async () => {
    setBusy(true)
    try {
      downloadBlob(await makeCard(), filename)
      toast('Diagnostic card saved', 'ok')
    } catch {
      toast('Could not render the card', 'error')
    } finally {
      setBusy(false)
    }
  }

  const onShare = async () => {
    setBusy(true)
    try {
      const blob = await makeCard()
      const file = new File([blob], filename, { type: 'image/png' })
      if (canShareFiles(file)) {
        await navigator.share({ files: [file], title: 'Reality Debugger', text: report.final_diagnosis })
      } else {
        downloadBlob(blob, filename)
        toast('Sharing is not supported here - card downloaded instead', 'info')
      }
    } catch (error) {
      if (!(error instanceof DOMException && error.name === 'AbortError')) toast('Sharing failed', 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="report-actions">
      <button type="button" className="key key--small" onClick={onCopy}>
        <Icon name="copy" size={16} /> Copy report
      </button>
      <button type="button" className="key key--small" onClick={onDownload} disabled={busy}>
        <Icon name="download" size={16} /> Download card
      </button>
      <button type="button" className="key key--small" onClick={onShare} disabled={busy}>
        <Icon name="share" size={16} /> Share
      </button>
      {image ? (
        <label className="report-actions__toggle t-data">
          <input type="checkbox" checked={includePhoto} onChange={(e) => setIncludePhoto(e.target.checked)} /> include
          photo on card
        </label>
      ) : null}
    </div>
  )
}

export function ReportView({
  report,
  image = null,
  focusedId = null,
  onFocus,
  showTimes = false,
  heading = 'Reality diagnostic',
}: {
  report: Report
  image?: CardImage | null
  focusedId?: string | null
  onFocus?: (id: string | null) => void
  showTimes?: boolean
  heading?: string
}) {
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(report.findings.slice(0, 2).map((f) => f.id)))
  const objectLabel = useMemo(() => new Map(report.objects.map((o) => [o.id, o.label])), [report.objects])
  const toggle = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  return (
    <section className="report" aria-label="Diagnostic report">
      <header className="report__head">
        <div>
          <div className="t-label">{heading}</div>
          <div className="report__meta t-data">
            {report.report_id.toUpperCase()} · {new Date(report.created_at).toLocaleTimeString([], { hour12: false })}
            {report.latency_ms > 0 ? ` · ${formatMs(report.latency_ms)}` : ''}
          </div>
        </div>
        {report.simulated ? <DemoBadge /> : <span className="report__engine t-data">{report.model}</span>}
      </header>

      <ScoreReadout report={report} />

      <blockquote className="report__diagnosis">
        <span className="t-label">Final diagnosis</span>
        <p>{report.final_diagnosis}</p>
      </blockquote>
      {report.scene.summary ? <p className="report__summary">{report.scene.summary}</p> : null}

      <div className="report__section">
        <div className="report__section-head">
          <span className="t-label">Findings</span>
          <span className="t-data t-muted">{report.findings.length} filed</span>
        </div>
        {report.findings.length === 0 ? (
          <p className="t-muted">No findings. Either this place is immaculate, or the frame was too ambiguous.</p>
        ) : (
          <div className="report__findings">
            {report.findings.map((f) => (
              <FindingCard
                key={f.id}
                finding={f}
                expanded={expanded.has(f.id)}
                onToggle={() => toggle(f.id)}
                focused={focusedId === f.id}
                onFocus={onFocus}
                showTimes={showTimes}
              />
            ))}
          </div>
        )}
      </div>

      {report.optimizations.length ? (
        <div className="report__section">
          <div className="report__section-head">
            <span className="t-label">Optimizations</span>
            <span className="t-data t-muted">not bugs, just upside</span>
          </div>
          <ul className="optimizations">
            {report.optimizations.map((o) => (
              <li key={o.id}>
                <span className="optimizations__effort t-data">{o.effort}</span>
                <div>
                  <strong>{o.title}</strong>
                  {o.description ? <p>{o.description}</p> : null}
                  {o.impact ? <p className="t-muted">{o.impact}</p> : null}
                </div>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {report.objects.length ? (
        <div className="report__section">
          <div className="report__section-head">
            <span className="t-label">Scene graph</span>
            <span className="t-data t-muted">
              {report.objects.length} objects{report.relationships.length ? ` · ${report.relationships.length} relations` : ''}
            </span>
          </div>
          <div className="scene-graph">
            {report.objects.map((o) => (
              <span key={o.id} className="scene-graph__node t-data">
                {o.label} <em>{formatPercent(o.confidence)}</em>
              </span>
            ))}
          </div>
          {report.relationships.length ? (
            <ul className="relations t-data">
              {report.relationships.map((r, i) => (
                <li key={i}>
                  <b>{objectLabel.get(r.subject) ?? r.subject}</b> <span>— {r.relation} →</span>{' '}
                  <b>{objectLabel.get(r.object) ?? r.object}</b>
                  {r.observation ? <small> {r.observation}</small> : null}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}

      {report.warnings.length ? (
        <ul className="report__warnings t-data">
          {report.warnings.map((w) => (
            <li key={w}>⚠ {w}</li>
          ))}
        </ul>
      ) : null}

      <footer className="report__foot t-data">
        {report.simulated
          ? 'DEMO MODE: produced by fixed heuristics over on-device detections. Not AI output.'
          : `Reasoned by ${report.provider} · ${report.model}. Observations are separated from inferences; confidence is the model's own estimate.`}
      </footer>

      <ReportActions report={report} image={image} />
    </section>
  )
}
