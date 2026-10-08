import { useEffect, useRef, useState } from 'react'
import { CATEGORY_LABEL, formatClock, formatPercent } from '../lib/format'
import type { Finding, FindingStatus } from '../lib/schemas'
import { SeverityTag, StatusChip } from './Bits'

const STAMP: Partial<Record<FindingStatus, string>> = {
  CONFIRMED: 'Confirmed',
  RESOLVED: 'Resolved',
  TRACKING: 'Tracking',
}

export function FindingCard({
  finding,
  expanded,
  onToggle,
  onFocus,
  focused = false,
  showTimes = false,
  onJump,
}: {
  finding: Finding
  expanded: boolean
  onToggle?: () => void
  onFocus?: (id: string | null) => void
  focused?: boolean
  showTimes?: boolean
  onJump?: (seconds: number) => void
}) {
  const previous = useRef<FindingStatus>(finding.status)
  const [flash, setFlash] = useState<FindingStatus | null>(null)

  useEffect(() => {
    if (previous.current === finding.status) return
    previous.current = finding.status
    setFlash(finding.status)
    const timer = window.setTimeout(() => setFlash(null), 1700)
    return () => window.clearTimeout(timer)
  }, [finding.status])

  const related = finding.related_objects.slice(0, 3).join(', ')
  return (
    <article
      className="finding"
      data-severity={finding.severity}
      data-status={finding.status}
      data-flash={flash ?? undefined}
      data-focused={focused || undefined}
      data-out={finding.out_of_view || undefined}
      onMouseEnter={onFocus ? () => onFocus(finding.id) : undefined}
      onMouseLeave={onFocus ? () => onFocus(null) : undefined}
    >
      <button type="button" className="finding__head" onClick={onToggle} aria-expanded={expanded}>
        <span className="finding__id t-data">{finding.id}</span>
        <SeverityTag severity={finding.severity} />
        <span className="finding__cat t-label">{CATEGORY_LABEL[finding.category] ?? finding.category}</span>
        <StatusChip status={finding.status} />
      </button>
      <h3 className="finding__title" onClick={onToggle}>
        {finding.title}
      </h3>
      <div className="finding__meta t-data">
        <span>{formatPercent(finding.confidence)} conf</span>
        {finding.sightings > 1 ? <span>seen {finding.sightings}×</span> : null}
        {showTimes && finding.first_seen_s != null ? (
          onJump ? (
            <button type="button" className="finding__jump" onClick={() => onJump(finding.first_seen_s!)}>
              ▶ {formatClock(finding.first_seen_s)}
              {finding.last_seen_s != null && finding.last_seen_s !== finding.first_seen_s
                ? ` → ${formatClock(finding.last_seen_s)}`
                : ''}
            </button>
          ) : (
            <span>
              {formatClock(finding.first_seen_s)}
              {finding.last_seen_s != null && finding.last_seen_s !== finding.first_seen_s
                ? ` → ${formatClock(finding.last_seen_s)}`
                : ''}
            </span>
          )
        ) : null}
        {finding.out_of_view ? <span>out of view</span> : null}
        {related ? <span className="finding__related">{related}</span> : null}
      </div>
      {expanded ? (
        <dl className="finding__body">
          <div>
            <dt className="t-label">Evidence</dt>
            <dd>{finding.evidence || '—'}</dd>
          </div>
          <div>
            <dt className="t-label">Inference</dt>
            <dd>{finding.inference || '—'}</dd>
          </div>
          <div>
            <dt className="t-label">Impact</dt>
            <dd>{finding.impact || '—'}</dd>
          </div>
          <div className="finding__fix">
            <dt className="t-label">Recommended fix</dt>
            <dd>{finding.recommendation || '—'}</dd>
          </div>
          {finding.quip ? <p className="finding__quip">“{finding.quip}”</p> : null}
          {finding.resolved_note && finding.status === 'RESOLVED' ? (
            <p className="finding__resolved t-data">✓ {finding.resolved_note}</p>
          ) : null}
        </dl>
      ) : null}
      {flash && STAMP[flash] ? (
        <span className={`finding__stamp finding__stamp--${flash}`} aria-hidden="true">
          {STAMP[flash]}
        </span>
      ) : null}
    </article>
  )
}
