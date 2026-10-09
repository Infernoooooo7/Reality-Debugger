import type { FindingStatus, Personality, Severity, SystemStatus } from '../lib/schemas'

export function StatusChip({ status }: { status: FindingStatus | SystemStatus }) {
  return <span className={`chip chip--${status}`}>{status}</span>
}

export function SeverityTag({ severity }: { severity: Severity }) {
  return <span className={`sev sev--${severity}`}>{severity}</span>
}

/** Where a finding came from: measured by local CV, or added by the optional AI layer. */
export function SourceTag({ source }: { source: 'local' | 'ai' }) {
  return (
    <span
      className="source-tag"
      data-source={source}
      title={source === 'local' ? 'Measured on this device by the local CV engine' : 'Interpretation by the optional AI reasoning layer'}
    >
      {source === 'local' ? 'LOCAL CV' : 'AI'}
    </span>
  )
}

/** The always-on local pipeline. */
export function LocalBadge({ text = 'Local CV active' }: { text?: string }) {
  return (
    <span className="mode-badge" data-mode="local" title="Detection, tracking and diagnostics run locally - no API key needed.">
      {text}
    </span>
  )
}

/** The optional AI layer: off (normal), ready, or unavailable (local CV continues). */
export function AIBadge({ mode, provider }: { mode: 'off' | 'ready' | 'unavailable' | 'unknown'; provider?: string | null }) {
  const text =
    mode === 'ready'
      ? `AI reasoning · ${provider ?? 'on'}`
      : mode === 'unavailable'
        ? 'AI reasoning unavailable'
        : mode === 'off'
          ? 'AI reasoning off'
          : 'AI reasoning ?'
  const title =
    mode === 'off'
      ? 'No AI provider is configured. Everything runs on local computer vision.'
      : mode === 'unavailable'
        ? 'The AI provider failed; local computer vision keeps running.'
        : 'An optional AI layer adds explanations on top of the local results.'
  return (
    <span className="mode-badge" data-mode={`ai-${mode}`} title={title}>
      {text}
    </span>
  )
}

const PERSONALITIES: { id: Personality; label: string; hint: string }[] = [
  { id: 'serious', label: 'Serious', hint: 'Professional and practical' },
  { id: 'brutal', label: 'Brutal', hint: 'Blunt and funny' },
  { id: 'unhinged', label: 'Unhinged', hint: 'Creative, still evidence-based' },
]

export function PersonalitySwitch({
  value,
  onChange,
  compact = false,
}: {
  value: Personality
  onChange: (p: Personality) => void
  compact?: boolean
}) {
  return (
    <div className="switch" role="radiogroup" aria-label="Debug personality">
      {PERSONALITIES.map((p) => (
        <button
          key={p.id}
          type="button"
          role="radio"
          aria-checked={value === p.id}
          className="switch__opt"
          title={p.hint}
          onClick={() => onChange(p.id)}
        >
          {compact ? p.label.slice(0, 3) : p.label}
        </button>
      ))}
    </div>
  )
}
