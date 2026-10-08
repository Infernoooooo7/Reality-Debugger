import type { FindingStatus, Personality, Severity, SystemStatus } from '../lib/schemas'

export function StatusChip({ status }: { status: FindingStatus | SystemStatus }) {
  return <span className={`chip chip--${status}`}>{status}</span>
}

export function SeverityTag({ severity }: { severity: Severity }) {
  return <span className={`sev sev--${severity}`}>{severity}</span>
}

export function DemoBadge({ text = 'Demo mode · simulated' }: { text?: string }) {
  return (
    <span className="demo-badge" title="No vision model was used. Results are simulated from on-device detections.">
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
