export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes)) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
}

/** mm:ss (or h:mm:ss) */
export function formatClock(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return '--:--'
  const s = Math.max(0, Math.floor(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  const mm = String(m).padStart(2, '0')
  const ss = String(sec).padStart(2, '0')
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`
}

export function formatPercent(value: number | null | undefined, digits = 0): string {
  if (value == null || !Number.isFinite(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

export function pad2(n: number): string {
  return String(Math.max(0, Math.round(n))).padStart(2, '0')
}

export function formatMs(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms)) return '—'
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(1)} s`
}

export function timeOfDay(date = new Date()): string {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })
}

export function titleCase(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1).toLowerCase()
}

export const CATEGORY_LABEL: Record<string, string> = {
  EFFICIENCY: 'Efficiency',
  ORGANIZATION: 'Organization',
  CONSISTENCY: 'Visual consistency',
  WORKFLOW: 'Workflow',
  SAFETY: 'Safety',
  ERGONOMICS: 'Ergonomics',
  AESTHETICS: 'Aesthetics',
  SPATIAL: 'Spatial',
  TECH_DEBT: 'Technical debt',
  ABSURD: 'Absurd',
}

export function providerLabel(provider: string, model: string, simulated: boolean): string {
  if (simulated) return 'DEMO MODE · simulated heuristics'
  return `${provider} · ${model}`
}
