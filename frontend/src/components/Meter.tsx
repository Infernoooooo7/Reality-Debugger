import type { CSSProperties } from 'react'

export function Meter({
  value,
  segments = 20,
  color,
  height,
  label,
}: {
  value: number | null
  segments?: number
  color?: string
  height?: number
  label?: string
}) {
  const on = value == null ? 0 : Math.round(Math.max(0, Math.min(1, value)) * segments)
  const style = {
    '--segments': segments,
    ...(color ? { '--meter-color': color } : {}),
    ...(height ? { '--meter-h': `${height}px` } : {}),
  } as CSSProperties
  return (
    <div
      className="meter"
      style={style}
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={value == null ? undefined : Math.round(value * 100)}
    >
      {Array.from({ length: segments }, (_, i) => (
        <span key={i} className="meter__seg" data-on={i < on} />
      ))}
    </div>
  )
}

export function scoreColor(score: number | null | undefined): string {
  if (score == null) return 'var(--chalk-3)'
  if (score >= 80) return 'var(--phosphor)'
  if (score >= 50) return 'var(--amber)'
  return 'var(--critical)'
}
