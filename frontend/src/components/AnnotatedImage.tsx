import type { CSSProperties } from 'react'
import type { Finding } from '../lib/schemas'

/** An image with the findings' boxes drawn on top (percentage-positioned). */
export function AnnotatedImage({
  src,
  findings,
  focusedId,
  onFocus,
  alt,
}: {
  src: string
  findings: Finding[]
  focusedId: string | null
  onFocus?: (id: string | null) => void
  alt: string
}) {
  return (
    <figure className="annotated">
      <img src={src} alt={alt} draggable={false} />
      {findings
        .filter((f) => f.box)
        .map((f) => {
          const box = f.box!
          const style = {
            left: `${box.x * 100}%`,
            top: `${box.y * 100}%`,
            width: `${box.w * 100}%`,
            height: `${box.h * 100}%`,
          } as CSSProperties
          return (
            <button
              key={f.id}
              type="button"
              className="annotated__box"
              data-severity={f.severity}
              data-status={f.status}
              data-focused={focusedId === f.id || undefined}
              data-dim={focusedId !== null && focusedId !== f.id ? true : undefined}
              style={style}
              onMouseEnter={() => onFocus?.(f.id)}
              onMouseLeave={() => onFocus?.(null)}
              onClick={() => onFocus?.(focusedId === f.id ? null : f.id)}
              aria-label={`${f.id}: ${f.title}`}
            >
              <span className="annotated__tag">
                {f.id} · {f.severity}
              </span>
            </button>
          )
        })}
    </figure>
  )
}
