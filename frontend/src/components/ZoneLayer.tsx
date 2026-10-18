import { useRef, useState, type PointerEvent } from 'react'
import type { ZonePayload } from '../vision/scene'
import type { NBox } from '../vision/types'

/**
 * Keep-clear areas drawn over an image. While `editing`, dragging draws a new
 * area (normalised coordinates); otherwise the layer is see-through and the
 * findings underneath stay clickable.
 */
export function ZoneLayer({
  zones,
  editing,
  onAdd,
  onRemove,
}: {
  zones: ZonePayload[]
  editing: boolean
  onAdd: (box: NBox) => void
  onRemove: (id: string) => void
}) {
  const ref = useRef<HTMLDivElement | null>(null)
  const start = useRef<{ x: number; y: number } | null>(null)
  const [draft, setDraft] = useState<NBox | null>(null)

  const point = (e: PointerEvent) => {
    const r = ref.current!.getBoundingClientRect()
    return { x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) }
  }
  const boxFrom = (a: { x: number; y: number }, b: { x: number; y: number }): NBox => ({
    x: Math.min(a.x, b.x),
    y: Math.min(a.y, b.y),
    w: Math.abs(a.x - b.x),
    h: Math.abs(a.y - b.y),
  })

  return (
    <div
      ref={ref}
      className="zones"
      data-editing={editing || undefined}
      onPointerDown={(e) => {
        if (!editing) return
        e.currentTarget.setPointerCapture(e.pointerId)
        start.current = point(e)
        setDraft(null)
      }}
      onPointerMove={(e) => {
        if (editing && start.current) setDraft(boxFrom(start.current, point(e)))
      }}
      onPointerUp={(e) => {
        if (!editing || !start.current) return
        const box = boxFrom(start.current, point(e))
        start.current = null
        setDraft(null)
        if (box.w >= 0.02 && box.h >= 0.02) onAdd(box)
      }}
      aria-label={editing ? 'Drag to mark an area that should stay clear' : undefined}
    >
      {[...zones.map((z) => ({ key: z.id, box: z.box, name: z.name })), ...(draft ? [{ key: 'draft', box: draft, name: '' }] : [])].map((z) => (
        <span
          key={z.key}
          className="zones__area"
          data-draft={z.key === 'draft' || undefined}
          style={{ left: `${z.box.x * 100}%`, top: `${z.box.y * 100}%`, width: `${z.box.w * 100}%`, height: `${z.box.h * 100}%` }}
        >
          {z.name ? <em>{z.name}</em> : null}
          {editing && z.key !== 'draft' ? (
            <button
              type="button"
              className="zones__remove"
              onPointerDown={(e) => e.stopPropagation()}
              onClick={() => onRemove(z.key)}
              aria-label={`Remove ${z.name}`}
            >
              ×
            </button>
          ) : null}
        </span>
      ))}
    </div>
  )
}
