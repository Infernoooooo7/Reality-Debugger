import type { NBox } from './types'

export function iou(a: NBox, b: NBox): number {
  const x1 = Math.max(a.x, b.x)
  const y1 = Math.max(a.y, b.y)
  const x2 = Math.min(a.x + a.w, b.x + b.w)
  const y2 = Math.min(a.y + a.h, b.y + b.h)
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const union = a.w * a.h + b.w * b.h - inter
  return union > 0 ? inter / union : 0
}

/** Euclidean gap between two boxes (0 when touching/overlapping). */
export function gap(a: NBox, b: NBox): number {
  const dx = Math.max(0, Math.max(a.x, b.x) - Math.min(a.x + a.w, b.x + b.w))
  const dy = Math.max(0, Math.max(a.y, b.y) - Math.min(a.y + a.h, b.y + b.h))
  return Math.hypot(dx, dy)
}

export function center(b: NBox): [number, number] {
  return [b.x + b.w / 2, b.y + b.h / 2]
}

export function union(boxes: NBox[]): NBox | null {
  if (!boxes.length) return null
  let x1 = Infinity
  let y1 = Infinity
  let x2 = -Infinity
  let y2 = -Infinity
  for (const b of boxes) {
    x1 = Math.min(x1, b.x)
    y1 = Math.min(y1, b.y)
    x2 = Math.max(x2, b.x + b.w)
    y2 = Math.max(y2, b.y + b.h)
  }
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 }
}

export function clampBox(b: NBox): NBox {
  const x = Math.min(1, Math.max(0, b.x))
  const y = Math.min(1, Math.max(0, b.y))
  return { x, y, w: Math.max(0, Math.min(1 - x, b.w)), h: Math.max(0, Math.min(1 - y, b.h)) }
}

export function lerpBox(a: NBox, b: NBox, t: number): NBox {
  return {
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
    w: a.w + (b.w - a.w) * t,
    h: a.h + (b.h - a.h) * t,
  }
}
