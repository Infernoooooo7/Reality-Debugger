import type { NBox } from './types'

export function area(b: NBox): number {
  return b.w * b.h
}

export function intersection(a: NBox, b: NBox): number {
  const w = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)
  const h = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y)
  return w > 0 && h > 0 ? w * h : 0
}

export function iou(a: NBox, b: NBox): number {
  const inter = intersection(a, b)
  const union = a.w * a.h + b.w * b.h - inter
  return union > 0 ? inter / union : 0
}

/** Share of `inner` that lies inside `outer`. */
export function containment(inner: NBox, outer: NBox): number {
  const a = area(inner)
  return a > 0 ? intersection(inner, outer) / a : 0
}

/** Euclidean gap between two boxes (0 when touching/overlapping). */
export function gap(a: NBox, b: NBox): number {
  const dx = Math.max(0, Math.max(a.x, b.x) - Math.min(a.x + a.w, b.x + b.w))
  const dy = Math.max(0, Math.max(a.y, b.y) - Math.min(a.y + a.h, b.y + b.h))
  return Math.hypot(dx, dy)
}

/** Gap relative to the larger box (sqrt of its area): a scale-invariant "nearness". */
export function relativeGap(a: NBox, b: NBox): number {
  return gap(a, b) / Math.max(1e-6, Math.sqrt(Math.max(area(a), area(b))))
}

export function center(b: NBox): [number, number] {
  return [b.x + b.w / 2, b.y + b.h / 2]
}

/** Whether the bottom-centre of `inner` (where an object stands) is inside `outer`. */
export function baseInside(inner: NBox, outer: NBox, tolerance = 0.02): boolean {
  const bx = inner.x + inner.w / 2
  const by = inner.y + inner.h
  return bx >= outer.x && bx <= outer.x + outer.w && by >= outer.y && by <= outer.y + outer.h + tolerance
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

export function roundBox(b: NBox, digits = 4): NBox {
  const f = 10 ** digits
  const r = (n: number) => Math.round(n * f) / f
  return { x: r(b.x), y: r(b.y), w: r(b.w), h: r(b.h) }
}
