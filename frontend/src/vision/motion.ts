/**
 * Global (camera-induced) image motion between two grey thumbnails.
 *
 * Estimates one translation for the whole frame by coarse-to-fine block
 * matching (sum of absolute differences) with a parabolic sub-pixel fit:
 * cheap enough to run on every frame in the vision worker. The tracker uses
 * it to compensate camera motion before associating detections (the idea of
 * BoT-SORT's camera-motion compensation, which fits an affine model with
 * sparse optical flow; a translation is the part of hand-held phone motion
 * that matters most between consecutive frames at 6-15 fps).
 *
 * Not modelled: rotation, zoom and parallax. When the scene has too little
 * texture or a large object moves, the match is weak and `confidence` is
 * low; callers must not compensate then.
 *
 * The backend has a line-by-line port (backend/app/vision/motion.py) for
 * offline evaluation; a shared test pattern keeps both in agreement.
 */

export interface GlobalShift {
  /** Shift of the image content from the previous to the current frame, in frame widths (+ = right). */
  dx: number
  /** In frame heights (+ = down). */
  dy: number
  /** 0..1: how distinct the best match is. */
  confidence: number
}

export interface MotionOptions {
  /** Largest shift searched, as a fraction of the thumbnail width. */
  maxShift: number
  /** Border (fraction of each side) excluded from matching. */
  border: number
}

export const MOTION_DEFAULTS: MotionOptions = { maxShift: 0.12, border: 0.08 }

function downsample2(src: Float32Array, w: number, h: number): Float32Array {
  const w2 = w >> 1
  const h2 = h >> 1
  const out = new Float32Array(w2 * h2)
  for (let y = 0; y < h2; y++) {
    for (let x = 0; x < w2; x++) {
      const i = 2 * y * w + 2 * x
      out[y * w2 + x] = 0.25 * (src[i]! + src[i + 1]! + src[i + w]! + src[i + w + 1]!)
    }
  }
  return out
}

/** Mean absolute difference between prev shifted by (sx, sy) and cur over the inner region. */
function sad(prev: Float32Array, cur: Float32Array, w: number, h: number, sx: number, sy: number, margin: number): number {
  let sum = 0
  let n = 0
  for (let y = margin; y < h - margin; y++) {
    const py = y - sy
    if (py < 0 || py >= h) continue
    for (let x = margin; x < w - margin; x++) {
      const px = x - sx
      if (px < 0 || px >= w) continue
      sum += Math.abs(cur[y * w + x]! - prev[py * w + px]!)
      n++
    }
  }
  return n ? sum / n : Infinity
}

function search(prev: Float32Array, cur: Float32Array, w: number, h: number, cx: number, cy: number, radius: number, margin: number) {
  let best = Infinity
  let bx = cx
  let by = cy
  const costs = new Map<string, number>()
  for (let sy = cy - radius; sy <= cy + radius; sy++) {
    for (let sx = cx - radius; sx <= cx + radius; sx++) {
      const c = sad(prev, cur, w, h, sx, sy, margin)
      costs.set(`${sx},${sy}`, c)
      if (c < best) {
        best = c
        bx = sx
        by = sy
      }
    }
  }
  return { bx, by, best, costs }
}

function parabola(cm: number, c0: number, cp: number): number {
  const denom = cm - 2 * c0 + cp
  return denom > 1e-9 ? Math.max(-0.5, Math.min(0.5, (0.5 * (cm - cp)) / denom)) : 0
}

/** Estimate the global shift between two grey images of size w x h (values 0..1). */
export function estimateShift(prev: Float32Array, cur: Float32Array, w: number, h: number, opts: MotionOptions = MOTION_DEFAULTS): GlobalShift {
  const margin = Math.max(1, Math.round(opts.border * w))
  // Coarse level: half resolution, full search range.
  const w2 = w >> 1
  const h2 = h >> 1
  const p2 = downsample2(prev, w, h)
  const c2 = downsample2(cur, w, h)
  const r2 = Math.max(1, Math.round((opts.maxShift * w) / 2))
  const coarse = search(p2, c2, w2, h2, 0, 0, r2, Math.max(1, margin >> 1))
  // Fine level: refine around the doubled coarse estimate.
  const fine = search(prev, cur, w, h, coarse.bx * 2, coarse.by * 2, 1, margin)
  const get = (x: number, y: number) => fine.costs.get(`${x},${y}`) ?? sad(prev, cur, w, h, x, y, margin)
  const fx = fine.bx + parabola(get(fine.bx - 1, fine.by), fine.best, get(fine.bx + 1, fine.by))
  const fy = fine.by + parabola(get(fine.bx, fine.by - 1), fine.best, get(fine.bx, fine.by + 1))
  // Confidence: how much better the best match is than the typical candidate of the coarse search.
  const values = [...coarse.costs.values()].filter(Number.isFinite).sort((a, b) => a - b)
  const median = values[Math.floor(values.length / 2)] ?? 0
  const confidence = median > 1e-6 ? Math.max(0, Math.min(1, 1 - coarse.best / median)) : 0
  return { dx: fx / w, dy: fy / h, confidence: Math.round(confidence * 1000) / 1000 }
}
