/**
 * How much of an image's visible structure lies outside every recognised object.
 *
 * A detection count cannot tell whether a scan looked at "everything": the
 * detectors know 80 COCO categories, and small objects vanish when a large
 * photo is downscaled. This measures the visible structure (intensity edges)
 * that no recognised object's box explains. It is reported as evidence of
 * uninspected detail, never as a recall estimate (measured: it correlates only
 * weakly with missed objects, see docs/SCORING.md).
 *
 * Same definition as backend/app/vision/coverage.py (parity-tested on a shared
 * pattern; float32 arithmetic is emulated with Math.fround):
 * 1. luminance 0.299 R + 0.587 G + 0.114 B on 0..1, long side <= 256 px;
 * 2. central-difference gradients (zero on the border), "structure" where the
 *    gradient magnitude >= 0.05;
 * 3. boxes rasterised with floor/ceil;
 * 4. unexplainedShare = structure outside every box / all structure.
 */
import type { NBox } from './types'

export const STRUCTURE_SIDE = 256
export const EDGE_THRESHOLD = 0.05

export interface StructureStats {
  edgeDensity: number
  /** null when the image has no structure at all */
  unexplainedShare: number | null
  boxCoverage: number
  width: number
  height: number
}

const f = Math.fround

export function structureMask(gray: Float32Array, w: number, h: number, threshold = EDGE_THRESHOLD): Uint8Array {
  const mask = new Uint8Array(w * h)
  const t = f(threshold)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const gx = x > 0 && x < w - 1 ? f(f(gray[y * w + x + 1]! - gray[y * w + x - 1]!) * 0.5) : 0
      const gy = y > 0 && y < h - 1 ? f(f(gray[(y + 1) * w + x]! - gray[(y - 1) * w + x]!) * 0.5) : 0
      if (f(Math.sqrt(f(f(gx * gx) + f(gy * gy)))) >= t) mask[y * w + x] = 1
    }
  }
  return mask
}

export function boxMask(w: number, h: number, boxes: NBox[]): Uint8Array {
  const mask = new Uint8Array(w * h)
  for (const b of boxes) {
    const x1 = Math.max(0, Math.floor(b.x * w))
    const y1 = Math.max(0, Math.floor(b.y * h))
    const x2 = Math.min(w, Math.ceil((b.x + b.w) * w))
    const y2 = Math.min(h, Math.ceil((b.y + b.h) * h))
    for (let y = y1; y < y2; y++) mask.fill(1, y * w + x1, y * w + Math.max(x1, x2))
  }
  return mask
}

export function structureStats(gray: Float32Array, w: number, h: number, boxes: NBox[], threshold = EDGE_THRESHOLD): StructureStats {
  const edges = structureMask(gray, w, h, threshold)
  const covered = boxMask(w, h, boxes)
  let nEdges = 0
  let outside = 0
  let nCovered = 0
  for (let i = 0; i < edges.length; i++) {
    if (covered[i]) nCovered++
    if (edges[i]) {
      nEdges++
      if (!covered[i]) outside++
    }
  }
  const total = w * h
  return {
    edgeDensity: total ? nEdges / total : 0,
    unexplainedShare: nEdges ? outside / nEdges : null,
    boxCoverage: total ? nCovered / total : 0,
    width: w,
    height: h,
  }
}

/** Luminance of an image reduced so that its long side is at most STRUCTURE_SIDE px. */
export function luminanceSmall(source: CanvasImageSource, srcWidth: number, srcHeight: number): { gray: Float32Array; w: number; h: number } {
  const scale = Math.min(1, STRUCTURE_SIDE / Math.max(srcWidth, srcHeight))
  const w = Math.max(1, Math.round(srcWidth * scale))
  const h = Math.max(1, Math.round(srcHeight * scale))
  const canvas = new OffscreenCanvas(w, h)
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) throw new Error('2D canvas unavailable')
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(source, 0, 0, w, h)
  const data = ctx.getImageData(0, 0, w, h).data
  const gray = new Float32Array(w * h)
  for (let i = 0; i < gray.length; i++) gray[i] = (0.299 * data[i * 4]! + 0.587 * data[i * 4 + 1]! + 0.114 * data[i * 4 + 2]!) / 255
  return { gray, w, h }
}

/** Measure on an image (the caller keeps ownership of the source). */
export function measureCoverage(source: CanvasImageSource, srcWidth: number, srcHeight: number, boxes: NBox[]): StructureStats {
  const { gray, w, h } = luminanceSmall(source, srcWidth, srcHeight)
  return structureStats(gray, w, h, boxes)
}

/** The pattern shared with backend/app/vision/coverage.py test_pattern(). */
export function testPattern(w = 64, h = 48): Float32Array {
  const img = new Float32Array(w * h).fill(0.2)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      if (y >= 8 && y < 20 && x >= 6 && x < 22) img[i] = 0.9
      if (x + y >= 40 && x + y < 52) img[i] = 0.2 + 0.6 * ((x + y - 40) / 12)
      const checker = (Math.floor(x / 2) + Math.floor(y / 2)) % 2 === 0
      if (checker && x >= 44 && x < 60 && y >= 28 && y < 44) img[i] = 0.7
    }
  }
  return img
}
