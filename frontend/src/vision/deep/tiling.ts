/**
 * Tiled deep pass for large stills (Image Debug, video keyframes).
 *
 * A whole-image pass letterboxes a 12-megapixel photo into 640 px, so small
 * objects lose almost all their pixels. Here the deep detector also runs on
 * overlapping tiles of the original image; boxes are mapped back to the full
 * image and merged with class-aware NMS. Tiles are sized so that at most
 * `maxGrid` per side cover the image (bounded cost, config/vision.json
 * deep.tiling). Measured on held-out COCO mosaics (1920 x 1920): recall at
 * the display threshold 0.26 -> 0.55, precision 0.74 -> 0.63, AP 18.8 -> 36.9.
 *
 * Window layout matches backend/app/vision/tiling.py tile_windows().
 */
import { VISION } from '../../config'
import { iou } from '../geometry'
import type { Detection } from '../types'
import type { DeepResult } from './protocol'

export interface Window {
  x: number
  y: number
  w: number
  h: number
}

export interface TilingOptions {
  /** false: whole-image pass only */
  enabled?: boolean
  minLongSide: number
  maxGrid: number
  overlap: number
  mergeIou: number
  timeBudgetMs: number
}

export interface TiledResult {
  detections: Detection[]
  /** network passes actually run (whole image + tiles) */
  passes: number
  /** tile side in source pixels, null when the image was not tiled */
  tilePx: number | null
  /** tiles planned (excluding the whole-image pass) */
  planned: number
  /** true when the time budget stopped tiling early */
  incomplete: boolean
  totalMs: number
  boxesBeforeMerge: number
}

export const TILING_DEFAULTS: TilingOptions = VISION.deep.tiling

/** Overlapping windows covering the image; the last row/column is aligned to the border. */
export function tileWindows(width: number, height: number, tile: number, overlap: number): Window[] {
  if (width <= tile && height <= tile) return [{ x: 0, y: 0, w: width, h: height }]
  const step = Math.max(1, Math.round(tile * (1 - overlap)))
  const starts = (size: number): number[] => {
    if (size <= tile) return [0]
    const out: number[] = []
    for (let s = 0; s < size - tile; s += step) out.push(s)
    out.push(size - tile)
    return [...new Set(out)].sort((a, b) => a - b)
  }
  const out: Window[] = []
  for (const y of starts(height)) {
    for (const x of starts(width)) out.push({ x, y, w: Math.min(width, x + tile) - x, h: Math.min(height, y + tile) - y })
  }
  return out
}

/** Tile side so that at most `maxGrid` tiles (with overlap) span the long side; null when no tiling is needed. */
export function planTile(width: number, height: number, inputSize: number, opts: Pick<TilingOptions, 'minLongSide' | 'maxGrid' | 'overlap'>): number | null {
  const long = Math.max(width, height)
  if (long <= opts.minLongSide) return null
  return Math.max(inputSize, Math.ceil(long / (opts.maxGrid - (opts.maxGrid - 1) * opts.overlap)))
}

/** Class-aware greedy NMS, highest score first. */
export function mergeDetections(detections: Detection[], iouThreshold: number): Detection[] {
  const sorted = [...detections].sort((a, b) => b.score - a.score)
  const kept: Detection[] = []
  for (const d of sorted) {
    if (kept.some((k) => k.label === d.label && iou(k.box, d.box) > iouThreshold)) continue
    kept.push(d)
  }
  return kept
}

/** Map a detection from tile-normalised to image-normalised coordinates. */
export function fromTile(d: Detection, win: Window, width: number, height: number): Detection {
  return {
    label: d.label,
    score: d.score,
    box: { x: (win.x + d.box.x * win.w) / width, y: (win.y + d.box.y * win.h) / height, w: (d.box.w * win.w) / width, h: (d.box.h * win.h) / height },
  }
}

/**
 * Whole-image pass, then tiles until done or out of time. `detect` consumes the bitmap it is given;
 * `source` stays owned by the caller.
 */
export async function detectTiled(
  source: ImageBitmap,
  detect: (bitmap: ImageBitmap) => Promise<DeepResult>,
  inputSize: number,
  opts: TilingOptions = TILING_DEFAULTS,
  now: () => number = () => performance.now(),
): Promise<TiledResult> {
  const started = now()
  const whole = await detect(await createImageBitmap(source))
  const all: Detection[] = [...whole.detections]
  const tile = opts.enabled === false ? null : planTile(source.width, source.height, inputSize, opts)
  if (tile === null) {
    return { detections: whole.detections, passes: 1, tilePx: null, planned: 0, incomplete: false, totalMs: now() - started, boxesBeforeMerge: all.length }
  }
  const windows = tileWindows(source.width, source.height, tile, opts.overlap)
  let passes = 1
  let incomplete = false
  for (const win of windows) {
    if (now() - started > opts.timeBudgetMs) {
      incomplete = true
      break
    }
    const crop = await createImageBitmap(source, win.x, win.y, win.w, win.h)
    const result = await detect(crop)
    passes++
    for (const d of result.detections) all.push(fromTile(d, win, source.width, source.height))
  }
  return {
    detections: mergeDetections(all, opts.mergeIou),
    passes,
    tilePx: tile,
    planned: windows.length,
    incomplete,
    totalMs: now() - started,
    boxesBeforeMerge: all.length,
  }
}
