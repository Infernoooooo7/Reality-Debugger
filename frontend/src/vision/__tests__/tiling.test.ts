import { afterEach, describe, expect, it, vi } from 'vitest'
import { detectTiled, fromTile, mergeDetections, planTile, tileWindows } from '../deep/tiling'
import type { DeepResult } from '../deep/protocol'
import type { Detection } from '../types'

const opts = { minLongSide: 960, maxGrid: 3, overlap: 0.2, mergeIou: 0.5, timeBudgetMs: 20000 }

describe('tile layout (same windows as backend/app/vision/tiling.py)', () => {
  it('matches the Python windows', () => {
    const w = tileWindows(1920, 1920, 739, 0.2)
    expect(w).toHaveLength(9)
    expect(w[1]).toEqual({ x: 591, y: 0, w: 739, h: 739 })
    expect(w[8]).toEqual({ x: 1181, y: 1181, w: 739, h: 739 })
    const phone = tileWindows(4032, 3024, 1551, 0.2)
    expect(phone).toHaveLength(9)
    expect(phone[8]).toEqual({ x: 2481, y: 1473, w: 1551, h: 1551 })
    expect(tileWindows(500, 400, 640, 0.2)).toEqual([{ x: 0, y: 0, w: 500, h: 400 }])
  })

  it('sizes tiles so at most maxGrid span the long side', () => {
    expect(planTile(640, 480, 640, opts)).toBeNull() // fits the detector input: no tiling
    expect(planTile(1920, 1920, 640, opts)).toBe(739)
    expect(planTile(4032, 3024, 640, opts)).toBe(1551)
    expect(planTile(1280, 960, 640, opts)).toBe(640) // never smaller than the network input
  })
})

describe('merging', () => {
  it('maps tile boxes back to image coordinates', () => {
    const d = fromTile({ label: 'cup', score: 0.8, box: { x: 0.5, y: 0.5, w: 0.1, h: 0.2 } }, { x: 1000, y: 500, w: 1000, h: 1000 }, 4000, 3000)
    expect(d.box.x).toBeCloseTo(0.375)
    expect(d.box.y).toBeCloseTo(1000 / 3000)
    expect(d.box.w).toBeCloseTo(0.025)
    expect(d.box.h).toBeCloseTo(200 / 3000)
  })

  it('removes duplicates of one class but keeps overlapping objects of different classes', () => {
    const box = { x: 0.1, y: 0.1, w: 0.2, h: 0.2 }
    const merged = mergeDetections(
      [
        { label: 'cup', score: 0.6, box },
        { label: 'cup', score: 0.9, box: { ...box, x: 0.11 } },
        { label: 'bottle', score: 0.5, box },
      ],
      0.5,
    )
    expect(merged.map((d) => `${d.label}:${d.score}`)).toEqual(['cup:0.9', 'bottle:0.5'])
  })
})

describe('detectTiled', () => {
  afterEach(() => vi.unstubAllGlobals())

  function fakeBitmap(width: number, height: number): ImageBitmap {
    return { width, height, close: () => undefined } as unknown as ImageBitmap
  }

  it('runs the whole image plus every tile and reports the passes', async () => {
    vi.stubGlobal('createImageBitmap', async (_src: unknown, _x?: number, _y?: number, w?: number, h?: number) => fakeBitmap(w ?? 1920, h ?? 1920))
    const calls: [number, number][] = []
    const detect = async (b: ImageBitmap): Promise<DeepResult> => {
      calls.push([b.width, b.height])
      const det: Detection = { label: 'cup', score: 0.7, box: { x: 0.4, y: 0.4, w: 0.1, h: 0.1 } }
      return { detections: [det], preprocessMs: 0, inferenceMs: 0, postprocessMs: 0, totalMs: 1 }
    }
    const result = await detectTiled(fakeBitmap(1920, 1920), detect, 640, opts)
    expect(result.passes).toBe(10)
    expect(result.planned).toBe(9)
    expect(result.tilePx).toBe(739)
    expect(result.incomplete).toBe(false)
    expect(calls[0]).toEqual([1920, 1920])
    expect(result.boxesBeforeMerge).toBe(10)
  })

  it('stops at the time budget and says so', async () => {
    vi.stubGlobal('createImageBitmap', async (_src: unknown, _x?: number, _y?: number, w?: number, h?: number) => fakeBitmap(w ?? 1920, h ?? 1920))
    let t = 0
    const detect = async (): Promise<DeepResult> => {
      t += 1000
      return { detections: [], preprocessMs: 0, inferenceMs: 0, postprocessMs: 0, totalMs: 1000 }
    }
    const result = await detectTiled(fakeBitmap(1920, 1920), detect, 640, { ...opts, timeBudgetMs: 2500 }, () => t)
    expect(result.incomplete).toBe(true)
    expect(result.passes).toBeLessThan(10)
  })
})
