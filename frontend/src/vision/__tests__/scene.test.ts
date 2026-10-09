import { describe, expect, it } from 'vitest'
import type { FusedObject } from '../fusion'
import { detectorRun } from '../runs'
import { stillScene } from '../scene'

const facts = { inputSize: 640, vocabulary: 80 }

function objects(n: number): FusedObject[] {
  return Array.from({ length: n }, (_, i) => ({
    id: `o${i + 1}`,
    label: 'cup',
    confidence: (i % 10) / 10 + 0.05,
    box: { x: 0.01 * (i % 50), y: 0.5, w: 0.01, h: 0.02 },
    source: 'deep' as const,
    verified: false,
  }))
}

describe('detector run reports', () => {
  it('records what a detector did, with input size and vocabulary from its manifest', () => {
    const ok = detectorRun('yolox_s', 'deep', 'ok', facts, { boxes: 23, ms: 4768.4, passes: 10, tile_px: 1477 })
    expect(ok).toEqual({
      model: 'yolox_s',
      role: 'deep',
      status: 'ok',
      boxes: 23,
      ms: 4768,
      input_size: 640,
      passes: 10,
      tile_px: 1477,
      incomplete: false,
      vocabulary: 80,
      note: null,
    })
    const failed = detectorRun('yolox_s', 'deep', 'failed', facts, { note: 'x'.repeat(300) })
    expect(failed.passes).toBe(0) // a failed detector made no pass
    expect(failed.note).toHaveLength(160) // backend limit
  })
})

describe('still scene payload', () => {
  it('sends the most confident objects up to the backend limit and reports the total', () => {
    const scene = stillScene(objects(120), null, { width: 3840, height: 2880, detectors: ['yolox_s'] })
    expect(scene.objects).toHaveLength(80)
    expect(scene.stats.objects_total).toBe(120)
    expect(scene.objects[0]!.confidence).toBeGreaterThanOrEqual(scene.objects[79]!.confidence)
    expect(Math.min(...scene.objects.map((o) => o.confidence))).toBeGreaterThanOrEqual(0.25)
  })

  it('is built even when no detector produced a result, with signals unmeasured', () => {
    const runs = [detectorRun('efficientdet_lite0', 'fast', 'failed', facts, { note: 'model failed to load' })]
    const scene = stillScene([], null, { width: 640, height: 480, detectors: [], runs, coverage: null })
    expect(scene.objects).toEqual([])
    expect(scene.signals.brightness).toBeNull()
    expect(scene.stats.runs).toEqual(runs)
    expect(scene.stats.coverage).toBeNull()
  })
})
