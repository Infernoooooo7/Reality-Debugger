import { describe, expect, it } from 'vitest'
import { DETECTION } from '../../config'
import { fuseStill } from '../fusion'
import type { Detection, NBox } from '../types'

const A: NBox = { x: 0.1, y: 0.1, w: 0.2, h: 0.2 }
const A2: NBox = { x: 0.11, y: 0.1, w: 0.2, h: 0.21 } // same object, slightly different box
const B: NBox = { x: 0.6, y: 0.6, w: 0.2, h: 0.2 }
const det = (label: string, score: number, box: NBox): Detection => ({ label, score, box })

describe('fuseStill (fast + deep detector)', () => {
  it('uses the operating points from config/detection.json', () => {
    expect(DETECTION.fusion).toMatchObject({ matchIou: 0.3, deepOnlyMinScore: 0.4, relabelMargin: 0.15 })
    expect(DETECTION.fast.scoreThreshold).toBe(0.3)
  })

  it('verifies an object both detectors agree on', () => {
    const [o, ...rest] = fuseStill([det('cup', 0.5, A)], [det('cup', 0.9, A2)])
    expect(rest).toHaveLength(0)
    expect(o).toMatchObject({ label: 'cup', confidence: 0.9, source: 'fused', verified: true, box: A2 })
  })

  it('takes the deep label only when it is clearly more confident', () => {
    // cup (kitchen) and vase (indoor) are different COCO supercategories
    expect(fuseStill([det('cup', 0.4, A)], [det('vase', 0.8, A2)])[0]).toMatchObject({ label: 'vase', verified: true })
    expect(fuseStill([det('cup', 0.6, A)], [det('vase', 0.7, A2)])[0]).toMatchObject({ label: 'cup', source: 'fast', verified: false })
  })

  it('keeps confident deep-only objects and drops weak ones', () => {
    const fused = fuseStill([], [det('laptop', 0.6, A), det('mouse', 0.35, B)])
    expect(fused.map((o) => [o.label, o.source])).toEqual([['laptop', 'deep']])
  })

  it('keeps fast-only objects at the fast operating point', () => {
    const fused = fuseStill([det('book', 0.5, A), det('book', 0.25, B)], null)
    expect(fused.map((o) => [o.label, o.confidence, o.source])).toEqual([['book', 0.5, 'fast']])
  })
})
