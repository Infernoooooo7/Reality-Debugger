import { describe, expect, it } from 'vitest'
import { structureMask, structureStats, testPattern } from '../coverage'

// Reference values computed by backend/app/vision/coverage.py on the same pattern.
const PY = {
  edges: 934,
  edgeDensity: 0.3040364583333333,
  oneBox: { unexplained: 0.892933618843683, coverage: 0.0830078125 },
  twoBoxes: { unexplained: 0.5867237687366167, coverage: 0.1884765625 },
}
const BOX_A = { x: 0.1, y: 0.15, w: 0.25, h: 0.3 }
const BOX_B = { x: 0.68, y: 0.58, w: 0.26, h: 0.35 }

describe('structure coverage (parity with the Python implementation)', () => {
  const pattern = testPattern()

  it('marks the same structure pixels', () => {
    const mask = structureMask(pattern, 64, 48)
    expect(mask.reduce((n, v) => n + v, 0)).toBe(PY.edges)
  })

  it('matches the Python statistics', () => {
    const none = structureStats(pattern, 64, 48, [])
    expect(none.edgeDensity).toBeCloseTo(PY.edgeDensity, 12)
    expect(none.unexplainedShare).toBe(1)
    expect(none.boxCoverage).toBe(0)
    const one = structureStats(pattern, 64, 48, [BOX_A])
    expect(one.unexplainedShare).toBeCloseTo(PY.oneBox.unexplained, 12)
    expect(one.boxCoverage).toBeCloseTo(PY.oneBox.coverage, 12)
    const two = structureStats(pattern, 64, 48, [BOX_A, BOX_B])
    expect(two.unexplainedShare).toBeCloseTo(PY.twoBoxes.unexplained, 12)
    expect(two.boxCoverage).toBeCloseTo(PY.twoBoxes.coverage, 12)
  })

  it('reports no share for a featureless image', () => {
    const flat = new Float32Array(64 * 48).fill(0.5)
    const stats = structureStats(flat, 64, 48, [])
    expect(stats.edgeDensity).toBe(0)
    expect(stats.unexplainedShare).toBeNull()
  })
})
