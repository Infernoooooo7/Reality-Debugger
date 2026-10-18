import { describe, expect, it } from 'vitest'
import { testPattern } from '../coverage'
import { zoneStructure } from '../zones'

describe('keep-clear area structure', () => {
  const gray = testPattern(64, 48)
  it('measures the detail inside an area and how much of it recognised objects explain', () => {
    const checker = { x: 44 / 64, y: 28 / 48, w: 16 / 64, h: 16 / 48 } // the checkerboard patch
    const busy = zoneStructure(gray, 64, 48, checker, [])
    expect(busy.edge_density).toBeGreaterThan(0.5)
    expect(busy.unexplained_share).toBe(1)
    expect(zoneStructure(gray, 64, 48, checker, [checker]).unexplained_share).toBe(0)
  })

  it('reports a flat area as having no detail', () => {
    const flat = zoneStructure(gray, 64, 48, { x: 0, y: 0, w: 0.08, h: 0.1 }, [])
    expect(flat.edge_density).toBe(0)
    expect(flat.unexplained_share).toBeNull()
  })
})
