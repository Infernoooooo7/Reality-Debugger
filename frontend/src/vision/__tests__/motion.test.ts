import { describe, expect, it } from 'vitest'
import { estimateShift } from '../motion'

const W = 128
const H = 96

/** Same pattern as backend/app/vision/motion.py test_pattern (hashed noise + 3x3 box blur). */
function hash32(x: number, y: number): number {
  let h = (Math.imul(x, 374761393) + Math.imul(y, 668265263)) >>> 0
  h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0
  return (h ^ (h >>> 16)) >>> 0
}

function pattern(shiftX = 0, shiftY = 0): Float32Array {
  const base = new Float64Array(W * H)
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) base[y * W + x] = (hash32(x - shiftX, y - shiftY) & 255) / 255
  const out = new Float32Array(W * H)
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let sum = 0
      let n = 0
      for (let yy = Math.max(0, y - 1); yy < Math.min(H, y + 2); yy++) {
        for (let xx = Math.max(0, x - 1); xx < Math.min(W, x + 2); xx++) {
          sum += base[yy * W + xx]!
          n++
        }
      }
      out[y * W + x] = sum / n
    }
  }
  return out
}

describe('estimateShift (global image motion)', () => {
  const ref = pattern()

  it('generates the same pattern as the Python port', () => {
    // backend: test_pattern()[10, 10:14] and mean
    expect([...ref.slice(10 * W + 10, 10 * W + 14)].map((v) => Math.round(v * 1e6) / 1e6)).toEqual([0.450545, 0.432244, 0.51024, 0.484096])
  })

  it.each([
    [3, -2, 0.02343547491914607, -0.02084305726970272],
    [-7, 4, -0.054689174747940314, 0.04166071550341955],
    [10, 0, 0.07812596782956774, 4.818206770395917e-6],
    [-12, -9, -0.09376030495359981, -0.09376069019985761],
  ])('recovers a (%i, %i) px shift and agrees with the Python port', (sx, sy, pyDx, pyDy) => {
    const shift = estimateShift(ref, pattern(sx, sy), W, H)
    expect(shift.dx * W).toBeCloseTo(sx, 1)
    expect(shift.dy * H).toBeCloseTo(sy, 1)
    expect(Math.abs(shift.dx - pyDx)).toBeLessThan(1e-4)
    expect(Math.abs(shift.dy - pyDy)).toBeLessThan(1e-4)
    expect(shift.confidence).toBeGreaterThan(0.5)
  })

  it('reports no motion between identical frames', () => {
    const shift = estimateShift(ref, ref, W, H)
    expect(Math.abs(shift.dx)).toBeLessThan(1e-4)
    expect(Math.abs(shift.dy)).toBeLessThan(1e-4)
  })

  it('has no confidence on a textureless frame', () => {
    const flat = new Float32Array(W * H).fill(0.5)
    expect(estimateShift(flat, flat, W, H).confidence).toBe(0)
  })
})
