import { describe, expect, it } from 'vitest'
import { KalmanFilter, invert4 } from '../kalman'

// ByteTrack's noise weights (config/tracking.json).
const kf = new KalmanFilter({ stdWeightPosition: 1 / 20, stdWeightVelocity: 1 / 160 })

describe('KalmanFilter', () => {
  it('starts at the measurement with zero velocity', () => {
    const { mean } = kf.initiate(Float64Array.of(0.5, 0.4, 1.2, 0.2))
    expect([...mean]).toEqual([0.5, 0.4, 1.2, 0.2, 0, 0, 0, 0])
  })

  it('predicts with constant velocity over dt nominal frames', () => {
    const { mean, cov } = kf.initiate(Float64Array.of(0.5, 0.4, 1, 0.2))
    mean[4] = 0.01
    kf.predict(mean, cov, 3)
    expect(mean[0]).toBeCloseTo(0.53, 10)
    expect(mean[1]).toBeCloseTo(0.4, 10)
  })

  it('estimates the velocity of a steadily moving box', () => {
    const { mean, cov } = kf.initiate(Float64Array.of(0.2, 0.5, 1, 0.2))
    for (let i = 1; i <= 40; i++) {
      kf.predict(mean, cov, 1)
      kf.update(mean, cov, Float64Array.of(0.2 + 0.01 * i, 0.5, 1, 0.2))
    }
    expect(Math.abs(mean[4]! - 0.01)).toBeLessThan(0.0005)
    expect(Math.abs(mean[5]!)).toBeLessThan(1e-9)
    expect(mean[0]).toBeCloseTo(0.6, 2)
  })
})

describe('invert4', () => {
  it('inverts a well-conditioned matrix', () => {
    const m = Float64Array.of(4, 1, 0, 0, 1, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 5)
    const inv = invert4(m)!
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        let sum = 0
        for (let k = 0; k < 4; k++) sum += m[r * 4 + k]! * inv[k * 4 + c]!
        expect(sum).toBeCloseTo(r === c ? 1 : 0, 10)
      }
    }
  })

  it('returns null for a singular matrix', () => {
    expect(invert4(new Float64Array(16))).toBeNull()
  })
})
