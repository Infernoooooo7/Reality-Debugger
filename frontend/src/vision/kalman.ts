/**
 * Constant-velocity Kalman filter over (cx, cy, aspect, height), as used by
 * SORT/DeepSORT/ByteTrack (state: position + velocity, 8 dims). Noise scales
 * with the box height (std_weight_position = 1/20, std_weight_velocity =
 * 1/160 in the ByteTrack reference; config/tracking.json).
 *
 * The reference assumes a fixed frame rate; our detector runs at 6-15 fps, so
 * the transition uses dt in "nominal frames" (elapsed / nominalFrameMs).
 */

export type Vec = Float64Array // length 8 (state) or 4 (measurement)
export type Mat = Float64Array // row-major

const N = 8
const M = 4

export interface KalmanOptions {
  stdWeightPosition: number
  stdWeightVelocity: number
}

function diag(values: number[]): Mat {
  const n = values.length
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++) out[i * n + i] = values[i]!
  return out
}

export class KalmanFilter {
  private readonly wp: number
  private readonly wv: number

  constructor(opts: KalmanOptions) {
    this.wp = opts.stdWeightPosition
    this.wv = opts.stdWeightVelocity
  }

  initiate(z: Vec): { mean: Vec; cov: Mat } {
    const mean = new Float64Array(N)
    mean.set(z, 0)
    const h = z[3]!
    const std = [
      2 * this.wp * h,
      2 * this.wp * h,
      1e-2,
      2 * this.wp * h,
      10 * this.wv * h,
      10 * this.wv * h,
      1e-5,
      10 * this.wv * h,
    ]
    return { mean, cov: diag(std.map((s) => s * s)) }
  }

  /** Propagate by `dt` nominal frames (in place). */
  predict(mean: Vec, cov: Mat, dt: number): void {
    const h = mean[3]!
    // mean = F * mean
    for (let i = 0; i < M; i++) mean[i] = mean[i]! + dt * mean[i + M]!
    // cov = F P F^T, F = [[I, dt I], [0, I]]
    const p = cov
    const out = new Float64Array(N * N)
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        // (F P)_ij
        let fp = p[i * N + j]!
        if (i < M) fp += dt * p[(i + M) * N + j]!
        out[i * N + j] = fp
      }
    }
    const res = new Float64Array(N * N)
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        // ((F P) F^T)_ij = (FP)_ij + dt * (FP)_i,(j+M) for j < M
        let v = out[i * N + j]!
        if (j < M) v += dt * out[i * N + j + M]!
        res[i * N + j] = v
      }
    }
    const q = [this.wp * h, this.wp * h, 1e-2, this.wp * h, this.wv * h, this.wv * h, 1e-5, this.wv * h]
    for (let i = 0; i < N; i++) res[i * N + i] = res[i * N + i]! + q[i]! * q[i]! * Math.max(dt, 1e-3)
    cov.set(res)
  }

  /** Correct with measurement z = (cx, cy, aspect, h) (in place). */
  update(mean: Vec, cov: Mat, z: Vec): void {
    const h = mean[3]!
    const r = [this.wp * h, this.wp * h, 1e-1, this.wp * h]
    // S = H P H^T + R (top-left 4x4 block of P)
    const s = new Float64Array(M * M)
    for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) s[i * M + j] = cov[i * N + j]!
    for (let i = 0; i < M; i++) s[i * M + i] = s[i * M + i]! + r[i]! * r[i]!
    const sInv = invert4(s)
    if (!sInv) return
    // K = P H^T S^-1  (8x4): P H^T = first 4 columns of P
    const k = new Float64Array(N * M)
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < M; j++) {
        let v = 0
        for (let l = 0; l < M; l++) v += cov[i * N + l]! * sInv[l * M + j]!
        k[i * M + j] = v
      }
    }
    const innovation = [z[0]! - mean[0]!, z[1]! - mean[1]!, z[2]! - mean[2]!, z[3]! - mean[3]!]
    for (let i = 0; i < N; i++) {
      let v = 0
      for (let j = 0; j < M; j++) v += k[i * M + j]! * innovation[j]!
      mean[i] = mean[i]! + v
    }
    // P = P - K S K^T = P - K (H P)  (H P = first 4 rows of P)
    const next = new Float64Array(N * N)
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        let v = 0
        for (let l = 0; l < M; l++) v += k[i * M + l]! * cov[l * N + j]!
        next[i * N + j] = cov[i * N + j]! - v
      }
    }
    cov.set(next)
  }
}

/** Inverse of a 4x4 matrix (Gauss-Jordan with partial pivoting); null if singular. */
export function invert4(m: Mat): Mat | null {
  const n = 4
  const a = new Float64Array(n * 2 * n)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) a[i * 2 * n + j] = m[i * n + j]!
    a[i * 2 * n + n + i] = 1
  }
  for (let col = 0; col < n; col++) {
    let pivot = col
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r * 2 * n + col]!) > Math.abs(a[pivot * 2 * n + col]!)) pivot = r
    const pv = a[pivot * 2 * n + col]!
    if (Math.abs(pv) < 1e-12) return null
    if (pivot !== col) {
      for (let j = 0; j < 2 * n; j++) {
        const t = a[col * 2 * n + j]!
        a[col * 2 * n + j] = a[pivot * 2 * n + j]!
        a[pivot * 2 * n + j] = t
      }
    }
    for (let j = 0; j < 2 * n; j++) a[col * 2 * n + j] = a[col * 2 * n + j]! / pv
    for (let r = 0; r < n; r++) {
      if (r === col) continue
      const f = a[r * 2 * n + col]!
      if (f === 0) continue
      for (let j = 0; j < 2 * n; j++) a[r * 2 * n + j] = a[r * 2 * n + j]! - f * a[col * 2 * n + j]!
    }
  }
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i * n + j] = a[i * 2 * n + n + j]!
  return out
}
