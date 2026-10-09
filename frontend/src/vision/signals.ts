/**
 * Cheap per-frame pixel signals computed on a 128x96 thumbnail:
 * motion (frame differencing), scene signature (colour histogram +
 * luminance grid), brightness, contrast and sharpness (Laplacian variance).
 *
 * All buffers are allocated once and reused.
 */
import { VISION } from '../config'
import { estimateShift } from './motion'
import type { FrameSignals, NBox, Signature } from './types'

// config/vision.json "signals" (shared with the backend's video fallback).
const S = VISION.signals
const W = S.thumbWidth
const H = S.thumbHeight
const GRID_W = S.gridWidth
const GRID_H = S.gridHeight
const BINS = S.histogramBins
const SHIFT = 8 - Math.log2(BINS)

type AnyCanvas = OffscreenCanvas | HTMLCanvasElement
type AnyContext = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D

function makeCanvas(w: number, h: number): AnyCanvas {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h)
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  return canvas
}

export function signatureDistance(a: Signature, b: Signature): number {
  let hist = 0
  for (let i = 0; i < a.hist.length; i++) hist += Math.abs(a.hist[i]! - b.hist[i]!)
  hist *= 0.5
  let grid = 0
  for (let i = 0; i < a.grid.length; i++) grid += Math.abs(a.grid[i]! - b.grid[i]!)
  grid /= a.grid.length
  return S.histogramWeight * hist + (1 - S.histogramWeight) * Math.min(1, grid * S.gridGain)
}

export class SignalAnalyzer {
  private readonly canvas: AnyCanvas
  private readonly ctx: AnyContext
  private gray = new Float32Array(W * H)
  private prevGray = new Float32Array(W * H)
  private hasPrev = false
  private prevSignature: Signature | null = null
  private anchor: Signature | null = null
  private last: Signature | null = null

  constructor() {
    this.canvas = makeCanvas(W, H)
    const ctx = this.canvas.getContext('2d', { willReadFrequently: true }) as AnyContext | null
    if (!ctx) throw new Error('2D canvas unavailable for signal analysis')
    this.ctx = ctx
  }

  /** Remember the most recent frame as the reference for scene change (start of a view). */
  setAnchor(): void {
    if (this.last) this.anchor = { hist: this.last.hist.slice(), grid: this.last.grid.slice() }
  }

  get hasAnchor(): boolean {
    return this.anchor !== null
  }

  reset(): void {
    this.hasPrev = false
    this.prevSignature = null
    this.anchor = null
    this.last = null
  }

  analyze(source: CanvasImageSource, { temporal = true } = {}): { signals: FrameSignals; signature: Signature } {
    this.ctx.drawImage(source, 0, 0, W, H)
    const data = this.ctx.getImageData(0, 0, W, H).data

    const hist = new Float32Array(BINS * BINS * BINS)
    const grid = new Float32Array(GRID_W * GRID_H)
    const gray = this.gray
    let sum = 0
    let sumSq = 0

    for (let i = 0, p = 0; i < W * H; i++, p += 4) {
      const r = data[p]!
      const g = data[p + 1]!
      const b = data[p + 2]!
      const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
      gray[i] = lum
      sum += lum
      sumSq += lum * lum
      hist[((r >> SHIFT) * BINS + (g >> SHIFT)) * BINS + (b >> SHIFT)]! += 1
      const gx = Math.floor((i % W) / (W / GRID_W))
      const gy = Math.floor(Math.floor(i / W) / (H / GRID_H))
      grid[gy * GRID_W + gx]! += lum
    }
    const n = W * H
    for (let i = 0; i < hist.length; i++) hist[i]! /= n
    const cell = n / (GRID_W * GRID_H)
    for (let i = 0; i < grid.length; i++) grid[i]! /= cell

    const brightness = sum / n
    const contrast = Math.sqrt(Math.max(0, sumSq / n - brightness * brightness))

    // Laplacian variance (sharpness).
    let lapSum = 0
    let lapSq = 0
    let lapN = 0
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const i = y * W + x
        const lap = gray[i - 1]! + gray[i + 1]! + gray[i - W]! + gray[i + W]! - 4 * gray[i]!
        lapSum += lap
        lapSq += lap * lap
        lapN++
      }
    }
    const lapMean = lapSum / lapN
    const lapVar = Math.max(0, lapSq / lapN - lapMean * lapMean)
    // Variance of the Laplacian on a log scale: ~1e-4 (very blurred) -> 0, ~0.04 (crisp) -> 1.
    const sharpness = Math.min(1, Math.max(0, (Math.log10(lapVar + 1e-6) + S.sharpnessLogOffset) / S.sharpnessLogSpan))

    // Motion: frame difference + bounding box of changed pixels.
    let motion = 0
    let motionBox: NBox | null = null
    let cameraShift: FrameSignals['cameraShift'] = null
    if (temporal && this.hasPrev) {
      const shift = estimateShift(this.prevGray, gray, W, H)
      cameraShift = { dx: round4(shift.dx), dy: round4(shift.dy), confidence: shift.confidence }
      let diffSum = 0
      let minX = W
      let minY = H
      let maxX = -1
      let maxY = -1
      let changed = 0
      for (let i = 0; i < n; i++) {
        const d = Math.abs(gray[i]! - this.prevGray[i]!)
        diffSum += d
        if (d > S.motionPixelDelta) {
          changed++
          const x = i % W
          const y = (i / W) | 0
          if (x < minX) minX = x
          if (x > maxX) maxX = x
          if (y < minY) minY = y
          if (y > maxY) maxY = y
        }
      }
      motion = Math.min(1, (diffSum / n) * S.motionGain)
      if (changed > n * S.motionMinFraction && maxX >= minX) {
        motionBox = { x: minX / W, y: minY / H, w: (maxX - minX + 1) / W, h: (maxY - minY + 1) / H }
      }
    }

    const signature: Signature = { hist, grid }
    const frameDelta = temporal && this.prevSignature ? signatureDistance(signature, this.prevSignature) : 0
    const sceneDelta = temporal && this.anchor ? signatureDistance(signature, this.anchor) : 0

    if (temporal) {
      const swap = this.prevGray
      this.prevGray = this.gray
      this.gray = swap
      this.hasPrev = true
      this.prevSignature = signature
      this.last = signature
    }

    return {
      signals: {
        motion: round3(motion),
        motionBox,
        frameDelta: round3(frameDelta),
        sceneDelta: round3(sceneDelta),
        brightness: round3(brightness),
        contrast: round3(Math.min(1, contrast * 2)),
        sharpness: round3(sharpness),
        cameraShift,
      },
      signature,
    }
  }
}

function round3(v: number): number {
  return Math.round(v * 1000) / 1000
}

function round4(v: number): number {
  return Math.round(v * 10000) / 10000
}
