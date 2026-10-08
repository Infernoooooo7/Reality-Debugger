/**
 * The local vision engine: detector + tracker + pixel signals.
 * Environment-agnostic so it can run inside the Web Worker (default) or on
 * the main thread as a fallback.
 */
import { createDetector, MODEL_NAME, type Detector } from './detector'
import { SignalAnalyzer } from './signals'
import { Tracker } from './tracker'
import type { Detection, EngineInfo, FrameResult, InitPayload, StillResult } from './types'

const LETTERBOX = 384

/**
 * EfficientDet takes a square input and MediaPipe stretches whatever it is
 * given. Camera frames are 4:3 / 16:9, and stretching measurably hurts
 * recall (e.g. laptops), so frames are letterboxed onto a square canvas and
 * boxes are mapped back to frame coordinates.
 */
class Letterbox {
  private canvas: OffscreenCanvas | HTMLCanvasElement
  private ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D
  // content rect inside the square, normalised
  private rx = 0
  private ry = 0
  private rw = 1
  private rh = 1

  constructor(size = LETTERBOX) {
    if (typeof OffscreenCanvas !== 'undefined') this.canvas = new OffscreenCanvas(size, size)
    else {
      const c = document.createElement('canvas')
      c.width = size
      c.height = size
      this.canvas = c
    }
    const ctx = this.canvas.getContext('2d') as OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null
    if (!ctx) throw new Error('2D canvas unavailable for letterboxing')
    this.ctx = ctx
  }

  draw(source: ImageBitmap): OffscreenCanvas | HTMLCanvasElement {
    const size = this.canvas.width
    const scale = Math.min(size / source.width, size / source.height)
    const w = source.width * scale
    const h = source.height * scale
    const x = (size - w) / 2
    const y = (size - h) / 2
    this.ctx.fillStyle = '#727272'
    this.ctx.fillRect(0, 0, size, size)
    this.ctx.drawImage(source, x, y, w, h)
    this.rx = x / size
    this.ry = y / size
    this.rw = w / size
    this.rh = h / size
    return this.canvas
  }

  unmap(detections: Detection[]): Detection[] {
    const out: Detection[] = []
    for (const d of detections) {
      const x1 = Math.max(0, (d.box.x - this.rx) / this.rw)
      const y1 = Math.max(0, (d.box.y - this.ry) / this.rh)
      const x2 = Math.min(1, (d.box.x + d.box.w - this.rx) / this.rw)
      const y2 = Math.min(1, (d.box.y + d.box.h - this.ry) / this.rh)
      if (x2 - x1 < 0.01 || y2 - y1 < 0.01) continue
      out.push({ ...d, box: { x: x1, y: y1, w: x2 - x1, h: y2 - y1 } })
    }
    return out
  }
}

export class VisionEngine {
  private detector: Detector | null = null
  private readonly tracker = new Tracker()
  private signals: SignalAnalyzer | null = null
  private letterbox: Letterbox | null = null
  private clock = 0

  async init(payload: InitPayload): Promise<Omit<EngineInfo, 'runtime'>> {
    const started = performance.now()
    this.signals = new SignalAnalyzer()
    this.letterbox = new Letterbox()
    const selection = await createDetector(payload)
    this.detector = selection.detector
    return {
      delegate: selection.detector.delegate,
      model: MODEL_NAME,
      loadMs: Math.round(performance.now() - started),
      selfTest: selection.selfTest,
      probes: selection.probes,
    }
  }

  private tick(timestamp: number): number {
    this.clock = Math.max(this.clock + 1, Math.round(timestamp))
    return this.clock
  }

  processFrame(frameId: number, bitmap: ImageBitmap, timestamp: number): FrameResult {
    if (!this.detector || !this.signals || !this.letterbox) throw new Error('Vision engine not initialised')
    const started = performance.now()
    try {
      const square = this.letterbox.draw(bitmap)
      const detections = this.letterbox.unmap(this.detector.detect(square, this.tick(timestamp)))
      const inferenceMs = performance.now() - started
      const tracks = this.tracker.update(detections, timestamp)
      const { signals } = this.signals.analyze(bitmap)
      return {
        frameId,
        timestamp,
        tracks,
        detections: detections.length,
        signals,
        inferenceMs: Math.round(inferenceMs * 10) / 10,
        totalMs: Math.round((performance.now() - started) * 10) / 10,
      }
    } finally {
      bitmap.close()
    }
  }

  /** One-off analysis of a still image (no tracking, no temporal signals). */
  analyzeStill(bitmap: ImageBitmap): StillResult {
    if (!this.detector || !this.signals || !this.letterbox) throw new Error('Vision engine not initialised')
    const started = performance.now()
    try {
      const square = this.letterbox.draw(bitmap)
      const detections = this.letterbox.unmap(this.detector.detect(square, this.tick(performance.now())))
      const inferenceMs = performance.now() - started
      const { signals, signature } = this.signals.analyze(bitmap, { temporal: false })
      return { detections, signals, signature, inferenceMs: Math.round(inferenceMs * 10) / 10 }
    } finally {
      bitmap.close()
    }
  }

  setAnchor(): void {
    this.signals?.setAnchor()
  }

  reset(): void {
    this.tracker.reset()
    this.signals?.reset()
  }
}
