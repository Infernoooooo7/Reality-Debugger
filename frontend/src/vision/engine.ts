/**
 * The local vision engine: fast detector + ByteTrack tracker + pixel signals
 * + view tracking. Environment-agnostic so it can run inside the Web Worker
 * (default) or on the main thread as a fallback.
 */
import { DETECTION, TEMPORAL, VISION } from '../config'
import { createDetector, type Detector } from './detector'
import { loadManifest } from './models'
import { SignalAnalyzer } from './signals'
import { Tracker } from './tracker'
import type { Detection, EngineInfo, FrameResult, FuseResult, InitPayload, StillResult } from './types'

/**
 * EfficientDet takes a square input and MediaPipe stretches whatever it is
 * given. Camera frames are 4:3 / 16:9, so frames are letterboxed onto a
 * square canvas of the model's input size and boxes are mapped back
 * (on coco128: AP 37.0 letterboxed vs 36.3 stretched, docs/benchmarks).
 */
class Letterbox {
  private canvas: OffscreenCanvas | HTMLCanvasElement
  private ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D
  private rx = 0
  private ry = 0
  private rw = 1
  private rh = 1

  constructor(size: number) {
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
  private viewId = 0

  async init(payload: InitPayload): Promise<Omit<EngineInfo, 'runtime'>> {
    const started = performance.now()
    const manifest = await loadManifest(payload.manifestUrl)
    this.signals = new SignalAnalyzer()
    const size = VISION.fast.letterbox ? (manifest.input?.width ?? 320) : 0
    this.letterbox = size ? new Letterbox(size) : null
    const selection = await createDetector({
      ...payload,
      scoreThreshold: DETECTION.fast.lowScoreFloor,
      maxResults: DETECTION.fast.maxResults,
    })
    this.detector = selection.detector
    return {
      delegate: selection.detector.delegate,
      model: manifest.name,
      modelId: manifest.id,
      classes: manifest.num_classes,
      loadMs: Math.round(performance.now() - started),
      selfTest: selection.selfTest,
      probes: selection.probes,
    }
  }

  private tick(timestamp: number): number {
    this.clock = Math.max(this.clock + 1, Math.round(timestamp))
    return this.clock
  }

  private detect(bitmap: ImageBitmap): Detection[] {
    if (!this.detector) throw new Error('Vision engine not initialised')
    if (!this.letterbox) return this.detector.detect(bitmap, this.tick(performance.now()))
    return this.letterbox.unmap(this.detector.detect(this.letterbox.draw(bitmap), this.tick(performance.now())))
  }

  processFrame(frameId: number, bitmap: ImageBitmap, timestamp: number): FrameResult {
    if (!this.signals) throw new Error('Vision engine not initialised')
    const started = performance.now()
    try {
      const detections = this.detect(bitmap)
      const inferenceMs = performance.now() - started
      const tracks = this.tracker.update(detections, timestamp)
      const { signals, signature } = this.signals.analyze(bitmap)
      this.updateView(signals.sceneDelta, signals.motion)
      return {
        frameId,
        timestamp,
        tracks,
        detections: detections.length,
        signals,
        signature,
        viewId: this.viewId,
        events: this.tracker.drainEvents(),
        inferenceMs: Math.round(inferenceMs * 10) / 10,
        totalMs: Math.round((performance.now() - started) * 10) / 10,
      }
    } finally {
      bitmap.close()
    }
  }

  /**
   * A view starts on the first settled frame; a new view starts when a
   * settled frame differs from the view's reference frame by more than
   * temporal.sceneChange.threshold (cuts and pans alike). Requiring a
   * settled frame keeps motion blur from creating spurious views.
   */
  private updateView(sceneDelta: number, motion: number): void {
    if (!this.signals) return
    const settled = motion < TEMPORAL.sceneChange.settleMotion
    if (!this.signals.hasAnchor) {
      if (settled) this.signals.setAnchor()
      return
    }
    if (settled && sceneDelta > TEMPORAL.sceneChange.threshold) {
      this.viewId += 1
      this.signals.setAnchor()
    }
  }

  /** One-off analysis of a still image (no tracking, no temporal signals). */
  analyzeStill(bitmap: ImageBitmap): StillResult {
    if (!this.signals) throw new Error('Vision engine not initialised')
    const started = performance.now()
    try {
      const detections = this.detect(bitmap)
      const inferenceMs = performance.now() - started
      const { signals, signature } = this.signals.analyze(bitmap, { temporal: false })
      return { detections, signals, signature, inferenceMs: Math.round(inferenceMs * 10) / 10 }
    } finally {
      bitmap.close()
    }
  }

  /** Merge a deep-detector pass into the live tracks. */
  fuse(detections: Detection[], timestamp: number, holdMs: number): FuseResult {
    const outcome = this.tracker.verify(detections, timestamp, {
      matchIou: DETECTION.fusion.matchIou,
      deepOnlyMinScore: DETECTION.fusion.deepOnlyMinScore,
      relabelMargin: DETECTION.fusion.relabelMargin,
      holdMs,
    })
    return { ...outcome, tracks: this.tracker.current(timestamp) }
  }

  reset(): void {
    this.tracker.reset()
    this.signals?.reset()
    this.viewId = 0
  }
}
