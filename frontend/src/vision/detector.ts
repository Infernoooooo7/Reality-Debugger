/**
 * Object detector backed by MediaPipe Tasks (EfficientDet-Lite0, 80 COCO
 * classes) running on WebAssembly (XNNPACK) or the WebGL GPU delegate.
 *
 * Delegate choice is verified, not assumed: a procedurally drawn stop sign is
 * run through each candidate delegate and GPU is only used when it actually
 * detects the probe and is faster than CPU (some GPUs/drivers silently return
 * nothing). The rest of the app only depends on the `Detector` interface, so
 * the model or runtime can be swapped here.
 */
import { ObjectDetector } from '@mediapipe/tasks-vision'
import type { Delegate, DelegatePreference, Detection, ProbeResult } from './types'

export const MODEL_NAME = 'EfficientDet-Lite0 int8 · COCO-80 · MediaPipe Tasks'

export interface Detector {
  readonly delegate: Delegate
  /** Detect objects in a frame. `timestampMs` must increase monotonically. */
  detect(source: TexImageSource, timestampMs: number): Detection[]
  close(): void
}

export interface DetectorConfig {
  modelUrl: string
  wasmLoaderUrl: string
  wasmBinaryUrl: string
  delegate: DelegatePreference
  scoreThreshold?: number
  maxResults?: number
}

type AnyCanvas = OffscreenCanvas | HTMLCanvasElement

function sourceSize(source: TexImageSource): [number, number] {
  const s = source as {
    videoWidth?: number
    videoHeight?: number
    naturalWidth?: number
    naturalHeight?: number
    displayWidth?: number
    displayHeight?: number
    width?: number
    height?: number
  }
  if (s.videoWidth) return [s.videoWidth, s.videoHeight ?? 0]
  if (s.naturalWidth) return [s.naturalWidth, s.naturalHeight ?? 0]
  if (s.displayWidth) return [s.displayWidth, s.displayHeight ?? 0]
  return [s.width ?? 0, s.height ?? 0]
}

class MediaPipeDetector implements Detector {
  private lastTs = 0

  private readonly task: ObjectDetector
  readonly delegate: Delegate

  constructor(task: ObjectDetector, delegate: Delegate) {
    this.task = task
    this.delegate = delegate
  }

  detect(source: TexImageSource, timestampMs: number): Detection[] {
    const [w, h] = sourceSize(source)
    if (!w || !h) return []
    // MediaPipe requires strictly increasing timestamps in VIDEO mode.
    const ts = Math.max(Math.round(timestampMs), this.lastTs + 1)
    this.lastTs = ts
    const result = this.task.detectForVideo(source, ts)
    const out: Detection[] = []
    for (const det of result.detections) {
      const category = det.categories[0]
      const box = det.boundingBox
      if (!category || !box) continue
      const x = Math.max(0, box.originX / w)
      const y = Math.max(0, box.originY / h)
      out.push({
        label: (category.categoryName || category.displayName || 'object').toLowerCase(),
        score: category.score,
        box: { x, y, w: Math.min(1 - x, box.width / w), h: Math.min(1 - y, box.height / h) },
      })
    }
    return out
  }

  close(): void {
    try {
      this.task.close()
    } catch {
      /* already closed */
    }
  }
}

function makeCanvas(w: number, h: number): AnyCanvas {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h)
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  return canvas
}

/** A stop sign on a pole: a COCO class every detector variant recognises. */
export function drawProbe(size = 320): AnyCanvas {
  const canvas = makeCanvas(size, size)
  const ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null
  if (!ctx) return canvas
  ctx.fillStyle = '#96aabe'
  ctx.fillRect(0, 0, size, size)
  ctx.fillStyle = '#78787d'
  ctx.fillRect(size * 0.47, size * 0.55, size * 0.06, size * 0.45)
  const octagon = (radius: number, color: string) => {
    ctx.beginPath()
    for (let i = 0; i < 8; i++) {
      const a = ((22.5 + 45 * i) * Math.PI) / 180
      const x = size / 2 + radius * Math.cos(a)
      const y = size * 0.4 + radius * Math.sin(a)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.closePath()
    ctx.fillStyle = color
    ctx.fill()
  }
  octagon(size * 0.3, '#ffffff')
  octagon(size * 0.27, '#c8101e')
  ctx.fillStyle = '#ffffff'
  ctx.font = `bold ${Math.round(size * 0.13)}px sans-serif`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText('STOP', size / 2, size * 0.4)
  return canvas
}

function runProbe(detector: Detector, probe: AnyCanvas): ProbeResult {
  detector.detect(probe, 1) // warm-up (shader compilation, allocations)
  const started = performance.now()
  const detections = detector.detect(probe, 2)
  const ms = performance.now() - started
  const best = detections.filter((d) => d.label === 'stop sign').sort((a, b) => b.score - a.score)[0]
  return {
    delegate: detector.delegate,
    ms: Math.round(ms * 10) / 10,
    label: best ? best.label : null,
    score: best ? Math.round(best.score * 100) / 100 : 0,
  }
}

function withTimeout<T>(promise: Promise<T>, ms: number, what: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${what} timed out after ${ms / 1000}s`)), ms)
    promise.then(
      (value) => {
        clearTimeout(timer)
        resolve(value)
      },
      (error: unknown) => {
        clearTimeout(timer)
        reject(error)
      },
    )
  })
}

/**
 * MediaPipe loads its WASM glue by evaluating a loader script that defines a
 * global `ModuleFactory`, and clears it after use. In a module worker the
 * loader is an ES module that only evaluates once, so before creating a
 * second detector (CPU + GPU probing) the factory is restored from the
 * cached module.
 */
async function ensureModuleFactory(loaderUrl: string): Promise<void> {
  if (typeof document !== 'undefined') return // main thread: classic loader re-runs itself
  const scope = globalThis as { ModuleFactory?: unknown }
  if (scope.ModuleFactory) return
  try {
    const mod = (await import(/* @vite-ignore */ loaderUrl)) as { default?: unknown }
    if (mod.default) scope.ModuleFactory = mod.default
  } catch {
    /* MediaPipe will report a clear error itself */
  }
}

async function create(config: DetectorConfig, delegate: Delegate): Promise<Detector> {
  await ensureModuleFactory(config.wasmLoaderUrl)
  // In a worker the GPU delegate needs an explicit OffscreenCanvas.
  const canvas =
    delegate === 'GPU' && typeof OffscreenCanvas !== 'undefined' && typeof document === 'undefined'
      ? new OffscreenCanvas(1, 1)
      : undefined
  const task = await ObjectDetector.createFromOptions(
    { wasmLoaderPath: config.wasmLoaderUrl, wasmBinaryPath: config.wasmBinaryUrl },
    {
      baseOptions: { modelAssetPath: config.modelUrl, delegate },
      runningMode: 'VIDEO',
      scoreThreshold: config.scoreThreshold ?? 0.3,
      maxResults: config.maxResults ?? 25,
      ...(canvas ? { canvas } : {}),
    },
  )
  return new MediaPipeDetector(task, delegate)
}

interface Candidate {
  detector: Detector
  probe: ProbeResult
}

async function candidate(config: DetectorConfig, delegate: Delegate, probe: AnyCanvas): Promise<Candidate | null> {
  try {
    const detector = await withTimeout(create(config, delegate), delegate === 'GPU' ? 25_000 : 45_000, `${delegate} delegate`)
    return { detector, probe: runProbe(detector, probe) }
  } catch (error) {
    console.warn(`[vision] ${delegate} delegate unavailable:`, error)
    return null
  }
}

export interface DetectorSelection {
  detector: Detector
  selfTest: ProbeResult
  probes: ProbeResult[]
}

/**
 * Create the detector.
 * - 'CPU' / 'GPU': use that delegate (GPU falls back to CPU if it fails the probe).
 * - 'auto': probe both, keep GPU only if it detects the probe and is >20% faster.
 */
export async function createDetector(config: DetectorConfig): Promise<DetectorSelection> {
  const probe = drawProbe()
  const probes: ProbeResult[] = []

  if (config.delegate === 'CPU') {
    const cpu = await candidate(config, 'CPU', probe)
    if (!cpu) throw new Error('The vision model could not be loaded.')
    return { detector: cpu.detector, selfTest: cpu.probe, probes: [cpu.probe] }
  }

  if (config.delegate === 'GPU') {
    const gpu = await candidate(config, 'GPU', probe)
    if (gpu && gpu.probe.label) return { detector: gpu.detector, selfTest: gpu.probe, probes: [gpu.probe] }
    if (gpu) probes.push(gpu.probe)
    gpu?.detector.close()
    const cpu = await candidate(config, 'CPU', probe)
    if (!cpu) throw new Error('The vision model could not be loaded.')
    return { detector: cpu.detector, selfTest: cpu.probe, probes: [...probes, cpu.probe] }
  }

  const cpu = await candidate(config, 'CPU', probe)
  const gpu = await candidate(config, 'GPU', probe)
  if (cpu) probes.push(cpu.probe)
  if (gpu) probes.push(gpu.probe)
  if (!cpu && !gpu) throw new Error('The vision model could not be loaded.')

  if (gpu && (!cpu || ((gpu.probe.label !== null || cpu.probe.label === null) && gpu.probe.ms < cpu.probe.ms * 0.8))) {
    cpu?.detector.close()
    return { detector: gpu.detector, selfTest: gpu.probe, probes }
  }
  gpu?.detector.close()
  return { detector: cpu!.detector, selfTest: cpu!.probe, probes }
}
