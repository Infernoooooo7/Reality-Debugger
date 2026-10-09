/**
 * Vision lab: measures the real on-device pipeline in this browser.
 *
 *   npm run dev  ->  /vision-lab.html?images=a.jpg,b.jpg&runs=5&deep=1&tracker=1
 *
 * Images are read from public/test-media/ (git-ignored). Results are printed
 * and exposed as `window.__lab`; scripts/vision-lab.mjs runs this page in
 * headless Chromium and saves them (docs/benchmarks/browser_bench.json was
 * assembled from such runs). Nothing here is used by the app itself.
 */
import { DETECTION } from '../config'
import { vision } from '../vision/client'
import { deepDetector } from '../vision/deep/client'
import { fuseStill } from '../vision/fusion'
import { Tracker } from '../vision/tracker'
import type { Detection } from '../vision/types'

interface LabState {
  done: boolean
  error: string | null
  results: Record<string, unknown>
}

const out = document.getElementById('out')!
const state: LabState = { done: false, error: null, results: {} }
;(window as unknown as { __lab: LabState }).__lab = state

const params = new URLSearchParams(window.location.search)
const images = (params.get('images') ?? '').split(',').filter(Boolean)
const runs = Math.max(1, Number(params.get('runs') ?? 5))

function log(line: string): void {
  out.textContent += `${line}\n`
}

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b)
  return s.length ? s[Math.floor((s.length - 1) / 2)]! : 0
}

function percentile(values: number[], p: number): number {
  const s = [...values].sort((a, b) => a - b)
  return s.length ? s[Math.min(s.length - 1, Math.round(p * (s.length - 1)))]! : 0
}

async function loadBitmap(name: string): Promise<ImageBitmap> {
  const res = await fetch(`/test-media/${name}`)
  if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`)
  return createImageBitmap(await res.blob())
}

const short = (dets: Detection[]) => dets.map((d) => `${d.label}:${d.score.toFixed(2)}`).slice(0, 12)

async function benchFast(): Promise<void> {
  const t0 = performance.now()
  const info = await vision.init()
  state.results.fast = { info, initMs: Math.round(performance.now() - t0), images: {} as Record<string, unknown> }
  log(`fast detector: ${info.model} · ${info.delegate} · ${info.runtime} · init ${Math.round(performance.now() - t0)} ms`)
  for (const name of images) {
    const times: number[] = []
    let detections: Detection[] = []
    for (let i = 0; i < runs + 1; i++) {
      const still = await vision.analyzeStill(await loadBitmap(name))
      if (i > 0) times.push(still.inferenceMs) // first run is warm-up
      detections = still.detections
    }
    ;(state.results.fast as { images: Record<string, unknown> }).images[name] = {
      medianMs: median(times),
      p95Ms: percentile(times, 0.95),
      detections: detections.length,
      aboveThreshold: detections.filter((d) => d.score >= DETECTION.fast.scoreThreshold).length,
      top: short(detections),
    }
    log(`  ${name}: ${median(times)} ms · ${short(detections).join(' ')}`)
  }
}

async function benchDeep(): Promise<void> {
  const t0 = performance.now()
  const info = await deepDetector.init()
  state.results.deep = { info, initMs: Math.round(performance.now() - t0), images: {} as Record<string, unknown> }
  log(`deep detector: ${info.model} · ${info.backend} · ${info.threads} thread(s) · isolated=${info.crossOriginIsolated} · warm-up ${info.warmupMs} ms`)
  for (const name of images) {
    const totals: number[] = []
    const inference: number[] = []
    let detections: Detection[] = []
    for (let i = 0; i < runs; i++) {
      const r = await deepDetector.detect(await loadBitmap(name))
      totals.push(r.totalMs)
      inference.push(r.inferenceMs)
      detections = r.detections
    }
    const fast = await vision.analyzeStill(await loadBitmap(name))
    const fused = fuseStill(fast.detections, detections)
    ;(state.results.deep as { images: Record<string, unknown> }).images[name] = {
      medianTotalMs: median(totals),
      medianInferenceMs: median(inference),
      p95TotalMs: percentile(totals, 0.95),
      detections: detections.length,
      top: short(detections),
      fused: { objects: fused.length, verified: fused.filter((o) => o.verified).length, deepOnly: fused.filter((o) => o.source === 'deep').length },
    }
    log(`  ${name}: total ${median(totals)} ms (inference ${median(inference)}) · ${short(detections).join(' ')}`)
  }
}

/**
 * Tracker speed estimates on synthetic sequences built from a real photo and
 * the real fast detector: identical frames, a static camera with ±jitter
 * (hand tremor), and the photo panned at known speeds. Used to calibrate
 * temporal.movement stillSpeed / movingSpeed.
 */
async function benchTracker(): Promise<void> {
  const name = images[0]
  if (!name) return
  await vision.init()
  const source = await loadBitmap(name)
  const W = 480
  const H = Math.round((source.height / source.width) * W)
  const canvas = new OffscreenCanvas(W, H)
  const ctx = canvas.getContext('2d')!
  const frameMs = 100
  const frames = 40

  const scenario = async (label: string, offset: (i: number) => [number, number]) => {
    const tracker = new Tracker()
    const speeds: number[] = []
    const movements: Record<string, number> = { static: 0, moving: 0, unknown: 0 }
    let ids = new Set<number>()
    for (let i = 0; i < frames; i++) {
      const [dx, dy] = offset(i)
      ctx.fillStyle = '#727272'
      ctx.fillRect(0, 0, W, H)
      // Draw the photo slightly zoomed so shifts never reveal the border.
      ctx.drawImage(source, -0.05 * W + dx * W, -0.05 * H + dy * H, 1.1 * W, 1.1 * H)
      const bitmap = await createImageBitmap(canvas)
      const still = await vision.analyzeStill(bitmap)
      const tracks = tracker.update(still.detections, i * frameMs)
      if (i >= 8) {
        for (const t of tracks) {
          if (t.state !== 'confirmed' || t.hits < 4) continue
          speeds.push(t.speed)
          movements[t.movement] = (movements[t.movement] ?? 0) + 1
        }
      }
      ids = new Set([...ids, ...tracks.filter((t) => t.state !== 'tentative').map((t) => t.id)])
    }
    const result = {
      samples: speeds.length,
      medianSpeed: +median(speeds).toFixed(4),
      p95Speed: +percentile(speeds, 0.95).toFixed(4),
      maxSpeed: +Math.max(0, ...speeds).toFixed(4),
      movements,
      trackIds: ids.size,
    }
    ;(state.results.tracker as Record<string, unknown>)[label] = result
    log(`  ${label}: median ${result.medianSpeed} · p95 ${result.p95Speed} · max ${result.maxSpeed} fw/s · ${JSON.stringify(movements)} · ids ${ids.size}`)
  }

  state.results.tracker = { image: name, frameMs, frames }
  log(`tracker (ByteTrack + Kalman) on ${name}, ${frames} frames at ${1000 / frameMs} fps:`)
  let seed = 7
  const rand = () => {
    seed = (seed * 16807) % 2147483647
    return seed / 2147483647 - 0.5
  }
  await scenario('identical', () => [0, 0])
  await scenario('jitter_0.5pct', () => [rand() * 0.01, rand() * 0.01])
  await scenario('jitter_1pct', () => [rand() * 0.02, rand() * 0.02])
  await scenario('pan_0.05fw_s', (i) => [(0.05 * i * frameMs) / 1000, 0])
  await scenario('pan_0.10fw_s', (i) => [(0.1 * i * frameMs) / 1000, 0])
  await scenario('pan_0.20fw_s', (i) => [(0.2 * i * frameMs) / 1000, 0])
}

async function run(): Promise<void> {
  out.textContent = ''
  log(`crossOriginIsolated=${String(window.crossOriginIsolated)} · cores=${navigator.hardwareConcurrency} · ${navigator.userAgent}`)
  state.results.env = {
    crossOriginIsolated: window.crossOriginIsolated,
    hardwareConcurrency: navigator.hardwareConcurrency,
    userAgent: navigator.userAgent,
    webgpu: 'gpu' in navigator,
  }
  await benchFast()
  if (params.get('deep') !== '0') await benchDeep()
  if (params.get('tracker') !== '0') await benchTracker()
  log('DONE')
}

run()
  .catch((error: unknown) => {
    state.error = error instanceof Error ? `${error.message}\n${error.stack}` : String(error)
    log(`ERROR ${state.error}`)
  })
  .finally(() => {
    state.done = true
  })
