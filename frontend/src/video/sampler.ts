/**
 * Browser-side video pipeline. The file never leaves the device: frames are
 * decoded by the <video> element (streamed from disk, never loaded into
 * memory as a whole), sampled at intervals, fingerprinted, run through the
 * local detector, segmented into scenes and de-duplicated. Only a handful of
 * representative keyframes are uploaded.
 */
import { ApiError } from '../lib/api'
import { FrameGrabber, JpegCapturer } from '../live/capture'
import { vision } from '../vision/client'
import { SignalAnalyzer, signatureDistance } from '../vision/signals'
import type { Detection, Signature } from '../vision/types'

export interface VideoInfo {
  duration: number
  width: number
  height: number
}

export interface Sample {
  t: number
  signature: Signature
  sharpness: number
  detections: Detection[]
}

export interface Keyframe {
  sample: number
  t: number
  scene: number
  reason: string
}

export interface Selection {
  scenes: { index: number; start_t: number; end_t: number }[]
  keyframes: Keyframe[]
  redundant: number
  events: { t: number; kind: 'SCENE_CHANGE' | 'OBJECT_ENTERED' | 'OBJECT_LEFT'; text: string }[]
}

export interface Progress {
  done: number
  total: number
  t: number
  scenes: number
  objects: number
}

const SCENE_THRESHOLD = 0.3
const DEDUPE_THRESHOLD = 0.06

function once(target: EventTarget, events: string[], timeoutMs: number): Promise<string> {
  return new Promise((resolve) => {
    const handlers: [string, () => void][] = []
    const finish = (name: string) => {
      handlers.forEach(([n, h]) => target.removeEventListener(n, h))
      window.clearTimeout(timer)
      resolve(name)
    }
    for (const name of events) {
      const handler = () => finish(name)
      handlers.push([name, handler])
      target.addEventListener(name, handler)
    }
    const timer = window.setTimeout(() => finish('timeout'), timeoutMs)
  })
}

/** Attach a file to a <video> element and read its metadata. */
export async function loadVideo(video: HTMLVideoElement, url: string): Promise<VideoInfo> {
  video.muted = true
  video.playsInline = true
  video.preload = 'auto'
  video.src = url
  const result = await once(video, ['loadedmetadata', 'error'], 12_000)
  if (result !== 'loadedmetadata' || !video.videoWidth) {
    throw new ApiError('VIDEO_UNREADABLE', 'This browser cannot decode the video.', {
      hint: 'It may be corrupted or use a codec this browser does not support (e.g. HEVC on some desktops).',
    })
  }
  let duration = video.duration
  if (!Number.isFinite(duration) || duration <= 0) {
    // Some recordings (e.g. MediaRecorder WebM) report Infinity until the end is reached.
    video.currentTime = 1e7
    await once(video, ['durationchange', 'seeked'], 4000)
    duration = video.duration
    video.currentTime = 0
    await once(video, ['seeked'], 3000)
  }
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new ApiError('VIDEO_UNREADABLE', 'The video duration could not be determined.', {
      hint: 'Try re-exporting the video as MP4.',
    })
  }
  if (video.readyState < 2) await once(video, ['loadeddata', 'error'], 8000)
  return { duration, width: video.videoWidth, height: video.videoHeight }
}

async function seek(video: HTMLVideoElement, t: number): Promise<void> {
  if (Math.abs(video.currentTime - t) < 0.001 && video.readyState >= 2) return
  video.currentTime = t
  const result = await once(video, ['seeked', 'error'], 5000)
  if (result === 'error') throw new ApiError('VIDEO_UNREADABLE', 'Decoding failed while seeking.', { hint: 'The file may be corrupted.' })
  // Make sure the decoded frame is actually presented before we read it.
  const v = video as HTMLVideoElement & { requestVideoFrameCallback?: (cb: () => void) => number }
  if (typeof v.requestVideoFrameCallback === 'function') {
    await Promise.race([
      new Promise<void>((resolve) => v.requestVideoFrameCallback!(() => resolve())),
      new Promise<void>((resolve) => window.setTimeout(resolve, 120)),
    ])
  }
}

/** Sample the video at regular intervals; detection runs on every sample. */
export async function sampleVideo(
  video: HTMLVideoElement,
  info: VideoInfo,
  opts: { maxSamples?: number; signal?: AbortSignal; onProgress?: (p: Progress) => void } = {},
): Promise<Sample[]> {
  const total = Math.max(8, Math.min(opts.maxSamples ?? 72, Math.round(info.duration / 0.5)))
  const step = info.duration / total
  const grabber = new FrameGrabber()
  let analyzer: SignalAnalyzer | null = null
  let useEngine = true
  try {
    await vision.init()
  } catch {
    useEngine = false
    analyzer = new SignalAnalyzer()
  }

  const samples: Sample[] = []
  const labelsSeen = new Set<string>()
  let scenes = 1
  for (let i = 0; i < total; i++) {
    if (opts.signal?.aborted) throw new DOMException('Aborted', 'AbortError')
    const t = Math.min(info.duration - 0.05, (i + 0.5) * step)
    await seek(video, t)
    const bitmap = await grabber.bitmap(video, 480)
    if (!bitmap) continue
    let sample: Sample
    if (useEngine) {
      const still = await vision.analyzeStill(bitmap)
      sample = { t, signature: still.signature, sharpness: still.signals.sharpness, detections: still.detections }
    } else {
      const { signals, signature } = analyzer!.analyze(bitmap, { temporal: false })
      bitmap.close()
      sample = { t, signature, sharpness: signals.sharpness, detections: [] }
    }
    const prev = samples[samples.length - 1]
    if (prev && signatureDistance(sample.signature, prev.signature) > SCENE_THRESHOLD) scenes++
    sample.detections.filter((d) => d.score >= 0.45).forEach((d) => labelsSeen.add(d.label))
    samples.push(sample)
    opts.onProgress?.({ done: i + 1, total, t, scenes, objects: labelsSeen.size })
  }
  if (!samples.length) {
    throw new ApiError('VIDEO_UNREADABLE', 'No frames could be decoded from this video.', { hint: 'The file may be corrupted.' })
  }
  return samples
}

function labelSet(sample: Sample): Set<string> {
  return new Set(sample.detections.filter((d) => d.score >= 0.45).map((d) => d.label))
}

/** Scene segmentation, redundancy removal and representative keyframe selection. */
export function selectKeyframes(samples: Sample[], maxKeyframes: number): Selection {
  const sceneOf: number[] = [0]
  const scenes: number[][] = [[0]]
  const events: Selection['events'] = []
  for (let i = 1; i < samples.length; i++) {
    const stepDist = signatureDistance(samples[i]!.signature, samples[i - 1]!.signature)
    const drift = signatureDistance(samples[i]!.signature, samples[scenes[scenes.length - 1]![0]!]!.signature)
    if (stepDist > SCENE_THRESHOLD || drift > SCENE_THRESHOLD * 1.6) {
      scenes.push([i])
      events.push({ t: samples[i]!.t, kind: 'SCENE_CHANGE', text: `Scene ${scenes.length} begins (local frame comparison)` })
    } else {
      scenes[scenes.length - 1]!.push(i)
    }
    sceneOf[i] = scenes.length - 1
  }

  // Redundant frames: near-identical to the last kept frame of the scene.
  let redundant = 0
  for (const scene of scenes) {
    let kept = scene[0]!
    for (const i of scene.slice(1)) {
      if (signatureDistance(samples[i]!.signature, samples[kept]!.signature) < DEDUPE_THRESHOLD) redundant++
      else kept = i
    }
  }

  // Object enter/leave events (debounced over two samples).
  const changePoints: { index: number; text: string }[] = []
  const present = new Map<string, number>()
  for (let i = 0; i < samples.length; i++) {
    const now = labelSet(samples[i]!)
    const next = samples[i + 1] ? labelSet(samples[i + 1]!) : now
    for (const label of now) {
      if (!present.has(label) && (next.has(label) || i === samples.length - 1)) {
        present.set(label, i)
        if (i > 0) {
          events.push({ t: samples[i]!.t, kind: 'OBJECT_ENTERED', text: `${label} enters the frame` })
          changePoints.push({ index: i, text: `${label} entered` })
        }
      }
    }
    for (const label of [...present.keys()]) {
      if (!now.has(label) && !next.has(label)) {
        present.delete(label)
        events.push({ t: samples[i]!.t, kind: 'OBJECT_LEFT', text: `${label} leaves the frame` })
        changePoints.push({ index: i, text: `${label} left` })
      }
    }
  }

  const representative = (scene: number[]): number => {
    const lo = Math.floor(scene.length * 0.2)
    const hi = Math.max(lo + 1, Math.floor(scene.length * 0.8))
    const core = scene.slice(lo, hi)
    return (core.length ? core : scene).reduce((best, i) => (samples[i]!.sharpness > samples[best]!.sharpness ? i : best))
  }

  let picks: Keyframe[] = scenes.map((scene, si) => ({ sample: representative(scene), t: 0, scene: si, reason: 'scene representative' }))
  if (picks.length > maxKeyframes) {
    const byLength = scenes.map((s, si) => [s.length, si] as const).sort((a, b) => b[0] - a[0])
    const keep = new Set(byLength.slice(0, maxKeyframes - 1).map(([, si]) => si))
    keep.add(0)
    picks = picks.filter((p) => keep.has(p.scene)).slice(0, maxKeyframes)
  } else {
    const chosen = new Set(picks.map((p) => p.sample))
    const farEnough = (i: number) => [...chosen].every((c) => Math.abs(samples[c]!.t - samples[i]!.t) > 0.75)
    // Object changes are where stories happen: spend budget there first.
    for (const cp of changePoints) {
      if (picks.length >= maxKeyframes) break
      if (!chosen.has(cp.index) && farEnough(cp.index)) {
        picks.push({ sample: cp.index, t: 0, scene: sceneOf[cp.index]!, reason: cp.text })
        chosen.add(cp.index)
      }
    }
    // Then the frames most unlike their scene's representative.
    const extra: { d: number; i: number; si: number }[] = []
    scenes.forEach((scene, si) => {
      const rep = representative(scene)
      for (const i of scene) if (i !== rep) extra.push({ d: signatureDistance(samples[i]!.signature, samples[rep]!.signature), i, si })
    })
    extra.sort((a, b) => b.d - a.d)
    for (const { d, i, si } of extra) {
      if (picks.length >= maxKeyframes || d < DEDUPE_THRESHOLD * 2) break
      if (!chosen.has(i) && farEnough(i)) {
        picks.push({ sample: i, t: 0, scene: si, reason: 'state change within scene' })
        chosen.add(i)
      }
    }
  }
  picks.sort((a, b) => samples[a.sample]!.t - samples[b.sample]!.t)
  for (const p of picks) p.t = samples[p.sample]!.t

  return {
    scenes: scenes.map((scene, index) => ({ index, start_t: samples[scene[0]!]!.t, end_t: samples[scene[scene.length - 1]!]!.t })),
    keyframes: picks,
    redundant,
    events: events.sort((a, b) => a.t - b.t),
  }
}

/** Seek to each keyframe and encode it as a JPEG for upload. */
export async function captureKeyframes(video: HTMLVideoElement, keyframes: Keyframe[], maxEdge = 1024): Promise<Blob[]> {
  const capturer = new JpegCapturer()
  const blobs: Blob[] = []
  for (const k of keyframes) {
    await seek(video, k.t)
    blobs.push((await capturer.capture(video, maxEdge, 0.85)).blob)
  }
  return blobs
}
