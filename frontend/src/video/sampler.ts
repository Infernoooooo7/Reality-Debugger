/**
 * Browser-side video pipeline. The file never leaves the device: frames are
 * decoded by the <video> element (streamed from disk, never loaded into
 * memory as a whole) and sampled at intervals (config/temporal.json "video").
 * Every sample runs through the same fast detector + ByteTrack tracker as
 * Live Scan, with video time as the clock, so objects keep their identity
 * across samples. Scene cuts are found from appearance fingerprints,
 * redundant frames are dropped and representative keyframes are selected;
 * the deep detector verifies the keyframes. The backend receives the
 * per-sample scene models - keyframe images only when AI reasoning is on.
 */
import { DETECTION, TEMPORAL, VISION } from '../config'
import { ApiError } from '../lib/api'
import { FrameGrabber, JpegCapturer } from '../live/capture'
import { vision } from '../vision/client'
import { measureCoverage } from '../vision/coverage'
import { detectorRun, modelFacts } from '../vision/runs'
import { coveragePayload, trackObject, visibleTracks, type CoveragePayload, type DetectorRunPayload, type SceneObjectPayload } from '../vision/scene'
import { SignalAnalyzer, signatureDistance } from '../vision/signals'
import type { Signature, TrackEvent } from '../vision/types'

export interface VideoInfo {
  duration: number
  width: number
  height: number
}

export interface Sample {
  t: number
  signature: Signature
  sharpness: number
  brightness: number
  motion: number
  objects: SceneObjectPayload[]
  events: TrackEvent[]
  /** What each detector did on this frame (the deep run is added when a keyframe is verified). */
  runs: DetectorRunPayload[]
  /** Visible detail the tracked objects explain (vision/coverage.ts); null when not measured. */
  coverage: CoveragePayload | null
}

export interface Keyframe {
  sample: number
  t: number
  scene: number
  reason: string
}

export interface Selection {
  scenes: { index: number; start_t: number; end_t: number }[]
  sceneOf: number[]
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

const V = TEMPORAL.video

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

/** Sample the video at regular intervals; detection and tracking run on every sample. */
export async function sampleVideo(
  video: HTMLVideoElement,
  info: VideoInfo,
  opts: { signal?: AbortSignal; onProgress?: (p: Progress) => void } = {},
): Promise<Sample[]> {
  const total = Math.max(V.minSamples, Math.min(V.maxSamples, Math.round(info.duration / V.sampleIntervalS)))
  const step = info.duration / total
  const grabber = new FrameGrabber()
  let analyzer: SignalAnalyzer | null = null
  let useEngine = true
  try {
    await vision.init()
    vision.resetTracking()
  } catch {
    useEngine = false
    analyzer = new SignalAnalyzer()
  }

  const samples: Sample[] = []
  const seen = new Set<string>()
  let scenes = 1
  const fastFacts = await modelFacts(VISION.fast.model)
  try {
    for (let i = 0; i < total; i++) {
      if (opts.signal?.aborted) throw new DOMException('Aborted', 'AbortError')
      const t = Math.min(info.duration - 0.05, (i + 0.5) * step)
      await seek(video, t)
      const bitmap = await grabber.bitmap(video, VISION.fast.grabWidth)
      if (!bitmap) continue
      let sample: Sample
      if (useEngine) {
        const result = await vision.processFrame(bitmap, t * 1000)
        const objects = visibleTracks(result.tracks).map((track) => trackObject(track, t * 1000))
        let coverage: CoveragePayload | null = null
        try {
          coverage = coveragePayload(measureCoverage(video, video.videoWidth, video.videoHeight, objects.map((o) => o.box)))
        } catch {
          coverage = null
        }
        sample = {
          t,
          signature: result.signature,
          sharpness: result.signals.sharpness,
          brightness: result.signals.brightness,
          motion: result.signals.motion,
          objects,
          events: result.events,
          runs: [detectorRun(VISION.fast.model, 'fast', 'ok', fastFacts, { boxes: result.detections, ms: result.inferenceMs })],
          coverage,
        }
      } else {
        const { signals, signature } = analyzer!.analyze(bitmap)
        bitmap.close()
        sample = {
          t,
          signature,
          sharpness: signals.sharpness,
          brightness: signals.brightness,
          motion: signals.motion,
          objects: [],
          events: [],
          runs: [detectorRun(VISION.fast.model, 'fast', 'unavailable', fastFacts, { note: 'the on-device detector could not be loaded' })],
          coverage: null,
        }
      }
      const prev = samples[samples.length - 1]
      if (prev && signatureDistance(sample.signature, prev.signature) > V.sceneBoundary) scenes++
      sample.objects.forEach((o) => seen.add(o.id))
      samples.push(sample)
      opts.onProgress?.({ done: i + 1, total, t, scenes, objects: seen.size })
    }
  } finally {
    if (useEngine) vision.resetTracking()
  }
  if (!samples.length) {
    throw new ApiError('VIDEO_UNREADABLE', 'No frames could be decoded from this video.', { hint: 'The file may be corrupted.' })
  }
  return samples
}

/** Scene segmentation, redundancy removal and representative keyframe selection. */
export function selectKeyframes(samples: Sample[], maxKeyframes: number): Selection {
  const sceneOf: number[] = [0]
  const scenes: number[][] = [[0]]
  const events: Selection['events'] = []
  for (let i = 1; i < samples.length; i++) {
    const stepDist = signatureDistance(samples[i]!.signature, samples[i - 1]!.signature)
    const drift = signatureDistance(samples[i]!.signature, samples[scenes[scenes.length - 1]![0]!]!.signature)
    if (stepDist > V.sceneBoundary || drift > V.sceneDrift) {
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
      if (signatureDistance(samples[i]!.signature, samples[kept]!.signature) < V.dedupe) redundant++
      else kept = i
    }
  }

  // Object enter/leave events from the tracker (confirmed / expired tracks).
  const changePoints: { index: number; text: string }[] = []
  samples.forEach((sample, i) => {
    for (const e of sample.events) {
      const name = `${e.label}·${e.sceneId.slice(1).padStart(2, '0')}`
      if (e.kind === 'entered' && i > 0) {
        events.push({ t: sample.t, kind: 'OBJECT_ENTERED', text: `${name} enters the frame` })
        changePoints.push({ index: i, text: `${e.label} entered` })
      } else if (e.kind === 'left') {
        events.push({ t: sample.t, kind: 'OBJECT_LEFT', text: `${name} leaves the frame` })
        changePoints.push({ index: i, text: `${e.label} left` })
      }
    }
  })

  const [wLo, wHi] = V.representativeWindow
  const representative = (scene: number[]): number => {
    const lo = Math.floor(scene.length * wLo)
    const hi = Math.max(lo + 1, Math.floor(scene.length * wHi))
    const core = scene.slice(lo, hi)
    return (core.length ? core : scene).reduce((best, i) => (samples[i]!.sharpness > samples[best]!.sharpness ? i : best))
  }

  let picks: Keyframe[] = scenes.map((scene, si) => ({ sample: representative(scene), t: 0, scene: si, reason: 'scene representative' }))
  if (picks.length > maxKeyframes) {
    const byLength = scenes.map((sc, si) => [sc.length, si] as const).sort((a, b) => b[0] - a[0])
    const keep = new Set(byLength.slice(0, maxKeyframes - 1).map(([, si]) => si))
    keep.add(0)
    picks = picks.filter((p) => keep.has(p.scene)).slice(0, maxKeyframes)
  } else {
    const chosen = new Set(picks.map((p) => p.sample))
    const farEnough = (i: number) => [...chosen].every((c) => Math.abs(samples[c]!.t - samples[i]!.t) > V.keyframeMinGapS)
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
      if (picks.length >= maxKeyframes || d < V.dedupe * 2) break
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
    sceneOf,
    keyframes: picks,
    redundant,
    events: events.sort((a, b) => a.t - b.t),
  }
}

/** Seek to each keyframe and encode it as a JPEG (thumbnails, deep detector, optional AI upload). */
export async function captureKeyframes(video: HTMLVideoElement, keyframes: Keyframe[]): Promise<Blob[]> {
  const capturer = new JpegCapturer()
  const blobs: Blob[] = []
  for (const k of keyframes) {
    await seek(video, k.t)
    blobs.push((await capturer.capture(video, VISION.capture.keyframeMaxEdge, VISION.capture.keyframeQuality)).blob)
  }
  return blobs
}

/** The JSON manifest for /api/analyze/video (numbers only; matches VideoManifest). */
export function buildManifest(
  file: File,
  info: VideoInfo,
  samples: Sample[],
  selection: Selection,
  detectors: string[],
): Record<string, unknown> {
  const tracks = new Map<string, { id: string; label: string; first_t: number; last_t: number; samples: number; sum: number }>()
  for (const sample of samples) {
    for (const o of sample.objects) {
      const entry = tracks.get(o.id) ?? { id: o.id, label: o.label, first_t: sample.t, last_t: sample.t, samples: 0, sum: 0 }
      entry.label = o.label
      entry.last_t = sample.t
      entry.samples += 1
      entry.sum += o.confidence
      tracks.set(o.id, entry)
    }
  }
  const r2 = (n: number) => Math.round(n * 100) / 100
  return {
    duration_s: info.duration,
    width: info.width,
    height: info.height,
    name: file.name,
    size_bytes: file.size,
    detector: detectors.join(' + ') || null,
    sampled_frames: samples.length,
    redundant_removed: selection.redundant,
    scenes: selection.scenes,
    samples: samples.map((s, i) => ({
      t: r2(s.t),
      scene: selection.sceneOf[i] ?? 0,
      objects: s.objects.slice(0, 40),
      brightness: s.brightness,
      sharpness: s.sharpness,
      motion: s.motion,
      detectors: s.runs.filter((r) => r.status === 'ok').map((r) => r.model),
      runs: s.runs,
      coverage: s.coverage,
    })),
    frames: selection.keyframes.map((k, i) => ({
      index: i,
      t: r2(k.t),
      scene: k.scene,
      reason: k.reason,
      sharpness: samples[k.sample]!.sharpness,
      brightness: samples[k.sample]!.brightness,
      detectors,
      objects: samples[k.sample]!.objects
        .filter((o) => o.confidence >= DETECTION.fast.scoreThreshold || o.verified)
        .slice(0, 30)
        .map((o) => ({ track_id: o.id, label: o.label, confidence: o.confidence, box: o.box })),
    })),
    events: selection.events.slice(0, 200),
    tracks: [...tracks.values()].slice(0, 300).map((t) => ({
      id: t.id,
      label: t.label,
      first_t: r2(t.first_t),
      last_t: r2(t.last_t),
      samples: t.samples,
      mean_confidence: Math.round((t.sum / t.samples) * 1000) / 1000,
    })),
  }
}
