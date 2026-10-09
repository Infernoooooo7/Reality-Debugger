/**
 * The structured scene model sent to the backend's local diagnostic engine
 * (and, when enabled, to the optional AI layer). Everything in it is measured
 * on this device: detector boxes and confidences, tracker state and pixel
 * signals. Matches backend/app/schemas/scene.py.
 */
import { TEMPORAL } from '../config'
import type { FusedObject } from './fusion'
import { roundBox } from './geometry'
import type { FrameResult, FrameSignals, NBox, Track } from './types'

export interface SceneObjectPayload {
  id: string
  label: string
  confidence: number
  box: NBox
  source: 'fast' | 'deep' | 'fused'
  verified: boolean
  age_ms: number
  persistent: boolean
  movement: 'static' | 'moving' | 'unknown'
  speed: number | null
  static_ms: number
  reversals: number
  occlusion: number
  occluded_ms: number
  truncated: boolean
}

export interface SceneEventPayload {
  kind: 'entered' | 'left' | 'moved' | 'scene_change'
  at_ms: number
  object_id?: string | null
  label?: string | null
}

export interface ScenePayload {
  version: 2
  at_ms: number
  view_id: number
  width: number | null
  height: number | null
  objects: SceneObjectPayload[]
  signals: {
    motion: number | null
    brightness: number | null
    sharpness: number | null
    scene_change: number | null
    motion_box: NBox | null
  }
  events: SceneEventPayload[]
  stats: {
    fps: number | null
    tentative_tracks: number
    mean_confidence: number | null
    detectors: string[]
  }
}

const MAX_OBJECTS = 60
const r3 = (n: number) => Math.round(n * 1000) / 1000

export function trackObject(t: Track, now: number): SceneObjectPayload {
  const age = Math.max(0, now - t.firstSeen)
  return {
    id: t.sceneId,
    label: t.label,
    confidence: r3(t.score),
    box: roundBox(t.box),
    source: t.source,
    verified: t.verified,
    age_ms: Math.round(age),
    persistent: age >= TEMPORAL.persistence.persistentAfterMs,
    movement: t.movement,
    speed: r3(t.speed),
    static_ms: Math.round(t.staticMs),
    reversals: t.reversals,
    occlusion: r3(t.occlusion),
    occluded_ms: Math.round(t.occludedMs),
    truncated: t.truncated,
  }
}

/** Tracks matched in this frame (a coasting "lost" track may already have left the view). */
export function visibleTracks(tracks: Track[]): Track[] {
  return tracks.filter((t) => t.state === 'confirmed')
}

function signalsPayload(s: FrameSignals): ScenePayload['signals'] {
  return {
    motion: r3(s.motion),
    brightness: r3(s.brightness),
    sharpness: r3(s.sharpness),
    scene_change: r3(s.sceneDelta),
    motion_box: s.motionBox ? roundBox(s.motionBox) : null,
  }
}

export function liveScene(
  result: FrameResult,
  opts: { atMs: number; width: number | null; height: number | null; fps: number | null; events: SceneEventPayload[]; detectors: string[] },
): ScenePayload {
  const tracks = visibleTracks(result.tracks).sort((a, b) => b.score - a.score).slice(0, MAX_OBJECTS)
  const objects = tracks.map((t) => trackObject(t, result.timestamp))
  const mean = objects.length ? objects.reduce((n, o) => n + o.confidence, 0) / objects.length : null
  return {
    version: 2,
    at_ms: Math.round(opts.atMs),
    view_id: result.viewId,
    width: opts.width,
    height: opts.height,
    objects,
    signals: signalsPayload(result.signals),
    events: opts.events.slice(-40),
    stats: {
      fps: opts.fps === null ? null : Math.round(opts.fps * 10) / 10,
      tentative_tracks: result.tracks.filter((t) => t.state === 'tentative').length,
      mean_confidence: mean === null ? null : r3(mean),
      detectors: opts.detectors,
    },
  }
}

export function stillScene(
  objects: FusedObject[],
  signals: FrameSignals,
  opts: { width: number | null; height: number | null; detectors: string[]; atMs?: number; viewId?: number },
): ScenePayload {
  const list = objects.slice(0, MAX_OBJECTS).map<SceneObjectPayload>((o) => ({
    id: o.id,
    label: o.label,
    confidence: r3(o.confidence),
    box: roundBox(o.box),
    source: o.source,
    verified: o.verified,
    age_ms: 0,
    persistent: false,
    movement: 'unknown',
    speed: null,
    static_ms: 0,
    reversals: 0,
    occlusion: 0,
    occluded_ms: 0,
    truncated: false,
  }))
  const mean = list.length ? list.reduce((n, o) => n + o.confidence, 0) / list.length : null
  return {
    version: 2,
    at_ms: Math.round(opts.atMs ?? 0),
    view_id: opts.viewId ?? 0,
    width: opts.width,
    height: opts.height,
    objects: list,
    signals: signalsPayload(signals),
    events: [],
    stats: { fps: null, tentative_tracks: 0, mean_confidence: mean === null ? null : r3(mean), detectors: opts.detectors },
  }
}
