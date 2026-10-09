/**
 * The structured scene model sent to the backend's local diagnostic engine
 * (and, when enabled, to the optional AI layer). Everything in it is measured
 * on this device: detector boxes and confidences, tracker state and pixel
 * signals. Matches backend/app/schemas/scene.py.
 */
import { TEMPORAL } from '../config'
import type { StructureStats } from './coverage'
import type { FusedObject } from './fusion'
import { roundBox } from './geometry'
import { isVisible, type FrameResult, type FrameSignals, type NBox, type Track } from './types'

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
  kind: 'entered' | 'left' | 'recovered' | 'moved' | 'scene_change'
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
    /** Objects recognised before the payload cap, when more than were sent. */
    objects_total?: number
    /** What each detector actually did (status, boxes, resolution); the backend's coverage rules read it. */
    runs?: DetectorRunPayload[]
    /** Visible structure the recognised objects explain (vision/coverage.ts). */
    coverage?: CoveragePayload | null
  }
  /** User-defined areas that should stay clear. */
  zones?: ZonePayload[]
}

export interface DetectorRunPayload {
  model: string
  role: 'fast' | 'deep' | 'server'
  status: 'ok' | 'failed' | 'skipped' | 'unavailable'
  boxes: number
  ms: number | null
  input_size: number | null
  passes: number
  tile_px: number | null
  /** true when a tiled pass stopped before covering every tile (time budget) */
  incomplete?: boolean
  vocabulary: number | null
  note: string | null
}

export interface CoveragePayload {
  edge_density: number
  unexplained_share: number | null
  box_coverage: number
}

export interface ZonePayload {
  id: string
  name: string
  kind: 'keep_clear'
  box: NBox
}

export function coveragePayload(stats: StructureStats): CoveragePayload {
  return {
    edge_density: r4(stats.edgeDensity),
    unexplained_share: stats.unexplainedShare === null ? null : r4(stats.unexplainedShare),
    box_coverage: r4(stats.boxCoverage),
  }
}

/** The backend accepts up to 80 objects (backend/app/schemas/scene.py); the most confident are sent. */
const MAX_OBJECTS = 80
const r3 = (n: number) => Math.round(n * 1000) / 1000
const r4 = (n: number) => Math.round(n * 10000) / 10000

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
  return tracks.filter(isVisible)
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
  opts: {
    atMs: number
    width: number | null
    height: number | null
    fps: number | null
    events: SceneEventPayload[]
    detectors: string[]
    runs?: DetectorRunPayload[]
    coverage?: CoveragePayload | null
  },
): ScenePayload {
  const visible = visibleTracks(result.tracks)
  const tracks = [...visible].sort((a, b) => b.score - a.score).slice(0, MAX_OBJECTS)
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
      objects_total: visible.length,
      runs: opts.runs ?? [],
      coverage: opts.coverage ?? null,
    },
  }
}

export function stillScene(
  objects: FusedObject[],
  signals: FrameSignals | null,
  opts: {
    width: number | null
    height: number | null
    detectors: string[]
    atMs?: number
    viewId?: number
    runs?: DetectorRunPayload[]
    coverage?: CoveragePayload | null
    zones?: ZonePayload[]
  },
): ScenePayload {
  const list = [...objects].sort((a, b) => b.confidence - a.confidence).slice(0, MAX_OBJECTS).map<SceneObjectPayload>((o) => ({
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
    signals: signals ? signalsPayload(signals) : { motion: null, brightness: null, sharpness: null, scene_change: null, motion_box: null },
    events: [],
    stats: {
      fps: null,
      tentative_tracks: 0,
      mean_confidence: mean === null ? null : r3(mean),
      detectors: opts.detectors,
      objects_total: objects.length,
      runs: opts.runs ?? [],
      coverage: opts.coverage ?? null,
    },
    zones: opts.zones ?? [],
  }
}
