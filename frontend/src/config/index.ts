/**
 * Shared parameters (repository `config/`), bundled at build time. The same
 * files configure the backend, so thresholds such as the tracker's score
 * bands or the scene-change distance are defined once, with their source.
 *
 * Every parameter in those files is `{ value, unit?, purpose, source, ref }`;
 * `values()` keeps only the values.
 */
import aiJson from '../../../config/ai.json'
import detectionJson from '../../../config/detection.json'
import diagnosticsJson from '../../../config/diagnostics.json'
import temporalJson from '../../../config/temporal.json'
import trackingJson from '../../../config/tracking.json'
import visionJson from '../../../config/vision.json'

type Json = null | boolean | number | string | Json[] | { [key: string]: Json }

export function values(node: unknown): Json {
  if (Array.isArray(node)) return node.map(values)
  if (node && typeof node === 'object') {
    const obj = node as Record<string, unknown>
    if ('value' in obj && ('purpose' in obj || 'source' in obj)) return values(obj.value)
    return Object.fromEntries(Object.entries(obj).map(([k, v]) => [k, values(v)]))
  }
  return node as Json
}

export interface VisionConfig {
  fast: { model: string; letterbox: boolean; maxFps: number; grabWidth: number }
  deep: {
    model: string
    mode: 'auto' | 'off'
    backend: 'auto' | 'webgpu' | 'wasm'
    wasmThreads: number
    liveCooldownMs: number
    liveMaxLatencyMs: number
    videoBudgetMs: number
  }
  capture: {
    aiFrameMaxEdge: number
    aiFrameQuality: number
    deepScanMaxEdge: number
    deepScanQuality: number
    keyframeMaxEdge: number
    keyframeQuality: number
  }
  signals: {
    thumbWidth: number
    thumbHeight: number
    gridWidth: number
    gridHeight: number
    histogramBins: number
    histogramWeight: number
    gridGain: number
    motionGain: number
    motionPixelDelta: number
    motionMinFraction: number
    sharpnessLogOffset: number
    sharpnessLogSpan: number
  }
}

export interface DetectionConfig {
  fast: { scoreThreshold: number; lowScoreFloor: number; maxResults: number; nmsIou: number }
  deep: {
    inputSize: number
    padValue: number
    channelOrder: 'BGR' | 'RGB'
    preNmsScore: number
    nmsIou: number
    classAgnosticNms: boolean
    scoreThreshold: number
  }
  fusion: { matchIou: number; deepOnlyMinScore: number; relabelMargin: number }
}

export interface TrackingConfig {
  method: string
  highScore: number
  lowScore: number
  newTrackScore: number
  firstMatchIou: number
  secondMatchIou: number
  unconfirmedMatchIou: number
  fuseScore: boolean
  duplicateIou: number
  lostBufferMs: number
  confirmHits: number
  tentativeMaxMisses: number
  kalmanStdWeightPosition: number
  kalmanStdWeightVelocity: number
  classCompatibility: 'supercategory' | 'label'
  cmc: boolean
  cmcMinConfidence: number
}

export interface TemporalConfig {
  movement: { movingSpeed: number; stillSpeed: number; reversalWindowMs: number }
  persistence: { persistentAfterMs: number }
  occlusion: { occludedFraction: number }
  sceneChange: { threshold: number; settleMotion: number; maxSettleWaitMs: number }
  observe: { intervalMs: number; heartbeatMs: number }
  lifecycle: Record<string, number>
  video: {
    sampleIntervalS: number
    minSamples: number
    maxSamples: number
    sceneBoundary: number
    sceneDrift: number
    dedupe: number
    maxKeyframes: number
    representativeWindow: [number, number]
    keyframeMinGapS: number
  }
}

export interface DiagnosticsConfig {
  proximity: { touchGap: number; nearRelativeGap: number }
}

export interface AIConfig {
  triggers: Record<string, boolean | number | string>
  budget: { cooldownMs: number; maxCallsPerMinute: number }
  images: { liveMaxEdge: number; deepMaxEdge: number; videoMaxEdge: number; jpegQuality: number }
}

export const VISION = values(visionJson) as unknown as VisionConfig
export const DETECTION = values(detectionJson) as unknown as DetectionConfig
export const TRACKING = values(trackingJson) as unknown as TrackingConfig
export const TEMPORAL = values(temporalJson) as unknown as TemporalConfig
export const DIAGNOSTICS = values(diagnosticsJson) as unknown as DiagnosticsConfig
export const AI = values(aiJson) as unknown as AIConfig

/** The documented form of one parameter, e.g. `describe(trackingJson, 'highScore')`, for the dev panel. */
export interface ParamDoc {
  path: string
  value: Json
  unit?: string
  purpose: string
  source: string
  ref?: string
}

export function documented(): ParamDoc[] {
  const out: ParamDoc[] = []
  const walk = (node: unknown, path: string) => {
    if (!node || typeof node !== 'object' || Array.isArray(node)) return
    const obj = node as Record<string, unknown>
    if ('value' in obj && 'purpose' in obj) {
      out.push({
        path,
        value: values(obj.value),
        unit: obj.unit as string | undefined,
        purpose: String(obj.purpose),
        source: String(obj.source),
        ref: obj.ref as string | undefined,
      })
      return
    }
    for (const [key, child] of Object.entries(obj)) walk(child, path ? `${path}.${key}` : key)
  }
  const files: [string, unknown][] = [
    ['vision', visionJson],
    ['detection', detectionJson],
    ['tracking', trackingJson],
    ['temporal', temporalJson],
    ['diagnostics', diagnosticsJson],
    ['ai', aiJson],
  ]
  for (const [name, json] of files) walk(json, name)
  return out
}
