import { create } from 'zustand'
import type { ApiError } from '../lib/api'
import type { AIRun, LifecycleEvent, Report, ScanState, Trigger } from '../lib/schemas'

export type Phase = 'booting' | 'running' | 'paused' | 'deep' | 'error' | 'ended'

export type Living =
  | 'STABLE'
  | 'LIMITED'
  | 'INCONCLUSIVE'
  | 'MONITORING'
  | 'ANOMALY'
  | 'INVESTIGATING'
  | 'DISCOVERED'
  | 'CONFIRMED'
  | 'RESOLVED'
  | 'FROZEN'
  | 'BOOTING'

/** The optional AI layer as seen by this scan. Local CV runs regardless. */
export type AIState = 'off' | 'idle' | 'waiting' | 'analyzing' | 'unavailable'

export interface Hud {
  fps: number
  inferenceMs: number
  objects: number
  tentative: number
  verified: number
  relations: number
  motion: number
  sceneDelta: number
  viewId: number
  brightness: number
  sharpness: number
  motionTrace: number[]
}

/** Developer metrics (measured in this browser). */
export interface DevMetrics {
  fastMs: number
  frameMs: number
  observeMs: number | null
  observations: number
  observePayloadBytes: number
  deepMs: number | null
  deepBackend: string | null
  deepRuns: number
  deepVerified: number
  deepAdded: number
  aiCalls: number
  aiSkipped: number
  lastSkipReason: string | null
}

export interface DeepState {
  status: 'detecting' | 'analyzing' | 'done' | 'error'
  imageUrl: string
  width: number
  height: number
  startedAt: number
  step: string
  report: Report | null
  error: ApiError | null
}

interface LiveStore {
  phase: Phase
  bootStep: string
  fatal: ApiError | null
  hud: Hud
  dev: DevMetrics
  ai: {
    state: AIState
    enabled: boolean
    provider: string | null
    startedAt: number
    lastLatencyMs: number | null
    lastTrigger: Trigger | null
    lastRun: AIRun | null
    detail: string | null
    error: ApiError | null
    analyses: number
  }
  living: { state: Living; detail: string | null; until: number }
  anomaly: string | null
  scan: ScanState | null
  events: LifecycleEvent[]
  deep: DeepState | null
  explaining: string | null
  set: (patch: Partial<LiveStore>) => void
  reset: () => void
}

const EMPTY_HUD: Hud = {
  fps: 0,
  inferenceMs: 0,
  objects: 0,
  tentative: 0,
  verified: 0,
  relations: 0,
  motion: 0,
  sceneDelta: 0,
  viewId: 0,
  brightness: 0,
  sharpness: 0,
  motionTrace: [],
}

const EMPTY_DEV: DevMetrics = {
  fastMs: 0,
  frameMs: 0,
  observeMs: null,
  observations: 0,
  observePayloadBytes: 0,
  deepMs: null,
  deepBackend: null,
  deepRuns: 0,
  deepVerified: 0,
  deepAdded: 0,
  aiCalls: 0,
  aiSkipped: 0,
  lastSkipReason: null,
}

const initial = {
  phase: 'booting' as Phase,
  bootStep: 'Requesting camera',
  fatal: null,
  hud: EMPTY_HUD,
  dev: EMPTY_DEV,
  ai: {
    state: 'off' as AIState,
    enabled: false,
    provider: null,
    startedAt: 0,
    lastLatencyMs: null,
    lastTrigger: null,
    lastRun: null,
    detail: null,
    error: null,
    analyses: 0,
  },
  living: { state: 'BOOTING' as Living, detail: null, until: 0 },
  anomaly: null,
  scan: null,
  events: [],
  deep: null,
  explaining: null,
}

export const useLive = create<LiveStore>((set) => ({
  ...initial,
  set: (patch) => set(patch),
  reset: () => set(initial),
}))

export const LIVING_TEXT: Record<Living, string> = {
  STABLE: 'System stable',
  LIMITED: 'Limited inspection',
  INCONCLUSIVE: 'Inconclusive',
  MONITORING: 'Monitoring',
  ANOMALY: 'Anomaly detected',
  INVESTIGATING: 'AI reasoning…',
  DISCOVERED: 'Bug discovered',
  CONFIRMED: 'Bug confirmed',
  RESOLVED: 'Issue resolved ✓',
  FROZEN: 'Frame frozen',
  BOOTING: 'Booting',
}
