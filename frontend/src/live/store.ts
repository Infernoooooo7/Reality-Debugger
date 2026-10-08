import { create } from 'zustand'
import type { ApiError } from '../lib/api'
import type { LifecycleEvent, Report, ScanState, Trigger } from '../lib/schemas'

export type Phase = 'booting' | 'running' | 'paused' | 'deep' | 'error' | 'ended'

export type Living =
  | 'STABLE'
  | 'MONITORING'
  | 'ANOMALY'
  | 'INVESTIGATING'
  | 'DISCOVERED'
  | 'CONFIRMED'
  | 'RESOLVED'
  | 'FROZEN'
  | 'BOOTING'

export type AIState = 'idle' | 'analyzing' | 'error' | 'budget' | 'blocked'

export interface Hud {
  fps: number
  inferenceMs: number
  objects: number
  tentative: number
  motion: number
  sceneDelta: number
  brightness: number
  sharpness: number
  motionTrace: number[]
}

export interface DeepState {
  status: 'analyzing' | 'done' | 'error'
  imageUrl: string
  width: number
  height: number
  startedAt: number
  report: Report | null
  error: ApiError | null
}

interface LiveStore {
  phase: Phase
  bootStep: string
  fatal: ApiError | null
  hud: Hud
  ai: { state: AIState; startedAt: number; lastLatencyMs: number | null; lastTrigger: Trigger | null; detail: string | null; error: ApiError | null; analyses: number }
  living: { state: Living; detail: string | null; until: number }
  anomaly: string | null
  scan: ScanState | null
  events: LifecycleEvent[]
  deep: DeepState | null
  set: (patch: Partial<LiveStore>) => void
  reset: () => void
}

const EMPTY_HUD: Hud = {
  fps: 0,
  inferenceMs: 0,
  objects: 0,
  tentative: 0,
  motion: 0,
  sceneDelta: 0,
  brightness: 0,
  sharpness: 0,
  motionTrace: [],
}

const initial = {
  phase: 'booting' as Phase,
  bootStep: 'Requesting camera',
  fatal: null,
  hud: EMPTY_HUD,
  ai: { state: 'idle' as AIState, startedAt: 0, lastLatencyMs: null, lastTrigger: null, detail: null, error: null, analyses: 0 },
  living: { state: 'BOOTING' as Living, detail: null, until: 0 },
  anomaly: null,
  scan: null,
  events: [],
  deep: null,
}

export const useLive = create<LiveStore>((set) => ({
  ...initial,
  set: (patch) => set(patch),
  reset: () => set(initial),
}))

export const LIVING_TEXT: Record<Living, string> = {
  STABLE: 'System stable',
  MONITORING: 'Monitoring',
  ANOMALY: 'Anomaly detected',
  INVESTIGATING: 'Investigating…',
  DISCOVERED: 'Bug discovered',
  CONFIRMED: 'Bug confirmed',
  RESOLVED: 'Issue resolved ✓',
  FROZEN: 'Frame frozen',
  BOOTING: 'Booting',
}
