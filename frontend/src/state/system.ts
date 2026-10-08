/**
 * Real subsystem status shown on the home screen's power-on self test:
 * backend reachability, AI provider, camera support and vision engine.
 */
import { create } from 'zustand'
import { checkAI, getHealth, toApiError, type ApiError } from '../lib/api'
import type { AICheck, Health } from '../lib/schemas'
import { readList, removeKey, writeJSON } from '../lib/storage'
import type { Report } from '../lib/schemas'

export interface HistoryEntry {
  id: string
  at: string
  mode: Report['mode']
  systemName: string
  score: number
  status: Report['status']
  bugs: number
  topIssue: string | null
  simulated: boolean
}

const HISTORY_KEY = 'rd.history.v1'

interface SystemStore {
  health: Health | null
  healthError: ApiError | null
  healthLatencyMs: number | null
  checkingHealth: boolean
  aiCheck: AICheck | null
  history: HistoryEntry[]
  refreshHealth: () => Promise<void>
  refreshAICheck: () => Promise<void>
  record: (report: Report) => void
  clearHistory: () => void
}

export const useSystem = create<SystemStore>((set, get) => ({
  health: null,
  healthError: null,
  healthLatencyMs: null,
  checkingHealth: false,
  aiCheck: null,
  history: readList<HistoryEntry>(HISTORY_KEY),

  refreshHealth: async () => {
    if (get().checkingHealth) return
    set({ checkingHealth: true })
    const started = performance.now()
    try {
      const health = await getHealth()
      set({ health, healthError: null, healthLatencyMs: Math.round(performance.now() - started) })
    } catch (error) {
      set({ health: null, healthError: toApiError(error), healthLatencyMs: null })
    } finally {
      set({ checkingHealth: false })
    }
  },

  refreshAICheck: async () => {
    try {
      set({ aiCheck: await checkAI() })
    } catch {
      set({ aiCheck: null })
    }
  },

  record: (report) => {
    const top = report.findings.find((f) => f.status !== 'RESOLVED' && f.severity !== 'INFO')
    const entry: HistoryEntry = {
      id: report.report_id,
      at: report.created_at,
      mode: report.mode,
      systemName: report.system_name,
      score: report.system_score,
      status: report.status,
      bugs: report.counts.active_bugs,
      topIssue: top?.title ?? null,
      simulated: report.simulated,
    }
    const history = [entry, ...get().history.filter((h) => h.id !== entry.id)].slice(0, 12)
    writeJSON(HISTORY_KEY, history)
    set({ history })
  },

  clearHistory: () => {
    removeKey(HISTORY_KEY)
    set({ history: [] })
  },
}))

/** True when the backend will simulate diagnostics (no provider or forced demo). */
export function isDemo(health: Health | null, forceDemo: boolean): boolean {
  return forceDemo || !health || health.ai.simulated
}
