/**
 * Real subsystem status shown on the home screen's power-on self test:
 * backend reachability, local CV engine, optional AI layer, camera support.
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
  /** Whether the optional AI layer contributed to the report. */
  ai: boolean
}

const HISTORY_KEY = 'rd.history.v2'

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
      ai: report.ai.status === 'ok' || report.ai.status === 'cached',
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

export type AIMode = 'off' | 'ready' | 'unavailable' | 'unknown'

/**
 * The optional AI layer as the UI presents it. Local CV never depends on it:
 * "off" is a normal configuration (no key), not an error.
 */
export function aiMode(health: Health | null): AIMode {
  if (!health) return 'unknown'
  if (!health.ai.configured || health.ai.state === 'off') return 'off'
  if (health.ai.state === 'unavailable') return 'unavailable'
  return 'ready'
}

export function aiEnabled(health: Health | null): boolean {
  return Boolean(health?.features.ai_reasoning)
}

const PROVIDER_LABEL: Record<string, string> = { gemini: 'Gemini', claude: 'Claude', openai: 'OpenAI-compatible' }

export function providerLabel(provider: string | null | undefined): string {
  return provider ? (PROVIDER_LABEL[provider] ?? provider) : 'AI'
}
