import { create } from 'zustand'
import { readJSON, writeJSON } from '../lib/storage'
import type { Personality } from '../lib/schemas'

export type AnalysisInterval = 0 | 10 | 20 | 40

export interface Settings {
  personality: Personality
  /** Force DEMO MODE even when an AI key is configured. */
  forceDemo: boolean
  /** Seconds between periodic re-checks of a stable scene (0 = off). */
  interval: AnalysisInterval
  /** Local vision frame-rate cap. */
  maxFps: 6 | 10 | 15
}

const KEY = 'rd.settings.v1'

const DEFAULTS: Settings = {
  personality: 'serious',
  forceDemo: false,
  interval: 20,
  maxFps: 10,
}

interface SettingsStore extends Settings {
  update: (patch: Partial<Settings>) => void
}

export const useSettings = create<SettingsStore>((set, get) => ({
  ...readJSON<Settings>(KEY, DEFAULTS),
  update: (patch) => {
    set(patch)
    const { personality, forceDemo, interval, maxFps } = { ...get(), ...patch }
    writeJSON(KEY, { personality, forceDemo, interval, maxFps })
  },
}))
