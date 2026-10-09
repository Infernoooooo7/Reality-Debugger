import { create } from 'zustand'
import { VISION } from '../config'
import { readJSON, writeJSON } from '../lib/storage'
import type { Personality } from '../lib/schemas'

export interface Settings {
  personality: Personality
  /** Local vision frame-rate cap (config/vision.json fast.maxFps options). */
  maxFps: 6 | 10 | 15
  /** Deep detector: auto = used for Deep Scan / Image / Video (and live checks when fast enough); off = never. */
  deepMode: 'auto' | 'off'
  /** Deep detector runtime preference. */
  deepBackend: 'auto' | 'webgpu' | 'wasm'
  /** Show the developer metrics panel in Live Scan. */
  devPanel: boolean
}

const KEY = 'rd.settings.v2'

const DEFAULTS: Settings = {
  personality: 'serious',
  maxFps: VISION.fast.maxFps as Settings['maxFps'],
  deepMode: VISION.deep.mode,
  deepBackend: VISION.deep.backend,
  devPanel: false,
}

interface SettingsStore extends Settings {
  update: (patch: Partial<Settings>) => void
}

function persist(s: Settings): void {
  const { personality, maxFps, deepMode, deepBackend, devPanel } = s
  writeJSON(KEY, { personality, maxFps, deepMode, deepBackend, devPanel })
}

export const useSettings = create<SettingsStore>((set, get) => ({
  ...DEFAULTS,
  ...readJSON<Partial<Settings>>(KEY, {}),
  update: (patch) => {
    set(patch)
    persist({ ...get(), ...patch })
  },
}))
