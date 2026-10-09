/**
 * Main-thread handle to the deep detector (YOLOX-S in its own worker).
 *
 * Loaded lazily - the model is 36 MB - when the user runs a Deep Scan,
 * Image Debug or Video Debug. The live loop only uses it for periodic
 * verification once it is loaded and fast enough on this device
 * (config/vision.json "deep").
 */
import { useSyncExternalStore } from 'react'
import { DETECTION, VISION } from '../../config'
import { useSettings } from '../../state/settings'
import { loadManifest, modelUrl } from '../models'
import type { DeepInfo, DeepRequest, DeepResponse, DeepResult } from './protocol'

export type DeepStatus = 'idle' | 'loading' | 'ready' | 'error' | 'off'

export interface DeepSnapshot {
  status: DeepStatus
  info: DeepInfo | null
  error: string | null
  progress: { loaded: number; total: number } | null
  /** Exponential moving average of total detection time on this device. */
  emaMs: number | null
  lastMs: number | null
  runs: number
}

type Pending = { resolve: (value: DeepResult) => void; reject: (reason: Error) => void }

class DeepClient {
  private snapshot: DeepSnapshot = { status: 'idle', info: null, error: null, progress: null, emaMs: null, lastMs: null, runs: 0 }
  private listeners = new Set<() => void>()
  private worker: Worker | null = null
  private initPromise: Promise<DeepInfo> | null = null
  private pending = new Map<number, Pending>()
  private seq = 0

  getSnapshot = (): DeepSnapshot => this.snapshot

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  private set(update: Partial<DeepSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...update }
    this.listeners.forEach((l) => l())
  }

  /** The user (or config) can switch the deep detector off entirely. */
  get enabled(): boolean {
    return useSettings.getState().deepMode !== 'off' && typeof Worker !== 'undefined' && typeof OffscreenCanvas !== 'undefined'
  }

  get ready(): boolean {
    return this.snapshot.status === 'ready'
  }

  /** Fast enough to run alongside the live loop on this device? */
  get liveCapable(): boolean {
    const ms = this.snapshot.emaMs ?? this.snapshot.info?.warmupMs ?? Infinity
    return this.ready && ms <= VISION.deep.liveMaxLatencyMs
  }

  init(): Promise<DeepInfo> {
    if (!this.enabled) {
      this.set({ status: 'off' })
      return Promise.reject(new Error('The deep detector is switched off.'))
    }
    if (!this.initPromise) {
      this.initPromise = this.load().catch((error: unknown) => {
        this.initPromise = null
        this.worker?.terminate()
        this.worker = null
        const message = error instanceof Error ? error.message : String(error)
        this.set({ status: 'error', error: message, progress: null })
        throw error instanceof Error ? error : new Error(message)
      })
    }
    return this.initPromise
  }

  private async load(): Promise<DeepInfo> {
    this.set({ status: 'loading', error: null, progress: null })
    const manifest = await loadManifest(VISION.deep.model)
    const worker = new Worker(new URL('./deep.worker.ts', import.meta.url), { type: 'module', name: 'deep-detector' })
    this.worker = worker
    const info = await new Promise<DeepInfo>((resolve, reject) => {
      worker.onerror = (event) => reject(new Error(event.message || 'The deep detector worker crashed while loading.'))
      worker.onmessage = (event: MessageEvent<DeepResponse>) => {
        const message = event.data
        if (message.type === 'progress') this.set({ progress: { loaded: message.loaded, total: message.total } })
        else if (message.type === 'ready') resolve(message.info)
        else if (message.type === 'init-error') reject(new Error(message.message))
        else this.route(message)
      }
      const preference = useSettings.getState().deepBackend
      worker.postMessage({
        type: 'init',
        payload: {
          modelUrl: new URL(modelUrl(manifest.file), window.location.href).href,
          sha256: manifest.sha256,
          modelName: manifest.name,
          backend: preference,
          threads: VISION.deep.wasmThreads,
          params: {
            inputSize: manifest.input?.width ?? DETECTION.deep.inputSize,
            padValue: manifest.input?.pad_value ?? DETECTION.deep.padValue,
            channelOrder: DETECTION.deep.channelOrder,
            strides: manifest.decode?.strides ?? [8, 16, 32],
            preNmsScore: DETECTION.deep.preNmsScore,
            nmsIou: DETECTION.deep.nmsIou,
            scoreThreshold: DETECTION.deep.scoreThreshold,
            labels: manifest.labels,
          },
        },
      } satisfies DeepRequest)
    })
    worker.onmessage = (event: MessageEvent<DeepResponse>) => this.route(event.data)
    worker.onerror = (event) => this.fail(new Error(event.message || 'The deep detector worker crashed.'))
    this.set({ status: 'ready', info, progress: null })
    return info
  }

  private route(message: DeepResponse): void {
    if (message.type !== 'detect' && message.type !== 'error') return
    const entry = this.pending.get(message.id)
    if (!entry) return
    this.pending.delete(message.id)
    if (message.type === 'detect') {
      const ms = message.result.totalMs
      const ema = this.snapshot.emaMs === null ? ms : 0.7 * this.snapshot.emaMs + 0.3 * ms
      this.set({ lastMs: ms, emaMs: Math.round(ema), runs: this.snapshot.runs + 1 })
      entry.resolve(message.result)
    } else {
      entry.reject(new Error(message.message))
    }
  }

  private fail(error: Error): void {
    for (const entry of this.pending.values()) entry.reject(error)
    this.pending.clear()
    this.worker?.terminate()
    this.worker = null
    this.initPromise = null
    this.set({ status: 'error', error: error.message })
  }

  /** Detect objects in a bitmap (consumed). Loads the model on first use. */
  async detect(bitmap: ImageBitmap): Promise<DeepResult> {
    try {
      await this.init()
    } catch (error) {
      bitmap.close()
      throw error
    }
    const id = ++this.seq
    return new Promise<DeepResult>((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      this.worker!.postMessage({ type: 'detect', id, bitmap } satisfies DeepRequest, [bitmap])
    })
  }
}

export const deepDetector = new DeepClient()

export function useDeep(): DeepSnapshot {
  return useSyncExternalStore(deepDetector.subscribe, deepDetector.getSnapshot)
}
