/**
 * Main-thread handle to the local vision engine.
 *
 * Runs the engine in a module Web Worker; if workers (or OffscreenCanvas in
 * workers) are unavailable it falls back to running the same engine on the
 * main thread.
 */
import { useSyncExternalStore } from 'react'
import moduleLoaderUrl from '@mediapipe/tasks-vision/vision_wasm_module_internal.js?url'
import moduleBinaryUrl from '@mediapipe/tasks-vision/vision_wasm_module_internal.wasm?url'
import classicLoaderUrl from '@mediapipe/tasks-vision/vision_wasm_internal.js?url'
import classicBinaryUrl from '@mediapipe/tasks-vision/vision_wasm_internal.wasm?url'
import { supportsWasm, supportsWorkers } from '../lib/env'
import type { VisionEngine } from './engine'
import type {
  Delegate,
  DelegatePreference,
  EngineInfo,
  FrameResult,
  InitPayload,
  StillResult,
  WorkerRequest,
  WorkerResponse,
} from './types'

export const MODEL_URL = `${import.meta.env.BASE_URL}models/efficientdet_lite0.tflite`

const PREF_KEY = 'rd.vision.delegate'
const CHOICE_KEY = 'rd.vision.autoChoice'
const CHOICE_TTL_MS = 14 * 24 * 60 * 60 * 1000

export type VisionStatus = 'idle' | 'loading' | 'ready' | 'error'

export interface VisionSnapshot {
  status: VisionStatus
  info: EngineInfo | null
  error: string | null
}

type Pending = { resolve: (value: never) => void; reject: (reason: Error) => void }

/** User preference: `?vision=cpu|gpu|auto` overrides the saved setting. */
export function delegatePreference(): DelegatePreference {
  try {
    const param = new URLSearchParams(window.location.search).get('vision')?.toUpperCase()
    if (param === 'CPU' || param === 'GPU') return param
    if (param === 'AUTO') return 'auto'
    const saved = window.localStorage.getItem(PREF_KEY)
    return saved === 'CPU' || saved === 'GPU' ? saved : 'auto'
  } catch {
    return 'auto'
  }
}

export function setDelegatePreference(pref: DelegatePreference): void {
  try {
    window.localStorage.setItem(PREF_KEY, pref)
    window.localStorage.removeItem(CHOICE_KEY)
  } catch {
    /* ignore */
  }
}

function cachedAutoChoice(): Delegate | null {
  try {
    const raw = window.localStorage.getItem(CHOICE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as { delegate?: Delegate; at?: number }
    if ((parsed.delegate === 'CPU' || parsed.delegate === 'GPU') && Date.now() - (parsed.at ?? 0) < CHOICE_TTL_MS) {
      return parsed.delegate
    }
  } catch {
    /* ignore */
  }
  return null
}

function rememberAutoChoice(delegate: Delegate): void {
  try {
    window.localStorage.setItem(CHOICE_KEY, JSON.stringify({ delegate, at: Date.now() }))
  } catch {
    /* ignore */
  }
}

function absolute(url: string): string {
  return new URL(url, window.location.href).href
}

class VisionClient {
  private snapshot: VisionSnapshot = { status: 'idle', info: null, error: null }
  private listeners = new Set<() => void>()
  private worker: Worker | null = null
  private engine: VisionEngine | null = null
  private pending = new Map<number, Pending>()
  private seq = 0
  private initPromise: Promise<EngineInfo> | null = null

  getSnapshot = (): VisionSnapshot => this.snapshot

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  private set(update: Partial<VisionSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...update }
    this.listeners.forEach((l) => l())
  }

  get ready(): boolean {
    return this.snapshot.status === 'ready'
  }

  /** Load the model (once). Safe to call repeatedly. */
  init(): Promise<EngineInfo> {
    if (!this.initPromise) {
      this.initPromise = this.load().catch((error: unknown) => {
        this.initPromise = null
        const message = error instanceof Error ? error.message : String(error)
        this.set({ status: 'error', error: message })
        throw error instanceof Error ? error : new Error(message)
      })
    }
    return this.initPromise
  }

  private async load(): Promise<EngineInfo> {
    if (!supportsWasm()) throw new Error('This browser does not support WebAssembly, which the vision engine needs.')
    this.set({ status: 'loading', error: null })

    const preference = delegatePreference()
    const cached = preference === 'auto' ? cachedAutoChoice() : null
    const delegate: DelegatePreference = cached ?? preference

    let info: EngineInfo | null = null
    if (supportsWorkers() && typeof OffscreenCanvas !== 'undefined') {
      try {
        const workerInfo = await this.startWorker({
          modelUrl: absolute(MODEL_URL),
          wasmLoaderUrl: absolute(moduleLoaderUrl),
          wasmBinaryUrl: absolute(moduleBinaryUrl),
          delegate,
        })
        info = { ...workerInfo, runtime: 'worker' }
      } catch (error) {
        console.warn('[vision] worker runtime failed, falling back to the main thread:', error)
        this.worker?.terminate()
        this.worker = null
      }
    }

    if (!info) {
      const { VisionEngine } = await import('./engine')
      const engine = new VisionEngine()
      const mainInfo = await engine.init({
        modelUrl: absolute(MODEL_URL),
        wasmLoaderUrl: absolute(classicLoaderUrl),
        wasmBinaryUrl: absolute(classicBinaryUrl),
        delegate,
      })
      this.engine = engine
      info = { ...mainInfo, runtime: 'main-thread' }
    }

    if (preference === 'auto' && !cached) rememberAutoChoice(info.delegate)
    this.set({ status: 'ready', info })
    return info
  }

  private startWorker(payload: InitPayload): Promise<Omit<EngineInfo, 'runtime'>> {
    const worker = new Worker(new URL('./vision.worker.ts', import.meta.url), { type: 'module', name: 'vision-engine' })
    this.worker = worker
    return new Promise((resolve, reject) => {
      const timer = window.setTimeout(() => reject(new Error('The vision engine did not start within 90 s.')), 90_000)
      worker.onerror = (event) => {
        window.clearTimeout(timer)
        reject(new Error(event.message || 'The vision worker crashed while loading.'))
      }
      worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        const message = event.data
        if (message.type === 'ready') {
          window.clearTimeout(timer)
          worker.onmessage = (e: MessageEvent<WorkerResponse>) => this.route(e.data)
          worker.onerror = (e) => this.fail(new Error(e.message || 'The vision worker crashed.'))
          resolve(message.info)
        } else if (message.type === 'init-error') {
          window.clearTimeout(timer)
          reject(new Error(message.message))
        }
      }
      worker.postMessage({ type: 'init', payload } satisfies WorkerRequest)
    })
  }

  private route(message: WorkerResponse): void {
    if (message.type === 'frame' || message.type === 'still') {
      const entry = this.pending.get(message.id)
      if (!entry) return
      this.pending.delete(message.id)
      entry.resolve(message.result as never)
    } else if (message.type === 'error') {
      const entry = this.pending.get(message.id)
      if (!entry) return
      this.pending.delete(message.id)
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

  private call<T>(request: WorkerRequest & { id: number }, transfer: Transferable[]): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.pending.set(request.id, { resolve: resolve as (v: never) => void, reject })
      this.worker!.postMessage(request, transfer)
    })
  }

  /** Detect + track + analyse one live frame. The bitmap is consumed. */
  async processFrame(bitmap: ImageBitmap, timestamp: number): Promise<FrameResult> {
    await this.init()
    const id = ++this.seq
    if (this.worker) return this.call<FrameResult>({ type: 'frame', id, bitmap, timestamp }, [bitmap])
    return this.engine!.processFrame(id, bitmap, timestamp)
  }

  /** Detect objects in a still image (no tracking). The bitmap is consumed. */
  async analyzeStill(bitmap: ImageBitmap): Promise<StillResult> {
    await this.init()
    const id = ++this.seq
    if (this.worker) return this.call<StillResult>({ type: 'still', id, bitmap }, [bitmap])
    return this.engine!.analyzeStill(bitmap)
  }

  /** Mark the current frame as the reference for scene-change detection. */
  setAnchor(): void {
    if (this.worker) this.worker.postMessage({ type: 'anchor' } satisfies WorkerRequest)
    else this.engine?.setAnchor()
  }

  resetTracking(): void {
    if (this.worker) this.worker.postMessage({ type: 'reset' } satisfies WorkerRequest)
    else this.engine?.reset()
  }
}

export const vision = new VisionClient()

export function useVision(): VisionSnapshot {
  return useSyncExternalStore(vision.subscribe, vision.getSnapshot)
}
