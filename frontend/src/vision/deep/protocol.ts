import type { Detection } from '../types'
import type { YoloxParams } from './yolox'

export type DeepBackend = 'webgpu' | 'wasm'

export interface DeepInitPayload {
  modelUrl: string
  sha256: string
  modelName: string
  backend: 'auto' | DeepBackend
  threads: number
  params: YoloxParams
}

export interface DeepInfo {
  model: string
  backend: DeepBackend
  threads: number
  crossOriginIsolated: boolean
  bytes: number
  cached: boolean
  downloadMs: number
  sessionMs: number
  warmupMs: number
  fallbackReason: string | null
}

export interface DeepResult {
  detections: Detection[]
  preprocessMs: number
  inferenceMs: number
  postprocessMs: number
  totalMs: number
}

export type DeepRequest = { type: 'init'; payload: DeepInitPayload } | { type: 'detect'; id: number; bitmap: ImageBitmap }

export type DeepResponse =
  | { type: 'progress'; loaded: number; total: number }
  | { type: 'ready'; info: DeepInfo }
  | { type: 'init-error'; message: string }
  | { type: 'detect'; id: number; result: DeepResult }
  | { type: 'error'; id: number; message: string }
