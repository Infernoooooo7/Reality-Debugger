/**
 * Typed client for the FastAPI backend. All calls go to same-origin `/api`
 * (proxied by Vite to the backend) unless VITE_API_BASE is set.
 */
import type { z } from 'zod'
import {
  AICheckSchema,
  ErrorBodySchema,
  HealthSchema,
  ReportSchema,
  ScanAnalysisSchema,
  ScanStateSchema,
  VideoReportSchema,
  type AICheck,
  type Health,
  type Personality,
  type Report,
  type ScanAnalysis,
  type ScanState,
  type Trigger,
  type VideoReport,
} from './schemas'

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, '') ?? ''

export class ApiError extends Error {
  readonly code: string
  readonly hint: string | null
  readonly status: number
  readonly retryable: boolean

  constructor(code: string, message: string, opts: { hint?: string | null; status?: number; retryable?: boolean } = {}) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.hint = opts.hint ?? null
    this.status = opts.status ?? 0
    this.retryable = opts.retryable ?? false
  }
}

export const BACKEND_START_HINT =
  'Start it in a terminal: cd backend, activate the venv, then run uvicorn app.main:app --host 0.0.0.0 --port 8000'

function backendUnreachable(): ApiError {
  return new ApiError('BACKEND_UNREACHABLE', 'Cannot reach the Reality Debugger backend.', {
    hint: BACKEND_START_HINT,
    retryable: true,
  })
}

export function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

interface RequestOptions {
  signal?: AbortSignal
  timeoutMs?: number
}

async function request<T>(
  path: string,
  init: RequestInit,
  schema: z.ZodType<T> | null,
  { signal, timeoutMs = 30_000 }: RequestOptions = {},
): Promise<T> {
  const controller = new AbortController()
  const onAbort = () => controller.abort(signal?.reason)
  signal?.addEventListener('abort', onAbort, { once: true })
  let timedOut = false
  const timer = window.setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)

  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, { ...init, signal: controller.signal })
  } catch {
    if (timedOut) {
      throw new ApiError('REQUEST_TIMEOUT', 'The backend took too long to answer.', {
        hint: 'The vision model may be slow right now. Try again.',
        retryable: true,
      })
    }
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')
    throw backendUnreachable()
  } finally {
    window.clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }

  if (response.status === 204) return undefined as T

  const text = await response.text()
  let data: unknown = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = null
  }

  if (!response.ok) {
    const parsed = ErrorBodySchema.safeParse((data as { error?: unknown } | null)?.error)
    if (parsed.success) {
      throw new ApiError(parsed.data.code, parsed.data.message, {
        hint: parsed.data.hint,
        status: response.status,
        retryable: parsed.data.retryable ?? response.status >= 500,
      })
    }
    // A non-JSON 5xx comes from the dev-server proxy when the backend is down.
    if (response.status >= 500) throw backendUnreachable()
    throw new ApiError(`HTTP_${response.status}`, `Request failed (HTTP ${response.status}).`, {
      status: response.status,
    })
  }

  if (!schema) return data as T
  const parsed = schema.safeParse(data)
  if (!parsed.success) {
    console.warn('Unexpected response shape', parsed.error.issues.slice(0, 5))
    throw new ApiError('INVALID_RESPONSE', 'The backend answered in an unexpected format.', {
      hint: 'Make sure the frontend and backend come from the same version of the project.',
    })
  }
  return parsed.data
}

// ---------------------------------------------------------------- endpoints

export function getHealth(opts?: RequestOptions): Promise<Health> {
  return request('/api/health', { method: 'GET' }, HealthSchema, { timeoutMs: 6000, ...opts })
}

export function checkAI(opts?: RequestOptions): Promise<AICheck> {
  return request('/api/health/ai', { method: 'GET' }, AICheckSchema, { timeoutMs: 20_000, ...opts })
}

export interface AnalyzeCommon {
  personality: Personality
  context?: unknown
  demo?: boolean
  signal?: AbortSignal
}

function form(fields: Record<string, string | Blob | undefined>, files?: [string, Blob, string][]): FormData {
  const data = new FormData()
  for (const [key, value] of Object.entries(fields)) {
    if (value !== undefined) data.append(key, value)
  }
  for (const [key, blob, name] of files ?? []) data.append(key, blob, name)
  return data
}

function common(opts: AnalyzeCommon): Record<string, string | undefined> {
  return {
    personality: opts.personality,
    context: opts.context === undefined ? undefined : JSON.stringify(opts.context),
    demo: opts.demo ? 'true' : undefined,
  }
}

export function analyzeImage(image: Blob, opts: AnalyzeCommon): Promise<Report> {
  return request(
    '/api/analyze/image',
    { method: 'POST', body: form(common(opts), [['image', image, 'image.jpg']]) },
    ReportSchema,
    { signal: opts.signal, timeoutMs: 200_000 },
  )
}

export function analyzeFrame(
  image: Blob,
  opts: AnalyzeCommon & { scanId: string | null; trigger: Trigger },
): Promise<ScanAnalysis> {
  return request(
    '/api/analyze/frame',
    {
      method: 'POST',
      body: form({ ...common(opts), scan_id: opts.scanId ?? undefined, trigger: opts.trigger }, [
        ['image', image, 'frame.jpg'],
      ]),
    },
    ScanAnalysisSchema,
    { signal: opts.signal, timeoutMs: 120_000 },
  )
}

export function deepScan(
  image: Blob,
  opts: AnalyzeCommon & { scanId: string | null; trigger: 'deep_scan' | 'freeze' },
): Promise<ScanAnalysis> {
  return request(
    '/api/scan/deep',
    {
      method: 'POST',
      body: form({ ...common(opts), scan_id: opts.scanId ?? undefined, trigger: opts.trigger }, [
        ['image', image, 'deep.jpg'],
      ]),
    },
    ScanAnalysisSchema,
    { signal: opts.signal, timeoutMs: 200_000 },
  )
}

export function getScan(scanId: string, opts?: RequestOptions): Promise<ScanState> {
  return request(`/api/scan/${encodeURIComponent(scanId)}`, { method: 'GET' }, ScanStateSchema, opts)
}

export function deleteScan(scanId: string): Promise<void> {
  return request(`/api/scan/${encodeURIComponent(scanId)}`, { method: 'DELETE' }, null, { timeoutMs: 8000 })
}

export function analyzeVideoKeyframes(
  frames: Blob[],
  manifest: unknown,
  opts: Omit<AnalyzeCommon, 'context'>,
): Promise<VideoReport> {
  return request(
    '/api/analyze/video',
    {
      method: 'POST',
      body: form(
        { personality: opts.personality, demo: opts.demo ? 'true' : undefined, manifest: JSON.stringify(manifest) },
        frames.map((blob, i) => ['frames', blob, `keyframe-${i + 1}.jpg`] as [string, Blob, string]),
      ),
    },
    VideoReportSchema,
    { signal: opts.signal, timeoutMs: 300_000 },
  )
}

/** Server-side fallback: uploads the whole video with progress reporting. */
export function analyzeVideoFile(
  file: File,
  opts: Omit<AnalyzeCommon, 'context'> & { onUploadProgress?: (fraction: number) => void },
): Promise<VideoReport> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/api/analyze/video`)
    xhr.timeout = 600_000
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) opts.onUploadProgress?.(event.loaded / event.total)
    }
    xhr.onerror = () => reject(backendUnreachable())
    xhr.ontimeout = () => reject(new ApiError('REQUEST_TIMEOUT', 'The video analysis took too long.', { retryable: true }))
    xhr.onabort = () => reject(new DOMException('Aborted', 'AbortError'))
    xhr.onload = () => {
      let data: unknown = null
      try {
        data = JSON.parse(xhr.responseText)
      } catch {
        data = null
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        const parsed = VideoReportSchema.safeParse(data)
        if (parsed.success) resolve(parsed.data)
        else reject(new ApiError('INVALID_RESPONSE', 'The backend answered in an unexpected format.'))
        return
      }
      const err = ErrorBodySchema.safeParse((data as { error?: unknown } | null)?.error)
      if (err.success) {
        reject(new ApiError(err.data.code, err.data.message, { hint: err.data.hint, status: xhr.status }))
      } else if (xhr.status >= 500) {
        reject(backendUnreachable())
      } else {
        reject(new ApiError(`HTTP_${xhr.status}`, `Request failed (HTTP ${xhr.status}).`, { status: xhr.status }))
      }
    }
    opts.signal?.addEventListener('abort', () => xhr.abort(), { once: true })
    const data = new FormData()
    data.append('personality', opts.personality)
    if (opts.demo) data.append('demo', 'true')
    data.append('video', file, file.name || 'video')
    xhr.send(data)
  })
}

/** Normalise anything thrown into an ApiError for display. */
export function toApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error
  if (error instanceof Error) return new ApiError('CLIENT_ERROR', error.message)
  return new ApiError('CLIENT_ERROR', 'Something unexpected happened.')
}
