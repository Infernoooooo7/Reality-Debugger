/**
 * Deep detector worker: YOLOX-S (ONNX) on onnxruntime-web, off the main
 * thread and separate from the fast detector so the live loop never waits
 * for it.
 *
 * Backend: WebGPU when the browser exposes a GPU adapter, otherwise
 * WebAssembly (multi-threaded when the page is cross-origin isolated). The
 * ONNX Runtime build for the chosen backend is imported dynamically, so only
 * that runtime is downloaded. The model is fetched once, verified against
 * the SHA-256 in its manifest and kept in Cache Storage.
 */
import type * as OrtModule from 'onnxruntime-web'
import type { DeepBackend, DeepInfo, DeepInitPayload, DeepRequest, DeepResponse, DeepResult } from './protocol'
import { decode, placement, toTensor, type YoloxParams } from './yolox'

type Ort = typeof OrtModule

interface WorkerScope {
  postMessage(message: DeepResponse, transfer?: Transferable[]): void
  onmessage: ((event: MessageEvent<DeepRequest>) => void) | null
  crossOriginIsolated?: boolean
}

const scope = self as unknown as WorkerScope
const CACHE = 'reality-debugger-models-v1'

let ort: Ort | null = null
let session: OrtModule.InferenceSession | null = null
let params: YoloxParams | null = null
let canvas: OffscreenCanvas | null = null
let ctx: OffscreenCanvasRenderingContext2D | null = null

function send(message: DeepResponse): void {
  scope.postMessage(message)
}

async function sha256Hex(bytes: Uint8Array): Promise<string | null> {
  if (typeof crypto === 'undefined' || !crypto.subtle) return null // not a secure context
  const digest = await crypto.subtle.digest('SHA-256', bytes as Uint8Array<ArrayBuffer>)
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('')
}

async function fetchModel(url: string, sha256: string): Promise<{ bytes: Uint8Array; cached: boolean }> {
  let cache: Cache | null = null
  try {
    if (typeof caches !== 'undefined') cache = await caches.open(CACHE)
  } catch {
    cache = null
  }
  const key = `${url}#sha256=${sha256}`
  if (cache) {
    const hit = await cache.match(key)
    if (hit) return { bytes: new Uint8Array(await hit.arrayBuffer()), cached: true }
  }
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(
      res.status === 404
        ? 'The deep detector model is not installed on this server (run: python tools/fetch_models.py --fetch).'
        : `Downloading the deep detector failed (HTTP ${res.status}).`,
    )
  }
  const total = Number(res.headers.get('content-length')) || 0
  const chunks: Uint8Array[] = []
  let loaded = 0
  const reader = res.body?.getReader()
  if (reader) {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value)
      loaded += value.length
      send({ type: 'progress', loaded, total })
    }
  } else {
    chunks.push(new Uint8Array(await res.arrayBuffer()))
  }
  const bytes = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0))
  let offset = 0
  for (const c of chunks) {
    bytes.set(c, offset)
    offset += c.length
  }
  const digest = await sha256Hex(bytes)
  if (digest && digest !== sha256) throw new Error('The deep detector model failed its SHA-256 check and was not used.')
  if (cache) {
    try {
      for (const request of await cache.keys()) if (request.url.split('#')[0] === url.split('#')[0]) await cache.delete(request)
      await cache.put(key, new Response(bytes, { headers: { 'content-type': 'application/octet-stream' } }))
    } catch {
      /* storage full or unavailable: the HTTP cache still helps */
    }
  }
  return { bytes, cached: false }
}

/** WebGPU slower than this on the warm-up run triggers a WebAssembly comparison. */
const SLOW_WEBGPU_WARMUP_MS = 3000

/**
 * A hardware WebGPU adapter, or a reason not to use WebGPU. Software adapters
 * (SwiftShader, llvmpipe) emulate the GPU on the CPU and were ~55x slower than
 * WebAssembly in our measurement (docs/benchmarks/browser_bench.json).
 */
async function webGpuAdapter(): Promise<{ ok: boolean; reason: string | null }> {
  type Adapter = { info?: { vendor?: string; architecture?: string; description?: string; isFallbackAdapter?: boolean }; isFallbackAdapter?: boolean }
  const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<Adapter | null> } }).gpu
  if (!gpu) return { ok: false, reason: 'WebGPU not supported by this browser' }
  try {
    const adapter = await gpu.requestAdapter()
    if (!adapter) return { ok: false, reason: 'no WebGPU adapter' }
    const info = adapter.info ?? {}
    const text = `${info.vendor ?? ''} ${info.architecture ?? ''} ${info.description ?? ''}`.toLowerCase()
    if (info.isFallbackAdapter || adapter.isFallbackAdapter || /swiftshader|llvmpipe|software|lavapipe/.test(text)) {
      return { ok: false, reason: `software WebGPU adapter (${text.trim() || 'fallback'}) - WebAssembly is faster` }
    }
    return { ok: true, reason: null }
  } catch (error) {
    return { ok: false, reason: `WebGPU adapter error: ${error instanceof Error ? error.message : String(error)}` }
  }
}

async function loadRuntime(backend: DeepBackend): Promise<Ort> {
  const mod = backend === 'webgpu' ? await import('onnxruntime-web/webgpu') : await import('onnxruntime-web/wasm')
  return mod as unknown as Ort
}

function prepare(bitmap: ImageBitmap, p: YoloxParams): { tensor: Float32Array; place: ReturnType<typeof placement> } {
  const size = p.inputSize
  if (!canvas || !ctx) {
    canvas = new OffscreenCanvas(size, size)
    ctx = canvas.getContext('2d', { willReadFrequently: true })
    if (!ctx) throw new Error('2D canvas unavailable in the deep detector worker')
  }
  const place = placement(bitmap.width, bitmap.height, size)
  ctx.fillStyle = `rgb(${p.padValue},${p.padValue},${p.padValue})`
  ctx.fillRect(0, 0, size, size)
  ctx.drawImage(bitmap, 0, 0, Math.round(bitmap.width * place.ratio), Math.round(bitmap.height * place.ratio))
  const rgba = ctx.getImageData(0, 0, size, size).data
  return { tensor: toTensor(rgba, size, p.channelOrder), place }
}

async function run(tensorData: Float32Array): Promise<Float32Array> {
  if (!ort || !session || !params) throw new Error('Deep detector not initialised')
  const size = params.inputSize
  const input = new ort.Tensor('float32', tensorData, [1, 3, size, size])
  const outputs = await session.run({ [session.inputNames[0]!]: input })
  const output = outputs[session.outputNames[0]!]!
  const data = output.data as Float32Array
  for (const tensor of Object.values(outputs)) tensor.dispose?.()
  input.dispose?.()
  return data
}

async function createSession(backend: DeepBackend, model: Uint8Array, threads: number): Promise<void> {
  ort = await loadRuntime(backend)
  ort.env.logLevel = 'error'
  ort.env.wasm.numThreads = scope.crossOriginIsolated ? Math.max(1, Math.min(threads, navigator.hardwareConcurrency || 2)) : 1
  session = await ort.InferenceSession.create(model, {
    executionProviders: backend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'],
    graphOptimizationLevel: 'all',
  })
}

async function init(payload: DeepInitPayload): Promise<DeepInfo> {
  params = payload.params
  const started = performance.now()
  const { bytes, cached } = await fetchModel(payload.modelUrl, payload.sha256)
  const downloadMs = performance.now() - started

  let backend: DeepBackend = 'wasm'
  let fallbackReason: string | null = null
  if (payload.backend !== 'wasm') {
    const gpu = await webGpuAdapter()
    if (gpu.ok) backend = 'webgpu'
    else fallbackReason = gpu.reason
  }
  const sessionStart = performance.now()
  let warmupMs = 0
  const blank = new Float32Array(3 * params.inputSize * params.inputSize).fill(params.padValue)
  // Time the second run: the first one includes one-time kernel/shader compilation.
  const warmUp = async () => {
    await run(blank)
    const warm = performance.now()
    await run(blank)
    return performance.now() - warm
  }
  try {
    await createSession(backend, bytes, payload.threads)
    warmupMs = await warmUp()
  } catch (error) {
    if (backend !== 'webgpu') throw error
    fallbackReason = `WebGPU failed (${error instanceof Error ? error.message : String(error)})`
    backend = 'wasm'
    await session?.release().catch(() => undefined)
    await createSession('wasm', bytes, payload.threads)
    warmupMs = await warmUp()
  }
  if (backend === 'webgpu' && warmupMs > SLOW_WEBGPU_WARMUP_MS && payload.backend === 'auto') {
    // Measured, not assumed: keep whichever runtime is faster on this device.
    const gpuSession = session
    const gpuOrt = ort
    await createSession('wasm', bytes, payload.threads)
    const wasmMs = await warmUp()
    if (wasmMs < warmupMs) {
      await gpuSession?.release().catch(() => undefined)
      fallbackReason = `WebGPU warm-up ${Math.round(warmupMs)} ms vs WebAssembly ${Math.round(wasmMs)} ms`
      backend = 'wasm'
      warmupMs = wasmMs
    } else {
      await session?.release().catch(() => undefined)
      session = gpuSession
      ort = gpuOrt
    }
  }
  return {
    model: payload.modelName,
    backend,
    threads: ort?.env.wasm.numThreads ?? 1,
    crossOriginIsolated: Boolean(scope.crossOriginIsolated),
    bytes: bytes.length,
    cached,
    downloadMs: Math.round(downloadMs),
    sessionMs: Math.round(performance.now() - sessionStart - warmupMs),
    warmupMs: Math.round(warmupMs),
    fallbackReason,
  }
}

async function detect(bitmap: ImageBitmap): Promise<DeepResult> {
  if (!params) throw new Error('Deep detector not initialised')
  const t0 = performance.now()
  const { tensor, place } = prepare(bitmap, params)
  const t1 = performance.now()
  const output = await run(tensor)
  const t2 = performance.now()
  const detections = decode(output, params, place)
  const t3 = performance.now()
  const r = (n: number) => Math.round(n * 10) / 10
  return { detections, preprocessMs: r(t1 - t0), inferenceMs: r(t2 - t1), postprocessMs: r(t3 - t2), totalMs: r(t3 - t0) }
}

let queue: Promise<unknown> = Promise.resolve()

scope.onmessage = (event) => {
  const message = event.data
  if (message.type === 'init') {
    init(message.payload).then(
      (info) => send({ type: 'ready', info }),
      (error: unknown) => send({ type: 'init-error', message: error instanceof Error ? error.message : String(error) }),
    )
    return
  }
  // One inference at a time; requests queue in order.
  queue = queue.then(async () => {
    try {
      const result = await detect(message.bitmap)
      send({ type: 'detect', id: message.id, result })
    } catch (error) {
      send({ type: 'error', id: message.id, message: error instanceof Error ? error.message : String(error) })
    } finally {
      message.bitmap.close()
    }
  })
}
