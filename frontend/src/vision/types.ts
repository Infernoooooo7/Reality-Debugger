/** Normalised box: 0..1 relative to the analysed frame, origin top-left. */
export interface NBox {
  x: number
  y: number
  w: number
  h: number
}

export interface Detection {
  label: string
  score: number
  box: NBox
}

export type TrackState = 'tentative' | 'confirmed' | 'lost'

export interface Track {
  id: number
  /** Display key such as "cup·03". */
  key: string
  label: string
  score: number
  box: NBox
  /** Velocity in normalised units per second. */
  vx: number
  vy: number
  hits: number
  misses: number
  state: TrackState
  firstSeen: number
  lastSeen: number
}

/** Compact appearance fingerprint used for scene-change detection. */
export interface Signature {
  hist: Float32Array // 4x4x4 RGB histogram (64 bins, sums to 1)
  grid: Float32Array // 16x12 luminance grid (0..1)
}

export interface FrameSignals {
  /** Mean absolute luminance change vs previous frame (0..1, scaled). */
  motion: number
  motionBox: NBox | null
  /** Appearance distance to the previous frame (0..1). */
  frameDelta: number
  /** Appearance distance to the last analysed "anchor" frame (0..1). */
  sceneDelta: number
  brightness: number
  contrast: number
  sharpness: number
}

export interface FrameResult {
  frameId: number
  timestamp: number
  tracks: Track[]
  detections: number
  signals: FrameSignals
  inferenceMs: number
  totalMs: number
}

export interface StillResult {
  detections: Detection[]
  signals: FrameSignals
  signature: Signature
  inferenceMs: number
}

export type Delegate = 'GPU' | 'CPU'
export type DelegatePreference = 'auto' | Delegate

/** Result of running the built-in self-test image through a delegate. */
export interface ProbeResult {
  delegate: Delegate
  ms: number
  label: string | null
  score: number
}

export interface EngineInfo {
  delegate: Delegate
  runtime: 'worker' | 'main-thread'
  model: string
  loadMs: number
  selfTest: ProbeResult
  probes: ProbeResult[]
}

// ---------------------------------------------------------------- messages

export interface InitPayload {
  modelUrl: string
  wasmLoaderUrl: string
  wasmBinaryUrl: string
  delegate: DelegatePreference
}

export type WorkerRequest =
  | { type: 'init'; payload: InitPayload }
  | { type: 'frame'; id: number; bitmap: ImageBitmap; timestamp: number }
  | { type: 'still'; id: number; bitmap: ImageBitmap }
  | { type: 'anchor' }
  | { type: 'reset' }

export type WorkerResponse =
  | { type: 'ready'; info: Omit<EngineInfo, 'runtime'> }
  | { type: 'init-error'; message: string }
  | { type: 'frame'; id: number; result: FrameResult }
  | { type: 'still'; id: number; result: StillResult }
  | { type: 'error'; id: number; message: string }
