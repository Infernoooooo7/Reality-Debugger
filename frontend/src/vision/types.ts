/** Normalised box: 0..1 relative to the analysed frame, origin top-left. */
export interface NBox {
  x: number
  y: number
  w: number
  h: number
}

export type DetectorRole = 'fast' | 'deep'

export interface Detection {
  label: string
  score: number
  box: NBox
}

/**
 * tentative: not yet confirmed (one detection); confirmed: matched in this frame;
 * recovered: matched again in this frame after being lost; occluded: not matched,
 * predicted box mostly covered by other tracks (probably hidden behind them);
 * lost: not matched, kept for the lost buffer. Terminated tracks are removed
 * (a 'left' event). Identities are not guaranteed to stay correct indefinitely.
 */
export type TrackState = 'tentative' | 'confirmed' | 'recovered' | 'occluded' | 'lost'

/** Tracks seen in the current frame. */
export function isVisible(t: { state: TrackState }): boolean {
  return t.state === 'confirmed' || t.state === 'recovered'
}

/** Tracks coasting on their prediction (not seen in this frame). */
export function isCoasting(t: { state: TrackState }): boolean {
  return t.state === 'lost' || t.state === 'occluded'
}
export type Movement = 'static' | 'moving' | 'unknown'

export interface Track {
  id: number
  /** Display key such as "cup·03". */
  key: string
  /** Id used in the scene model sent to the backend ("t3"). */
  sceneId: string
  label: string
  score: number
  /** Kalman-filtered box. */
  box: NBox
  /** Kalman velocity in normalised units per second. */
  vx: number
  vy: number
  /** Speed relative to the scene (camera motion compensated when CMC is on), frame units per second. */
  speed: number
  /** Image-space speed including camera motion, frame units per second. */
  apparentSpeed: number
  hits: number
  misses: number
  state: TrackState
  firstSeen: number
  lastSeen: number
  movement: Movement
  staticMs: number
  reversals: number
  /** Share of the box overlapped by other tracked boxes (0..1). */
  occlusion: number
  occludedMs: number
  /** The box touches the frame edge (object partly out of view). */
  truncated: boolean
  /** fast = fast detector only, deep = deep detector only (held), fused = both detectors. */
  source: 'fast' | 'deep' | 'fused'
  verified: boolean
}

export interface TrackEvent {
  kind: 'entered' | 'left' | 'recovered'
  at: number
  trackId: number
  sceneId: string
  label: string
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
  /** Appearance distance to the reference frame of the current view (0..1). */
  sceneDelta: number
  brightness: number
  contrast: number
  sharpness: number
  /** Global image motion since the previous frame (frame widths/heights); null on the first frame. */
  cameraShift: { dx: number; dy: number; confidence: number } | null
}

export interface FrameResult {
  frameId: number
  timestamp: number
  tracks: Track[]
  detections: number
  signals: FrameSignals
  /** Appearance fingerprint of the frame (video keyframe selection). */
  signature: Signature
  /** Increments whenever the camera settles on a substantially different view. */
  viewId: number
  events: TrackEvent[]
  inferenceMs: number
  totalMs: number
}

export interface StillResult {
  detections: Detection[]
  signals: FrameSignals
  signature: Signature
  inferenceMs: number
}

export interface FuseResult {
  verified: number
  relabeled: number
  added: number
  tracks: Track[]
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
  modelId: string
  /** Number of classes the model reports in its own metadata. */
  classes: number
  loadMs: number
  selfTest: ProbeResult
  probes: ProbeResult[]
}

// ---------------------------------------------------------------- messages

export interface InitPayload {
  modelUrl: string
  manifestUrl: string
  wasmLoaderUrl: string
  wasmBinaryUrl: string
  delegate: DelegatePreference
}

export type WorkerRequest =
  | { type: 'init'; payload: InitPayload }
  | { type: 'frame'; id: number; bitmap: ImageBitmap; timestamp: number }
  | { type: 'still'; id: number; bitmap: ImageBitmap }
  | { type: 'fuse'; id: number; detections: Detection[]; timestamp: number; holdMs: number }
  | { type: 'reset' }

export type WorkerResponse =
  | { type: 'ready'; info: Omit<EngineInfo, 'runtime'> }
  | { type: 'init-error'; message: string }
  | { type: 'frame'; id: number; result: FrameResult }
  | { type: 'still'; id: number; result: StillResult }
  | { type: 'fuse'; id: number; result: FuseResult }
  | { type: 'error'; id: number; message: string }
