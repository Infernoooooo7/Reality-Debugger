/**
 * Live Scan session: camera → local vision (every frame) → director →
 * selective vision-model analysis → finding lifecycle → overlay + HUD.
 *
 * Plain TypeScript (no React) so the per-frame path never triggers renders;
 * React reads a throttled snapshot from the `useLive` store.
 */
import { analyzeFrame, ApiError, deepScan, deleteScan, toApiError } from '../lib/api'
import type { Finding, LifecycleEvent, ScanState } from '../lib/schemas'
import { useSettings } from '../state/settings'
import { useSystem } from '../state/system'
import { vision } from '../vision/client'
import { gap, iou, union } from '../vision/geometry'
import { spatialRelations, type Relation } from '../vision/relations'
import type { FrameResult, NBox, Track } from '../vision/types'
import { CameraController } from './camera'
import { FrameGrabber, JpegCapturer } from './capture'
import { ScanDirector, type Decision } from './director'
import { cocoLabels } from './labels'
import { OverlayRenderer, type Anchor } from './overlay'
import { useLive, type Living } from './store'

const BLOCKING_CODES = new Set(['AI_AUTH_FAILED', 'AI_NOT_CONFIGURED', 'AI_MODEL_NOT_FOUND', 'AI_REQUEST_REJECTED', 'INVALID_RESPONSE'])

function round(n: number, digits = 3): number {
  const f = 10 ** digits
  return Math.round(n * f) / f
}

function roundBox(b: NBox): NBox {
  return { x: round(b.x), y: round(b.y), w: round(b.w), h: round(b.h) }
}

export class LiveSession {
  readonly camera: CameraController
  private readonly grabber = new FrameGrabber()
  private readonly jpeg = new JpegCapturer()
  private readonly overlay: OverlayRenderer
  private readonly director: ScanDirector
  private raf = 0
  private running = false
  private processing = false
  private lastFrameAt = 0
  private lastHudAt = 0
  private fpsEma = 0
  private lastResultAt = 0
  private lastResult: FrameResult | null = null
  private relations: Relation[] = []
  private scanId: string | null = null
  private findings: Finding[] = []
  private disposed = false
  private failures = 0
  private forceDemo: boolean
  private motionTrace: number[] = []
  private readonly onVisibility = () => {
    if (document.hidden) this.lastFrameAt = performance.now() + 500
  }

  private readonly video: HTMLVideoElement

  constructor(video: HTMLVideoElement, canvas: HTMLCanvasElement) {
    this.video = video
    this.camera = new CameraController(video)
    this.overlay = new OverlayRenderer(canvas, video)
    const settings = useSettings.getState()
    this.forceDemo = settings.forceDemo
    this.director = new ScanDirector({ intervalMs: settings.interval * 1000 })
    document.addEventListener('visibilitychange', this.onVisibility)
  }

  private get store() {
    return useLive.getState()
  }

  // ---------------------------------------------------------------- lifecycle

  async boot(): Promise<void> {
    const { set } = this.store
    set({ phase: 'booting', bootStep: 'Requesting camera', fatal: null })
    try {
      await this.camera.start('environment')
    } catch (error) {
      if (this.disposed) return
      set({ phase: 'error', fatal: toApiError(error) })
      return
    }
    if (this.disposed) return
    set({ bootStep: 'Loading on-device vision model' })
    try {
      await vision.init()
    } catch (error) {
      if (this.disposed) return
      set({
        phase: 'error',
        fatal: new ApiError('VISION_FAILED', toApiError(error).message, {
          hint: 'Reload the page. If it keeps failing, set "Vision backend" to CPU in the home screen settings.',
        }),
      })
      return
    }
    if (this.disposed) return
    vision.resetTracking()
    this.director.reset()
    this.overlay.start()
    set({ phase: 'running', bootStep: 'Ready' })
    this.running = true
    this.loop()
  }

  dispose(): void {
    this.disposed = true
    this.running = false
    cancelAnimationFrame(this.raf)
    this.overlay.stop()
    this.camera.stop()
    document.removeEventListener('visibilitychange', this.onVisibility)
    const deep = this.store.deep
    if (deep) URL.revokeObjectURL(deep.imageUrl)
  }

  setForceDemo(on: boolean): void {
    this.forceDemo = on
    if (this.store.ai.state === 'blocked') {
      this.director.markDone(true)
      this.store.set({ ai: { ...this.store.ai, state: 'idle', error: null } })
    }
  }

  configure(intervalSeconds: number): void {
    this.director.configure({ intervalMs: intervalSeconds * 1000 })
  }

  // ---------------------------------------------------------------- frame loop

  private loop = (): void => {
    if (!this.running || this.disposed) return
    this.raf = requestAnimationFrame(this.loop)
    if (this.processing || document.hidden) return
    const now = performance.now()
    const interval = 1000 / useSettings.getState().maxFps
    if (now - this.lastFrameAt < interval) return
    this.lastFrameAt = now
    void this.processFrame(now)
  }

  private async processFrame(now: number): Promise<void> {
    this.processing = true
    try {
      const bitmap = await this.grabber.bitmap(this.video, 480)
      if (!bitmap) return
      const result = await vision.processFrame(bitmap, now)
      if (!this.running || this.disposed) return
      this.onResult(result)
    } catch (error) {
      console.warn('[live] frame failed', error)
    } finally {
      this.processing = false
    }
  }

  private onResult(result: FrameResult): void {
    const now = performance.now()
    if (this.lastResultAt) {
      const instant = 1000 / Math.max(1, now - this.lastResultAt)
      this.fpsEma = this.fpsEma ? this.fpsEma * 0.85 + instant * 0.15 : instant
    }
    this.lastResultAt = now
    this.lastResult = result
    this.relations = spatialRelations(result.tracks)
    this.overlay.setTracks(result.tracks, result.signals.motion > 0.08 ? result.signals.motionBox : null)
    this.overlay.setAnchors(this.anchors(result.tracks, result.signals.sceneDelta))
    this.motionTrace.push(result.signals.motion)
    if (this.motionTrace.length > 64) this.motionTrace.shift()

    const { decision, anomaly } = this.director.observe(result, this.relations, now)
    if (decision) void this.analyze(decision, result)

    if (now - this.lastHudAt > 200) {
      this.lastHudAt = now
      this.store.set({
        anomaly,
        hud: {
          fps: round(this.fpsEma, 1),
          inferenceMs: result.inferenceMs,
          objects: result.tracks.filter((t) => t.state !== 'tentative').length,
          tentative: result.tracks.filter((t) => t.state === 'tentative').length,
          motion: result.signals.motion,
          sceneDelta: result.signals.sceneDelta,
          brightness: result.signals.brightness,
          sharpness: result.signals.sharpness,
          motionTrace: [...this.motionTrace],
        },
      })
      this.refreshLiving(anomaly)
    }
  }

  /** Anchor each open finding to live tracks when possible, else its static box. */
  private anchors(tracks: Track[], sceneDelta: number): Anchor[] {
    const anchors: Anchor[] = []
    const usable = tracks.filter((t) => t.state !== 'tentative')
    for (const finding of this.findings) {
      if (finding.status === 'RESOLVED') continue
      const labels = cocoLabels(finding.related_objects)
      const candidates = usable.filter((t) => labels.includes(t.label))
      if (candidates.length) {
        const near = finding.box
          ? candidates.filter((t) => iou(t.box, finding.box!) > 0.03 || gap(t.box, finding.box!) < 0.08)
          : candidates
        const box = union((near.length ? near : candidates).slice(0, 3).map((t) => t.box))
        if (box) anchors.push({ finding, box, live: true })
      } else if (finding.box && !finding.out_of_view && sceneDelta < 0.22) {
        anchors.push({ finding, box: finding.box, live: false })
      }
    }
    return anchors
  }

  // ---------------------------------------------------------------- living state

  private refreshLiving(anomaly: string | null = this.store.anomaly): void {
    const { living, ai, phase, scan } = this.store
    const now = performance.now()
    if (now < living.until) return
    let state: Living
    let detail: string | null = null
    if (phase === 'paused' || phase === 'deep') state = 'FROZEN'
    else if (ai.state === 'analyzing') {
      state = 'INVESTIGATING'
      detail = ai.detail
    } else if (anomaly) {
      state = 'ANOMALY'
      detail = anomaly
    } else if (scan && scan.counts.active_bugs > 0) {
      state = 'MONITORING'
      const top = scan.findings.find((f) => f.status !== 'RESOLVED' && f.severity !== 'INFO')
      detail = top ? `top: ${top.id} · ${top.title}` : null
    } else state = 'STABLE'
    if (state !== living.state || detail !== living.detail) this.store.set({ living: { state, detail, until: 0 } })
  }

  private pulseLiving(state: Living, detail: string, holdMs: number): void {
    this.store.set({ living: { state, detail, until: performance.now() + holdMs } })
  }

  // ---------------------------------------------------------------- AI analysis

  private context(result: FrameResult | null, detail?: string) {
    // Only objects matched in this frame: a 'lost' track is coasting on its
    // prediction and may already have left the scene (e.g. after a hard cut).
    const tracks = (result?.tracks ?? []).filter((t) => t.state === 'confirmed').slice(0, 30)
    return {
      detector: vision.getSnapshot().info?.model ?? 'on-device detector',
      objects: tracks.map((t) => ({ track_id: t.key, label: t.label, confidence: round(t.score, 2), box: roundBox(t.box) })),
      motion: result ? round(result.signals.motion) : null,
      scene_change: result ? round(result.signals.sceneDelta) : null,
      brightness: result ? round(result.signals.brightness) : null,
      sharpness: result ? round(result.signals.sharpness) : null,
      relationships: this.relations.map((r) => r.text).slice(0, 10),
      trigger_detail: detail ?? null,
      fps: round(this.fpsEma, 1),
    }
  }

  private async analyze(decision: Decision, result: FrameResult): Promise<void> {
    const started = performance.now()
    this.director.markSent(decision, result, this.relations, started)
    vision.setAnchor()
    this.store.set({
      ai: { ...this.store.ai, state: 'analyzing', startedAt: Date.now(), lastTrigger: decision.trigger, detail: decision.detail, error: null },
    })
    this.overlay.setInvestigating(true)
    this.refreshLiving()
    try {
      const shot = await this.jpeg.capture(this.video, 1280, 0.82)
      const response = await analyzeFrame(shot.blob, {
        scanId: this.scanId,
        trigger: decision.trigger,
        personality: useSettings.getState().personality,
        context: this.context(result, decision.detail),
        demo: this.forceDemo,
      })
      if (this.disposed) return
      this.failures = 0
      this.director.markDone(true)
      this.applyScan(response.scan, response.events)
      this.store.set({
        ai: {
          ...this.store.ai,
          state: 'idle',
          lastLatencyMs: Math.round(performance.now() - started),
          analyses: this.store.ai.analyses + 1,
          error: null,
        },
      })
    } catch (error) {
      if (this.disposed) return
      const err = toApiError(error)
      this.failures += 1
      if (BLOCKING_CODES.has(err.code)) {
        this.director.markDone(false, 10 * 60_000)
        this.store.set({ ai: { ...this.store.ai, state: 'blocked', error: err } })
      } else if (err.code === 'SCAN_RATE_LIMITED' || err.code === 'AI_RATE_LIMITED') {
        this.director.markDone(false, 15_000)
        this.store.set({ ai: { ...this.store.ai, state: 'budget', error: err } })
      } else {
        this.director.markDone(false, Math.min(60_000, 4000 * 2 ** (this.failures - 1)))
        this.store.set({ ai: { ...this.store.ai, state: 'error', error: err } })
      }
    } finally {
      this.overlay.setInvestigating(false)
      this.refreshLiving()
    }
  }

  private applyScan(scan: ScanState, events: LifecycleEvent[]): void {
    const previous = new Map(this.findings.map((f) => [f.id, f]))
    this.scanId = scan.scan_id
    this.findings = scan.findings
    this.director.syncFindings(scan.findings)
    for (const event of events) {
      if (event.type === 'RESOLVED') {
        const before = previous.get(event.finding_id)
        const box = before?.box ?? scan.findings.find((f) => f.id === event.finding_id)?.box
        if (box) this.overlay.flashResolved(event.finding_id, box)
      }
    }
    const recent = [...events, ...this.store.events].slice(0, 30)
    this.store.set({ scan, events: recent })
    const resolved = events.find((e) => e.type === 'RESOLVED')
    const confirmed = events.find((e) => e.type === 'CONFIRMED')
    const discovered = events.find((e) => e.type === 'DISCOVERED' || e.type === 'REOPENED')
    if (resolved) this.pulseLiving('RESOLVED', `${resolved.finding_id} · ${resolved.title}`, 2800)
    else if (confirmed) this.pulseLiving('CONFIRMED', `${confirmed.finding_id} · ${confirmed.title}`, 2800)
    else if (discovered) this.pulseLiving('DISCOVERED', `${discovered.finding_id} · ${discovered.title}`, 2400)
    if (this.lastResult) this.overlay.setAnchors(this.anchors(this.lastResult.tracks, this.lastResult.signals.sceneDelta))
  }

  // ---------------------------------------------------------------- controls

  pause(): void {
    if (this.store.phase !== 'running') return
    this.running = false
    cancelAnimationFrame(this.raf)
    this.camera.pause()
    this.overlay.setFrozen(true)
    this.director.setPaused(true)
    this.store.set({ phase: 'paused', anomaly: null })
    this.refreshLiving(null)
  }

  async resume(): Promise<void> {
    if (this.store.phase !== 'paused') return
    await this.camera.resume()
    this.overlay.setFrozen(false)
    this.director.setPaused(false)
    this.store.set({ phase: 'running' })
    this.running = true
    this.lastFrameAt = 0
    this.loop()
    this.refreshLiving()
  }

  async switchCamera(): Promise<void> {
    const wasRunning = this.running
    this.running = false
    cancelAnimationFrame(this.raf)
    try {
      await this.camera.switch()
      vision.resetTracking()
      this.overlay.clear()
    } catch (error) {
      this.store.set({ phase: 'error', fatal: toApiError(error) })
      return
    }
    if (wasRunning) {
      this.running = true
      this.loop()
    }
  }

  /** Freeze the current frame and run the most thorough analysis available. */
  async deepScan(): Promise<void> {
    const store = this.store
    if (store.deep?.status === 'analyzing' || store.phase === 'booting' || store.phase === 'error') return
    const frozenBefore = store.phase === 'paused'
    this.running = false
    cancelAnimationFrame(this.raf)
    this.camera.pause()
    this.overlay.setFrozen(true)
    this.director.setPaused(true)
    let shot: { blob: Blob; width: number; height: number }
    try {
      shot = await this.jpeg.capture(this.video, 1600, 0.88)
    } catch (error) {
      store.set({ deep: null })
      void this.resumeAfterDeep()
      useLive.getState().set({ ai: { ...this.store.ai, state: 'error', error: toApiError(error) } })
      return
    }
    const imageUrl = URL.createObjectURL(shot.blob)
    this.store.set({
      phase: 'deep',
      deep: { status: 'analyzing', imageUrl, width: shot.width, height: shot.height, startedAt: Date.now(), report: null, error: null },
    })
    this.refreshLiving()
    try {
      const response = await deepScan(shot.blob, {
        scanId: this.scanId,
        trigger: frozenBefore ? 'freeze' : 'deep_scan',
        personality: useSettings.getState().personality,
        context: this.context(this.lastResult, 'user requested a deep scan of the frozen frame'),
        demo: this.forceDemo,
      })
      if (this.disposed) return
      this.applyScan(response.scan, response.events)
      useSystem.getState().record(response.report)
      const deep = this.store.deep
      if (deep) this.store.set({ deep: { ...deep, status: 'done', report: response.report } })
    } catch (error) {
      if (this.disposed) return
      const deep = this.store.deep
      if (deep) this.store.set({ deep: { ...deep, status: 'error', error: toApiError(error) } })
    }
  }

  async resumeAfterDeep(): Promise<void> {
    const deep = this.store.deep
    if (deep) URL.revokeObjectURL(deep.imageUrl)
    this.store.set({ deep: null, phase: 'running' })
    await this.camera.resume()
    this.overlay.setFrozen(false)
    this.director.setPaused(false)
    this.running = true
    this.lastFrameAt = 0
    this.loop()
    this.refreshLiving()
  }

  /** Privacy: forget this scan on the backend and locally. */
  async clearSession(): Promise<void> {
    const id = this.scanId
    this.scanId = null
    this.findings = []
    this.director.reset()
    vision.resetTracking()
    this.overlay.clear()
    this.store.set({ scan: null, events: [], ai: { ...this.store.ai, analyses: 0, error: null, state: 'idle' } })
    if (id) await deleteScan(id).catch(() => undefined)
  }

  get currentScanId(): string | null {
    return this.scanId
  }
}
