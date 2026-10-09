/**
 * Live Scan session.
 *
 *   camera -> fast detector + ByteTrack + pixel signals (every frame, worker)
 *          -> scene model -> backend local diagnostic engine (/scan/observe,
 *             numbers only, every ~1.5 s) -> finding lifecycle -> overlay + HUD
 *
 * The optional AI layer is consulted only when the backend suggests it (a
 * confirmed finding, a new view, an ambiguous scene) or the user asks (Deep
 * Scan, Explain); only then is a frame captured and uploaded. Without an AI
 * key nothing but the scene model ever leaves the device.
 *
 * Plain TypeScript (no React) so the per-frame path never triggers renders;
 * React reads a throttled snapshot from the `useLive` store.
 */
import { TEMPORAL, VISION } from '../config'
import { analyzeFrame, ApiError, deepScan, deleteScan, observeScene, toApiError } from '../lib/api'
import type { AISuggestion, Finding, LifecycleEvent, ScanAnalysis, ScanState, Trigger } from '../lib/schemas'
import { useSettings } from '../state/settings'
import { aiEnabled, useSystem } from '../state/system'
import { vision } from '../vision/client'
import { deepDetector } from '../vision/deep/client'
import { union } from '../vision/geometry'
import { spatialRelations } from '../vision/relations'
import { liveScene, type SceneEventPayload, type ScenePayload } from '../vision/scene'
import type { FrameResult, Track } from '../vision/types'
import { CameraController } from './camera'
import { FrameGrabber, JpegCapturer } from './capture'
import { OverlayRenderer, type Anchor } from './overlay'
import { useLive, type Living } from './store'

function round(n: number, digits = 3): number {
  const f = 10 ** digits
  return Math.round(n * f) / f
}

/** Compact fingerprint of what an observation would report (decides "changed"). */
function sceneKey(scene: ScenePayload): string {
  const objects = scene.objects
    .map((o) => `${o.id}:${o.label}:${Math.round(o.box.x * 20)},${Math.round(o.box.y * 20)},${Math.round(o.box.w * 20)}:${o.movement}`)
    .sort()
    .join('|')
  const s = scene.signals
  return `${scene.view_id}#${objects}#${Math.round((s.brightness ?? 0) * 10)}:${Math.round((s.sharpness ?? 0) * 10)}`
}

export class LiveSession {
  readonly camera: CameraController
  private readonly grabber = new FrameGrabber()
  private readonly deepGrabber = new FrameGrabber()
  private readonly jpeg = new JpegCapturer()
  private readonly overlay: OverlayRenderer
  private raf = 0
  private running = false
  private processing = false
  private lastFrameAt = 0
  private lastHudAt = 0
  private fpsEma = 0
  private lastResultAt = 0
  private lastResult: FrameResult | null = null
  private sessionStart = performance.now()
  private scanId: string | null = null
  private findings: Finding[] = []
  private disposed = false
  private motionTrace: number[] = []
  private pendingEvents: SceneEventPayload[] = []
  private lastViewId = 0
  private observing = false
  private lastObserveAt = 0
  private lastObservedKey = ''
  private suggestion: (AISuggestion & { since: number }) | null = null
  private aiBusy = false
  private deepBusy = false
  private lastDeepCheckAt = 0
  private observeFailures = 0
  private readonly onVisibility = () => {
    if (document.hidden) this.lastFrameAt = performance.now() + 500
  }

  private readonly video: HTMLVideoElement

  constructor(video: HTMLVideoElement, canvas: HTMLCanvasElement) {
    this.video = video
    this.camera = new CameraController(video)
    this.overlay = new OverlayRenderer(canvas, video)
    document.addEventListener('visibilitychange', this.onVisibility)
  }

  private get store() {
    return useLive.getState()
  }

  private get aiOn(): boolean {
    return aiEnabled(useSystem.getState().health)
  }

  private atMs(now = performance.now()): number {
    return Math.max(0, now - this.sessionStart)
  }

  // ---------------------------------------------------------------- lifecycle

  async boot(): Promise<void> {
    const { set } = this.store
    set({ phase: 'booting', bootStep: 'Requesting camera', fatal: null })
    void useSystem.getState().refreshHealth()
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
    this.sessionStart = performance.now()
    this.overlay.start()
    this.syncAIState()
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

  /** Reflect the backend's AI configuration (off is a normal, calm state). */
  syncAIState(): void {
    const health = useSystem.getState().health
    const enabled = this.aiOn
    const ai = this.store.ai
    const state = !enabled ? 'off' : ai.state === 'off' ? (health?.ai.state === 'unavailable' ? 'unavailable' : 'idle') : ai.state
    this.store.set({ ai: { ...ai, enabled, provider: health?.ai.provider ?? null, state } })
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
      const bitmap = await this.grabber.bitmap(this.video, VISION.fast.grabWidth)
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
    for (const e of result.events) {
      this.pendingEvents.push({ kind: e.kind, at_ms: Math.round(this.atMs(e.at)), object_id: e.sceneId, label: e.label })
    }
    if (result.viewId !== this.lastViewId) {
      this.pendingEvents.push({ kind: 'scene_change', at_ms: Math.round(this.atMs(now)) })
      this.lastViewId = result.viewId
    }
    if (this.pendingEvents.length > 60) this.pendingEvents.splice(0, this.pendingEvents.length - 60)

    const relations = spatialRelations(result.tracks)
    this.overlay.setTracks(result.tracks, result.signals.motion > 0.08 ? result.signals.motionBox : null)
    this.overlay.setAnchors(this.anchors(result.tracks, result.viewId))
    this.motionTrace.push(result.signals.motion)
    if (this.motionTrace.length > 64) this.motionTrace.shift()

    this.maybeObserve(now, result)
    this.maybeAskAI(now, result)
    this.maybeDeepCheck(now)

    if (now - this.lastHudAt > 200) {
      this.lastHudAt = now
      const visible = result.tracks.filter((t) => t.state === 'confirmed')
      this.store.set({
        hud: {
          fps: round(this.fpsEma, 1),
          inferenceMs: result.inferenceMs,
          objects: visible.length,
          tentative: result.tracks.filter((t) => t.state === 'tentative').length,
          verified: visible.filter((t) => t.verified).length,
          relations: relations.length,
          motion: result.signals.motion,
          sceneDelta: result.signals.sceneDelta,
          viewId: result.viewId,
          brightness: result.signals.brightness,
          sharpness: result.signals.sharpness,
          motionTrace: [...this.motionTrace],
        },
        dev: { ...this.store.dev, fastMs: result.inferenceMs, frameMs: result.totalMs },
      })
      this.refreshLiving()
    }
  }

  /** Anchor open findings to the live tracks they involve; else their last box (same view only). */
  private anchors(tracks: Track[], viewId: number): Anchor[] {
    const anchors: Anchor[] = []
    const usable = tracks.filter((t) => t.state !== 'tentative')
    for (const finding of this.findings) {
      if (finding.status === 'RESOLVED') continue
      const involved = finding.object_ids.length ? usable.filter((t) => finding.object_ids.includes(t.sceneId)) : []
      const box = union(involved.map((t) => t.box))
      if (box) anchors.push({ finding, box, live: true })
      else if (finding.box && !finding.out_of_view && viewId === this.lastViewId && finding.source === 'local') {
        anchors.push({ finding, box: finding.box, live: false })
      }
    }
    return anchors
  }

  private scene(result: FrameResult, events: SceneEventPayload[]): ScenePayload {
    const detectors = [vision.getSnapshot().info?.modelId ?? VISION.fast.model]
    if (deepDetector.getSnapshot().runs > 0) detectors.push(VISION.deep.model)
    return liveScene(result, {
      atMs: this.atMs(result.timestamp),
      width: this.video.videoWidth || null,
      height: this.video.videoHeight || null,
      fps: this.fpsEma || null,
      events,
      detectors,
    })
  }

  // ---------------------------------------------------------------- local observations

  private maybeObserve(now: number, result: FrameResult): void {
    if (this.observing || !this.running) return
    const sinceLast = now - this.lastObserveAt
    if (sinceLast < TEMPORAL.observe.intervalMs) return
    const scene = this.scene(result, this.pendingEvents)
    const key = sceneKey(scene)
    const changed = key !== this.lastObservedKey || this.pendingEvents.length > 0
    if (!changed && sinceLast < TEMPORAL.observe.heartbeatMs) return
    // Back off while the backend is unreachable.
    if (this.observeFailures && sinceLast < Math.min(30_000, 2000 * 2 ** this.observeFailures)) return
    void this.observe(scene, key)
  }

  private async observe(scene: ScenePayload, key: string): Promise<void> {
    this.observing = true
    this.lastObserveAt = performance.now()
    const events = this.pendingEvents
    this.pendingEvents = []
    const started = performance.now()
    try {
      const response = await observeScene(scene, { scanId: this.scanId, personality: useSettings.getState().personality })
      if (this.disposed) return
      this.observeFailures = 0
      this.lastObservedKey = key
      const ms = Math.round(performance.now() - started)
      const dev = this.store.dev
      this.store.set({
        dev: { ...dev, observeMs: ms, observations: dev.observations + 1, observePayloadBytes: JSON.stringify(scene).length },
      })
      this.applyScan(response.scan, response.events)
      if (response.ai_suggestion && this.aiOn && !this.suggestion && !this.aiBusy) {
        this.suggestion = { ...response.ai_suggestion, since: performance.now() }
        this.store.set({ ai: { ...this.store.ai, state: 'waiting', detail: response.ai_suggestion.reason }, anomaly: response.ai_suggestion.reason })
      }
    } catch (error) {
      if (this.disposed) return
      this.pendingEvents = [...events, ...this.pendingEvents].slice(-60)
      this.observeFailures += 1
      const err = toApiError(error)
      if (this.observeFailures === 1) console.warn('[live] observation failed', err.code, err.message)
    } finally {
      this.observing = false
    }
  }

  // ---------------------------------------------------------------- optional AI reasoning

  /** Send the frame once it is settled (or after maxSettleWaitMs). */
  private maybeAskAI(now: number, result: FrameResult): void {
    const pending = this.suggestion
    if (!pending || this.aiBusy || !this.running) return
    const settled = result.signals.motion < TEMPORAL.sceneChange.settleMotion && !result.tracks.some((t) => t.state === 'lost')
    if (!settled && now - pending.since < TEMPORAL.sceneChange.maxSettleWaitMs) return
    this.suggestion = null
    void this.askAI(pending.trigger, result, null)
  }

  /** User asked: explain one finding (AI reasoning on the current frame). */
  async explain(findingId: string): Promise<void> {
    if (!this.aiOn || this.aiBusy || !this.lastResult) return
    this.store.set({ explaining: findingId })
    try {
      await this.askAI('user_explain', this.lastResult, findingId)
    } finally {
      this.store.set({ explaining: null })
    }
  }

  private async askAI(trigger: Trigger, result: FrameResult, focus: string | null): Promise<void> {
    if (!this.aiOn || this.disposed) return
    this.aiBusy = true
    const started = performance.now()
    this.store.set({
      ai: { ...this.store.ai, state: 'analyzing', startedAt: Date.now(), lastTrigger: trigger, error: null },
      anomaly: null,
    })
    this.overlay.setInvestigating(true)
    this.refreshLiving()
    try {
      const shot = await this.jpeg.capture(this.video, VISION.capture.aiFrameMaxEdge, VISION.capture.aiFrameQuality)
      const response = await analyzeFrame(shot.blob, {
        scanId: this.scanId,
        trigger,
        scene: this.scene(result, []),
        personality: useSettings.getState().personality,
        focus,
      })
      if (this.disposed) return
      this.applyResponse(response, started)
    } catch (error) {
      if (this.disposed) return
      const err = toApiError(error)
      this.store.set({ ai: { ...this.store.ai, state: 'unavailable', error: err, detail: err.message } })
    } finally {
      this.aiBusy = false
      this.overlay.setInvestigating(false)
      this.refreshLiving()
    }
  }

  private applyResponse(response: ScanAnalysis, started: number): void {
    const run = response.report.ai
    this.applyScan(response.scan, response.events)
    const dev = this.store.dev
    const called = run.status === 'ok' || run.status === 'unavailable'
    this.store.set({
      dev: {
        ...dev,
        aiCalls: dev.aiCalls + (called ? 1 : 0),
        aiSkipped: dev.aiSkipped + (run.status === 'skipped' ? 1 : 0),
        lastSkipReason: run.status === 'skipped' ? (run.reason ?? null) : dev.lastSkipReason,
      },
      ai: {
        ...this.store.ai,
        state: run.status === 'unavailable' ? 'unavailable' : run.status === 'off' ? 'off' : 'idle',
        lastRun: run,
        lastLatencyMs: run.status === 'ok' ? Math.round(performance.now() - started) : this.store.ai.lastLatencyMs,
        analyses: this.store.ai.analyses + (run.status === 'ok' ? 1 : 0),
        error: run.error ? new ApiError(run.error.code, run.error.message, { hint: run.error.hint }) : null,
        detail: run.reason ?? null,
      },
    })
    if (run.status === 'unavailable' || run.status === 'ok') void useSystem.getState().refreshHealth()
  }

  // ---------------------------------------------------------------- deep detector

  /** Periodic verification by the deep detector - only once loaded and fast enough here. */
  private maybeDeepCheck(now: number): void {
    if (this.deepBusy || !deepDetector.liveCapable || this.store.phase !== 'running') return
    if (now - this.lastDeepCheckAt < VISION.deep.liveCooldownMs) return
    if ((this.lastResult?.signals.motion ?? 1) >= TEMPORAL.sceneChange.settleMotion) return
    this.lastDeepCheckAt = now
    void this.deepCheck()
  }

  private async deepCheck(): Promise<void> {
    this.deepBusy = true
    try {
      const bitmap = await this.deepGrabber.bitmap(this.video, 640)
      if (!bitmap) return
      const at = performance.now()
      const result = await deepDetector.detect(bitmap)
      const fused = await vision.fuse(result.detections, at, VISION.deep.liveCooldownMs)
      const dev = this.store.dev
      this.store.set({
        dev: {
          ...dev,
          deepMs: result.totalMs,
          deepBackend: deepDetector.getSnapshot().info?.backend ?? null,
          deepRuns: dev.deepRuns + 1,
          deepVerified: dev.deepVerified + fused.verified,
          deepAdded: dev.deepAdded + fused.added,
        },
      })
    } catch (error) {
      console.warn('[live] deep check failed', error)
    } finally {
      this.deepBusy = false
    }
  }

  // ---------------------------------------------------------------- living state

  private refreshLiving(): void {
    const { living, ai, phase, scan, anomaly } = this.store
    const now = performance.now()
    if (now < living.until) return
    let state: Living
    let detail: string | null = null
    if (phase === 'paused' || phase === 'deep') state = 'FROZEN'
    else if (ai.state === 'analyzing') {
      state = 'INVESTIGATING'
      detail = ai.lastTrigger ? ai.lastTrigger.replace('_', ' ') : null
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

  private applyScan(scan: ScanState, events: LifecycleEvent[]): void {
    const previous = new Map(this.findings.map((f) => [f.id, f]))
    this.scanId = scan.scan_id
    this.findings = scan.findings
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
    if (this.lastResult) this.overlay.setAnchors(this.anchors(this.lastResult.tracks, this.lastResult.viewId))
  }

  // ---------------------------------------------------------------- controls

  pause(): void {
    if (this.store.phase !== 'running') return
    this.running = false
    cancelAnimationFrame(this.raf)
    this.camera.pause()
    this.overlay.setFrozen(true)
    this.suggestion = null
    this.store.set({ phase: 'paused', anomaly: null })
    this.refreshLiving()
  }

  async resume(): Promise<void> {
    if (this.store.phase !== 'paused') return
    await this.camera.resume()
    this.overlay.setFrozen(false)
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
      this.lastViewId = 0
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

  /**
   * Freeze the current frame, run the deep detector on it (loading it on
   * first use), fuse the result into the tracks, and send the fused scene to
   * the local engine - plus the frame for AI reasoning when that is on.
   */
  async deepScan(): Promise<void> {
    const store = this.store
    if (store.deep && store.deep.status !== 'done' && store.deep.status !== 'error') return
    if (store.phase === 'booting' || store.phase === 'error') return
    const frozenBefore = store.phase === 'paused'
    this.running = false
    cancelAnimationFrame(this.raf)
    this.camera.pause()
    this.overlay.setFrozen(true)
    this.suggestion = null
    let shot: { blob: Blob; width: number; height: number }
    try {
      shot = await this.jpeg.capture(this.video, VISION.capture.deepScanMaxEdge, VISION.capture.deepScanQuality)
    } catch (error) {
      store.set({ deep: null })
      void this.resumeAfterDeep()
      this.store.set({ ai: { ...this.store.ai, error: toApiError(error) } })
      return
    }
    const imageUrl = URL.createObjectURL(shot.blob)
    const step = deepDetector.enabled ? (deepDetector.ready ? 'Deep detector running' : 'Loading deep detector') : 'Fast detector only'
    this.store.set({
      phase: 'deep',
      deep: { status: 'detecting', imageUrl, width: shot.width, height: shot.height, startedAt: Date.now(), step, report: null, error: null },
    })
    this.refreshLiving()
    try {
      let result = this.lastResult
      if (deepDetector.enabled && result) {
        try {
          const bitmap = await createImageBitmap(shot.blob)
          const at = performance.now()
          const deep = await deepDetector.detect(bitmap)
          const fused = await vision.fuse(deep.detections, at, VISION.deep.liveCooldownMs)
          result = { ...result, tracks: fused.tracks, timestamp: at }
          const dev = this.store.dev
          this.store.set({
            dev: { ...dev, deepMs: deep.totalMs, deepBackend: deepDetector.getSnapshot().info?.backend ?? null, deepRuns: dev.deepRuns + 1 },
          })
        } catch (error) {
          console.warn('[live] deep detector unavailable for this scan', error)
        }
      }
      if (!result) throw new ApiError('NO_FRAME', 'No analysed frame is available yet.', { hint: 'Wait for the camera preview, then retry.' })
      const deepState = this.store.deep
      if (deepState) this.store.set({ deep: { ...deepState, status: 'analyzing', step: this.aiOn ? 'Local diagnostics + AI reasoning' : 'Local diagnostics' } })
      const started = performance.now()
      const response = await deepScan(this.aiOn ? shot.blob : null, {
        scanId: this.scanId,
        trigger: frozenBefore ? 'freeze' : 'deep_scan',
        scene: this.scene(result, []),
        personality: useSettings.getState().personality,
      })
      if (this.disposed) return
      this.applyResponse(response, started)
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
    this.suggestion = null
    this.pendingEvents = []
    vision.resetTracking()
    this.overlay.clear()
    this.store.set({ scan: null, events: [], ai: { ...this.store.ai, analyses: 0, error: null } })
    if (id) await deleteScan(id).catch(() => undefined)
  }

  get currentScanId(): string | null {
    return this.scanId
  }
}
