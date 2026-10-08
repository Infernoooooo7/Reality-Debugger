/**
 * Scan director: decides *when* a live frame deserves the vision model.
 *
 * Local ML runs on every frame; the LLM is only called when something
 * meaningful happens (first look, a new object, an interesting spatial
 * relationship, a substantial scene change, a finding that needs confirming
 * or a possible resolution, or a periodic re-check), never more often than
 * `minGapMs`, and preferably on a stable (low-motion) frame.
 */
import type { Finding, Trigger } from '../lib/schemas'
import type { Relation } from '../vision/relations'
import type { FrameResult } from '../vision/types'
import { cocoLabels } from './labels'

export interface DirectorConfig {
  minGapMs: number
  intervalMs: number
  stableMotion: number
  maxWaitStableMs: number
  sceneChange: number
  firstLookDelayMs: number
  confirmAfterMs: number
}

export const DEFAULT_DIRECTOR: DirectorConfig = {
  minGapMs: 5000,
  intervalMs: 20000,
  stableMotion: 0.07,
  maxWaitStableMs: 2200,
  sceneChange: 0.32,
  firstLookDelayMs: 1400,
  confirmAfterMs: 4500,
}

export interface Decision {
  trigger: Trigger
  detail: string
  findingId?: string
  priority: number
}

interface Watch {
  id: string
  title: string
  status: Finding['status']
  labels: string[]
  createdAt: number
  lastCheckAt: number
  missingSince: number | null
}

const ANOMALY_TRIGGERS = new Set<Trigger>(['new_object', 'relationship', 'scene_change'])

export class ScanDirector {
  private cfg: DirectorConfig
  private startedAt = performance.now()
  private lastAnalysisAt = 0
  private analyses = 0
  private knownLabels = new Set<string>()
  private knownRelations = new Set<string>()
  private pending: (Decision & { since: number }) | null = null
  private watches = new Map<string, Watch>()
  private backoffUntil = 0
  private paused = false
  inFlight = false

  constructor(config: Partial<DirectorConfig> = {}) {
    this.cfg = { ...DEFAULT_DIRECTOR, ...config }
  }

  configure(config: Partial<DirectorConfig>): void {
    this.cfg = { ...this.cfg, ...config }
  }

  setPaused(paused: boolean): void {
    this.paused = paused
    if (paused) this.pending = null
  }

  get lastAnalysis(): number {
    return this.lastAnalysisAt
  }

  get pendingDecision(): Decision | null {
    return this.pending
  }

  get backoffRemainingMs(): number {
    return Math.max(0, this.backoffUntil - performance.now())
  }

  private candidate(result: FrameResult, relations: Relation[], now: number): Decision | null {
    const { tracks, signals } = result
    const confirmed = tracks.filter((t) => t.state === 'confirmed')

    if (this.analyses === 0) {
      return now - this.startedAt > this.cfg.firstLookDelayMs
        ? { trigger: 'first_look', detail: 'initial survey of the scene', priority: 9 }
        : null
    }

    const fresh = confirmed.find(
      (t) => !this.knownLabels.has(t.label) && t.box.w * t.box.h >= 0.006 && now - t.firstSeen > 300,
    )
    if (fresh) return { trigger: 'new_object', detail: `${fresh.key} entered the frame`, priority: 8 }

    const relation = relations.find((r) => !this.knownRelations.has(r.key))
    if (relation) return { trigger: 'relationship', detail: relation.text, priority: 7 }

    // Possible resolution: every object a finding depends on has vanished
    // while the camera still looks at roughly the same scene.
    const present = new Set(tracks.filter((t) => t.state !== 'tentative').map((t) => t.label))
    for (const watch of this.watches.values()) {
      if (!watch.labels.length) continue
      const missing = watch.labels.every((label) => !present.has(label))
      if (!missing) {
        watch.missingSince = null
        continue
      }
      watch.missingSince ??= now
      if (now - watch.missingSince > 2500 && signals.sceneDelta < 0.3 && now - watch.lastCheckAt > 15000) {
        return {
          trigger: 'confirmation',
          detail: `verify whether ${watch.id} ("${watch.title}") is resolved - its objects are no longer detected`,
          findingId: watch.id,
          priority: 6,
        }
      }
    }

    if (signals.sceneDelta > this.cfg.sceneChange) {
      return { trigger: 'scene_change', detail: `scene changed (Δ ${signals.sceneDelta.toFixed(2)})`, priority: 5 }
    }

    for (const watch of this.watches.values()) {
      if (watch.status === 'DISCOVERED' && now - watch.createdAt > this.cfg.confirmAfterMs && now - watch.lastCheckAt > 12000) {
        return {
          trigger: 'confirmation',
          detail: `confirm or reject ${watch.id} ("${watch.title}")`,
          findingId: watch.id,
          priority: 4,
        }
      }
    }

    if (this.cfg.intervalMs > 0 && now - this.lastAnalysisAt > this.cfg.intervalMs) {
      return { trigger: 'interval', detail: 'periodic re-check of a stable scene', priority: 1 }
    }
    return null
  }

  /**
   * Feed every local result. Returns the anomaly being considered (for the
   * HUD) and, when it is time, the decision to analyse the current frame.
   */
  observe(result: FrameResult, relations: Relation[], now = performance.now()): { decision: Decision | null; anomaly: string | null } {
    if (this.paused) return { decision: null, anomaly: null }
    const candidate = this.candidate(result, relations, now)
    if (candidate && (!this.pending || candidate.priority > this.pending.priority)) {
      this.pending = { ...candidate, since: this.pending?.since ?? now }
    }
    const pending = this.pending
    const anomaly = pending && ANOMALY_TRIGGERS.has(pending.trigger) ? pending.detail : null
    if (!pending || this.inFlight || now < this.backoffUntil) return { decision: null, anomaly }
    if (this.lastAnalysisAt && now - this.lastAnalysisAt < this.cfg.minGapMs) return { decision: null, anomaly }
    const stable = result.signals.motion < this.cfg.stableMotion
    if (!stable && now - pending.since < this.cfg.maxWaitStableMs) return { decision: null, anomaly }
    this.pending = null
    const { since: _since, ...decision } = pending
    void _since
    return { decision, anomaly }
  }

  markSent(decision: Decision, result: FrameResult | null, relations: Relation[], now = performance.now()): void {
    this.inFlight = true
    this.lastAnalysisAt = now
    this.analyses += 1
    for (const t of result?.tracks ?? []) if (t.state === 'confirmed') this.knownLabels.add(t.label)
    for (const r of relations) this.knownRelations.add(r.key)
    if (decision.findingId) {
      const watch = this.watches.get(decision.findingId)
      if (watch) watch.lastCheckAt = now
    }
    if (decision.trigger === 'interval' || decision.trigger === 'scene_change' || decision.trigger === 'first_look') {
      for (const watch of this.watches.values()) watch.lastCheckAt = now
    }
  }

  markDone(ok: boolean, retryAfterMs = 0): void {
    this.inFlight = false
    if (!ok) this.backoffUntil = performance.now() + retryAfterMs
  }

  /** Keep the watch list in sync with the backend's lifecycle state. */
  syncFindings(findings: Finding[], now = performance.now()): void {
    const seen = new Set<string>()
    for (const f of findings) {
      if (f.status === 'RESOLVED') continue
      seen.add(f.id)
      const existing = this.watches.get(f.id)
      if (existing) {
        existing.status = f.status
        existing.title = f.title
        existing.labels = cocoLabels(f.related_objects)
      } else {
        this.watches.set(f.id, {
          id: f.id,
          title: f.title,
          status: f.status,
          labels: cocoLabels(f.related_objects),
          createdAt: now,
          lastCheckAt: now,
          missingSince: null,
        })
      }
    }
    for (const id of [...this.watches.keys()]) if (!seen.has(id)) this.watches.delete(id)
  }

  reset(): void {
    this.startedAt = performance.now()
    this.lastAnalysisAt = 0
    this.analyses = 0
    this.knownLabels.clear()
    this.knownRelations.clear()
    this.pending = null
    this.watches.clear()
    this.backoffUntil = 0
    this.inFlight = false
  }
}
