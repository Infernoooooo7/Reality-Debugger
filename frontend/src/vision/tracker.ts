/**
 * Multi-object tracker: ByteTrack (Zhang et al., ECCV 2022) with a
 * constant-velocity Kalman filter (SORT / DeepSORT) and optimal assignment.
 *
 * Per frame:
 *  1. Kalman-predict every track to the frame time.
 *  2. Associate tracked + lost tracks with HIGH-score detections
 *     (IoU fused with the detection score).
 *  3. Associate the remaining tracked tracks with LOW-score detections
 *     (plain IoU) - ByteTrack's key idea: occluded or blurred objects are
 *     often still detected, just with low confidence.
 *  4. Associate unconfirmed tracks with the leftover high-score detections;
 *     a track is confirmed once it has been matched `confirmHits` times.
 *  5. Start tentative tracks from unmatched detections above `newTrackScore`;
 *     drop lost tracks after `lostBufferMs`.
 *
 * ByteTrack is single-class; here a detection may only extend a track of a
 * compatible class (same label or COCO supercategory, config/ontology).
 * On top of the association the tracker derives temporal attributes from
 * the Kalman state: speed, moving/static (with hysteresis), time static,
 * direction reversals and occlusion. All thresholds: config/tracking.json and
 * config/temporal.json.
 */
import { TEMPORAL, TRACKING, VISION } from '../config'
import { linearAssignment } from './assignment'
import { area, intersection, iou } from './geometry'
import { KalmanFilter, type Mat, type Vec } from './kalman'
import { compatible } from './ontology'
import type { Detection, Movement, NBox, Track, TrackEvent } from './types'

export interface TrackerOptions {
  highScore: number
  lowScore: number
  newTrackScore: number
  firstMatchIou: number
  secondMatchIou: number
  unconfirmedMatchIou: number
  fuseScore: boolean
  duplicateIou: number
  lostBufferMs: number
  confirmHits: number
  classCompatibility: 'supercategory' | 'label'
  kalmanStdWeightPosition: number
  kalmanStdWeightVelocity: number
  /** Duration of one "frame" for the Kalman model (ms). */
  nominalFrameMs: number
  movingSpeed: number
  stillSpeed: number
  reversalWindowMs: number
  occludedFraction: number
  /** Camera-motion compensation: shift predictions by the estimated global image motion. */
  cmc: boolean
  /** Minimum confidence of the global motion estimate for it to be applied. */
  cmcMinConfidence: number
}

/** Global image motion between the previous and the current frame (see motion.ts). */
export interface CameraMotion {
  dx: number
  dy: number
  confidence: number
}

export const TRACKER_DEFAULTS: TrackerOptions = {
  highScore: TRACKING.highScore,
  lowScore: TRACKING.lowScore,
  newTrackScore: TRACKING.newTrackScore,
  firstMatchIou: TRACKING.firstMatchIou,
  secondMatchIou: TRACKING.secondMatchIou,
  unconfirmedMatchIou: TRACKING.unconfirmedMatchIou,
  fuseScore: TRACKING.fuseScore,
  duplicateIou: TRACKING.duplicateIou,
  lostBufferMs: TRACKING.lostBufferMs,
  confirmHits: TRACKING.confirmHits,
  classCompatibility: TRACKING.classCompatibility,
  kalmanStdWeightPosition: TRACKING.kalmanStdWeightPosition,
  kalmanStdWeightVelocity: TRACKING.kalmanStdWeightVelocity,
  nominalFrameMs: 1000 / VISION.fast.maxFps,
  movingSpeed: TEMPORAL.movement.movingSpeed,
  stillSpeed: TEMPORAL.movement.stillSpeed,
  reversalWindowMs: TEMPORAL.movement.reversalWindowMs,
  occludedFraction: TEMPORAL.occlusion.occludedFraction,
  cmc: TRACKING.cmc,
  cmcMinConfidence: TRACKING.cmcMinConfidence,
}

const EDGE = 0.005
const MAX_DT_FRAMES = 10

function toMeasurement(b: NBox): Vec {
  const h = Math.max(b.h, 1e-4)
  return Float64Array.of(b.x + b.w / 2, b.y + b.h / 2, b.w / h, h)
}

function toBox(mean: Vec): NBox {
  const h = Math.max(mean[3]!, 1e-4)
  const w = Math.max(mean[2]! * h, 1e-4)
  const x = mean[0]! - w / 2
  const y = mean[1]! - h / 2
  const x1 = Math.max(0, x)
  const y1 = Math.max(0, y)
  const x2 = Math.min(1, x + w)
  const y2 = Math.min(1, y + h)
  return { x: x1, y: y1, w: Math.max(1e-4, x2 - x1), h: Math.max(1e-4, y2 - y1) }
}

/** Share of `box` covered by the boxes of other tracks matched in this frame (occlusion proxy). */
function coverage(box: NBox, tracks: STrack[], self: STrack): number {
  const own = area(box)
  if (own <= 0) return 0
  let covered = 0
  for (const other of tracks) if (other !== self && other.matched) covered += intersection(box, other.box)
  return Math.min(1, covered / own)
}

class STrack {
  readonly id: number
  label: string
  readonly votes = new Map<string, number>()
  score: number
  readonly mean: Vec
  readonly cov: Mat
  activated = false
  state: 'tracked' | 'lost' = 'tracked'
  hits = 1
  misses = 0
  readonly firstSeen: number
  lastSeen: number
  predictedAt: number
  matched = true
  source: 'fast' | 'deep' | 'fused' = 'fast'
  verified = false
  holdUntil = 0
  movement: Movement = 'unknown'
  staticSince: number
  lastSign = 0
  reversalTimes: number[] = []
  occlusion = 0
  occludedSince: number | null = null
  truncated = false
  recoveredAt: number | null = null

  constructor(id: number, det: Detection, now: number, kf: KalmanFilter) {
    this.id = id
    this.label = det.label
    this.score = det.score
    this.votes.set(det.label, det.score)
    const { mean, cov } = kf.initiate(toMeasurement(det.box))
    this.mean = mean
    this.cov = cov
    this.firstSeen = now
    this.lastSeen = now
    this.predictedAt = now
    this.staticSince = now
  }

  get box(): NBox {
    return toBox(this.mean)
  }

  get historyMs(): number {
    return this.lastSeen - this.firstSeen
  }
}

export class Tracker {
  private tracks: STrack[] = []
  private nextId = 1
  private readonly kf: KalmanFilter
  private readonly opts: TrackerOptions
  private events: TrackEvent[] = []
  /** Camera velocity (frame units per second) of the last update, when compensated. */
  private cameraVelocity: [number, number] = [0, 0]
  private lastUpdateAt: number | null = null

  constructor(options: Partial<TrackerOptions> = {}) {
    this.opts = { ...TRACKER_DEFAULTS, ...options }
    this.kf = new KalmanFilter({
      stdWeightPosition: this.opts.kalmanStdWeightPosition,
      stdWeightVelocity: this.opts.kalmanStdWeightVelocity,
    })
  }

  reset(): void {
    this.tracks = []
    this.nextId = 1
    this.events = []
    this.cameraVelocity = [0, 0]
    this.lastUpdateAt = null
  }

  /** Events since the last call (entered / recovered / left = terminated). */
  drainEvents(): TrackEvent[] {
    const out = this.events
    this.events = []
    return out
  }

  private compatible(a: string, b: string): boolean {
    return compatible(a, b, this.opts.classCompatibility)
  }

  private cost(tracks: STrack[], dets: Detection[], fuse: boolean): number[][] {
    return tracks.map((t) => {
      const box = t.box
      return dets.map((d) => {
        if (!this.compatible(t.label, d.label)) return 1e9
        const overlap = iou(box, d.box)
        return 1 - (fuse ? overlap * d.score : overlap)
      })
    })
  }

  private compensate(camera: CameraMotion | null, now: number): void {
    const dt = this.lastUpdateAt === null ? 0 : (now - this.lastUpdateAt) / 1000
    this.lastUpdateAt = now
    if (!this.opts.cmc || !camera || camera.confidence < this.opts.cmcMinConfidence) {
      this.cameraVelocity = [0, 0]
      return
    }
    for (const t of this.tracks) {
      t.mean[0] = t.mean[0]! + camera.dx
      t.mean[1] = t.mean[1]! + camera.dy
    }
    this.cameraVelocity = dt > 0 ? [camera.dx / dt, camera.dy / dt] : [0, 0]
  }

  private predict(now: number): void {
    for (const t of this.tracks) {
      const dt = Math.min(MAX_DT_FRAMES, Math.max(0, (now - t.predictedAt) / this.opts.nominalFrameMs))
      if (dt <= 0) continue
      if (t.state !== 'tracked') t.mean[7] = 0 // ByteTrack: lost tracks keep their height
      this.kf.predict(t.mean, t.cov, dt)
      t.predictedAt = now
    }
  }

  private apply(t: STrack, det: Detection, now: number): void {
    if (t.state === 'lost' && t.activated) {
      t.recoveredAt = now
      this.events.push({ kind: 'recovered', at: now, trackId: t.id, sceneId: `t${t.id}`, label: t.label })
    }
    this.kf.update(t.mean, t.cov, toMeasurement(det.box))
    t.score = 0.5 * t.score + 0.5 * det.score
    t.votes.set(det.label, (t.votes.get(det.label) ?? 0) + det.score)
    let best = t.label
    let bestVotes = -1
    for (const [label, votes] of t.votes) {
      if (votes > bestVotes) {
        best = label
        bestVotes = votes
      }
    }
    t.label = best
    t.hits += 1
    t.misses = 0
    t.lastSeen = now
    t.state = 'tracked'
    t.matched = true
    if (t.source === 'deep') {
      // The fast detector now sees an object the deep detector found: both agree.
      t.source = 'fused'
      t.verified = true
      t.holdUntil = 0
    }
  }

  private activate(t: STrack, now: number): void {
    if (t.activated) return
    t.activated = true
    this.events.push({ kind: 'entered', at: now, trackId: t.id, sceneId: `t${t.id}`, label: t.label })
  }

  /**
   * One frame. `camera` is the global image motion since the previous frame
   * (motion.ts); with `cmc` on and a confident estimate, every prediction is
   * shifted by it before association, so the Kalman state - and the speed and
   * movement derived from it - describes motion relative to the scene, not
   * the camera's own motion.
   */
  update(detections: Detection[], now: number, camera?: CameraMotion | null): Track[] {
    const o = this.opts
    this.predict(now)
    this.compensate(camera ?? null, now)
    for (const t of this.tracks) t.matched = false

    const high = detections.filter((d) => d.score >= o.highScore)
    const low = detections.filter((d) => d.score >= o.lowScore && d.score < o.highScore)
    const pool = this.tracks.filter((t) => t.activated)
    const unconfirmed = this.tracks.filter((t) => !t.activated)

    // 1. tracked + lost vs high-score detections
    const first = linearAssignment(this.cost(pool, high, o.fuseScore), 1 - o.firstMatchIou, high.length)
    for (const [ti, di] of first.matches) this.apply(pool[ti]!, high[di]!, now)
    const remainingHigh = first.unmatchedCols.map((i) => high[i]!)

    // 2. remaining tracked tracks vs low-score detections
    const rTracked = first.unmatchedRows.map((i) => pool[i]!).filter((t) => t.state === 'tracked')
    const second = linearAssignment(this.cost(rTracked, low, false), 1 - o.secondMatchIou, low.length)
    for (const [ti, di] of second.matches) this.apply(rTracked[ti]!, low[di]!, now)
    for (const i of second.unmatchedRows) {
      const t = rTracked[i]!
      if (now < t.holdUntil) continue // deep-only object: held until the next deep check
      t.state = 'lost'
    }

    // 3. unconfirmed tracks vs leftover high-score detections
    const third = linearAssignment(this.cost(unconfirmed, remainingHigh, o.fuseScore), 1 - o.unconfirmedMatchIou, remainingHigh.length)
    const used = new Set<number>()
    for (const [ti, di] of third.matches) {
      const t = unconfirmed[ti]!
      this.apply(t, remainingHigh[di]!, now)
      if (t.hits >= o.confirmHits) this.activate(t, now)
      used.add(di)
    }
    const removed = new Set<STrack>(third.unmatchedRows.map((i) => unconfirmed[i]!)) // unconfirmed and missed: drop

    // 4. new tentative tracks
    remainingHigh.forEach((det, di) => {
      if (used.has(di) || det.score < o.newTrackScore) return
      const t = new STrack(this.nextId++, det, now, this.kf)
      if (o.confirmHits <= 1) this.activate(t, now)
      this.tracks.push(t)
    })

    // 5. expire lost tracks, remove duplicates
    for (const t of this.tracks) {
      if (t.state === 'lost') t.misses += 1
      if (t.state === 'lost' && now - t.lastSeen > o.lostBufferMs) removed.add(t)
    }
    const tracked = this.tracks.filter((t) => t.state === 'tracked' && t.activated && !removed.has(t))
    const lost = this.tracks.filter((t) => t.state === 'lost' && !removed.has(t))
    for (const a of tracked) {
      for (const b of lost) {
        if (iou(a.box, b.box) > o.duplicateIou) removed.add(a.historyMs >= b.historyMs ? b : a)
      }
    }
    for (const t of removed) {
      if (t.activated) this.events.push({ kind: 'left', at: now, trackId: t.id, sceneId: `t${t.id}`, label: t.label })
    }
    this.tracks = this.tracks.filter((t) => !removed.has(t))

    this.updateTemporal(now)
    return this.snapshot(now)
  }

  /**
   * Fuse a deep-detector pass into the tracks: a compatible overlapping deep
   * detection verifies a track; a clearly more confident conflicting one
   * relabels it; confident deep-only detections become held tracks.
   */
  verify(
    detections: Detection[],
    now: number,
    opts: { matchIou: number; deepOnlyMinScore: number; relabelMargin: number; holdMs: number },
  ): { verified: number; relabeled: number; added: number } {
    let verified = 0
    let relabeled = 0
    let added = 0
    const candidates = this.tracks.filter((t) => t.activated)
    const claimed = new Set<STrack>()
    for (const det of [...detections].sort((a, b) => b.score - a.score)) {
      let best: STrack | null = null
      let bestIou = opts.matchIou
      for (const t of candidates) {
        if (claimed.has(t)) continue
        const overlap = iou(t.box, det.box)
        if (overlap >= bestIou) {
          best = t
          bestIou = overlap
        }
      }
      if (best) {
        claimed.add(best)
        if (this.compatible(best.label, det.label)) {
          best.verified = true
          if (best.source === 'fast') best.source = 'fused'
          verified += 1
        } else if (det.score >= best.score + opts.relabelMargin) {
          best.votes.clear()
          best.votes.set(det.label, det.score)
          best.label = det.label
          best.verified = true
          best.source = 'fused'
          relabeled += 1
        }
        continue
      }
      if (det.score < opts.deepOnlyMinScore) continue
      const t = new STrack(this.nextId++, det, now, this.kf)
      t.source = 'deep'
      t.holdUntil = now + opts.holdMs
      this.activate(t, now)
      this.tracks.push(t)
      claimed.add(t)
      added += 1
    }
    this.updateTemporal(now)
    return { verified, relabeled, added }
  }

  private updateTemporal(now: number): void {
    const o = this.opts
    const perSecond = 1000 / o.nominalFrameMs
    const visible = this.tracks.filter((t) => t.activated)
    for (const t of visible) {
      const vx = t.mean[4]! * perSecond
      const vy = t.mean[5]! * perSecond
      const speed = Math.hypot(vx, vy)
      if (t.hits >= 3 && t.source !== 'deep') {
        if (speed > o.movingSpeed) t.movement = 'moving'
        else if (speed < o.stillSpeed) t.movement = 'static'
      }
      if (t.movement === 'moving') {
        t.staticSince = now
        const sign = Math.abs(vx) > o.stillSpeed ? Math.sign(vx) : 0
        if (sign !== 0) {
          if (t.lastSign !== 0 && sign !== t.lastSign) t.reversalTimes.push(now)
          t.lastSign = sign
        }
      }
      t.reversalTimes = t.reversalTimes.filter((at) => now - at <= o.reversalWindowMs)
      const box = t.box
      const own = area(box)
      let covered = 0
      for (const other of visible) if (other !== t) covered += intersection(box, other.box)
      t.occlusion = own > 0 ? Math.min(1, covered / own) : 0
      t.truncated = box.x <= EDGE || box.y <= EDGE || box.x + box.w >= 1 - EDGE || box.y + box.h >= 1 - EDGE
      if (t.occlusion >= o.occludedFraction) t.occludedSince ??= now
      else t.occludedSince = null
    }
  }

  /** Current tracks without a detector update (e.g. right after `verify`). */
  current(now: number): Track[] {
    return this.snapshot(now)
  }

  private snapshot(now: number): Track[] {
    const perSecond = 1000 / this.opts.nominalFrameMs
    return this.tracks.map((t) => {
      const vx = t.mean[4]! * perSecond
      const vy = t.mean[5]! * perSecond
      const held = t.source === 'deep' && now < t.holdUntil
      const visibleNow = t.matched || held
      let state: Track['state']
      if (!t.activated) state = 'tentative'
      else if (visibleNow) state = t.recoveredAt === now ? 'recovered' : 'confirmed'
      else if (t.state === 'lost') state = coverage(t.box, this.tracks, t) >= this.opts.occludedFraction ? 'occluded' : 'lost'
      else state = 'confirmed'
      return {
        id: t.id,
        key: `${t.label}·${String(t.id).padStart(2, '0')}`,
        sceneId: `t${t.id}`,
        label: t.label,
        score: Math.round(t.score * 1000) / 1000,
        box: t.box,
        vx,
        vy,
        speed: Math.hypot(vx, vy),
        apparentSpeed: Math.hypot(vx + this.cameraVelocity[0], vy + this.cameraVelocity[1]),
        hits: t.hits,
        misses: t.misses,
        state,
        firstSeen: t.firstSeen,
        lastSeen: t.lastSeen,
        movement: t.movement,
        staticMs: t.movement === 'moving' ? 0 : Math.max(0, now - t.staticSince),
        reversals: t.reversalTimes.length,
        occlusion: Math.round(t.occlusion * 100) / 100,
        occludedMs: t.occludedSince === null ? 0 : now - t.occludedSince,
        truncated: t.truncated,
        source: t.source,
        verified: t.verified,
      }
    })
  }
}
