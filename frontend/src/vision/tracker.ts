/**
 * Lightweight multi-object tracker (IoU association with constant-velocity
 * prediction, class voting and tentative/confirmed/lost states).
 *
 * Deliberately simple: it runs every frame inside the vision worker and only
 * has to keep identities stable for a few seconds so the UI and the scan
 * director can reason about "new" objects and persisting issues.
 */
import { center, iou } from './geometry'
import type { Detection, NBox, Track } from './types'

const CONFUSABLE: string[][] = [
  ['cup', 'wine glass', 'bowl', 'vase', 'bottle'],
  ['tv', 'laptop'],
  ['cell phone', 'remote'],
  ['couch', 'chair', 'bench'],
  ['dining table', 'bed'],
  ['car', 'truck', 'bus'],
  ['cat', 'dog', 'teddy bear'],
  ['handbag', 'backpack', 'suitcase'],
]

function compatible(a: string, b: string): boolean {
  if (a === b) return true
  return CONFUSABLE.some((group) => group.includes(a) && group.includes(b))
}

interface TrackerOptions {
  iouMatch: number
  maxMissMs: number
  confirmHits: number
  minNewScore: number
  boxSmoothing: number
}

const DEFAULTS: TrackerOptions = {
  iouMatch: 0.12,
  maxMissMs: 1100,
  confirmHits: 3,
  minNewScore: 0.42,
  boxSmoothing: 0.6,
}

interface InternalTrack extends Track {
  votes: Map<string, number>
  updatedAt: number
}

function shift(box: NBox, dx: number, dy: number): NBox {
  return { x: box.x + dx, y: box.y + dy, w: box.w, h: box.h }
}

export class Tracker {
  private tracks: InternalTrack[] = []
  private nextId = 1
  private readonly opts: TrackerOptions

  constructor(options: Partial<TrackerOptions> = {}) {
    this.opts = { ...DEFAULTS, ...options }
  }

  reset(): void {
    this.tracks = []
    this.nextId = 1
  }

  update(detections: Detection[], now: number): Track[] {
    const { iouMatch, maxMissMs, confirmHits, minNewScore, boxSmoothing } = this.opts

    // 1. Constant-velocity prediction.
    const predicted = this.tracks.map((t) => {
      const dt = Math.min(0.5, Math.max(0, (now - t.updatedAt) / 1000))
      return shift(t.box, t.vx * dt, t.vy * dt)
    })

    // 2. Candidate pairs scored by overlap (fallback: centre distance).
    const pairs: { ti: number; di: number; score: number }[] = []
    this.tracks.forEach((track, ti) => {
      detections.forEach((det, di) => {
        if (!compatible(track.label, det.label)) return
        const overlap = iou(predicted[ti]!, det.box)
        let score = overlap
        if (overlap < iouMatch) {
          const [tx, ty] = center(predicted[ti]!)
          const [dx, dy] = center(det.box)
          const scale = Math.sqrt(Math.max(track.box.w * track.box.h, det.box.w * det.box.h, 1e-4))
          const distance = Math.hypot(tx - dx, ty - dy) / scale
          if (distance > 0.6) return
          score = 0.08 * (1 - distance)
        }
        if (track.label !== det.label) score *= 0.8
        pairs.push({ ti, di, score })
      })
    })
    pairs.sort((a, b) => b.score - a.score)

    const usedTracks = new Set<number>()
    const usedDets = new Set<number>()
    for (const { ti, di } of pairs) {
      if (usedTracks.has(ti) || usedDets.has(di)) continue
      usedTracks.add(ti)
      usedDets.add(di)
      this.updateTrack(this.tracks[ti]!, detections[di]!, now, boxSmoothing, confirmHits)
    }

    // 3. Age unmatched tracks.
    const survivors: InternalTrack[] = []
    this.tracks.forEach((track, ti) => {
      if (!usedTracks.has(ti)) {
        track.misses += 1
        if (track.state === 'tentative' && track.misses >= 2) return
        if (now - track.lastSeen > maxMissMs) return
        if (track.state === 'confirmed') track.state = 'lost'
        // Keep coasting on the prediction while lost.
        track.box = predicted[ti]!
        track.vx *= 0.6
        track.vy *= 0.6
        track.updatedAt = now
      }
      survivors.push(track)
    })
    this.tracks = survivors

    // 4. Spawn tracks for confident unmatched detections.
    detections.forEach((det, di) => {
      if (usedDets.has(di) || det.score < minNewScore) return
      const id = this.nextId++
      this.tracks.push({
        id,
        key: `${det.label}·${String(id).padStart(2, '0')}`,
        label: det.label,
        score: det.score,
        box: { ...det.box },
        vx: 0,
        vy: 0,
        hits: 1,
        misses: 0,
        state: 'tentative',
        firstSeen: now,
        lastSeen: now,
        votes: new Map([[det.label, det.score]]),
        updatedAt: now,
      })
    })

    return this.snapshot()
  }

  private updateTrack(track: InternalTrack, det: Detection, now: number, alpha: number, confirmHits: number): void {
    const dt = Math.max(1e-3, (now - track.updatedAt) / 1000)
    const [ox, oy] = center(track.box)
    const box: NBox = {
      x: track.box.x + (det.box.x - track.box.x) * alpha,
      y: track.box.y + (det.box.y - track.box.y) * alpha,
      w: track.box.w + (det.box.w - track.box.w) * alpha,
      h: track.box.h + (det.box.h - track.box.h) * alpha,
    }
    const [nx, ny] = center(box)
    track.vx = 0.5 * track.vx + 0.5 * ((nx - ox) / dt)
    track.vy = 0.5 * track.vy + 0.5 * ((ny - oy) / dt)
    track.box = box
    track.score = 0.7 * track.score + 0.3 * det.score
    track.hits += 1
    track.misses = 0
    track.lastSeen = now
    track.updatedAt = now
    track.votes.set(det.label, (track.votes.get(det.label) ?? 0) + det.score)
    let best = track.label
    let bestVotes = -1
    for (const [label, votes] of track.votes) {
      if (votes > bestVotes) {
        best = label
        bestVotes = votes
      }
    }
    if (best !== track.label) {
      track.label = best
      track.key = `${best}·${String(track.id).padStart(2, '0')}`
    }
    if (track.state !== 'confirmed' && track.hits >= confirmHits) track.state = 'confirmed'
    else if (track.state === 'lost') track.state = 'confirmed'
  }

  private snapshot(): Track[] {
    return this.tracks.map((t) => ({
      id: t.id,
      key: t.key,
      label: t.label,
      score: Math.round(t.score * 1000) / 1000,
      box: { ...t.box },
      vx: t.vx,
      vy: t.vy,
      hits: t.hits,
      misses: t.misses,
      state: t.state,
      firstSeen: t.firstSeen,
      lastSeen: t.lastSeen,
    }))
  }
}
