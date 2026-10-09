/**
 * Late fusion of the fast and deep detectors for still frames (Image Debug,
 * video keyframes). Live Scan fuses into the tracker instead
 * (Tracker.verify). Thresholds: config/detection.json "fusion".
 *
 * - A deep box overlapping a fast box (IoU >= matchIou) with a compatible
 *   class verifies it: both models agree ("fused", verified).
 * - With an incompatible class, the deep label wins only if its score beats
 *   the fast one by relabelMargin (the deep model is far more accurate).
 * - Deep-only boxes are kept when score >= deepOnlyMinScore.
 * - Fast-only boxes are kept at the fast detector's own operating point.
 */
import { DETECTION } from '../config'
import { iou } from './geometry'
import { compatible } from './ontology'
import type { SceneObjectPayload } from './scene'
import type { Detection, NBox } from './types'

export interface FusedObject {
  id: string
  label: string
  confidence: number
  box: NBox
  source: 'fast' | 'deep' | 'fused'
  verified: boolean
}

export function fuseStill(fast: Detection[], deep: Detection[] | null): FusedObject[] {
  const f = DETECTION.fusion
  const out: FusedObject[] = []
  const usedFast = new Set<number>()
  let n = 0
  const nextId = () => `o${++n}`

  for (const d of [...(deep ?? [])].sort((a, b) => b.score - a.score)) {
    let best = -1
    let bestIou = f.matchIou
    fast.forEach((candidate, i) => {
      if (usedFast.has(i)) return
      const overlap = iou(candidate.box, d.box)
      if (overlap >= bestIou) {
        best = i
        bestIou = overlap
      }
    })
    if (best >= 0) {
      const match = fast[best]!
      usedFast.add(best)
      if (compatible(match.label, d.label)) {
        out.push({ id: nextId(), label: d.score >= match.score ? d.label : match.label, confidence: Math.max(d.score, match.score), box: d.box, source: 'fused', verified: true })
      } else if (d.score >= match.score + f.relabelMargin) {
        out.push({ id: nextId(), label: d.label, confidence: d.score, box: d.box, source: 'fused', verified: true })
      } else if (match.score >= DETECTION.fast.scoreThreshold) {
        out.push({ id: nextId(), label: match.label, confidence: match.score, box: match.box, source: 'fast', verified: false })
      }
      continue
    }
    if (d.score >= f.deepOnlyMinScore) {
      out.push({ id: nextId(), label: d.label, confidence: d.score, box: d.box, source: 'deep', verified: false })
    }
  }
  fast.forEach((d, i) => {
    if (usedFast.has(i) || d.score < DETECTION.fast.scoreThreshold) return
    out.push({ id: nextId(), label: d.label, confidence: d.score, box: d.box, source: 'fast', verified: false })
  })
  return out.sort((a, b) => b.confidence - a.confidence)
}

/**
 * Verify tracked objects of one frame with a deep-detector pass (video
 * keyframes): same rules as `fuseStill`, but the fast side keeps its track
 * ids and temporal attributes.
 */
export function verifyObjects(objects: SceneObjectPayload[], deep: Detection[], idPrefix: string): SceneObjectPayload[] {
  const f = DETECTION.fusion
  const out = objects.map((o) => ({ ...o }))
  const claimed = new Set<number>()
  let added = 0
  for (const d of [...deep].sort((a, b) => b.score - a.score)) {
    let best = -1
    let bestIou = f.matchIou
    out.forEach((o, i) => {
      if (claimed.has(i)) return
      const overlap = iou(o.box, d.box)
      if (overlap >= bestIou) {
        best = i
        bestIou = overlap
      }
    })
    if (best >= 0) {
      claimed.add(best)
      const o = out[best]!
      if (compatible(o.label, d.label)) {
        o.verified = true
        o.source = 'fused'
        o.confidence = Math.max(o.confidence, Math.round(d.score * 1000) / 1000)
      } else if (d.score >= o.confidence + f.relabelMargin) {
        o.label = d.label
        o.verified = true
        o.source = 'fused'
        o.confidence = Math.round(d.score * 1000) / 1000
      }
      continue
    }
    if (d.score < f.deepOnlyMinScore) continue
    out.push({
      id: `${idPrefix}${++added}`,
      label: d.label,
      confidence: Math.round(d.score * 1000) / 1000,
      box: d.box,
      source: 'deep',
      verified: false,
      age_ms: 0,
      persistent: false,
      movement: 'unknown',
      speed: null,
      static_ms: 0,
      reversals: 0,
      occlusion: 0,
      occluded_ms: 0,
      truncated: false,
    })
  }
  return out
}
