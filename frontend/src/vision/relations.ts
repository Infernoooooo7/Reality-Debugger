/**
 * Spatial relations between tracked objects, computed from boxes only (the
 * same measures and thresholds as the backend engine, config/diagnostics.json
 * "proximity"). Used for the live HUD and the developer panel; findings
 * themselves come from the backend's local diagnostic engine.
 */
import { DIAGNOSTICS } from '../config'
import { area, baseInside, containment, gap, iou, relativeGap } from './geometry'
import { hasAttribute } from './ontology'
import type { Track } from './types'

export interface Relation {
  key: string
  subject: Track
  object: Track
  kind: 'on' | 'overlaps' | 'touching' | 'near'
  relativeGap: number
  text: string
}

const RANK = { on: 0, overlaps: 1, touching: 2, near: 3 }

export function spatialRelations(tracks: Track[], limit = 12): Relation[] {
  const { touchGap, nearRelativeGap } = DIAGNOSTICS.proximity
  const live = tracks.filter((t) => t.state === 'confirmed')
  const out: Relation[] = []
  for (let i = 0; i < live.length; i++) {
    for (let j = i + 1; j < live.length; j++) {
      const a = live[i]!
      const b = live[j]!
      const rel = relativeGap(a.box, b.box)
      const overlap = iou(a.box, b.box)
      const touching = overlap > 0 || gap(a.box, b.box) <= touchGap
      if (!touching && rel > nearRelativeGap) continue
      let [subject, object] = area(a.box) <= area(b.box) ? [a, b] : [b, a]
      let kind: Relation['kind'] = overlap > 0 ? 'overlaps' : touching ? 'touching' : 'near'
      for (const [x, y] of [
        [a, b],
        [b, a],
      ] as const) {
        if (hasAttribute(y.label, 'surface') && !hasAttribute(x.label, 'surface') && containment(x.box, y.box) >= 0.5 && baseInside(x.box, y.box)) {
          subject = x
          object = y
          kind = 'on'
          break
        }
      }
      out.push({
        key: `${kind}:${subject.id}:${object.id}`,
        subject,
        object,
        kind,
        relativeGap: touching ? 0 : rel,
        text: `${subject.key} ${kind === 'on' ? 'on' : kind === 'near' ? 'near' : kind} ${object.key}`,
      })
    }
  }
  return out.sort((x, y) => RANK[x.kind] - RANK[y.kind] || x.relativeGap - y.relativeGap).slice(0, limit)
}
