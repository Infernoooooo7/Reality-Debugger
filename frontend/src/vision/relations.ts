/**
 * Local spatial heuristics over tracked objects. They never produce findings
 * on their own - they flag "interesting" relationships so the scan director
 * can ask the vision model to take a look, and they are passed to the model
 * as hints.
 */
import { gap } from './geometry'
import type { Track } from './types'

const ELECTRONICS = new Set(['laptop', 'keyboard', 'mouse', 'cell phone', 'tv', 'remote', 'microwave', 'toaster', 'oven'])
const CONTAINERS = new Set(['cup', 'bottle', 'wine glass', 'bowl', 'vase'])
const FOOD = new Set(['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'])
const SHARP = new Set(['knife', 'scissors'])

export interface Relation {
  key: string
  text: string
}

export function spatialRelations(tracks: Track[]): Relation[] {
  const live = tracks.filter((t) => t.state !== 'tentative')
  const out: Relation[] = []
  for (const a of live) {
    for (const b of live) {
      if (a.id === b.id) continue
      const d = gap(a.box, b.box)
      if (CONTAINERS.has(a.label) && ELECTRONICS.has(b.label) && d < 0.05) {
        out.push({ key: `spill:${a.id}:${b.id}`, text: `${a.key} is ${d === 0 ? 'touching' : 'next to'} ${b.key}` })
      } else if (FOOD.has(a.label) && ELECTRONICS.has(b.label) && d < 0.06) {
        out.push({ key: `food:${a.id}:${b.id}`, text: `${a.key} is next to ${b.key}` })
      }
    }
    if (SHARP.has(a.label)) out.push({ key: `sharp:${a.id}`, text: `${a.key} lying in the open` })
  }
  const containers = live.filter((t) => CONTAINERS.has(t.label))
  if (containers.length >= 3) out.push({ key: `stack:containers:${containers.length}`, text: `${containers.length} drink containers in view` })
  if (live.length >= 9) out.push({ key: `density:${Math.floor(live.length / 3)}`, text: `${live.length} objects in view` })
  return out.slice(0, 10)
}
