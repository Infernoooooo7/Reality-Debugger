/**
 * Keep-clear areas: parts of the image the user wants empty (the backend's
 * keep_clear_zone rule). Areas are image-relative, so they only make sense
 * for photos framed the same way; the last set is remembered on this device.
 *
 * Besides the recognised objects inside an area, the visible detail inside it
 * is measured (vision/coverage.ts), so that an area with no recognised object
 * is not reported as clear when it is full of unexplained structure.
 */
import { boxMask, luminanceSmall, structureMask } from './coverage'
import type { ZonePayload } from './scene'
import type { NBox } from './types'

const KEY = 'rd.zones.v1'
const MAX_ZONES = 8

export function loadZones(): ZonePayload[] {
  try {
    const raw = window.localStorage.getItem(KEY)
    const parsed = raw ? (JSON.parse(raw) as unknown) : []
    if (!Array.isArray(parsed)) return []
    return parsed
      .filter((z): z is ZonePayload => !!z && typeof z === 'object' && typeof (z as ZonePayload).id === 'string' && !!(z as ZonePayload).box)
      .slice(0, MAX_ZONES)
      .map((z) => ({ id: z.id, name: String(z.name ?? 'keep-clear area').slice(0, 48), kind: 'keep_clear', box: z.box }))
  } catch {
    return []
  }
}

export function saveZones(zones: ZonePayload[]): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(zones.map(({ id, name, kind, box }) => ({ id, name, kind, box }))))
  } catch {
    /* storage unavailable: zones last for this page only */
  }
}

export function newZone(box: NBox, existing: ZonePayload[]): ZonePayload | null {
  if (existing.length >= MAX_ZONES) return null
  const n = existing.length + 1
  return { id: `z${Date.now().toString(36)}${n}`, name: `keep-clear area ${n}`, kind: 'keep_clear', box }
}

/** Edge density inside the area, and the share of that detail outside every recognised box. */
export function zoneStructure(gray: Float32Array, w: number, h: number, zone: NBox, objects: NBox[]): { edge_density: number; unexplained_share: number | null } {
  const edges = structureMask(gray, w, h)
  const inZone = boxMask(w, h, [zone])
  const covered = boxMask(w, h, objects)
  let n = 0
  let e = 0
  let outside = 0
  for (let i = 0; i < edges.length; i++) {
    if (!inZone[i]) continue
    n++
    if (edges[i]) {
      e++
      if (!covered[i]) outside++
    }
  }
  const r4 = (v: number) => Math.round(v * 10000) / 10000
  return { edge_density: n ? r4(e / n) : 0, unexplained_share: e ? r4(outside / e) : null }
}

/** Zones with their measured structure, ready for the scene payload (measurement failures leave it out). */
export function measureZones(source: CanvasImageSource, width: number, height: number, zones: ZonePayload[], objects: NBox[]): ZonePayload[] {
  if (!zones.length) return []
  try {
    const { gray, w, h } = luminanceSmall(source, width, height)
    return zones.map((z) => ({ ...z, structure: zoneStructure(gray, w, h, z.box, objects) }))
  } catch {
    return zones
  }
}
