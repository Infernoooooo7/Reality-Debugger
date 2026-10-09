import { describe, expect, it } from 'vitest'
import { Tracker } from '../tracker'
import type { Detection, NBox, Track } from '../types'

const STEP = 100 // ms per frame: 10 fps, the tracker's nominal frame
const det = (label: string, score: number, box: NBox): Detection => ({ label, score, box })
const CUP: NBox = { x: 0.4, y: 0.4, w: 0.1, h: 0.15 }

/** Feed `frames` frames starting at `start` ms; returns the tracks of every frame. */
function feed(tracker: Tracker, frames: number, make: (i: number) => Detection[], start = 0): Track[][] {
  const out: Track[][] = []
  for (let i = 0; i < frames; i++) out.push(tracker.update(make(i), start + i * STEP))
  return out
}

describe('Tracker (ByteTrack)', () => {
  it('confirms a steady object on its second hit and keeps one id', () => {
    const tracker = new Tracker()
    const frames = feed(tracker, 20, () => [det('cup', 0.8, CUP)])
    expect(frames[0]!.map((t) => t.state)).toEqual(['tentative'])
    expect(frames[1]!.map((t) => t.state)).toEqual(['confirmed'])
    const ids = new Set(frames.flatMap((tracks) => tracks.map((t) => t.id)))
    expect(ids.size).toBe(1)
    const last = frames.at(-1)![0]!
    expect(last.movement).toBe('static')
    expect(last.speed).toBeLessThan(0.01)
    expect(tracker.drainEvents().map((e) => e.kind)).toEqual(['entered'])
  })

  it('measures the speed of a moving object and calls it moving', () => {
    const tracker = new Tracker()
    // 0.02 frame widths per 100 ms = 0.2 frame widths per second
    const frames = feed(tracker, 20, (i) => [det('cup', 0.8, { ...CUP, x: 0.1 + 0.02 * i })])
    const last = frames.at(-1)![0]!
    expect(last.movement).toBe('moving')
    expect(last.speed).toBeGreaterThan(0.18)
    expect(last.speed).toBeLessThan(0.22)
  })

  it('keeps a track alive on low-score detections (ByteTrack second association)', () => {
    const tracker = new Tracker()
    feed(tracker, 5, () => [det('cup', 0.8, CUP)])
    const [id] = tracker.current(400).map((t) => t.id)
    const low = feed(tracker, 3, () => [det('cup', 0.2, CUP)], 500)
    for (const tracks of low) {
      expect(tracks).toHaveLength(1)
      expect(tracks[0]).toMatchObject({ id, state: 'confirmed' })
    }
  })

  it('re-identifies an object after a short occlusion', () => {
    const tracker = new Tracker()
    feed(tracker, 5, () => [det('cup', 0.8, CUP)])
    const [id] = tracker.current(400).map((t) => t.id)
    const hidden = feed(tracker, 3, () => [], 500)
    expect(hidden.at(-1)![0]).toMatchObject({ id, state: 'lost' })
    const back = tracker.update([det('cup', 0.8, CUP)], 800)
    expect(back).toHaveLength(1)
    expect(back[0]).toMatchObject({ id, state: 'recovered' })
  })

  it('drops a lost track after the lost buffer and reports that it left', () => {
    const tracker = new Tracker()
    feed(tracker, 5, () => [det('cup', 0.8, CUP)]) // last seen at 400 ms
    tracker.drainEvents()
    feed(tracker, 10, () => [], 500) // up to 1400 ms: 1000 ms unseen, still within the buffer
    expect(tracker.current(1400)).toHaveLength(1)
    expect(tracker.update([], 1500)).toHaveLength(0)
    expect(tracker.drainEvents().map((e) => e.kind)).toEqual(['left'])
  })

  it('never extends a track with a detection of an unrelated class', () => {
    const tracker = new Tracker()
    feed(tracker, 5, () => [det('laptop', 0.8, CUP)])
    const [laptop] = tracker.current(400).map((t) => t.id)
    const tracks = tracker.update([det('cup', 0.8, CUP)], 500)
    // Not extended by the cup; it is coasting, and 'occluded' because the cup's box covers its prediction.
    expect(tracks.find((t) => t.id === laptop)).toMatchObject({ label: 'laptop', state: 'occluded' })
    expect(tracks.find((t) => t.label === 'cup')?.id).not.toBe(laptop)
  })

  it('compensates camera motion: a static object stays static while the camera pans', () => {
    // The camera pans so image content moves +0.02 frame widths per 100 ms (0.2 fw/s); the object is still.
    const pan = (i: number) => [det('cup', 0.8, { ...CUP, x: 0.1 + 0.02 * i })]
    const camera = { dx: 0.02, dy: 0, confidence: 0.9 }
    const withCmc = new Tracker({ cmc: true })
    let last: Track[] = []
    for (let i = 0; i < 20; i++) last = withCmc.update(pan(i), i * STEP, i ? camera : null)
    expect(last[0]!.speed).toBeLessThan(0.02)
    expect(last[0]!.movement).toBe('static')
    expect(last[0]!.apparentSpeed).toBeGreaterThan(0.18)

    const withoutCmc = new Tracker({ cmc: false })
    for (let i = 0; i < 20; i++) last = withoutCmc.update(pan(i), i * STEP, i ? camera : null)
    expect(last[0]!.movement).toBe('moving')
  })

  it('ignores an unreliable camera-motion estimate', () => {
    const tracker = new Tracker({ cmc: true, cmcMinConfidence: 0.25 })
    let last: Track[] = []
    for (let i = 0; i < 20; i++) last = tracker.update([det('cup', 0.8, { ...CUP, x: 0.1 + 0.02 * i })], i * STEP, { dx: 0.02, dy: 0, confidence: 0.1 })
    expect(last[0]!.movement).toBe('moving')
  })

  it('marks a re-acquired track as recovered for one frame and reports it', () => {
    const tracker = new Tracker()
    feed(tracker, 5, () => [det('cup', 0.8, CUP)])
    tracker.drainEvents()
    feed(tracker, 2, () => [], 500)
    const back = tracker.update([det('cup', 0.8, CUP)], 700)
    expect(back[0]).toMatchObject({ state: 'recovered' })
    expect(tracker.drainEvents().map((e) => e.kind)).toEqual(['recovered'])
    expect(tracker.update([det('cup', 0.8, CUP)], 800)[0]).toMatchObject({ state: 'confirmed' })
  })

  it('calls a lost track occluded when another tracked object covers its predicted box', () => {
    const tracker = new Tracker()
    const front: NBox = { x: 0.35, y: 0.35, w: 0.25, h: 0.3 }
    feed(tracker, 5, () => [det('cup', 0.8, CUP), det('laptop', 0.9, front)])
    const [cupId] = tracker.current(400).filter((t) => t.label === 'cup').map((t) => t.id)
    const tracks = tracker.update([det('laptop', 0.9, front)], 500) // the cup is hidden behind the laptop
    expect(tracks.find((t) => t.id === cupId)).toMatchObject({ state: 'occluded' })
  })

  it('lets a detection of the same COCO supercategory continue a track', () => {
    const tracker = new Tracker()
    feed(tracker, 5, () => [det('cup', 0.8, CUP)])
    const [id] = tracker.current(400).map((t) => t.id)
    const tracks = tracker.update([det('wine glass', 0.8, CUP)], 500) // both "kitchen"
    expect(tracks).toHaveLength(1)
    expect(tracks[0]).toMatchObject({ id, label: 'cup', state: 'confirmed' }) // label votes keep "cup"
  })
})
