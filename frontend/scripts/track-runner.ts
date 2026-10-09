/**
 * Runs the production tracker (src/vision/tracker.ts) over detection
 * sequences for offline evaluation (tools/evaluate.py tracking).
 *
 *   npm run build:tools          # bundles this file to .tools/track-runner.js
 *   node .tools/track-runner.js input.json output.json
 *
 * input:  { options?: Partial<TrackerOptions>, sequences: [{ id, frames: [{ t_ms, dets: [{label, score, box:{x,y,w,h}}], camera?: {dx,dy,confidence} }] }] }
 * output: { sequences: [{ id, frames: [{ tracks: [{ id, label, score, box, state }] }] }] }
 *
 * Only tracks seen in the frame (confirmed / recovered) are reported, as
 * ByteTrack reports activated tracks matched in the current frame.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { Tracker, type TrackerOptions } from '../src/vision/tracker'
import { isVisible, type Detection } from '../src/vision/types'

interface InFrame {
  t_ms: number
  dets: Detection[]
  camera?: { dx: number; dy: number; confidence: number } | null
}

interface Input {
  options?: Partial<TrackerOptions>
  sequences: { id: string; frames: InFrame[] }[]
}

const [input, output] = process.argv.slice(2)
if (!input || !output) {
  console.error('usage: node track-runner.js input.json output.json')
  process.exit(2)
}
const data = JSON.parse(readFileSync(input, 'utf8')) as Input
const sequences = data.sequences.map((seq) => {
  const tracker = new Tracker(data.options ?? {})
  const frames = seq.frames.map((f) => ({
    tracks: tracker
      .update(f.dets, f.t_ms, f.camera ?? null)
      .filter(isVisible)
      .map((t) => ({ id: t.id, label: t.label, score: t.score, box: t.box, state: t.state })),
  }))
  return { id: seq.id, frames }
})
writeFileSync(output, JSON.stringify({ sequences }))
