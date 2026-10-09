// Scores the vision lab's still-pipeline run (vision-lab.html?still=1) against ground truth and writes a
// benchmark record to docs/benchmarks/records/.
//
//   python tools/evaluate.py composite --cols 6 --rows 6 --out data/interim/composites/desk_composite_6x6.jpg
//   cp data/interim/composites/*.jpg frontend/public/test-media/
//   node scripts/vision-lab.mjs "http://localhost:5173/vision-lab.html?still=1&images=desk_composite_6x6.jpg" lab.json
//   node scripts/still-eval.mjs lab.json ../data/interim/composites/desk_composite_6x6.gt.json [...]
//
// Matching: same label, IoU >= 0.5, greedy by score. Crowd regions are neither misses nor false positives.
// Reported at the display threshold 0.3 (what Image Debug shows and analyses).
import { execSync } from 'node:child_process'
import { readFileSync, writeFileSync } from 'node:fs'
import { basename, dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const [labFile, ...gtFiles] = process.argv.slice(2)
if (!labFile || !gtFiles.length) {
  console.error('usage: node scripts/still-eval.mjs <lab.json> <image.gt.json> [...]')
  process.exit(2)
}
const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..')
const lab = JSON.parse(readFileSync(labFile, 'utf8'))
const still = lab.results.still
const THRESHOLD = 0.3

const iou = (a, b) => {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]))
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]))
  const inter = ix * iy
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
  return union > 0 ? inter / union : 0
}
const size = (o) => (o.area_px < 32 * 32 ? 'small' : o.area_px < 96 * 96 ? 'medium' : 'large')
const r3 = (x) => (x === null ? null : Math.round(x * 1000) / 1000)

function score(dets, gt, width, height) {
  const objects = gt.objects.filter((o) => !o.iscrowd)
  const crowd = gt.objects.filter((o) => o.iscrowd)
  const used = new Set()
  const total = { small: 0, medium: 0, large: 0 }
  const hit = { small: 0, medium: 0, large: 0 }
  for (const o of objects) total[size(o)]++
  let tp = 0
  let fp = 0
  for (const d of dets.filter((x) => x.score >= THRESHOLD).sort((a, b) => b.score - a.score)) {
    const box = [d.box.x * width, d.box.y * height, (d.box.x + d.box.w) * width, (d.box.y + d.box.h) * height]
    let best = -1
    let bestIou = 0.5
    objects.forEach((o, i) => {
      if (used.has(i) || o.label !== d.label) return
      const v = iou(box, o.box)
      if (v >= bestIou) {
        bestIou = v
        best = i
      }
    })
    if (best >= 0) {
      used.add(best)
      tp++
      hit[size(objects[best])]++
    } else if (!crowd.some((c) => c.label === d.label && iou(box, c.box) >= 0.5)) fp++
  }
  return {
    detections: tp + fp,
    true_positives: tp,
    false_positives: fp,
    missed: objects.length - tp,
    precision: tp + fp ? r3(tp / (tp + fp)) : null,
    recall: objects.length ? r3(tp / objects.length) : null,
    recall_by_size: Object.fromEntries(Object.keys(total).map((k) => [k, { objects: total[k], recall: total[k] ? r3(hit[k] / total[k]) : null }])),
  }
}

const PIPELINES = {
  fast: 'fast detector alone',
  deepWhole: 'deep detector, one whole-image pass',
  deepTiled: 'deep detector, whole image + tiles (NMS)',
  fusedWhole: 'Image Debug before: fast + deep whole-image pass, fused',
  fusedTiled: 'Image Debug now: fast + deep tiled pass, fused',
}
const images = {}
for (const gtFile of gtFiles) {
  const gt = JSON.parse(readFileSync(gtFile, 'utf8'))
  for (const o of gt.objects) o.area_px ??= (o.box[2] - o.box[0]) * (o.box[3] - o.box[1])
  const name = basename(gtFile).replace('.gt.json', '.jpg')
  const r = still[name]
  if (!r) throw new Error(`${name} is not in ${labFile}`)
  const entry = {
    width: r.width,
    height: r.height,
    source_photos: gt.sources.length,
    annotated_objects: gt.objects.filter((o) => !o.iscrowd).length,
    timing_ms: { fast: r.fast.ms, deep_whole: r.deepWhole.ms, deep_tiled: r.deepTiled.ms },
    tiling: { passes: r.deepTiled.passes, tile_px: r.deepTiled.tilePx, incomplete: r.deepTiled.incomplete, boxes_before_merge: r.deepTiled.boxesBeforeMerge },
  }
  const dets = { fast: r.fast.detections, deepWhole: r.deepWhole.detections, deepTiled: r.deepTiled.detections, fusedWhole: r.fusedWhole, fusedTiled: r.fusedTiled }
  for (const key of Object.keys(PIPELINES)) entry[key] = score(dets[key], gt, r.width, r.height)
  images[name] = entry
  console.log(`\n${name} ${r.width}x${r.height} · ${entry.annotated_objects} annotated objects · tiled: ${r.deepTiled.passes} passes of ${r.deepTiled.tilePx ?? '-'} px, ${r.deepTiled.ms} ms (whole ${r.deepWhole.ms} ms)`)
  for (const key of Object.keys(PIPELINES)) {
    const s = entry[key]
    const sz = s.recall_by_size
    console.log(
      `  ${key.padEnd(10)} ${String(s.detections).padStart(4)} det · TP ${String(s.true_positives).padStart(3)} · FP ${String(s.false_positives).padStart(3)} · ` +
        `P ${s.precision} · R ${s.recall} (small ${sz.small.recall} · medium ${sz.medium.recall} · large ${sz.large.recall})`,
    )
  }
}

let commit = null
try {
  commit = execSync('git rev-parse --short HEAD', { cwd: ROOT }).toString().trim()
  if (execSync('git status --porcelain', { cwd: ROOT }).toString().trim()) commit += '+uncommitted'
} catch {
  commit = null
}
const env = lab.results.env ?? {}
const record = {
  schema_version: 1,
  task: 'still-pipeline',
  date: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
  model: { id: 'efficientdet_lite0+yolox_s', name: 'Image Debug detection pipeline in the browser', version: 'deployed' },
  dataset: { id: 'coco_val2017-desk-composites', split: Object.keys(images).join(',') },
  config: {
    display_threshold: THRESHOLD,
    match: 'same label, IoU >= 0.5, greedy by score; crowd regions ignored; COCO size buckets 32^2 / 96^2 px (in composite pixels)',
    pipelines: PIPELINES,
    tiling: 'config/vision.json deep.tiling',
    composites: 'tools/evaluate.py composite (COCO val2017 desk scenes, never upscaled)',
  },
  config_name: 'default',
  environment: {
    runtime: 'browser (vision-lab.html)',
    user_agent: env.userAgent ?? null,
    cores: env.hardwareConcurrency ?? null,
    cross_origin_isolated: env.crossOriginIsolated ?? null,
    fast: lab.results.fastInfo ? { delegate: lab.results.fastInfo.delegate, runtime: lab.results.fastInfo.runtime } : null,
    deep: lab.results.deepInfo ? { backend: lab.results.deepInfo.backend, threads: lab.results.deepInfo.threads } : null,
  },
  metrics: images,
  notes: [
    'Composites are made of real photos; they stand in for a high-resolution photo of a cluttered desk (the reported failure image is not available).',
    'COCO annotations are incomplete, so some "false positives" are real, unannotated objects.',
  ],
  git_commit: commit,
}
const out = join(ROOT, 'docs', 'benchmarks', 'records', `still-pipeline-efficientdet-lite0-yolox-s-coco-val2017-desk-composites-browser.json`)
writeFileSync(out, `${JSON.stringify(record, null, 2)}\n`)
console.log(`\nwrote ${out.replace(`${ROOT}/`, '')}`)
