/**
 * Map free-text object labels from the vision model ("coffee mug", "monitor")
 * onto the on-device detector's COCO vocabulary ("cup", "tv") so a finding can
 * be anchored to - and resolved by - live tracks.
 */
const SYNONYMS: Record<string, string> = {
  mug: 'cup',
  'coffee mug': 'cup',
  'coffee cup': 'cup',
  glass: 'cup',
  tumbler: 'cup',
  'water bottle': 'bottle',
  can: 'bottle',
  monitor: 'tv',
  screen: 'tv',
  display: 'tv',
  television: 'tv',
  'computer monitor': 'tv',
  notebook: 'laptop',
  macbook: 'laptop',
  computer: 'laptop',
  phone: 'cell phone',
  smartphone: 'cell phone',
  'mobile phone': 'cell phone',
  iphone: 'cell phone',
  'computer mouse': 'mouse',
  trackpad: 'mouse',
  'office chair': 'chair',
  stool: 'chair',
  sofa: 'couch',
  table: 'dining table',
  desk: 'dining table',
  plant: 'potted plant',
  houseplant: 'potted plant',
  books: 'book',
  'remote control': 'remote',
  controller: 'remote',
  bag: 'backpack',
  fridge: 'refrigerator',
}

const COCO = new Set([
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
  'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
  'scissors', 'teddy bear', 'hair drier', 'toothbrush',
])

export function toCoco(label: string): string | null {
  const clean = label.toLowerCase().replace(/[#·_]\d+$/, '').replace(/[^a-z ]/g, ' ').replace(/\s+/g, ' ').trim()
  if (!clean) return null
  if (COCO.has(clean)) return clean
  if (SYNONYMS[clean]) return SYNONYMS[clean]!
  // last word ("ceramic coffee mug" -> "mug" -> "cup")
  const words = clean.split(' ')
  for (let i = words.length - 1; i >= 0; i--) {
    const w = words[i]!
    if (COCO.has(w)) return w
    if (SYNONYMS[w]) return SYNONYMS[w]!
    if (w.endsWith('s') && COCO.has(w.slice(0, -1))) return w.slice(0, -1)
  }
  return null
}

export function cocoLabels(labels: string[]): string[] {
  const out = new Set<string>()
  for (const label of labels) {
    const coco = toCoco(label)
    if (coco && coco !== 'person' && coco !== 'dining table') out.add(coco)
  }
  return [...out]
}
