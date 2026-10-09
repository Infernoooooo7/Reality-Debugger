/**
 * YOLOX pre- and post-processing, following the official ONNX demo
 * (YOLOX demo/ONNXRuntime/onnx_inference.py, yolox/data/data_augment.py
 * preproc() and yolox/utils/demo_utils.py demo_postprocess/multiclass_nms):
 *
 *  - resize keeping the aspect ratio, paste at the top-left of a 640x640
 *    canvas filled with 114, BGR channel order, float32 0-255 (no
 *    normalisation), NCHW;
 *  - decode (cx, cy, w, h) per anchor point of the stride-8/16/32 grids
 *    (sigmoid is already applied to objectness and class scores inside the
 *    exported model), score = objectness x best class score;
 *  - class-agnostic NMS.
 *
 * Every number comes from config/detection.json "deep" and the model manifest.
 */
import type { Detection } from '../types'

export interface YoloxParams {
  inputSize: number
  padValue: number
  channelOrder: 'BGR' | 'RGB'
  strides: number[]
  preNmsScore: number
  nmsIou: number
  scoreThreshold: number
  labels: (string | null)[]
}

/** Pixel layout of the letterboxed input. */
export interface Placement {
  ratio: number
  width: number // source width (px)
  height: number // source height (px)
}

export function placement(width: number, height: number, inputSize: number): Placement {
  return { ratio: Math.min(inputSize / width, inputSize / height), width, height }
}

/**
 * RGBA pixels of the padded square input (already drawn at the top-left with
 * the pad colour around it) -> NCHW float32 tensor data.
 */
export function toTensor(rgba: Uint8ClampedArray, inputSize: number, channelOrder: 'BGR' | 'RGB'): Float32Array {
  const plane = inputSize * inputSize
  const out = new Float32Array(3 * plane)
  const [c0, c2] = channelOrder === 'BGR' ? [2, 0] : [0, 2]
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    out[i] = rgba[p + c0]!
    out[plane + i] = rgba[p + 1]!
    out[2 * plane + i] = rgba[p + c2]!
  }
  return out
}

function iouXYXY(a: Float32Array, ai: number, b: Float32Array, bi: number): number {
  const x1 = Math.max(a[ai]!, b[bi]!)
  const y1 = Math.max(a[ai + 1]!, b[bi + 1]!)
  const x2 = Math.min(a[ai + 2]!, b[bi + 2]!)
  const y2 = Math.min(a[ai + 3]!, b[bi + 3]!)
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const areaA = (a[ai + 2]! - a[ai]!) * (a[ai + 3]! - a[ai + 1]!)
  const areaB = (b[bi + 2]! - b[bi]!) * (b[bi + 3]! - b[bi + 1]!)
  const union = areaA + areaB - inter
  return union > 0 ? inter / union : 0
}

/**
 * Decode the raw [1, anchors, 5 + classes] output into detections with
 * boxes normalised to the source image.
 */
export function decode(output: Float32Array, params: YoloxParams, place: Placement): Detection[] {
  const numClasses = params.labels.length
  const stride = 5 + numClasses
  const anchors = output.length / stride
  const boxes: number[] = []
  const scores: number[] = []
  const classes: number[] = []

  let index = 0
  for (const s of params.strides) {
    const size = Math.round(params.inputSize / s)
    for (let gy = 0; gy < size; gy++) {
      for (let gx = 0; gx < size; gx++, index++) {
        if (index >= anchors) break
        const o = index * stride
        const objectness = output[o + 4]!
        if (objectness < params.preNmsScore) continue
        let best = 0
        let bestScore = 0
        for (let c = 0; c < numClasses; c++) {
          const v = output[o + 5 + c]!
          if (v > bestScore) {
            bestScore = v
            best = c
          }
        }
        const score = objectness * bestScore
        if (score < params.preNmsScore) continue
        const cx = (output[o]! + gx) * s
        const cy = (output[o + 1]! + gy) * s
        const w = Math.exp(output[o + 2]!) * s
        const h = Math.exp(output[o + 3]!) * s
        boxes.push(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        scores.push(score)
        classes.push(best)
      }
    }
  }

  // Class-agnostic greedy NMS.
  const order = scores.map((_, i) => i).sort((a, b) => scores[b]! - scores[a]!)
  const xyxy = Float32Array.from(boxes)
  const keep: number[] = []
  const suppressed = new Uint8Array(scores.length)
  for (let a = 0; a < order.length; a++) {
    const i = order[a]!
    if (suppressed[i]) continue
    keep.push(i)
    for (let b = a + 1; b < order.length; b++) {
      const j = order[b]!
      if (!suppressed[j] && iouXYXY(xyxy, i * 4, xyxy, j * 4) > params.nmsIou) suppressed[j] = 1
    }
  }

  const out: Detection[] = []
  for (const i of keep) {
    if (scores[i]! < params.scoreThreshold) continue
    const label = params.labels[classes[i]!]
    if (!label) continue
    const x1 = Math.max(0, xyxy[i * 4]! / place.ratio / place.width)
    const y1 = Math.max(0, xyxy[i * 4 + 1]! / place.ratio / place.height)
    const x2 = Math.min(1, xyxy[i * 4 + 2]! / place.ratio / place.width)
    const y2 = Math.min(1, xyxy[i * 4 + 3]! / place.ratio / place.height)
    if (x2 - x1 < 0.005 || y2 - y1 < 0.005) continue
    out.push({ label, score: Math.round(scores[i]! * 1000) / 1000, box: { x: x1, y: y1, w: x2 - x1, h: y2 - y1 } })
  }
  return out
}
