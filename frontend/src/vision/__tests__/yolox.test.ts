import { describe, expect, it } from 'vitest'
import { decode, placement, toTensor, type YoloxParams } from '../deep/yolox'

// A tiny 64x64 "model" with three class slots (one unused, like COCO's gaps).
const params: YoloxParams = {
  inputSize: 64,
  padValue: 114,
  channelOrder: 'BGR',
  strides: [8, 16, 32],
  preNmsScore: 0.1,
  nmsIou: 0.45,
  scoreThreshold: 0.3,
  labels: ['cup', null, 'laptop'],
}
const WIDTH = 5 + params.labels.length
const ANCHORS = params.strides.reduce((n, s) => n + (params.inputSize / s) ** 2, 0)

function anchorIndex(stride: number, gx: number, gy: number): number {
  let index = 0
  for (const s of params.strides) {
    const size = params.inputSize / s
    if (s === stride) return index + gy * size + gx
    index += size * size
  }
  throw new Error(`no stride ${stride}`)
}

/** Raw anchor values as the exported model emits them (offsets, log sizes, sigmoid scores). */
function setAnchor(out: Float32Array, stride: number, gx: number, gy: number, a: { dx: number; dy: number; w: number; h: number; obj: number; cls: number[] }) {
  const o = anchorIndex(stride, gx, gy) * WIDTH
  out.set([a.dx, a.dy, Math.log(a.w / stride), Math.log(a.h / stride), a.obj, ...a.cls], o)
}

describe('YOLOX decode', () => {
  // Source image 128x64 -> ratio 0.5: it fills the top 32 rows of the 64x64 input.
  const place = placement(128, 64, params.inputSize)
  const out = new Float32Array(ANCHORS * WIDTH)
  // laptop, stride 8 cell (3,2): centre (28, 20), 16x12 px, score 0.9 x 0.8
  setAnchor(out, 8, 3, 2, { dx: 0.5, dy: 0.5, w: 16, h: 12, obj: 0.9, cls: [0.1, 0, 0.8] })
  // the same box from the stride-16 head with a lower score: removed by NMS
  setAnchor(out, 16, 1, 1, { dx: 0.75, dy: 0.25, w: 16, h: 12, obj: 0.9, cls: [0, 0, 0.6] })
  // cup, stride 32 cell (1,0): centre (48, 16), 8x8 px, score 0.95 x 0.9
  setAnchor(out, 32, 1, 0, { dx: 0.5, dy: 0.5, w: 8, h: 8, obj: 0.95, cls: [0.9, 0, 0] })
  // best class is the unused slot: dropped
  setAnchor(out, 8, 6, 1, { dx: 0.5, dy: 0.5, w: 8, h: 8, obj: 0.9, cls: [0, 0.9, 0] })
  // below the pre-NMS score: dropped
  setAnchor(out, 8, 0, 0, { dx: 0.5, dy: 0.5, w: 8, h: 8, obj: 0.3, cls: [0.2, 0, 0] })

  const detections = decode(out, params, place)

  it('keeps the best box per object, highest score first', () => {
    expect(detections.map((d) => [d.label, d.score])).toEqual([
      ['cup', 0.855],
      ['laptop', 0.72],
    ])
  })

  it('maps boxes back to the source image (top-left placement)', () => {
    const laptop = detections.find((d) => d.label === 'laptop')!.box
    expect(laptop.x).toBeCloseTo(20 / 0.5 / 128, 5)
    expect(laptop.y).toBeCloseTo(14 / 0.5 / 64, 5)
    expect(laptop.w).toBeCloseTo(16 / 0.5 / 128, 5)
    expect(laptop.h).toBeCloseTo(12 / 0.5 / 64, 5)
    const cup = detections.find((d) => d.label === 'cup')!.box
    expect(cup.x).toBeCloseTo(44 / 0.5 / 128, 5)
    expect(cup.y).toBeCloseTo(12 / 0.5 / 64, 5)
  })
})

describe('YOLOX preprocessing', () => {
  it('scales by the limiting side', () => {
    expect(placement(1280, 720, 640).ratio).toBe(0.5)
    expect(placement(480, 640, 640).ratio).toBe(1)
  })

  it('writes planar channels in BGR order by default (official preproc)', () => {
    const rgba = Uint8ClampedArray.of(10, 20, 30, 255)
    expect([...toTensor(rgba, 1, 'BGR')]).toEqual([30, 20, 10])
    expect([...toTensor(rgba, 1, 'RGB')]).toEqual([10, 20, 30])
  })
})
