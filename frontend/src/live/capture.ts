/**
 * Frame capture with reused canvases (no per-frame buffer churn):
 * - small ImageBitmaps for the vision worker (transferred, zero-copy)
 * - JPEG blobs for the backend, only when a frame is selected
 */

type Drawable = HTMLVideoElement | HTMLImageElement | HTMLCanvasElement | ImageBitmap

function sizeOf(source: Drawable): [number, number] {
  if (source instanceof HTMLVideoElement) return [source.videoWidth, source.videoHeight]
  if (source instanceof HTMLImageElement) return [source.naturalWidth, source.naturalHeight]
  return [source.width, source.height]
}

export function fit(width: number, height: number, maxEdge: number): [number, number] {
  const scale = Math.min(1, maxEdge / Math.max(width, height))
  return [Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale))]
}

export class FrameGrabber {
  private canvas: HTMLCanvasElement | null = null
  private resizeSupported = true

  /** Downscaled bitmap for the local engine (long edge `maxEdge`). */
  async bitmap(source: Drawable, maxEdge = 480): Promise<ImageBitmap | null> {
    const [w, h] = sizeOf(source)
    if (!w || !h) return null
    const [tw, th] = fit(w, h, maxEdge)
    if (this.resizeSupported) {
      try {
        return await createImageBitmap(source, { resizeWidth: tw, resizeHeight: th, resizeQuality: 'low' })
      } catch {
        this.resizeSupported = false
      }
    }
    const canvas = (this.canvas ??= document.createElement('canvas'))
    if (canvas.width !== tw) canvas.width = tw
    if (canvas.height !== th) canvas.height = th
    const ctx = canvas.getContext('2d')
    if (!ctx) return null
    ctx.drawImage(source, 0, 0, tw, th)
    return createImageBitmap(canvas)
  }
}

export class JpegCapturer {
  private readonly canvas = document.createElement('canvas')

  async capture(source: Drawable, maxEdge: number, quality = 0.85): Promise<{ blob: Blob; width: number; height: number }> {
    const [w, h] = sizeOf(source)
    if (!w || !h) throw new Error('No frame available yet')
    const [tw, th] = fit(w, h, maxEdge)
    if (this.canvas.width !== tw) this.canvas.width = tw
    if (this.canvas.height !== th) this.canvas.height = th
    const ctx = this.canvas.getContext('2d')
    if (!ctx) throw new Error('Canvas unavailable')
    ctx.drawImage(source, 0, 0, tw, th)
    const blob = await new Promise<Blob | null>((resolve) => this.canvas.toBlob(resolve, 'image/jpeg', quality))
    if (!blob) throw new Error('Could not encode the frame')
    return { blob, width: tw, height: th }
  }
}
