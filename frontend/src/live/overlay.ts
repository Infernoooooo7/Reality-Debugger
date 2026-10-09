/**
 * Instrument overlay drawn on a canvas above the camera preview: tracked
 * objects (corner brackets + tags; ✓ = confirmed by both detectors,
 * ◆ = deep detector only), finding anchors and live signals.
 * Boxes are interpolated between vision results so motion stays smooth even
 * when the detector runs at 6-10 fps.
 */
import type { Finding } from '../lib/schemas'
import { lerpBox } from '../vision/geometry'
import type { NBox, Track } from '../vision/types'

export interface Anchor {
  finding: Finding
  box: NBox
  live: boolean // attached to a live track (vs. the static box from the analysed frame)
}

interface Smoothed {
  box: NBox
  track: Track
  seenAt: number
}

const SEVERITY_COLOR: Record<string, string> = {
  CRITICAL: '#ff3a5e',
  HIGH: '#ff5a36',
  MEDIUM: '#f3b33d',
  LOW: '#8db3cf',
  INFO: '#b0b7af',
}

export class OverlayRenderer {
  private ctx: CanvasRenderingContext2D
  private smoothed = new Map<number, Smoothed>()
  private anchors: Anchor[] = []
  private resolvedFlashes: { box: NBox; until: number; id: string }[] = []
  private motionBox: NBox | null = null
  private investigating = false
  private frozen = false
  private raf = 0
  private lastResultAt = 0
  private dpr = 1

  private readonly canvas: HTMLCanvasElement
  private readonly video: HTMLVideoElement

  constructor(canvas: HTMLCanvasElement, video: HTMLVideoElement) {
    this.canvas = canvas
    this.video = video
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Overlay canvas unavailable')
    this.ctx = ctx
  }

  start(): void {
    const loop = () => {
      this.draw()
      this.raf = requestAnimationFrame(loop)
    }
    this.raf = requestAnimationFrame(loop)
  }

  stop(): void {
    cancelAnimationFrame(this.raf)
  }

  setTracks(tracks: Track[], motionBox: NBox | null): void {
    const now = performance.now()
    this.lastResultAt = now
    const alive = new Set<number>()
    for (const track of tracks) {
      alive.add(track.id)
      const existing = this.smoothed.get(track.id)
      if (existing) {
        existing.track = track
        existing.seenAt = now
      } else {
        this.smoothed.set(track.id, { box: { ...track.box }, track, seenAt: now })
      }
    }
    for (const id of [...this.smoothed.keys()]) if (!alive.has(id)) this.smoothed.delete(id)
    this.motionBox = motionBox
  }

  setAnchors(anchors: Anchor[]): void {
    this.anchors = anchors
  }

  flashResolved(id: string, box: NBox): void {
    this.resolvedFlashes.push({ id, box, until: performance.now() + 2600 })
  }

  setInvestigating(on: boolean): void {
    this.investigating = on
  }

  setFrozen(on: boolean): void {
    this.frozen = on
  }

  clear(): void {
    this.smoothed.clear()
    this.anchors = []
    this.motionBox = null
  }

  /** Map normalised frame coords to canvas pixels, honouring object-fit: cover. */
  private mapper(): ((b: NBox) => [number, number, number, number]) | null {
    const vw = this.video.videoWidth
    const vh = this.video.videoHeight
    const rect = this.canvas.getBoundingClientRect()
    if (!vw || !vh || !rect.width || !rect.height) return null
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    const cw = Math.round(rect.width * dpr)
    const ch = Math.round(rect.height * dpr)
    if (this.canvas.width !== cw || this.canvas.height !== ch || this.dpr !== dpr) {
      this.canvas.width = cw
      this.canvas.height = ch
      this.dpr = dpr
    }
    const contain = getComputedStyle(this.video).objectFit === 'contain'
    const scale = contain ? Math.min(cw / vw, ch / vh) : Math.max(cw / vw, ch / vh)
    const dx = (cw - vw * scale) / 2
    const dy = (ch - vh * scale) / 2
    return (b) => [dx + b.x * vw * scale, dy + b.y * vh * scale, b.w * vw * scale, b.h * vh * scale]
  }

  private draw(): void {
    const map = this.mapper()
    const { ctx, canvas } = this
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (!map) return
    const now = performance.now()
    const d = this.dpr

    // Motion region
    if (this.motionBox && !this.frozen && now - this.lastResultAt < 600) {
      const [x, y, w, h] = map(this.motionBox)
      ctx.save()
      ctx.setLineDash([2 * d, 5 * d])
      ctx.strokeStyle = 'rgba(243,179,61,0.45)'
      ctx.lineWidth = 1 * d
      ctx.strokeRect(x, y, w, h)
      ctx.restore()
    }

    // Tracked objects
    const extrapolate = this.frozen ? 0 : Math.min(0.15, (now - this.lastResultAt) / 1000)
    for (const s of this.smoothed.values()) {
      const t = s.track
      const target: NBox = { x: t.box.x + t.vx * extrapolate, y: t.box.y + t.vy * extrapolate, w: t.box.w, h: t.box.h }
      s.box = this.frozen ? s.box : lerpBox(s.box, target, 0.35)
      const [x, y, w, h] = map(s.box)
      const alpha = t.state === 'confirmed' ? 0.95 : t.state === 'lost' ? 0.35 : 0.45
      // Deep-detector-only objects are drawn in the signal blue; both-detector agreement gets a check mark.
      const rgb = t.source === 'deep' ? '141,179,207' : '232,235,228'
      this.brackets(x, y, w, h, `rgba(${rgb},${alpha})`, t.state === 'tentative' || t.source === 'deep')
      if (t.state !== 'tentative') {
        const mark = t.verified ? ' ✓' : t.source === 'deep' ? ' ◆' : ''
        this.tag(x, y, `${t.key}  ${Math.round(t.score * 100)}%${mark}`, 'rgba(12,15,14,0.72)', `rgba(${rgb},${alpha})`)
      }
    }

    // Finding anchors (tags of anchors sharing a box are stacked, not overdrawn)
    const placed: [number, number][] = []
    for (const anchor of this.anchors) {
      const f = anchor.finding
      const color = SEVERITY_COLOR[f.severity] ?? '#f3b33d'
      const [x, y, w, h] = map(anchor.box)
      ctx.save()
      ctx.strokeStyle = color
      ctx.lineWidth = 2 * d
      if (f.status === 'DISCOVERED') ctx.setLineDash([7 * d, 5 * d])
      if (!anchor.live) ctx.globalAlpha = 0.75
      ctx.strokeRect(x, y, w, h)
      if (f.status === 'TRACKING') {
        const pulse = 0.5 + 0.5 * Math.sin(now / 260)
        ctx.globalAlpha = 0.25 + 0.5 * pulse
        ctx.lineWidth = 1 * d
        ctx.setLineDash([])
        ctx.strokeRect(x - 5 * d, y - 5 * d, w + 10 * d, h + 10 * d)
      }
      ctx.restore()
      const label = `${f.id} · ${f.severity}${f.status === 'DISCOVERED' ? ' ?' : ''}`
      let tagY = y + h
      while (placed.some(([px, py]) => Math.abs(px - x) < 40 * d && Math.abs(py - tagY) < 18 * d)) tagY += 19 * d
      placed.push([x, tagY])
      this.tag(x, tagY, label, color, '#070908', true)
    }

    // Recently resolved flashes
    this.resolvedFlashes = this.resolvedFlashes.filter((r) => r.until > now)
    for (const r of this.resolvedFlashes) {
      const [x, y, w, h] = map(r.box)
      const life = (r.until - now) / 2600
      ctx.save()
      ctx.globalAlpha = life
      ctx.strokeStyle = '#63e2a3'
      ctx.lineWidth = 3 * d
      ctx.strokeRect(x, y, w, h)
      ctx.restore()
      ctx.save()
      ctx.globalAlpha = life
      this.tag(x, y + h, `✓ ${r.id} RESOLVED`, '#63e2a3', '#070908', true)
      ctx.restore()
    }

    // Investigating sweep
    if (this.investigating) {
      const period = 1600
      const yPos = ((now % period) / period) * canvas.height
      const grad = ctx.createLinearGradient(0, yPos - 60 * d, 0, yPos)
      grad.addColorStop(0, 'rgba(255,90,54,0)')
      grad.addColorStop(1, 'rgba(255,90,54,0.22)')
      ctx.fillStyle = grad
      ctx.fillRect(0, yPos - 60 * d, canvas.width, 60 * d)
      ctx.fillStyle = 'rgba(255,90,54,0.7)'
      ctx.fillRect(0, yPos, canvas.width, 1 * d)
    }
  }

  private brackets(x: number, y: number, w: number, h: number, color: string, dashed: boolean): void {
    const { ctx } = this
    const d = this.dpr
    const len = Math.max(6 * d, Math.min(18 * d, w * 0.22, h * 0.22))
    ctx.save()
    ctx.strokeStyle = color
    ctx.lineWidth = 1.6 * d
    if (dashed) ctx.setLineDash([2 * d, 3 * d])
    ctx.beginPath()
    ctx.moveTo(x, y + len)
    ctx.lineTo(x, y)
    ctx.lineTo(x + len, y)
    ctx.moveTo(x + w - len, y)
    ctx.lineTo(x + w, y)
    ctx.lineTo(x + w, y + len)
    ctx.moveTo(x + w, y + h - len)
    ctx.lineTo(x + w, y + h)
    ctx.lineTo(x + w - len, y + h)
    ctx.moveTo(x + len, y + h)
    ctx.lineTo(x, y + h)
    ctx.lineTo(x, y + h - len)
    ctx.stroke()
    ctx.restore()
  }

  private tag(x: number, y: number, text: string, bg: string, fg: string, below = false): void {
    const { ctx } = this
    const d = this.dpr
    ctx.save()
    ctx.font = `500 ${10.5 * d}px "Martian Mono Variable", ui-monospace, monospace`
    const padX = 5 * d
    const h = 17 * d
    const w = ctx.measureText(text).width + padX * 2
    const ty = below ? Math.min(y + 2 * d, this.canvas.height - h) : Math.max(0, y - h - 2 * d)
    const tx = Math.min(Math.max(0, x), this.canvas.width - w)
    ctx.fillStyle = bg
    ctx.fillRect(tx, ty, w, h)
    ctx.fillStyle = fg
    ctx.textBaseline = 'middle'
    ctx.fillText(text, tx + padX, ty + h / 2 + 0.5 * d)
    ctx.restore()
  }
}
