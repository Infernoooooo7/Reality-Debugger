/**
 * Shareable diagnostic: a 1080×1350 PNG card rendered on a canvas, plus a
 * plain-text report for the clipboard.
 */
import { engineLabel, pad2 } from './format'
import type { Report } from './schemas'

const W = 1080
const H = 1350

const COLORS = {
  ink: '#0c0f0e',
  panel: '#111513',
  line: '#29302b',
  chalk: '#e8ebe4',
  chalk2: '#b0b7af',
  chalk3: '#737c75',
  signal: '#ff5a36',
  amber: '#f3b33d',
  phosphor: '#63e2a3',
  critical: '#ff3a5e',
  hazard: '#f4d10b',
  steel: '#8db3cf',
}

const DISPLAY = '"Big Shoulders Display Variable", "Arial Narrow", sans-serif'
const BODY = '"Instrument Sans Variable", system-ui, sans-serif'
const DATA = '"Martian Mono Variable", ui-monospace, monospace'

function scoreColor(score: number): string {
  if (score >= 80) return COLORS.phosphor
  if (score >= 50) return COLORS.amber
  return COLORS.critical
}

function statusColor(status: Report['status']): string {
  const colors: Record<Report['status'], string> = {
    STABLE: COLORS.phosphor,
    LIMITED: COLORS.steel,
    INCONCLUSIVE: COLORS.chalk2,
    DEGRADED: COLORS.amber,
    CRITICAL: COLORS.critical,
  }
  return colors[status]
}

function statusText(status: Report['status']): string {
  return status === 'LIMITED' ? 'LIMITED INSPECTION' : status
}

function wrap(ctx: CanvasRenderingContext2D, text: string, maxWidth: number, maxLines: number): string[] {
  const words = text.split(/\s+/).filter(Boolean)
  const lines: string[] = []
  let line = ''
  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word
    if (ctx.measureText(candidate).width <= maxWidth) {
      line = candidate
      continue
    }
    if (line) lines.push(line)
    line = word
    if (lines.length === maxLines) break
  }
  if (lines.length < maxLines && line) lines.push(line)
  if (lines.length === maxLines && words.join(' ') !== lines.join(' ')) {
    let last = lines[maxLines - 1] ?? ''
    while (last && ctx.measureText(`${last}…`).width > maxWidth) last = last.slice(0, -1)
    lines[maxLines - 1] = `${last.trimEnd()}…`
  }
  return lines
}

function squiggle(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, color: string): void {
  ctx.save()
  ctx.strokeStyle = color
  ctx.lineWidth = 3
  ctx.beginPath()
  for (let i = 0; i <= width; i += 2) {
    const yy = y + Math.sin((i / 9) * Math.PI) * 3.5
    if (i === 0) ctx.moveTo(x + i, yy)
    else ctx.lineTo(x + i, yy)
  }
  ctx.stroke()
  ctx.restore()
}

async function ensureFonts(): Promise<void> {
  if (!document.fonts?.load) return
  await Promise.allSettled([
    document.fonts.load(`800 120px ${DISPLAY}`),
    document.fonts.load(`600 40px ${BODY}`),
    document.fonts.load(`400 24px ${DATA}`),
  ])
}

export interface CardImage {
  source: CanvasImageSource
  width: number
  height: number
}

export async function renderShareCard(report: Report, image: CardImage | null): Promise<Blob> {
  await ensureFonts()
  const canvas = document.createElement('canvas')
  canvas.width = W
  canvas.height = H
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Canvas unavailable')

  // Background + dot grid
  ctx.fillStyle = COLORS.ink
  ctx.fillRect(0, 0, W, H)
  ctx.fillStyle = 'rgba(232,235,228,0.07)'
  for (let y = 9; y < H; y += 18) for (let x = 9; x < W; x += 18) ctx.fillRect(x, y, 1.4, 1.4)

  const M = 64
  // Header
  ctx.fillStyle = COLORS.chalk
  ctx.font = `800 46px ${DISPLAY}`
  ctx.textBaseline = 'alphabetic'
  ctx.fillText('REALITY DEBUGGER', M, 96)
  ctx.fillStyle = COLORS.chalk3
  ctx.font = `400 20px ${DATA}`
  const stamp = new Date(report.created_at)
  const when = Number.isNaN(stamp.getTime()) ? '' : stamp.toISOString().slice(0, 16).replace('T', ' ')
  ctx.textAlign = 'right'
  ctx.fillText(`${report.mode.toUpperCase()} · ${when}`, W - M, 94)
  ctx.textAlign = 'left'

  // Tick ruler
  ctx.strokeStyle = COLORS.line
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(M, 124)
  ctx.lineTo(W - M, 124)
  ctx.stroke()
  for (let x = M, i = 0; x <= W - M; x += 12, i++) {
    ctx.beginPath()
    ctx.moveTo(x + 0.5, 124)
    ctx.lineTo(x + 0.5, i % 5 === 0 ? 136 : 130)
    ctx.stroke()
  }

  let y = 168
  // Optional photo panel with bug boxes
  if (image) {
    const panelW = W - 2 * M
    const panelH = 400
    const scale = Math.max(panelW / image.width, panelH / image.height)
    const dw = image.width * scale
    const dh = image.height * scale
    const dx = M + (panelW - dw) / 2
    const dy = y + (panelH - dh) / 2
    ctx.save()
    ctx.beginPath()
    ctx.rect(M, y, panelW, panelH)
    ctx.clip()
    ctx.drawImage(image.source, dx, dy, dw, dh)
    ctx.fillStyle = 'rgba(12,15,14,0.25)'
    ctx.fillRect(M, y, panelW, panelH)
    for (const f of report.findings) {
      if (!f.box || f.status === 'RESOLVED') continue
      const color = f.severity === 'CRITICAL' ? COLORS.critical : f.severity === 'HIGH' ? COLORS.signal : f.severity === 'MEDIUM' ? COLORS.amber : COLORS.chalk2
      ctx.strokeStyle = color
      ctx.lineWidth = 3
      const bx = dx + f.box.x * dw
      const by = dy + f.box.y * dh
      ctx.strokeRect(bx, by, f.box.w * dw, f.box.h * dh)
      ctx.fillStyle = color
      ctx.font = `700 20px ${DATA}`
      const label = ` ${f.id} `
      const tw = ctx.measureText(label).width
      ctx.fillRect(bx, by - 28, tw, 28)
      ctx.fillStyle = COLORS.ink
      ctx.fillText(label, bx, by - 8)
    }
    ctx.restore()
    ctx.strokeStyle = COLORS.line
    ctx.strokeRect(M + 0.5, y + 0.5, panelW - 1, panelH - 1)
    y += panelH + 90
  } else {
    y += 60
  }

  // System name (kept as-is: the lowercase "v" is part of the version)
  ctx.fillStyle = COLORS.chalk
  ctx.font = `800 ${image ? 72 : 96}px ${DISPLAY}`
  ctx.fillText(report.system_name, M, y)
  y += image ? 24 : 40

  // Score block: a number only when the inspection justifies one (docs/SCORING.md)
  const scoreSize = image ? 230 : 300
  const scoreY = y + scoreSize * 0.78
  let scoreW: number
  if (report.system_score == null) {
    ctx.font = `900 ${Math.round(scoreSize * 0.42)}px ${DISPLAY}`
    ctx.fillStyle = COLORS.chalk2
    ctx.fillText('UNRATED', M - 4, scoreY)
    scoreW = ctx.measureText('UNRATED').width - 150
  } else {
    const score = Math.round(report.system_score)
    ctx.font = `900 ${scoreSize}px ${DISPLAY}`
    ctx.fillStyle = scoreColor(score)
    ctx.fillText(String(score), M - 6, scoreY)
    scoreW = ctx.measureText(String(score)).width
    ctx.fillStyle = COLORS.chalk3
    ctx.font = `700 44px ${DISPLAY}`
    ctx.fillText('/ 100', M + scoreW + 8, scoreY)
  }

  // Status + counts column
  const colX = M + scoreW + 190
  ctx.font = `800 54px ${DISPLAY}`
  ctx.fillStyle = statusColor(report.status)
  ctx.fillText(statusText(report.status), colX, scoreY - scoreSize * 0.5)
  ctx.font = `400 24px ${DATA}`
  ctx.fillStyle = COLORS.chalk2
  ctx.fillText(`${pad2(report.counts.active_bugs)} ACTIVE BUGS`, colX, scoreY - scoreSize * 0.5 + 50)
  ctx.fillText(`${pad2(report.counts.high_priority)} HIGH PRIORITY`, colX, scoreY - scoreSize * 0.5 + 86)
  ctx.fillText(`${pad2(report.counts.optimizations)} OPTIMIZATIONS`, colX, scoreY - scoreSize * 0.5 + 122)
  y = scoreY + 70

  // Top issue (or an explicit all-clear)
  const top = report.findings.find((f) => f.status !== 'RESOLVED' && f.severity !== 'INFO')
  ctx.fillStyle = COLORS.chalk3
  ctx.font = `700 24px ${DISPLAY}`
  if (top) {
    ctx.fillText(`TOP ISSUE · ${top.id} · ${top.severity}`, M, y)
    y += 52
    ctx.fillStyle = COLORS.chalk
    ctx.font = `600 42px ${BODY}`
    for (const line of wrap(ctx, top.title, W - 2 * M, 2)) {
      ctx.fillText(line, M, y)
      squiggle(ctx, M, y + 12, Math.min(W - 2 * M, ctx.measureText(line).width), COLORS.signal)
      y += 58
    }
  } else {
    ctx.fillText('TOP ISSUE', M, y)
    y += 52
    ctx.fillStyle = COLORS.phosphor
    ctx.font = `600 42px ${BODY}`
    ctx.fillText('No active bugs. Suspiciously clean.', M, y)
    y += 58
  }
  y += 18

  // Final diagnosis
  ctx.fillStyle = COLORS.chalk3
  ctx.font = `700 24px ${DISPLAY}`
  ctx.fillText('FINAL DIAGNOSIS', M, y)
  y += 48
  ctx.fillStyle = COLORS.chalk
  ctx.font = `400 36px ${BODY}`
  const remainingLines = Math.max(1, Math.floor((H - 120 - y) / 48))
  for (const line of wrap(ctx, `“${report.final_diagnosis}”`, W - 2 * M, Math.min(4, remainingLines))) {
    ctx.fillText(line, M, y)
    y += 48
  }

  // Footer
  ctx.strokeStyle = COLORS.line
  ctx.beginPath()
  ctx.moveTo(M, H - 92)
  ctx.lineTo(W - M, H - 92)
  ctx.stroke()
  ctx.font = `400 20px ${DATA}`
  ctx.fillStyle = COLORS.chalk3
  ctx.fillText(engineLabel(report).toUpperCase().slice(0, 64), M, H - 52)
  ctx.textAlign = 'right'
  ctx.fillStyle = COLORS.chalk3
  ctx.fillText('YOUR WORLD HAS BUGS.', W - M, H - 52)
  ctx.textAlign = 'left'

  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('Could not encode the card'))), 'image/png')
  })
}

export function reportText(report: Report): string {
  const rule = '────────────────────────────────'
  const row = (k: string, v: string) => `${k.padEnd(15)}${v}`
  const lines = [
    'REALITY DIAGNOSTIC',
    rule,
    row('SYSTEM', report.system_name),
    row('STATUS', statusText(report.status)),
    row('SCENE SCORE', report.system_score == null ? 'UNRATED' : `${Math.round(report.system_score)} / 100`),
    ...(report.issue_score != null ? [row('ISSUE SCORE', `${report.issue_score} / 100 (open findings only)`)] : []),
    ...(report.inspection ? [row('ANALYSIS', `${report.inspection.analysis_status.replace('_', ' ')} · coverage ${report.inspection.coverage}`)] : []),
    row('ACTIVE BUGS', pad2(report.counts.active_bugs)),
    row('HIGH PRIORITY', pad2(report.counts.high_priority)),
    row('OPTIMIZATIONS', pad2(report.counts.optimizations)),
    rule,
  ]
  for (const f of report.findings) {
    lines.push(`[${f.severity}] ${f.id} · ${f.source === 'local' ? 'LOCAL CV' : 'AI'} · ${f.category} · ${f.status} · ${Math.round(f.confidence * 100)}%`)
    lines.push(`  ${f.title}`)
    if (f.evidence) lines.push(`  evidence: ${f.evidence}`)
    if (f.impact) lines.push(`  impact:   ${f.impact}`)
    if (f.recommendation) lines.push(`  fix:      ${f.recommendation}`)
    const numbers = Object.entries(f.measurements)
    if (numbers.length) lines.push(`  measured: ${numbers.map(([k, v]) => `${k}=${v}`).join(', ')}`)
    if (f.ai_note) lines.push(`  AI note:  ${f.ai_note}`)
    lines.push('')
  }
  if (report.optimizations.length) {
    lines.push('OPTIMIZATIONS')
    for (const o of report.optimizations) lines.push(`  + ${o.title} (${o.effort.toLowerCase()} effort)`)
    lines.push('')
  }
  if (report.inspection?.reasons.length) {
    lines.push('INSPECTION LIMITS')
    for (const r of report.inspection.reasons) lines.push(`  - ${r.message}`)
    lines.push('')
  }
  lines.push('FINAL DIAGNOSIS', `“${report.final_diagnosis}”`, rule)
  lines.push(`Reality Debugger · ${engineLabel(report)} · ${report.engine}`)
  return lines.join('\n')
}

export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    /* fall through */
  }
  const area = document.createElement('textarea')
  area.value = text
  area.setAttribute('readonly', '')
  area.style.position = 'fixed'
  area.style.opacity = '0'
  document.body.appendChild(area)
  area.select()
  let ok = false
  try {
    ok = document.execCommand('copy')
  } catch {
    ok = false
  }
  area.remove()
  return ok
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 2000)
}
