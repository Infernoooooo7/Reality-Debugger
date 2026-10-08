import { expect, test, type Page } from '@playwright/test'

/** Draw a stop sign (a COCO class) in the page and return it as base64 PNG. */
async function stopSignPng(page: Page): Promise<Buffer> {
  const b64 = await page.evaluate(() => {
    const c = document.createElement('canvas')
    c.width = 640
    c.height = 480
    const ctx = c.getContext('2d')!
    ctx.fillStyle = '#8fa6bb'
    ctx.fillRect(0, 0, 640, 480)
    ctx.fillStyle = '#77777c'
    ctx.fillRect(305, 250, 30, 230)
    const oct = (r: number, color: string) => {
      ctx.beginPath()
      for (let i = 0; i < 8; i++) {
        const a = ((22.5 + 45 * i) * Math.PI) / 180
        ctx.lineTo(320 + r * Math.cos(a), 190 + r * Math.sin(a))
      }
      ctx.closePath()
      ctx.fillStyle = color
      ctx.fill()
    }
    oct(150, '#fff')
    oct(135, '#c8101e')
    ctx.fillStyle = '#fff'
    ctx.font = 'bold 64px sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText('STOP', 320, 190)
    return c.toDataURL('image/png').split(',')[1]!
  })
  return Buffer.from(b64, 'base64')
}

/** Record a short WebM in the page: a stop sign, then a hard cut to a plain scene. */
async function recordedClip(page: Page): Promise<Buffer> {
  const b64 = await page.evaluate(async () => {
    const c = document.createElement('canvas')
    c.width = 480
    c.height = 360
    const ctx = c.getContext('2d')!
    const stream = c.captureStream(15)
    const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' })
    const chunks: Blob[] = []
    recorder.ondataavailable = (e) => chunks.push(e.data)
    const done = new Promise<void>((resolve) => (recorder.onstop = () => resolve()))
    recorder.start(200)
    const started = performance.now()
    await new Promise<void>((resolve) => {
      const draw = () => {
        const t = (performance.now() - started) / 1000
        if (t < 2.2) {
          ctx.fillStyle = '#8fa6bb'
          ctx.fillRect(0, 0, 480, 360)
          ctx.fillStyle = '#c8101e'
          ctx.beginPath()
          for (let i = 0; i < 8; i++) {
            const a = ((22.5 + 45 * i) * Math.PI) / 180
            ctx.lineTo(240 + Math.sin(t) * 20 + 110 * Math.cos(a), 160 + 110 * Math.sin(a))
          }
          ctx.fill()
          ctx.fillStyle = '#fff'
          ctx.font = 'bold 48px sans-serif'
          ctx.textAlign = 'center'
          ctx.fillText('STOP', 240 + Math.sin(t) * 20, 176)
        } else {
          ctx.fillStyle = '#2b3a2f'
          ctx.fillRect(0, 0, 480, 360)
          ctx.fillStyle = '#e8c547'
          ctx.fillRect(60 + t * 10, 220, 160, 60)
        }
        if (t < 4) requestAnimationFrame(draw)
        else resolve()
      }
      draw()
    })
    recorder.stop()
    await done
    const buffer = await new Blob(chunks, { type: 'video/webm' }).arrayBuffer()
    let binary = ''
    const bytes = new Uint8Array(buffer)
    for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]!)
    return btoa(binary)
  })
  return Buffer.from(b64, 'base64')
}

test.beforeEach(async ({ request }) => {
  const res = await request.get('/api/health').catch(() => null)
  test.skip(!res || !res.ok(), 'Backend is not reachable on :8000 - start it first (DEMO MODE is fine).')
})

test('home screen runs a real power-on self test', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: /your world has bugs/i })).toBeVisible()
  await expect(page.locator('.post__line', { hasText: 'Backend link' })).toHaveAttribute('data-state', 'ok')
  await expect(page.locator('.post__line', { hasText: 'Vision engine' })).toHaveAttribute('data-state', 'ok', { timeout: 120_000 })
  await expect(page.locator('.post__line', { hasText: 'Detector self-test' })).toContainText('PASS')
})

test('live scan: camera, local ML, AI analysis, pause, deep scan, clear session', async ({ page }) => {
  const firstAnalysis = page.waitForResponse((r) => r.url().includes('/api/analyze/frame') && r.ok(), { timeout: 150_000 })
  await page.goto('/#/live')
  await expect(page.locator('.live')).toHaveAttribute('data-phase', 'running', { timeout: 120_000 })
  await expect(page.locator('.hud-fps__value')).not.toHaveText('--', { timeout: 30_000 })
  await firstAnalysis

  await page.getByRole('button', { name: 'Pause' }).click()
  await expect(page.locator('.live')).toHaveAttribute('data-phase', 'paused')
  await expect(page.locator('.plate__text')).toHaveAttribute('aria-label', /frozen/i)
  await page.getByRole('button', { name: 'Resume' }).click()
  await expect(page.locator('.live')).toHaveAttribute('data-phase', 'running')

  await page.locator('button.ctl--deep').click()
  await expect(page.locator('.deep .report')).toBeVisible({ timeout: 120_000 })
  await expect(page.locator('.deep .readout__value')).toBeVisible()
  await page.getByRole('button', { name: /return to live/i }).click()
  await expect(page.locator('.deep')).toHaveCount(0)
  await expect(page.locator('.live')).toHaveAttribute('data-phase', 'running')

  await page.getByRole('button', { name: 'End scan' }).click()
  const cleared = page.waitForResponse((r) => r.request().method() === 'DELETE' && r.url().includes('/api/scan/'))
  await page.getByRole('button', { name: /clear session/i }).click()
  expect((await cleared).status()).toBe(204)
  await expect(page).toHaveURL(/#\/$/)
})

test('image debug: local detection + backend diagnostic', async ({ page }) => {
  await page.goto('/#/image')
  await page.locator('input[type=file]').first().setInputFiles({ name: 'stop.png', mimeType: 'image/png', buffer: await stopSignPng(page) })
  await expect(page.locator('.report')).toBeVisible({ timeout: 150_000 })
  await expect(page.locator('.stages li[data-state="done"]')).toHaveCount(6)
  await expect(page.locator('.stages li').nth(1)).toContainText('objects')
  await expect(page.locator('.report .readout__value')).toBeVisible()
})

test('image debug rejects files that are not images', async ({ page }) => {
  await page.goto('/#/image')
  await page.locator('input[type=file]').first().setInputFiles({ name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('hello') })
  await expect(page.locator('.error-panel')).toContainText('Unsupported file')
})

test('AI failures are surfaced with recovery actions', async ({ page }) => {
  await page.route('**/api/analyze/image', (route) =>
    route.fulfill({
      status: 503,
      contentType: 'application/json',
      body: JSON.stringify({
        error: { code: 'AI_UNAVAILABLE', message: 'The vision model is temporarily unavailable.', hint: 'Try again in a moment.', retryable: true },
      }),
    }),
  )
  await page.goto('/#/image')
  await page.locator('input[type=file]').first().setInputFiles({ name: 'stop.png', mimeType: 'image/png', buffer: await stopSignPng(page) })
  await expect(page.locator('.error-panel')).toContainText('AI unavailable', { timeout: 150_000 })
  await expect(page.getByRole('button', { name: /retry/i })).toBeVisible()
  await expect(page.getByRole('button', { name: /demo mode/i })).toBeVisible()
})

test('video debug: on-device sampling, scene detection, timeline', async ({ page }) => {
  await page.goto('/#/video')
  const clip = await recordedClip(page)
  await page.locator('input[type=file]').first().setInputFiles({ name: 'clip.webm', mimeType: 'video/webm', buffer: clip })
  const scan = page.getByRole('button', { name: /scan video/i })
  await expect(scan).toBeEnabled({ timeout: 30_000 })
  await scan.click()
  await expect(page.locator('.scanstats__phase b')).toHaveText(/complete/i, { timeout: 170_000 })
  await expect(page.locator('.timeline li').first()).toBeVisible()
  await expect(page.locator('.keyframes img').first()).toBeVisible()
})
