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
  test.skip(!res || !res.ok(), 'Backend is not reachable on :8000 - start it first (no API key needed).')
})

/** Collect API calls made by the page. */
function recordApi(page: Page): { url: string; method: string; hasImage: boolean }[] {
  const calls: { url: string; method: string; hasImage: boolean }[] = []
  page.on('request', (r) => {
    if (!r.url().includes('/api/')) return
    const body = r.postDataBuffer()
    calls.push({ url: r.url().replace(/^https?:\/\/[^/]+/, ''), method: r.method(), hasImage: Boolean(body?.includes(Buffer.from('image/jpeg'))) })
  })
  return calls
}

test('home screen: local CV active, AI optional', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: /your world has bugs/i })).toBeVisible()
  await expect(page.locator('.post__line', { hasText: 'Backend link' })).toHaveAttribute('data-state', 'ok')
  await expect(page.locator('.post__line', { hasText: 'Local CV engine' })).toHaveAttribute('data-state', 'ok')
  await expect(page.locator('.post__line', { hasText: 'Fast detector' })).toHaveAttribute('data-state', 'ok', { timeout: 120_000 })
  await expect(page.locator('.post__line', { hasText: 'Detector self-test' })).toContainText('PASS')
  // Without a key the AI layer is "off" - a normal state, never a fault.
  const ai = page.locator('.post__line', { hasText: 'AI reasoning' })
  const health = await (await page.request.get('/api/health')).json()
  if (!health.features.ai_reasoning) {
    await expect(ai).toContainText('OFF')
    await expect(ai).not.toHaveAttribute('data-state', 'fault')
  }
})

test('live scan: local observations only, deep scan, clear session', async ({ page }) => {
  const calls = recordApi(page)
  const firstObservation = page.waitForResponse((r) => r.url().includes('/api/scan/observe') && r.ok(), { timeout: 150_000 })
  await page.goto('/#/live')
  await expect(page.locator('.live')).toHaveAttribute('data-phase', 'running', { timeout: 120_000 })
  await expect(page.locator('.hud-fps__value')).not.toHaveText('--', { timeout: 30_000 })
  const observed = await (await firstObservation).json()
  expect(observed.report.engine).toMatch(/^local-diagnostics/)

  await page.getByRole('button', { name: 'Pause' }).click()
  await expect(page.locator('.live')).toHaveAttribute('data-phase', 'paused')
  await expect(page.locator('.plate__text')).toHaveAttribute('aria-label', /frozen/i)
  await page.getByRole('button', { name: 'Resume' }).click()
  await expect(page.locator('.live')).toHaveAttribute('data-phase', 'running')

  await page.locator('button.ctl--deep').click()
  await expect(page.locator('.deep .report')).toBeVisible({ timeout: 170_000 })
  await expect(page.locator('.deep .readout__value')).toBeVisible()
  await page.getByRole('button', { name: /return to live/i }).click()
  await expect(page.locator('.deep')).toHaveCount(0)

  const health = await (await page.request.get('/api/health')).json()
  if (!health.features.ai_reasoning) {
    // Local-only: no frame was ever uploaded.
    expect(calls.filter((c) => c.hasImage)).toEqual([])
    expect(calls.some((c) => c.url.startsWith('/api/scan/deep'))).toBe(true)
  }

  await page.getByRole('button', { name: 'End scan' }).click()
  const cleared = page.waitForResponse((r) => r.request().method() === 'DELETE' && r.url().includes('/api/scan/'))
  await page.getByRole('button', { name: /clear session/i }).click()
  expect((await cleared).status()).toBe(204)
  await expect(page).toHaveURL(/#\/$/)
})

test('image debug: fast + deep detectors, local diagnostic', async ({ page }) => {
  const calls = recordApi(page)
  await page.goto('/#/image')
  await page.locator('input[type=file]').first().setInputFiles({ name: 'stop.png', mimeType: 'image/png', buffer: await stopSignPng(page) })
  await expect(page.locator('.report')).toBeVisible({ timeout: 170_000 })
  await expect(page.locator('.stages li').nth(1)).toContainText('boxes')
  await expect(page.locator('.report .readout__value')).toBeVisible()
  await expect(page.locator('.report .scene-graph__node').first()).toContainText('stop sign')
  const health = await (await page.request.get('/api/health')).json()
  if (!health.features.ai_reasoning) {
    expect(calls.some((c) => c.url === '/api/analyze/scene')).toBe(true)
    expect(calls.filter((c) => c.hasImage)).toEqual([])
    await expect(page.locator('.ai-run')).toContainText('Local-only')
  }
})

test('image debug rejects files that are not images', async ({ page }) => {
  await page.goto('/#/image')
  await page.locator('input[type=file]').first().setInputFiles({ name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('hello') })
  await expect(page.locator('.error-panel')).toContainText('Unsupported file')
})

test('image debug: no findings is not a perfect score', async ({ page }) => {
  // Regression for the "false 100/100": recognised objects and no finding must not read as an all-clear.
  await page.goto('/#/image')
  await page.locator('input[type=file]').first().setInputFiles({ name: 'stop.png', mimeType: 'image/png', buffer: await stopSignPng(page) })
  await expect(page.locator('.report')).toBeVisible({ timeout: 170_000 })
  await expect(page.locator('.report .finding')).toHaveCount(0)
  await expect(page.locator('.report .readout__value')).toHaveText('UNRATED')
  await expect(page.locator('.report .chip--LIMITED')).toHaveText('LIMITED INSPECTION')
  await expect(page.locator('.report__diagnosis')).toContainText('not an all-clear')
  await expect(page.locator('.inspection')).toContainText('Only 80 object categories can be recognised')
  await expect(page.locator('.inspection')).toContainText('yolox_s')
})

test('image debug: a detector failure is reported, never as "no issues"', async ({ page }) => {
  await page.route('**/efficientdet_lite0.tflite', (route) => route.abort())
  await page.route('**/yolox_s.onnx', (route) => route.abort())
  await page.goto('/#/image')
  await page.locator('input[type=file]').first().setInputFiles({ name: 'stop.png', mimeType: 'image/png', buffer: await stopSignPng(page) })
  await expect(page.locator('.report')).toBeVisible({ timeout: 170_000 })
  await expect(page.locator('.stages li').nth(1)).toHaveAttribute('data-state', 'failed')
  await expect(page.locator('.stages li').nth(2)).toHaveAttribute('data-state', 'failed')
  await expect(page.locator('.report .chip--INCONCLUSIVE')).toBeVisible()
  await expect(page.locator('.report .readout__value')).toHaveText('UNRATED')
  await expect(page.locator('.report__diagnosis')).toContainText('No detector produced a result')
  await expect(page.locator('.report__diagnosis')).not.toContainText('No measurable issues')
})

test('AI failure keeps the local report and says so', async ({ page }) => {
  // Pretend an AI provider is configured, then make it fail.
  await page.route('**/api/health', async (route) => {
    const res = await route.fetch()
    const body = await res.json()
    body.ai = { ...body.ai, provider: 'gemini', model: 'gemini-flash-latest', configured: true, state: 'unverified', detail: 'test' }
    body.features = { ...body.features, ai_reasoning: true }
    await route.fulfill({ response: res, json: body })
  })
  await page.route('**/api/analyze/image', async (route) => {
    const form = route.request().postDataBuffer()
    expect(form?.includes(Buffer.from('name="scene"'))).toBe(true)
    // Forward to the real local-only backend, then mark the AI step as failed.
    const res = await route.fetch()
    const body = await res.json()
    body.ai = { status: 'unavailable', provider: 'gemini', model: 'gemini-flash-latest', error: { code: 'AI_RATE_LIMITED', message: 'The Gemini quota is exhausted.', retryable: true } }
    await route.fulfill({ response: res, json: body })
  })
  await page.goto('/#/image')
  await page.locator('input[type=file]').first().setInputFiles({ name: 'stop.png', mimeType: 'image/png', buffer: await stopSignPng(page) })
  await expect(page.locator('.report')).toBeVisible({ timeout: 170_000 })
  await expect(page.locator('.ai-run')).toContainText('AI reasoning unavailable')
  await expect(page.locator('.ai-run')).toContainText('Local CV results are complete')
  await expect(page.locator('.error-panel')).toHaveCount(0)
})

test('video debug: tracked samples, local timeline, nothing but numbers uploaded', async ({ page }) => {
  const calls = recordApi(page)
  await page.goto('/#/video')
  const clip = await recordedClip(page)
  await page.locator('input[type=file]').first().setInputFiles({ name: 'clip.webm', mimeType: 'video/webm', buffer: clip })
  const scan = page.getByRole('button', { name: /scan video/i })
  await expect(scan).toBeEnabled({ timeout: 30_000 })
  await scan.click()
  await expect(page.locator('.scanstats__phase b')).toHaveText(/complete/i, { timeout: 170_000 })
  await expect(page.locator('.timeline li').first()).toBeVisible()
  await expect(page.locator('.keyframes img').first()).toBeVisible()
  const health = await (await page.request.get('/api/health')).json()
  if (!health.features.ai_reasoning) expect(calls.filter((c) => c.hasImage)).toEqual([])
})
