// Runs frontend/vision-lab.html in headless Chromium and saves its results.
//
//   npm run dev                                   (terminal 1)
//   node scripts/vision-lab.mjs "http://localhost:5173/vision-lab.html?images=bus.jpg,cats_and_dogs.jpg&runs=3&tracker=0" out.json
//   node scripts/vision-lab.mjs "<url>" out.json --enable-unsafe-webgpu   (also exposes WebGPU on Linux)
//
// Images are read from public/test-media/ (git-ignored). Set PW_CHROMIUM_PATH
// to use an existing Chromium instead of `npx playwright install chromium`.
// docs/benchmarks/browser_bench.json was assembled from runs of this script.
import { writeFileSync } from 'node:fs'
import { chromium } from '@playwright/test'

const [url, outFile, ...extraArgs] = process.argv.slice(2)
if (!url) {
  console.error('usage: node scripts/vision-lab.mjs <vision-lab url> [out.json] [extra Chromium flags]')
  process.exit(2)
}

const browser = await chromium.launch({
  executablePath: process.env.PW_CHROMIUM_PATH || undefined,
  // Software WebGL/WebGPU when no GPU is present, so delegate probes still run.
  args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader', ...extraArgs],
})
try {
  const page = await browser.newPage()
  page.on('pageerror', (error) => console.error('[pageerror]', error.message))
  await page.goto(url)
  await page.waitForFunction(() => window.__lab?.done, null, { timeout: 600_000 })
  console.log(await page.textContent('#out'))
  const lab = await page.evaluate(() => window.__lab)
  if (lab.error) process.exitCode = 1
  if (outFile) writeFileSync(outFile, `${JSON.stringify(lab, null, 2)}\n`)
} finally {
  await browser.close()
}
