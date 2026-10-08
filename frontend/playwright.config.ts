import { defineConfig, devices } from '@playwright/test'

/**
 * End-to-end tests drive the real app in Chromium with a fake camera.
 *
 * Prerequisites:
 *   1. Backend running on :8000 (DEMO MODE is fine - no API key needed).
 *   2. `npx playwright install chromium` once (or set PW_CHROMIUM_PATH to an
 *      existing Chromium/Chrome binary).
 * Then: npm run test:e2e
 */
const executablePath = process.env.PW_CHROMIUM_PATH || undefined
const baseURL = process.env.E2E_BASE_URL || 'http://localhost:5173'

export default defineConfig({
  testDir: './e2e',
  timeout: 180_000,
  expect: { timeout: 30_000 },
  workers: 1,
  reporter: [['list']],
  outputDir: './test-results',
  use: {
    baseURL,
    trace: 'retain-on-failure',
    permissions: ['camera'],
    launchOptions: {
      executablePath,
      args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
    },
  },
  projects: [
    { name: 'mobile', use: { ...devices['Pixel 7'] } },
    { name: 'desktop', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } } },
  ],
  webServer: process.env.E2E_BASE_URL
    ? undefined
    : {
        command: 'npm run dev',
        url: baseURL,
        reuseExistingServer: true,
        timeout: 60_000,
      },
})
