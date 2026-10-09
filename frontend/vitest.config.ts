import { defineConfig } from 'vitest/config'

// Unit tests for the on-device vision code (pure TypeScript, run in Node).
// Browser end-to-end tests live in e2e/ and run with Playwright.
export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
  },
})
