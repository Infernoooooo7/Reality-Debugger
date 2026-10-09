import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'

// Bundles Node-side evaluation tools (scripts/track-runner.ts) that reuse the
// app's vision code. No public/ copy: the tools need no static assets.
export default defineConfig({
  publicDir: false,
  server: { fs: { allow: [fileURLToPath(new URL('..', import.meta.url))] } },
  build: {
    ssr: 'scripts/track-runner.ts',
    outDir: '.tools',
    emptyOutDir: true,
    target: 'node20',
  },
})
