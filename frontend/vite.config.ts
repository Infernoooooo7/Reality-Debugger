import basicSsl from '@vitejs/plugin-basic-ssl'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// Reality Debugger dev/preview server.
//
//  npm run dev        -> http://0.0.0.0:5173  (desktop: http://localhost:5173)
//  npm run dev:https  -> https://0.0.0.0:5173 with a self-signed certificate,
//                        required for camera access from a phone on the LAN.
//
// Requests to /api are proxied to the FastAPI backend, so the browser only
// ever talks to one origin (no CORS or mixed-content problems on phones).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const https = mode === 'https' || env.HTTPS === 'true' || env.HTTPS === '1'
  const backend = env.BACKEND_URL || 'http://127.0.0.1:8000'
  const port = Number(env.PORT || 5173)

  const proxy = {
    '/api': {
      target: backend,
      changeOrigin: true,
      // Uploads and vision-model calls can take a while.
      timeout: 180_000,
      proxyTimeout: 180_000,
    },
  }

  return {
    plugins: [react(), ...(https ? [basicSsl({ name: 'reality-debugger' })] : [])],
    server: {
      host: '0.0.0.0',
      port,
      strictPort: true,
      allowedHosts: ['.local', 'localhost'],
      proxy,
    },
    preview: {
      host: '0.0.0.0',
      port: Number(env.PREVIEW_PORT || 4173),
      strictPort: true,
      allowedHosts: ['.local', 'localhost'],
      proxy,
    },
    worker: {
      format: 'es',
    },
    build: {
      target: 'es2022',
      chunkSizeWarningLimit: 1600,
    },
  }
})
