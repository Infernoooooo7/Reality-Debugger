/// <reference types="vite/client" />

// Font package without bundled type declarations (side-effect CSS import).
declare module '@fontsource-variable/big-shoulders-display'

interface ImportMetaEnv {
  /** Optional absolute backend URL; defaults to same-origin /api via the dev proxy. */
  readonly VITE_API_BASE?: string
}
