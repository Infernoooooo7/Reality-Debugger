/** Browser capability checks used by the self-test and error screens. */

export interface CameraSupport {
  available: boolean
  reason: 'ok' | 'insecure' | 'unsupported'
  detail: string
}

export function cameraSupport(): CameraSupport {
  if (!window.isSecureContext) {
    return {
      available: false,
      reason: 'insecure',
      detail: 'Camera access needs HTTPS (or localhost). On a phone, open the https:// address from `npm run dev:https`.',
    }
  }
  if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
    return {
      available: false,
      reason: 'unsupported',
      detail: 'This browser does not expose camera APIs. Try a recent Chrome, Safari, Edge or Firefox.',
    }
  }
  return { available: true, reason: 'ok', detail: 'Camera API available.' }
}

export function supportsWorkers(): boolean {
  return typeof Worker !== 'undefined'
}

export function supportsWasm(): boolean {
  return typeof WebAssembly === 'object' && typeof WebAssembly.instantiate === 'function'
}

export function isTouchDevice(): boolean {
  return window.matchMedia?.('(pointer: coarse)').matches ?? false
}

export function canShareFiles(file: File): boolean {
  try {
    return typeof navigator.canShare === 'function' && navigator.canShare({ files: [file] })
  } catch {
    return false
  }
}
