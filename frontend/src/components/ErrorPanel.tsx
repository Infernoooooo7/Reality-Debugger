import type { ReactNode } from 'react'
import type { ApiError } from '../lib/api'

const TITLES: Record<string, string> = {
  BACKEND_UNREACHABLE: 'Backend offline',
  REQUEST_TIMEOUT: 'Response timed out',
  AI_NOT_CONFIGURED: 'AI reasoner not configured',
  AI_AUTH_FAILED: 'AI key rejected',
  AI_MODEL_NOT_FOUND: 'AI model not found',
  AI_RATE_LIMITED: 'AI rate limited',
  SCAN_RATE_LIMITED: 'Analysis budget reached',
  AI_TIMEOUT: 'AI took too long',
  AI_UNAVAILABLE: 'AI unavailable',
  AI_REFUSED: 'AI declined this frame',
  AI_MALFORMED_RESPONSE: 'Unreadable AI answer',
  AI_REQUEST_REJECTED: 'AI rejected the request',
  UNSUPPORTED_MEDIA_TYPE: 'Unsupported file',
  PAYLOAD_TOO_LARGE: 'File too large',
  INVALID_UPLOAD: 'Corrupted file',
  VIDEO_UNREADABLE: 'Cannot read this video',
  INVALID_RESPONSE: 'Version mismatch',
  CAMERA_DENIED: 'Camera permission denied',
  CAMERA_NOT_FOUND: 'No camera found',
  CAMERA_IN_USE: 'Camera busy',
  CAMERA_INSECURE: 'Camera needs HTTPS',
  CAMERA_UNSUPPORTED: 'Camera not supported',
  VISION_FAILED: 'Vision engine failed',
}

export function errorTitle(error: ApiError): string {
  return TITLES[error.code] ?? 'Something went wrong'
}

export function ErrorPanel({ error, actions }: { error: ApiError; actions?: ReactNode }) {
  return (
    <div className="error-panel" role="alert">
      <div className="error-panel__code">ERR · {error.code}</div>
      <h2 className="error-panel__title">{errorTitle(error)}</h2>
      <p className="error-panel__hint">
        {error.message}
        {error.hint ? (
          <>
            <br />
            <span className="t-muted">{error.hint}</span>
          </>
        ) : null}
      </p>
      {actions ? <div className="error-panel__actions">{actions}</div> : null}
    </div>
  )
}
