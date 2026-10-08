/** Camera access with explicit, user-facing error classification. */
import { ApiError } from '../lib/api'

export type Facing = 'environment' | 'user'

function cameraError(code: string, message: string, hint: string): ApiError {
  return new ApiError(code, message, { hint })
}

export function classifyCameraError(error: unknown): ApiError {
  const name = error instanceof DOMException || error instanceof Error ? error.name : ''
  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return cameraError(
        'CAMERA_DENIED',
        'Camera permission was denied.',
        'Allow camera access in the browser’s site settings (the icon next to the address bar), then retry. Image Debug and Video Debug work without the camera.',
      )
    case 'NotFoundError':
    case 'OverconstrainedError':
      return cameraError('CAMERA_NOT_FOUND', 'No usable camera was found.', 'Connect a camera, or use Image Debug / Video Debug.')
    case 'NotReadableError':
    case 'AbortError':
      return cameraError(
        'CAMERA_IN_USE',
        'The camera is busy or could not be started.',
        'Close other apps or tabs that use the camera, then retry.',
      )
    default:
      return cameraError('CAMERA_UNSUPPORTED', 'The camera could not be started.', error instanceof Error ? error.message : '')
  }
}

export class CameraController {
  private stream: MediaStream | null = null
  facing: Facing = 'environment'
  private deviceIds: string[] = []
  private deviceIndex = 0

  private readonly video: HTMLVideoElement

  constructor(video: HTMLVideoElement) {
    this.video = video
  }

  get active(): boolean {
    return !!this.stream && this.stream.getVideoTracks().some((t) => t.readyState === 'live')
  }

  async start(facing: Facing = this.facing, deviceId?: string): Promise<void> {
    if (!window.isSecureContext) {
      throw cameraError(
        'CAMERA_INSECURE',
        'Browsers only allow camera access on HTTPS or localhost.',
        'On a phone, start the frontend with `npm run dev:https` and open the https:// LAN address (accept the self-signed certificate).',
      )
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw cameraError('CAMERA_UNSUPPORTED', 'This browser does not support camera access.', 'Use a recent Chrome, Safari, Edge or Firefox.')
    }
    this.stop()
    const video: MediaTrackConstraints = deviceId
      ? { deviceId: { exact: deviceId } }
      : { facingMode: { ideal: facing } }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { ...video, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 30 } },
      })
    } catch (error) {
      if (!deviceId && error instanceof DOMException && error.name === 'OverconstrainedError') {
        this.stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true }).catch((e: unknown) => {
          throw classifyCameraError(e)
        })
      } else {
        throw classifyCameraError(error)
      }
    }
    this.facing = facing
    this.video.srcObject = this.stream
    this.video.muted = true
    this.video.playsInline = true
    this.video.setAttribute('playsinline', '')
    await this.video.play().catch(() => undefined)
    await this.waitForFrames()
  }

  private waitForFrames(): Promise<void> {
    if (this.video.readyState >= 2 && this.video.videoWidth > 0) return Promise.resolve()
    return new Promise((resolve) => {
      const done = () => {
        this.video.removeEventListener('loadeddata', done)
        resolve()
      }
      this.video.addEventListener('loadeddata', done)
      window.setTimeout(done, 4000)
    })
  }

  /** Switch between front/back cameras (or cycle devices on desktops). */
  async switch(): Promise<void> {
    try {
      const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput')
      this.deviceIds = devices.map((d) => d.deviceId).filter(Boolean)
    } catch {
      this.deviceIds = []
    }
    const hasFacing = this.stream?.getVideoTracks()[0]?.getSettings().facingMode
    if (hasFacing || this.deviceIds.length < 2) {
      await this.start(this.facing === 'environment' ? 'user' : 'environment')
      return
    }
    this.deviceIndex = (this.deviceIndex + 1) % this.deviceIds.length
    await this.start(this.facing, this.deviceIds[this.deviceIndex])
  }

  async canSwitch(): Promise<boolean> {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices()
      return devices.filter((d) => d.kind === 'videoinput').length > 1
    } catch {
      return false
    }
  }

  pause(): void {
    this.video.pause()
  }

  async resume(): Promise<void> {
    await this.video.play().catch(() => undefined)
  }

  stop(): void {
    this.stream?.getTracks().forEach((t) => t.stop())
    this.stream = null
    this.video.srcObject = null
  }
}
