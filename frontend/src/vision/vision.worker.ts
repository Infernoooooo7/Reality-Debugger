/**
 * Vision worker: keeps object detection, tracking and pixel analysis off the
 * main thread so the camera preview and UI never stutter.
 */
import { VisionEngine } from './engine'
import type { WorkerRequest, WorkerResponse } from './types'

interface WorkerScope {
  postMessage(message: WorkerResponse, transfer?: Transferable[]): void
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null
}

const scope = self as unknown as WorkerScope
const engine = new VisionEngine()
let ready = false

function send(message: WorkerResponse, transfer: Transferable[] = []): void {
  scope.postMessage(message, transfer)
}

scope.onmessage = async (event) => {
  const message = event.data
  switch (message.type) {
    case 'init': {
      try {
        const info = await engine.init(message.payload)
        ready = true
        send({ type: 'ready', info })
      } catch (error) {
        send({ type: 'init-error', message: error instanceof Error ? error.message : String(error) })
      }
      break
    }
    case 'frame': {
      if (!ready) {
        message.bitmap.close()
        send({ type: 'error', id: message.id, message: 'engine not ready' })
        break
      }
      try {
        const result = engine.processFrame(message.id, message.bitmap, message.timestamp)
        send({ type: 'frame', id: message.id, result })
      } catch (error) {
        send({ type: 'error', id: message.id, message: error instanceof Error ? error.message : String(error) })
      }
      break
    }
    case 'still': {
      if (!ready) {
        message.bitmap.close()
        send({ type: 'error', id: message.id, message: 'engine not ready' })
        break
      }
      try {
        const result = engine.analyzeStill(message.bitmap)
        send({ type: 'still', id: message.id, result }, [result.signature.hist.buffer, result.signature.grid.buffer])
      } catch (error) {
        send({ type: 'error', id: message.id, message: error instanceof Error ? error.message : String(error) })
      }
      break
    }
    case 'anchor':
      engine.setAnchor()
      break
    case 'reset':
      engine.reset()
      break
  }
}
