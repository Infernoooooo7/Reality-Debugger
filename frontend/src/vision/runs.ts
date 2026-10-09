/**
 * What each detector actually did for a scene: status, boxes, network input
 * size and tiling. Sent in the scene model (stats.runs) so the backend can
 * tell "nothing was found" apart from "nothing was looked at"
 * (backend/app/services/inspection.py). Input size and vocabulary come from
 * the model manifests, never from constants typed into the app.
 */
import { loadManifest } from './models'
import type { DetectorRunPayload } from './scene'

export interface ModelFacts {
  inputSize: number | null
  vocabulary: number | null
}

/** Network input side and number of categories, from the model manifest (null when it cannot be read). */
export async function modelFacts(id: string): Promise<ModelFacts> {
  try {
    const manifest = await loadManifest(id)
    const vocabulary = manifest.labels.filter((label) => label !== null && label !== '').length
    return { inputSize: manifest.input?.width ?? null, vocabulary: vocabulary || null }
  } catch {
    return { inputSize: null, vocabulary: null }
  }
}

export function detectorRun(
  model: string,
  role: DetectorRunPayload['role'],
  status: DetectorRunPayload['status'],
  facts: ModelFacts,
  extra: Partial<Pick<DetectorRunPayload, 'boxes' | 'ms' | 'passes' | 'tile_px' | 'incomplete' | 'note'>> = {},
): DetectorRunPayload {
  return {
    model,
    role,
    status,
    boxes: extra.boxes ?? 0,
    ms: extra.ms === undefined || extra.ms === null ? null : Math.round(extra.ms),
    input_size: facts.inputSize,
    passes: extra.passes ?? (status === 'ok' ? 1 : 0),
    tile_px: extra.tile_px ?? null,
    incomplete: extra.incomplete ?? false,
    vocabulary: facts.vocabulary,
    note: extra.note ? extra.note.slice(0, 160) : null,
  }
}
