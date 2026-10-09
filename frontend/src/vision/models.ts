/**
 * Model manifests (`public/models/<id>.manifest.json`, written by
 * `tools/fetch_models.py`): file, checksum, licence, paper, preprocessing
 * and - most importantly - the class list read from the model's own
 * metadata. Nothing about a model's vocabulary is typed into the app.
 */
import { z } from 'zod'

const ManifestSchema = z.object({
  id: z.string(),
  name: z.string(),
  role: z.enum(['fast', 'deep']),
  file: z.string(),
  sha256: z.string(),
  runtime: z.string(),
  license: z.string(),
  dataset: z.string(),
  paper: z.string(),
  bytes: z.number(),
  num_classes: z.number(),
  labels: z.array(z.string().nullable()),
  labels_source: z.string(),
  input: z
    .object({
      width: z.number(),
      height: z.number(),
      layout: z.string().optional(),
      channel_order: z.string().optional(),
      pad_value: z.number().optional(),
      placement: z.string().optional(),
    })
    .optional(),
  decode: z.object({ strides: z.array(z.number()) }).optional(),
  reported: z.record(z.string(), z.union([z.number(), z.string()])).optional(),
})

export type ModelManifest = z.infer<typeof ManifestSchema>

export function modelUrl(file: string): string {
  return `${import.meta.env.BASE_URL}models/${file}`
}

const cache = new Map<string, Promise<ModelManifest>>()

/** Load and validate a model manifest (absolute URL or model id). */
export function loadManifest(idOrUrl: string): Promise<ModelManifest> {
  const url = idOrUrl.includes('/') ? idOrUrl : modelUrl(`${idOrUrl}.manifest.json`)
  let pending = cache.get(url)
  if (!pending) {
    pending = fetch(url)
      .then(async (res) => {
        if (!res.ok) throw new Error(`Model manifest not found (${res.status}): ${url}`)
        return ManifestSchema.parse(await res.json())
      })
      .catch((error: unknown) => {
        cache.delete(url)
        throw error
      })
    cache.set(url, pending)
  }
  return pending
}
