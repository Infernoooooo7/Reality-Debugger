/**
 * Object ontology generated from model and dataset metadata
 * (`config/ontology.generated.json`, built by `tools/build_ontology.py`):
 * for every label the bundled detectors can output, its COCO supercategory
 * and its semantic attributes (`electronic`, `liquid_container`, ...).
 *
 * The browser uses it for class-compatible track association and fusion,
 * and to describe relations; it never tests label strings itself.
 */
import ontologyJson from '../../../config/ontology.generated.json'

interface LabelInfo {
  supercategory?: string | null
  attributes?: string[]
}

const LABELS = (ontologyJson as { labels: Record<string, LabelInfo> }).labels
const ATTRIBUTES = Object.keys((ontologyJson as { attributes: Record<string, unknown> }).attributes)
const EMPTY: readonly string[] = []

export function knownLabels(): string[] {
  return Object.keys(LABELS).sort()
}

export function attributeNames(): string[] {
  return [...ATTRIBUTES].sort()
}

export function attributes(label: string): readonly string[] {
  return LABELS[label.toLowerCase()]?.attributes ?? EMPTY
}

export function hasAttribute(label: string, attribute: string): boolean {
  return attributes(label).includes(attribute)
}

export function supercategory(label: string): string | null {
  return LABELS[label.toLowerCase()]?.supercategory ?? null
}

/** Same label, or same COCO supercategory (how detectors confuse classes). */
export function compatible(a: string, b: string, mode: 'supercategory' | 'label' = 'supercategory'): boolean {
  if (a === b) return true
  if (mode === 'label') return false
  const sa = supercategory(a)
  return sa !== null && sa === supercategory(b)
}
