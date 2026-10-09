/**
 * Runtime schemas for every backend response. The frontend never trusts the
 * network: responses are validated before they reach the UI.
 */
import { z } from 'zod'

export const SeveritySchema = z.enum(['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'])
export const CategorySchema = z.enum([
  'EFFICIENCY',
  'ORGANIZATION',
  'CONSISTENCY',
  'WORKFLOW',
  'SAFETY',
  'ERGONOMICS',
  'AESTHETICS',
  'SPATIAL',
  'TECH_DEBT',
  'ABSURD',
])
export const FindingStatusSchema = z.enum(['DISCOVERED', 'CONFIRMED', 'TRACKING', 'RESOLVED'])
export const SystemStatusSchema = z.enum(['STABLE', 'DEGRADED', 'CRITICAL'])
export const PersonalitySchema = z.enum(['serious', 'brutal', 'unhinged'])
export const ModeSchema = z.enum(['live', 'deep', 'image', 'video'])
export const TriggerSchema = z.enum([
  'first_look',
  'new_object',
  'scene_change',
  'relationship',
  'confirmation',
  'interval',
  'deep_scan',
  'freeze',
  'manual',
  'observe',
  'user_explain',
  'confirmed_finding',
  'ambiguous',
])

export const BoxSchema = z.object({ x: z.number(), y: z.number(), w: z.number(), h: z.number() })

export const StatusChangeSchema = z.object({
  status: FindingStatusSchema,
  at: z.string(),
  note: z.string().nullish(),
  video_t: z.number().nullish(),
})

export const FindingSchema = z.object({
  id: z.string(),
  severity: SeveritySchema,
  category: CategorySchema,
  title: z.string(),
  evidence: z.string(),
  inference: z.string(),
  impact: z.string(),
  recommendation: z.string(),
  confidence: z.number(),
  quip: z.string().default(''),
  status: FindingStatusSchema,
  box: BoxSchema.nullish(),
  related_objects: z.array(z.string()).default([]),
  object_ids: z.array(z.string()).default([]),
  source: z.enum(['local', 'ai']),
  rule: z.string().nullish(),
  measurements: z.record(z.string(), z.union([z.number(), z.string()])).default({}),
  ai_note: z.string().nullish(),
  ai_agrees: z.boolean().nullish(),
  sightings: z.number().default(1),
  out_of_view: z.boolean().default(false),
  first_seen_at: z.string().nullish(),
  last_seen_at: z.string().nullish(),
  first_seen_s: z.number().nullish(),
  last_seen_s: z.number().nullish(),
  resolved_note: z.string().nullish(),
  history: z.array(StatusChangeSchema).default([]),
})

export const DetectedObjectSchema = z.object({
  id: z.string(),
  label: z.string(),
  confidence: z.number(),
  box: BoxSchema.nullish(),
  source: z.enum(['fast', 'deep', 'fused', 'ai']),
  verified: z.boolean().default(false),
})

export const RelationshipSchema = z.object({
  subject: z.string(),
  relation: z.string(),
  object: z.string(),
  observation: z.string().default(''),
  source: z.enum(['local', 'ai']).default('local'),
})

export const OptimizationSchema = z.object({
  id: z.string(),
  title: z.string(),
  description: z.string().default(''),
  effort: z.enum(['LOW', 'MEDIUM', 'HIGH']),
  impact: z.string().default(''),
})

export const SceneSchema = z.object({
  name: z.string(),
  version: z.string(),
  summary: z.string(),
  confidence: z.number(),
})

export const CountsSchema = z.object({
  active_bugs: z.number(),
  high_priority: z.number(),
  optimizations: z.number(),
  resolved: z.number(),
})

export const ErrorBodySchema = z.object({
  code: z.string(),
  message: z.string(),
  hint: z.string().nullish(),
  retryable: z.boolean().optional(),
})

/** What the optional AI layer did for one request. */
export const AIRunSchema = z.object({
  status: z.enum(['off', 'ok', 'cached', 'skipped', 'unavailable']),
  provider: z.string().nullish(),
  model: z.string().nullish(),
  latency_ms: z.number().nullish(),
  trigger: z.string().nullish(),
  reason: z.string().nullish(),
  error: ErrorBodySchema.nullish(),
})

export const ReportSchema = z.object({
  report_id: z.string(),
  created_at: z.string(),
  mode: ModeSchema,
  personality: PersonalitySchema,
  provider: z.string(),
  model: z.string(),
  engine: z.string(),
  detectors: z.array(z.string()).default([]),
  ai: AIRunSchema,
  latency_ms: z.number(),
  system_name: z.string(),
  scene: SceneSchema,
  status: SystemStatusSchema,
  system_score: z.number(),
  ai_score: z.number().nullish(),
  counts: CountsSchema,
  objects: z.array(DetectedObjectSchema).default([]),
  relationships: z.array(RelationshipSchema).default([]),
  findings: z.array(FindingSchema).default([]),
  optimizations: z.array(OptimizationSchema).default([]),
  final_diagnosis: z.string(),
  trigger: TriggerSchema.nullish(),
  image: z
    .object({
      width: z.number(),
      height: z.number(),
      original_width: z.number().nullish(),
      original_height: z.number().nullish(),
      bytes_sent: z.number().nullish(),
    })
    .nullish(),
  warnings: z.array(z.string()).default([]),
})

export const LifecycleEventSchema = z.object({
  id: z.string(),
  type: z.enum(['DISCOVERED', 'CONFIRMED', 'TRACKING', 'RESOLVED', 'REOPENED']),
  finding_id: z.string(),
  title: z.string(),
  severity: SeveritySchema,
  at: z.string(),
  note: z.string().nullish(),
})

export const ScanStateSchema = z.object({
  scan_id: z.string(),
  created_at: z.string(),
  updated_at: z.string(),
  personality: PersonalitySchema,
  analyses: z.number(),
  observations: z.number().default(0),
  status: SystemStatusSchema,
  system_score: z.number().nullish(),
  system_name: z.string().nullish(),
  scene: SceneSchema.nullish(),
  final_diagnosis: z.string().nullish(),
  findings: z.array(FindingSchema).default([]),
  optimizations: z.array(OptimizationSchema).default([]),
  counts: CountsSchema,
  events: z.array(LifecycleEventSchema).default([]),
  ai: AIRunSchema.nullish(),
  engine: z.string().nullish(),
})

export const AISuggestionSchema = z.object({
  trigger: TriggerSchema,
  reason: z.string(),
  finding_ids: z.array(z.string()).default([]),
})

export const ScanAnalysisSchema = z.object({
  scan: ScanStateSchema,
  report: ReportSchema,
  events: z.array(LifecycleEventSchema),
  ai_suggestion: AISuggestionSchema.nullish(),
})

export const TimelineEventSchema = z.object({
  t: z.number(),
  frame_index: z.number().nullish(),
  kind: z.enum([
    'DISCOVERED',
    'CONFIRMED',
    'ESCALATED',
    'RESOLVED',
    'OBSERVATION',
    'SCENE_CHANGE',
    'OBJECT_ENTERED',
    'OBJECT_LEFT',
  ]),
  severity: SeveritySchema,
  finding_id: z.string().nullish(),
  text: z.string(),
  source: z.enum(['local', 'ai']),
})

export const VideoReportSchema = z.object({
  report: ReportSchema,
  timeline: z.array(TimelineEventSchema),
  video: z.object({
    duration_s: z.number(),
    width: z.number().nullish(),
    height: z.number().nullish(),
    fps: z.number().nullish(),
    name: z.string().nullish(),
    size_bytes: z.number().nullish(),
  }),
  sampling: z.object({
    processed_on: z.enum(['browser', 'server']),
    sampled_frames: z.number(),
    scenes: z.number(),
    redundant_removed: z.number(),
    keyframes: z.number(),
    tracked_objects: z.number().default(0),
  }),
  keyframes: z.array(z.object({ index: z.number(), t: z.number(), scene: z.number(), reason: z.string() })),
})

export const AIStatusSchema = z.object({
  provider: z.string(),
  model: z.string().nullish(),
  configured: z.boolean(),
  state: z.enum(['off', 'ready', 'unverified', 'unavailable']),
  detail: z.string(),
  fallback: z.string().nullish(),
  last_error: ErrorBodySchema.nullish(),
})

export const HealthSchema = z.object({
  status: z.literal('ok'),
  version: z.string(),
  time: z.string(),
  ai: AIStatusSchema,
  local: z.object({
    engine: z.string(),
    rules: z.array(z.string()),
    labels: z.number(),
    attributes: z.array(z.string()),
    detectors: z.array(z.string()),
  }),
  limits: z.record(z.string(), z.number()),
  features: z.record(z.string(), z.boolean()),
})

export const AICheckSchema = z.object({
  ok: z.boolean(),
  provider: z.string(),
  model: z.string().nullish(),
  latency_ms: z.number(),
  error: ErrorBodySchema.nullish(),
})

export const MetricsSchema = z.object({
  uptime_s: z.number(),
  sessions: z.number(),
  local: z.record(z.string(), z.unknown()),
  ai: z.record(z.string(), z.unknown()),
})

export type Severity = z.infer<typeof SeveritySchema>
export type Category = z.infer<typeof CategorySchema>
export type FindingStatus = z.infer<typeof FindingStatusSchema>
export type SystemStatus = z.infer<typeof SystemStatusSchema>
export type Personality = z.infer<typeof PersonalitySchema>
export type Trigger = z.infer<typeof TriggerSchema>
export type Box = z.infer<typeof BoxSchema>
export type Finding = z.infer<typeof FindingSchema>
export type DetectedObject = z.infer<typeof DetectedObjectSchema>
export type Relationship = z.infer<typeof RelationshipSchema>
export type Optimization = z.infer<typeof OptimizationSchema>
export type Report = z.infer<typeof ReportSchema>
export type LifecycleEvent = z.infer<typeof LifecycleEventSchema>
export type ScanState = z.infer<typeof ScanStateSchema>
export type ScanAnalysis = z.infer<typeof ScanAnalysisSchema>
export type TimelineEvent = z.infer<typeof TimelineEventSchema>
export type VideoReport = z.infer<typeof VideoReportSchema>
export type Health = z.infer<typeof HealthSchema>
export type AIStatus = z.infer<typeof AIStatusSchema>
export type AICheck = z.infer<typeof AICheckSchema>
export type AIRun = z.infer<typeof AIRunSchema>
export type AISuggestion = z.infer<typeof AISuggestionSchema>
export type Metrics = z.infer<typeof MetricsSchema>

// ------------------------------------------------------------ server-side vision (/api/vision)

export const AnalysisStateSchema = z.enum(['complete', 'analysis_incomplete', 'model_unavailable', 'insufficient_image_quality'])
export const ObjectStateSchema = z.enum(['detected', 'tentative', 'ambiguous'])
export const QueryStateSchema = z.enum([
  'detected',
  'tentative',
  'ambiguous',
  'not_detected',
  'unsupported_category',
  'insufficient_image_quality',
  'analysis_incomplete',
])

export const QualitySchema = z.object({
  ok: z.boolean(),
  brightness: z.number(),
  sharpness: z.number(),
  width: z.number(),
  height: z.number(),
  issues: z.array(z.object({ code: z.string(), message: z.string() })),
})

export const VisionModelStatusSchema = z.object({
  id: z.string(),
  name: z.string().nullable(),
  tasks: z.array(z.string()),
  available: z.boolean(),
  reason: z.string().nullable(),
  loaded: z.boolean(),
  load_error: z.string().nullable(),
  licence: z.string().nullable(),
})

export const VisionStatusSchema = z.object({
  enabled: z.boolean(),
  models: z.array(VisionModelStatusSchema),
  memory_budget_mb: z.number(),
  threads: z.number(),
  profiles: z.record(z.string(), z.object({ name: z.string(), status: z.string() })),
})

export const ProfileSchema = z.object({
  name: z.string(),
  status: z.enum(['available', 'limited', 'experimental', 'unavailable']),
  summary: z.string(),
  engines: z.array(z.string()),
  categories: z.union([z.literal('all'), z.array(z.string())]),
  default_mode: z.string(),
  unsupported: z.array(z.string()),
  evidence: z.array(z.string()),
  safety: z.string().optional(),
  notes: z.array(z.string()).optional(),
})
export const ProfilesSchema = z.record(z.string(), ProfileSchema)

export const VisionObjectSchema = z.object({
  id: z.number(),
  label: z.string(),
  class_id: z.number(),
  confidence: z.number(),
  box: z.tuple([z.number(), z.number(), z.number(), z.number()]),
  state: ObjectStateSchema,
  alternative_label: z.string().nullable(),
  in_profile: z.boolean(),
  source: z.string(),
})

export const QueryAnswerSchema = z.object({
  query: z.string(),
  state: QueryStateSchema,
  labels: z.array(z.string()),
  match: z.string().nullable(),
  count: z.number(),
  explanation: z.string(),
})

export const VisionDetectSchema = z.object({
  state: AnalysisStateSchema,
  mode: z.enum(['standard', 'precision']),
  profile: z.string(),
  model: z.object({ id: z.string(), version: z.string(), vocabulary: z.string(), operating_threshold: z.number() }),
  image: z.object({ width: z.number(), height: z.number() }),
  coordinate_system: z.string(),
  objects: z.array(VisionObjectSchema),
  queries: z.array(QueryAnswerSchema),
  quality: QualitySchema,
  timings_ms: z.record(z.string(), z.number()),
  notes: z.array(z.string()),
})

export const VisionCompareSchema = z.object({
  state: AnalysisStateSchema,
  verdict: z.enum(['anomalous', 'within_reference_variation']),
  score: z.number(),
  threshold: z.number(),
  score_to_threshold: z.number(),
  regions: z.array(z.object({ box: z.tuple([z.number(), z.number(), z.number(), z.number()]), area_px: z.number(), peak_score: z.number() })),
  references: z.number(),
  method: z.object({
    name: z.string(),
    features: z.string(),
    memory_bank_patches: z.number(),
    threshold_rule: z.string(),
    leave_one_out_scores: z.array(z.number()),
  }),
  heatmap: z.object({ width: z.number(), height: z.number(), encoding: z.string(), scale: z.string(), data: z.string().nullable() }),
  quality: QualitySchema,
  image: z.object({ width: z.number(), height: z.number() }),
  timings_ms: z.record(z.string(), z.number()),
  notes: z.array(z.string()),
  evidence: z.string(),
})

export type VisionStatus = z.infer<typeof VisionStatusSchema>
export type Profile = z.infer<typeof ProfileSchema>
export type Profiles = z.infer<typeof ProfilesSchema>
export type VisionObject = z.infer<typeof VisionObjectSchema>
export type QueryAnswer = z.infer<typeof QueryAnswerSchema>
export type VisionDetect = z.infer<typeof VisionDetectSchema>
export type VisionCompare = z.infer<typeof VisionCompareSchema>
