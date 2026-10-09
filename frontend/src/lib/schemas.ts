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
