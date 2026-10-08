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
  source: z.enum(['ai', 'demo']),
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
  source: z.enum(['ai', 'local']),
})

export const RelationshipSchema = z.object({
  subject: z.string(),
  relation: z.string(),
  object: z.string(),
  observation: z.string().default(''),
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

export const ReportSchema = z.object({
  report_id: z.string(),
  created_at: z.string(),
  mode: ModeSchema,
  personality: PersonalitySchema,
  provider: z.string(),
  model: z.string(),
  simulated: z.boolean(),
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
  status: SystemStatusSchema,
  system_score: z.number().nullish(),
  system_name: z.string().nullish(),
  scene: SceneSchema.nullish(),
  final_diagnosis: z.string().nullish(),
  findings: z.array(FindingSchema).default([]),
  optimizations: z.array(OptimizationSchema).default([]),
  counts: CountsSchema,
  events: z.array(LifecycleEventSchema).default([]),
  simulated: z.boolean(),
  provider: z.string().nullish(),
  model: z.string().nullish(),
})

export const ScanAnalysisSchema = z.object({
  scan: ScanStateSchema,
  report: ReportSchema,
  events: z.array(LifecycleEventSchema),
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
  source: z.enum(['ai', 'demo', 'local']),
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
  }),
  keyframes: z.array(z.object({ index: z.number(), t: z.number(), scene: z.number(), reason: z.string() })),
})

export const HealthSchema = z.object({
  status: z.literal('ok'),
  version: z.string(),
  time: z.string(),
  ai: z.object({
    provider: z.string(),
    model: z.string().nullish(),
    configured: z.boolean(),
    simulated: z.boolean(),
    detail: z.string(),
  }),
  limits: z.record(z.string(), z.number()),
  features: z.record(z.string(), z.boolean()),
})

export const ErrorBodySchema = z.object({
  code: z.string(),
  message: z.string(),
  hint: z.string().nullish(),
  retryable: z.boolean().optional(),
})

export const AICheckSchema = z.object({
  ok: z.boolean(),
  provider: z.string(),
  model: z.string().nullish(),
  latency_ms: z.number(),
  simulated: z.boolean(),
  error: ErrorBodySchema.nullish(),
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
export type AICheck = z.infer<typeof AICheckSchema>
