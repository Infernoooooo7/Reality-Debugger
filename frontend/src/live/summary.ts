import type { Report, ScanState } from '../lib/schemas'

/** Present a live scan's accumulated state as a report (for the summary and share card). */
export function scanToReport(scan: ScanState): Report {
  return {
    report_id: scan.scan_id,
    created_at: scan.updated_at,
    mode: 'live',
    personality: scan.personality,
    provider: scan.ai?.status === 'ok' && scan.ai.provider ? scan.ai.provider : 'local',
    model: scan.ai?.status === 'ok' && scan.ai.model ? scan.ai.model : (scan.engine ?? 'local-diagnostics'),
    engine: scan.engine ?? 'local-diagnostics',
    detectors: [],
    ai: scan.ai ?? { status: 'off' },
    latency_ms: 0,
    system_name: scan.system_name ?? 'UNKNOWN_SPACE_v0',
    scene: scan.scene ?? { name: 'UNKNOWN_SPACE', version: '0', summary: '', confidence: 0 },
    status: scan.status,
    system_score: scan.system_score ?? 0,
    ai_score: null,
    counts: scan.counts,
    objects: [],
    relationships: [],
    findings: scan.findings,
    optimizations: scan.optimizations,
    final_diagnosis: scan.final_diagnosis ?? 'Scan ended before a diagnosis was produced.',
    trigger: null,
    image: null,
    warnings: [],
  }
}
