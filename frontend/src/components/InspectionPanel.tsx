/**
 * What a scan examined and whether that supports a conclusion
 * (backend/app/services/inspection.py, docs/SCORING.md). Shown with every
 * report so that "no findings" is never read as "nothing is wrong" when the
 * detectors failed, saw a downscaled image, or cannot recognise what is there.
 */
import { formatMs, formatPercent } from '../lib/format'
import type { DetectorReport, Inspection } from '../lib/schemas'

type Tone = 'ok' | 'warn' | 'bad'

const DETECTION: Record<Inspection['detection'], [string, Tone]> = {
  ok: ['succeeded', 'ok'],
  partial: ['partial - a detector did not run', 'warn'],
  failed: ['failed - nothing was examined', 'bad'],
  not_run: ['not run - nothing was examined', 'bad'],
}

const COVERAGE: Record<Inspection['coverage'], [string, Tone]> = {
  sufficient: ['no measured limitation', 'ok'],
  limited: ['limited', 'warn'],
  insufficient: ['insufficient', 'bad'],
}

const EVIDENCE: Record<Inspection['coverage'], [string, Tone]> = {
  sufficient: ['supports the result', 'ok'],
  limited: ['absent findings are not an all-clear', 'warn'],
  insufficient: ['insufficient for any conclusion', 'bad'],
}

const CHECK_LABEL: Record<string, string> = {
  dense_region: 'object concentration',
  overlap_cluster: 'overlapping groups',
  keep_clear_zone: 'keep-clear areas',
  spill_risk: 'drink near electronics',
  cable_congestion: 'cables',
  liquid_near_outlet: 'liquid near outlet',
}

function detectorLine(d: DetectorReport): string {
  const parts = [`${d.role}`, d.status === 'ok' ? `${d.boxes} boxes` : d.status]
  if (d.input_size) parts.push(`${d.input_size} px input`)
  if (d.passes > 1) parts.push(`${d.passes} passes (${d.tile_px ?? '?'} px tiles${d.incomplete ? ', stopped early' : ''})`)
  if (d.effective_scale != null) parts.push(`${formatPercent(d.effective_scale)} of full detail`)
  if (d.ms != null) parts.push(formatMs(d.ms))
  return parts.join(' · ')
}

/** "No findings" is only good news when the inspection could have found something. */
function findingsFact(inspection: Inspection): [string, string, Tone] {
  if (inspection.findings === 'findings') return ['Findings', 'findings detected', 'warn']
  if (inspection.analysis_status === 'detection_failed') return ['Findings', 'none - nothing was examined', 'bad']
  return ['Findings', 'no findings detected', inspection.coverage === 'sufficient' ? 'ok' : inspection.coverage === 'limited' ? 'warn' : 'bad']
}

export function InspectionPanel({ inspection }: { inspection: Inspection }) {
  const facts: [string, string, Tone][] = [
    ['Processing', 'completed', 'ok'],
    ['Detection', ...DETECTION[inspection.detection]],
    ['Coverage', ...COVERAGE[inspection.coverage]],
    findingsFact(inspection),
    inspection.findings === 'findings' && inspection.coverage === 'limited'
      ? ['Evidence', 'other issues may exist', 'warn']
      : ['Evidence', ...EVIDENCE[inspection.coverage]],
  ]
  const conf = inspection.confidence
  const structure = inspection.structure
  const quality = inspection.quality
  const image = inspection.image

  return (
    <div className="report__section inspection" data-coverage={inspection.coverage}>
      <div className="report__section-head">
        <span className="t-label">Inspection coverage</span>
        <span className="t-data t-muted">what was examined</span>
      </div>

      <dl className="inspection__facts t-data">
        {facts.map(([k, v, tone]) => (
          <div key={k} data-tone={tone}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>

      {inspection.reasons.length ? (
        <ul className="inspection__reasons">
          {inspection.reasons.map((r) => (
            <li key={r.code} data-level={r.level}>
              <span className="inspection__level t-data">{r.level === 'insufficient' ? 'Insufficient' : 'Limited'}</span>
              <span>{r.message}</span>
            </li>
          ))}
        </ul>
      ) : null}

      <dl className="inspection__numbers t-data">
        {inspection.detectors.map((d) => (
          <div key={`${d.role}-${d.model}`} data-status={d.status}>
            <dt>{d.model}</dt>
            <dd>
              {detectorLine(d)}
              {d.note && d.status !== 'ok' ? <em> - {d.note}</em> : null}
            </dd>
          </div>
        ))}
        <div>
          <dt>Recognised</dt>
          <dd>
            {inspection.objects} object{inspection.objects === 1 ? '' : 's'} in {inspection.categories.length} categor
            {inspection.categories.length === 1 ? 'y' : 'ies'}
            {inspection.categories.length ? `: ${inspection.categories.join(', ')}` : ''}
          </dd>
        </div>
        {conf ? (
          <div>
            <dt>Confidence</dt>
            <dd>
              min {formatPercent(conf.min)} · median {formatPercent(conf.median)} · max {formatPercent(conf.max)} · {conf.below_threshold ?? 0} below{' '}
              {formatPercent(conf.threshold)}
            </dd>
          </div>
        ) : null}
        {image ? (
          <div>
            <dt>Image</dt>
            <dd>
              {image.width}×{image.height} px
            </dd>
          </div>
        ) : null}
        {structure ? (
          <div>
            <dt>Detail outside objects</dt>
            <dd>
              {structure.unexplained_share == null ? 'no visible detail' : `${formatPercent(structure.unexplained_share)} of the visible detail`} · objects cover{' '}
              {formatPercent(structure.box_coverage)} of the image
            </dd>
          </div>
        ) : null}
        {quality && (quality.brightness != null || quality.sharpness != null) ? (
          <div>
            <dt>Image quality</dt>
            <dd>
              brightness {quality.brightness?.toFixed(2) ?? '—'} · sharpness {quality.sharpness?.toFixed(2) ?? '—'}
            </dd>
          </div>
        ) : null}
      </dl>

      {inspection.vocabulary ? <p className="inspection__vocab t-muted">{inspection.vocabulary}</p> : null}

      <details className="inspection__checks">
        <summary className="t-data">
          {inspection.checks_run.length} check{inspection.checks_run.length === 1 ? '' : 's'} run · {inspection.skipped.length} analys
          {inspection.skipped.length === 1 ? 'is' : 'es'} not run
        </summary>
        <p className="t-data">Run: {inspection.checks_run.map((c) => CHECK_LABEL[c] ?? c.replace(/_/g, ' ')).join(', ') || 'none'}</p>
        <ul>
          {inspection.skipped.map((s) => (
            <li key={s.analysis}>
              <b>Not run: {s.analysis}</b> - {s.reason}
            </li>
          ))}
        </ul>
        <p className="t-muted t-data">Score and status rules: {inspection.basis}</p>
      </details>
    </div>
  )
}
