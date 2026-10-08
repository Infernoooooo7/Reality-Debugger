"""Diagnostic engine: turns validated model output into findings with a
lifecycle (DISCOVERED -> CONFIRMED -> TRACKING -> RESOLVED), scores and reports.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Iterable, Sequence
from datetime import datetime

from app.schemas.analysis import AIDiagnosis, AIFinding
from app.schemas.common import (
    SEVERITY_RANK,
    AnalysisMode,
    FindingStatus,
    Personality,
    ScanTrigger,
    Severity,
    SystemStatus,
)
from app.schemas.diagnostics import (
    Counts,
    DetectedObject,
    DiagnosticReport,
    Finding,
    ImageMeta,
    LifecycleEvent,
    Optimization,
    Relationship,
    ScanState,
    SceneInfo,
    StatusChange,
)
from app.services.ai_service import DiagnoseResult
from app.services.prompts import ActiveFindingBrief
from app.services.scan_store import ScanSession, TrackedFinding, utcnow
from app.services.vision_service import PreparedImage
from app.utils import coerce

SEVERITY_PENALTY = {
    Severity.CRITICAL: 30,
    Severity.HIGH: 16,
    Severity.MEDIUM: 8,
    Severity.LOW: 3,
    Severity.INFO: 0,
}

CONFIRM_ON_FIRST_SIGHT = 0.8  # deep/image analyses this confident start CONFIRMED


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _penalty(findings: Iterable[Finding | TrackedFinding]) -> float:
    return sum(
        SEVERITY_PENALTY[f.severity] * (0.5 + 0.5 * f.confidence)
        for f in findings
        if f.status != FindingStatus.RESOLVED
    )


def compute_score(ai_score: int | None, findings: Sequence[Finding | TrackedFinding]) -> int:
    """Blend the model's holistic score with a severity-weighted penalty.

    The penalty part makes the score move when individual bugs are resolved,
    and keeps an over-generous model score honest when serious bugs are open.
    """
    computed = max(5.0, 100.0 - _penalty(findings))
    if ai_score is None:
        return int(round(computed))
    return int(round(0.5 * ai_score + 0.5 * computed))


def system_status(score: int, findings: Sequence[Finding | TrackedFinding]) -> SystemStatus:
    active = [f for f in findings if f.status != FindingStatus.RESOLVED]
    if score < 45 or any(f.severity == Severity.CRITICAL for f in active):
        return SystemStatus.CRITICAL
    if score < 80 or any(f.severity == Severity.HIGH for f in active):
        return SystemStatus.DEGRADED
    return SystemStatus.STABLE


def count(findings: Sequence[Finding | TrackedFinding], optimizations: Sequence[Optimization]) -> Counts:
    active = [f for f in findings if f.status != FindingStatus.RESOLVED and f.severity != Severity.INFO]
    return Counts(
        active_bugs=len(active),
        high_priority=sum(1 for f in active if SEVERITY_RANK[f.severity] >= SEVERITY_RANK[Severity.HIGH]),
        optimizations=len(optimizations),
        resolved=sum(1 for f in findings if f.status == FindingStatus.RESOLVED),
    )


def system_name(scene: SceneInfo) -> str:
    name = re.sub(r"[^A-Z0-9]+", "_", scene.name.upper()).strip("_") or "UNKNOWN_SPACE"
    version = re.sub(r"[^0-9A-Za-z.\-]", "", scene.version.lstrip("vV")) or "1.0"
    return f"{name}_v{version}"


def _sort_key(f: Finding | TrackedFinding) -> tuple[int, int, float]:
    return (
        1 if f.status == FindingStatus.RESOLVED else 0,
        -SEVERITY_RANK[f.severity],
        -f.confidence,
    )


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------


def scene_info(diagnosis: AIDiagnosis) -> SceneInfo:
    s = diagnosis.scene
    return SceneInfo(name=s.name or "UNKNOWN_SPACE", version=s.version or "1.0", summary=s.summary, confidence=s.confidence)


def objects_from(diagnosis: AIDiagnosis, simulated: bool) -> list[DetectedObject]:
    return [
        DetectedObject(
            id=coerce.identifier(o.id, f"obj_{i + 1:02d}"),
            label=o.label,
            confidence=o.confidence,
            box=o.box,
            source="local" if simulated else "ai",
        )
        for i, o in enumerate(diagnosis.objects)
    ]


def relationships_from(diagnosis: AIDiagnosis) -> list[Relationship]:
    return [
        Relationship(subject=r.subject, relation=r.relation, object=r.object, observation=r.observation)
        for r in diagnosis.relationships
        if r.subject and r.relation and r.object
    ]


def optimizations_from(diagnosis: AIDiagnosis) -> list[Optimization]:
    return [
        Optimization(
            id=coerce.identifier(o.id, f"opt_{i + 1:02d}"),
            title=o.title,
            description=o.description,
            effort=o.effort,
            impact=o.impact,
        )
        for i, o in enumerate(diagnosis.optimizations[:5])
    ]


def stateless_findings(diagnosis: AIDiagnosis, *, simulated: bool, now: datetime) -> tuple[list[Finding], dict[str, str]]:
    """Findings for one-shot analyses (image mode). Returns findings and a map
    from the model's ids to canonical BUG-xxx ids."""
    findings: list[Finding] = []
    id_map: dict[str, str] = {}
    for i, af in enumerate(diagnosis.findings):
        canonical = f"BUG-{i + 1:03d}"
        if af.id:
            id_map[af.id] = canonical
        status = FindingStatus.CONFIRMED if af.confidence >= 0.75 else FindingStatus.DISCOVERED
        findings.append(_finding(af, canonical, status, simulated, now))
    findings.sort(key=_sort_key)
    return findings, id_map


def _finding(af: AIFinding, finding_id: str, status: FindingStatus, simulated: bool, now: datetime) -> Finding:
    return Finding(
        id=finding_id,
        severity=af.severity,
        category=af.category,
        title=af.title,
        evidence=af.evidence,
        inference=af.inference,
        impact=af.impact,
        recommendation=af.recommendation,
        confidence=af.confidence,
        quip=af.quip,
        status=status,
        box=af.box,
        related_objects=list(af.related_objects),
        source="demo" if simulated else "ai",
        first_seen_at=now,
        last_seen_at=now,
        history=[StatusChange(status=status, at=now)],
    )


def build_report(
    result: DiagnoseResult,
    *,
    mode: AnalysisMode,
    personality: Personality,
    findings: list[Finding],
    trigger: ScanTrigger | None = None,
    image: PreparedImage | None = None,
    score_findings: Sequence[Finding | TrackedFinding] | None = None,
) -> DiagnosticReport:
    diagnosis = result.diagnosis
    scene = scene_info(diagnosis)
    optimizations = optimizations_from(diagnosis)
    scored = score_findings if score_findings is not None else findings
    score = compute_score(diagnosis.system_score, scored)
    final = diagnosis.final_diagnosis or "Diagnosis complete."
    return DiagnosticReport(
        report_id=f"rpt_{secrets.token_hex(6)}",
        created_at=utcnow(),
        mode=mode,
        personality=personality,
        provider=result.provider,
        model=result.model,
        simulated=result.simulated,
        latency_ms=result.latency_ms,
        system_name=system_name(scene),
        scene=scene,
        status=system_status(score, scored),
        system_score=score,
        ai_score=diagnosis.system_score,
        counts=count(scored, optimizations),
        objects=objects_from(diagnosis, result.simulated),
        relationships=relationships_from(diagnosis),
        findings=sorted(findings, key=_sort_key),
        optimizations=optimizations,
        final_diagnosis=final,
        trigger=trigger,
        image=(
            ImageMeta(
                width=image.width,
                height=image.height,
                original_width=image.original_width,
                original_height=image.original_height,
                bytes_sent=len(image.jpeg),
            )
            if image
            else None
        ),
        warnings=result.warnings,
    )


# --------------------------------------------------------------------------
# Live-scan lifecycle
# --------------------------------------------------------------------------


def active_briefs(session: ScanSession) -> tuple[list[ActiveFindingBrief], dict[str, str]]:
    briefs: list[ActiveFindingBrief] = []
    demo_keys: dict[str, str] = {}
    for tf in session.findings.values():
        if tf.status == FindingStatus.RESOLVED:
            continue
        briefs.append(
            ActiveFindingBrief(
                id=tf.id,
                title=tf.title,
                severity=tf.severity,
                category=tf.category,
                status=tf.status.value,
                box=tf.box,
                related_objects=tf.related_objects,
            )
        )
        if tf.demo_key:
            demo_keys[tf.id] = tf.demo_key
    return briefs, demo_keys


def _fuzzy_match(session: ScanSession, af: AIFinding) -> TrackedFinding | None:
    best: tuple[float, TrackedFinding] | None = None
    for tf in session.findings.values():
        if tf.category != af.category:
            continue
        sim = coerce.similarity(tf.title, af.title)
        overlap = coerce.iou(tf.box, af.box)
        shared = bool(set(tf.related_objects) & set(af.related_objects))
        score = sim + 0.6 * overlap + (0.15 if shared else 0.0)
        if (sim >= 0.45 or (overlap >= 0.4 and sim >= 0.15) or (shared and overlap >= 0.25)) and (
            best is None or score > best[0]
        ):
            best = (score, tf)
    return best[1] if best else None


def _sight(session: ScanSession, tf: TrackedFinding, now: datetime, events: list[LifecycleEvent], *, strong: bool) -> None:
    tf.sightings += 1
    tf.last_seen_at = now
    tf.out_of_view = False
    if tf.status == FindingStatus.DISCOVERED and (tf.sightings >= 2 or strong):
        tf.transition(FindingStatus.CONFIRMED, now, "Observed again - confirmed.")
        events.append(session.add_event("CONFIRMED", tf, now, "Observed again - confirmed."))
    elif tf.status == FindingStatus.CONFIRMED and tf.sightings >= 3:
        tf.transition(FindingStatus.TRACKING, now, "Persistent - now tracking.")
        events.append(session.add_event("TRACKING", tf, now, "Persistent - now tracking."))


def merge_into_scan(
    session: ScanSession, result: DiagnoseResult, *, mode: AnalysisMode
) -> tuple[list[Finding], list[LifecycleEvent]]:
    """Apply one analysis to the session. Returns this analysis' findings
    (with canonical ids and lifecycle state) and the lifecycle events."""
    now = utcnow()
    diagnosis = result.diagnosis
    events: list[LifecycleEvent] = []
    touched: set[str] = set()
    source = "demo" if result.simulated else "ai"
    thorough = mode == AnalysisMode.DEEP

    for update in diagnosis.status_updates:
        tf = session.resolve_id(update.id)
        if tf is None or tf.status == FindingStatus.RESOLVED or tf.id in touched:
            continue
        if update.state == "RESOLVED":
            note = update.observation or "No longer visible in an unobstructed view."
            tf.transition(FindingStatus.RESOLVED, now, note)
            tf.last_seen_at = now
            events.append(session.add_event("RESOLVED", tf, now, note))
            touched.add(tf.id)
        elif update.state == "PRESENT":
            _sight(session, tf, now, events, strong=thorough)
            touched.add(tf.id)
        else:
            tf.out_of_view = True

    current: list[TrackedFinding] = []
    for af in diagnosis.findings:
        tf = session.resolve_id(af.id) or _fuzzy_match(session, af)
        if tf is not None and tf.id in {c.id for c in current}:
            tf = None  # two findings in one answer must not collapse into one
        if tf is not None:
            if tf.status == FindingStatus.RESOLVED:
                tf.update_from(af)
                tf.sightings += 1
                tf.last_seen_at = now
                tf.transition(FindingStatus.DISCOVERED, now, "Issue reappeared.")
                events.append(session.add_event("REOPENED", tf, now, "Issue reappeared."))
            else:
                tf.update_from(af)
                if tf.id not in touched:
                    _sight(session, tf, now, events, strong=thorough and af.confidence >= CONFIRM_ON_FIRST_SIGHT)
        else:
            initial = (
                FindingStatus.CONFIRMED
                if thorough and af.confidence >= CONFIRM_ON_FIRST_SIGHT
                else FindingStatus.DISCOVERED
            )
            tf = TrackedFinding.from_ai(session.allocate_id(), af, source=source, status=initial, now=now)
            session.findings[tf.id] = tf
            events.append(session.add_event("DISCOVERED", tf, now))
            if initial == FindingStatus.CONFIRMED:
                events.append(session.add_event("CONFIRMED", tf, now, "High-confidence deep scan."))
        touched.add(tf.id)
        current.append(tf)

    session.analyses += 1
    session.updated_at = now
    session.scene = scene_info(diagnosis)
    session.system_name = system_name(session.scene)
    session.last_ai_score = diagnosis.system_score
    session.final_diagnosis = diagnosis.final_diagnosis or session.final_diagnosis
    session.optimizations = optimizations_from(diagnosis) or session.optimizations
    session.simulated = session.simulated or result.simulated
    session.provider = result.provider
    session.model = result.model
    return [tf.to_model() for tf in current], events


def scan_state(session: ScanSession) -> ScanState:
    tracked = sorted(session.findings.values(), key=_sort_key)
    if session.analyses:
        score = compute_score(session.last_ai_score, tracked)
        status = system_status(score, tracked)
    else:
        score = None
        status = SystemStatus.STABLE
    return ScanState(
        scan_id=session.scan_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        personality=session.personality,
        analyses=session.analyses,
        status=status,
        system_score=score,
        system_name=session.system_name,
        scene=session.scene,
        final_diagnosis=session.final_diagnosis,
        findings=[tf.to_model() for tf in tracked],
        optimizations=session.optimizations,
        counts=count(tracked, session.optimizations),
        events=list(session.events)[-40:],
        simulated=session.simulated,
        provider=session.provider,
        model=session.model,
    )
