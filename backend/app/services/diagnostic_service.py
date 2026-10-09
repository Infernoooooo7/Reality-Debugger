"""Diagnostic service: local findings with a temporal lifecycle, the optional
AI merge, scores and reports.

Live scans: every observation (a scene model measured in the browser) runs the
local engine. Local findings move DISCOVERED -> CONFIRMED -> TRACKING ->
RESOLVED by the rules in ``config/temporal.json`` ("lifecycle"): a finding is
confirmed only after repeated observations over time, resolved only after its
condition has been absent for a while in the same view, and marked NOT VISIBLE
(not resolved) when the camera looks elsewhere. AI findings (source "ai") keep
the state the AI layer reports, and AI notes are attached to local findings
without changing their measurements.
"""

from __future__ import annotations

import re
import secrets
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from app.runtime_config import RuntimeConfig
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
    AIRun,
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
from app.schemas.scene import SceneModel, SceneObject
from app.services import geometry as geo
from app.services.ai_service import DiagnoseResult
from app.services.local_diagnostics import (
    ENGINE_VERSION,
    LocalDiagnosticEngine,
    LocalFinding,
    final_diagnosis,
    scene_name,
    summary_text,
    version_for,
)
from app.services.metrics import Metrics
from app.services.ontology import Ontology
from app.services.prompts import FindingBrief, RelationBrief
from app.services.scan_store import ScanSession, TrackedFinding, canonical_bug_id, utcnow
from app.services.vision_service import PreparedImage
from app.utils import coerce

STRONG_CONFIDENCE = 0.75  # one-shot findings at least this confident start CONFIRMED
AI_CONFIRM_ON_FIRST_SIGHT = 0.8  # AI findings in a deep scan
FALLBACK_MATCH_IOU = 0.3  # same rule + same labels + overlapping box = same finding


def _sort_key(f: Finding | TrackedFinding) -> tuple[int, int, float]:
    return (1 if f.status == FindingStatus.RESOLVED else 0, -SEVERITY_RANK[f.severity], -f.confidence)


def system_name(scene: SceneInfo) -> str:
    name = re.sub(r"[^A-Z0-9]+", "_", scene.name.upper()).strip("_") or "UNKNOWN_SPACE"
    version = re.sub(r"[^0-9A-Za-z.\-]", "", scene.version.lstrip("vV")) or "1.0"
    return f"{name}_v{version}"


def count(findings: Sequence[Finding | TrackedFinding], optimizations: Sequence[Optimization]) -> Counts:
    active = [f for f in findings if f.status != FindingStatus.RESOLVED and f.severity != Severity.INFO]
    return Counts(
        active_bugs=len(active),
        high_priority=sum(1 for f in active if SEVERITY_RANK[f.severity] >= SEVERITY_RANK[Severity.HIGH]),
        optimizations=len(optimizations),
        resolved=sum(1 for f in findings if f.status == FindingStatus.RESOLVED),
    )


def objects_from_scene(scene: SceneModel | None) -> list[DetectedObject]:
    if scene is None:
        return []
    return [
        DetectedObject(id=o.id, label=o.label, confidence=o.confidence, box=o.box, source=o.source, verified=o.verified)
        for o in scene.objects
    ]


def ai_objects(diagnosis: AIDiagnosis) -> list[DetectedObject]:
    return [
        DetectedObject(
            id=coerce.identifier(o.id, f"ai_{i + 1:02d}"), label=o.label, confidence=o.confidence, box=o.box, source="ai"
        )
        for i, o in enumerate(diagnosis.objects[:15])
    ]


def ai_relationships(diagnosis: AIDiagnosis) -> list[Relationship]:
    return [
        Relationship(subject=r.subject, relation=r.relation, object=r.object, observation=r.observation, source="ai")
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


def ai_scene_info(diagnosis: AIDiagnosis) -> SceneInfo:
    s = diagnosis.scene
    return SceneInfo(name=s.name or "UNKNOWN_SPACE", version=s.version or "1.0", summary=s.summary, confidence=s.confidence)


def _lower(items: Sequence[str]) -> set[str]:
    return {i.strip().lower() for i in items if i}


def _same_issue(af: AIFinding, category: str, title: str, box, related: Sequence[str]) -> bool:  # noqa: ANN001
    """Whether an AI finding describes an issue that a local finding already covers."""
    if af.category.value != category:
        return False
    shared = _lower(af.related_objects) & _lower(related)
    sim = coerce.similarity(af.title, title)
    overlap = coerce.iou(af.box, box)
    return sim >= 0.45 or len(shared) >= 2 or (bool(shared) and overlap >= 0.25)


@dataclass(slots=True)
class Observation:
    """Result of applying one scene model to a live session."""

    findings: list[Finding]
    events: list[LifecycleEvent]
    local: list[LocalFinding]
    eval_ms: float


class DiagnosticService:
    def __init__(self, config: RuntimeConfig, ontology: Ontology, engine: LocalDiagnosticEngine, metrics: Metrics | None = None) -> None:
        self.config = config
        self.ontology = ontology
        self.engine = engine
        self.metrics = metrics or Metrics()
        lc = "temporal.lifecycle"
        self.confirm_observations = int(config.get(f"{lc}.confirmObservations"))
        self.confirm_span_ms = float(config.get(f"{lc}.confirmMinSpanMs"))
        self.tracking_observations = int(config.get(f"{lc}.trackingObservations"))
        self.resolve_after_ms = float(config.get(f"{lc}.resolveAfterMs"))
        self.resolve_observations = int(config.get(f"{lc}.resolveObservations"))
        self.exit_edge_margin = float(config.get(f"{lc}.exitEdgeMargin"))
        weights = config.get("diagnostics.score.severityWeights")
        self.weights = {Severity(k): float(v) for k, v in weights.items()}
        self.score_floor = float(config.get("diagnostics.score.floor"))
        self.ai_blend = float(config.get("diagnostics.score.aiBlend"))
        self.critical_below = float(config.get("diagnostics.score.criticalBelow"))
        self.degraded_below = float(config.get("diagnostics.score.degradedBelow"))

    # -- scoring -------------------------------------------------------------------

    def score(self, ai_score: int | None, findings: Sequence[Finding | TrackedFinding]) -> int:
        """Severity-weighted penalty of open findings, blended with the AI's
        holistic score when there is one (weights in config/diagnostics.json)."""
        penalty = sum(
            self.weights[f.severity] * (0.5 + 0.5 * f.confidence) for f in findings if f.status != FindingStatus.RESOLVED
        )
        computed = max(self.score_floor, 100.0 - penalty)
        if ai_score is None:
            return int(round(computed))
        return int(round(self.ai_blend * ai_score + (1 - self.ai_blend) * computed))

    def status_for(self, score: int, findings: Sequence[Finding | TrackedFinding]) -> SystemStatus:
        active = [f for f in findings if f.status != FindingStatus.RESOLVED]
        if score < self.critical_below or any(f.severity == Severity.CRITICAL for f in active):
            return SystemStatus.CRITICAL
        if score < self.degraded_below or any(f.severity == Severity.HIGH for f in active):
            return SystemStatus.DEGRADED
        return SystemStatus.STABLE

    # -- local engine ------------------------------------------------------------------

    def evaluate(
        self, scene: SceneModel, *, mode: AnalysisMode, personality: Personality, timers: dict[str, float] | None = None
    ) -> tuple[list[LocalFinding], float]:
        started = time.perf_counter()
        local = self.engine.evaluate(scene, mode=mode, personality=personality, timers=timers)
        elapsed = (time.perf_counter() - started) * 1000
        self.metrics.record_local(elapsed, [f.rule for f in local])
        return local, elapsed

    def relations(self, scene: SceneModel | None, limit: int = 20) -> list[Relationship]:
        if scene is None:
            return []
        return [
            Relationship(subject=r.subject, relation=r.relation, object=r.object, observation=r.observation, source="local")
            for r in self.engine.relations(scene, limit)
        ]

    def relation_briefs(self, scene: SceneModel | None) -> list[RelationBrief]:
        if scene is None:
            return []
        return [
            RelationBrief(subject=f"{r.subject} ({r.subject_label})", relation=r.relation, object=f"{r.object} ({r.object_label})", observation=r.observation)
            for r in self.engine.relations(scene, 12)
        ]

    def local_scene_info(self, scene: SceneModel | None, finding_count: int, score: int) -> SceneInfo:
        if scene is None:
            return SceneInfo(name="UNKNOWN_SPACE", version=version_for(score), summary="No scene model.", confidence=0.0)
        confidences = [o.confidence for o in scene.objects]
        return SceneInfo(
            name=scene_name(scene, self.ontology),
            version=version_for(score),
            summary=summary_text(scene, finding_count),
            confidence=round(sum(confidences) / len(confidences), 3) if confidences else 0.5,
        )

    @staticmethod
    def _verified(lf: LocalFinding, by_id: dict[str, SceneObject]) -> bool:
        involved = [by_id[i] for i in lf.object_ids if i in by_id]
        return bool(involved) and all(o.verified for o in involved)

    @staticmethod
    def _local_model(lf: LocalFinding, finding_id: str, status: FindingStatus, now: datetime, note: str | None = None) -> Finding:
        return Finding(
            id=finding_id,
            severity=lf.severity,
            category=lf.category,
            title=lf.title,
            evidence=lf.evidence,
            inference=lf.inference,
            impact=lf.impact,
            recommendation=lf.recommendation,
            confidence=lf.confidence,
            quip=lf.quip,
            status=status,
            box=lf.box,
            related_objects=list(lf.related_objects),
            source="local",
            rule=lf.rule,
            measurements=dict(lf.measurements),
            first_seen_at=now,
            last_seen_at=now,
            history=[StatusChange(status=status, at=now, note=note)],
        )

    # -- one-shot analyses (image) ---------------------------------------------------------

    def stateless_local(self, local: Sequence[LocalFinding], scene: SceneModel, *, now: datetime) -> list[Finding]:
        """Findings of a single image: CONFIRMED when both detectors agree on every
        object involved or the evidence is strong, otherwise DISCOVERED."""
        by_id = {o.id: o for o in scene.objects}
        out = []
        for i, lf in enumerate(local):
            verified = self._verified(lf, by_id)
            strong = verified or lf.confidence >= STRONG_CONFIDENCE
            note = "Both detectors agree on every object involved." if verified else None
            status = FindingStatus.CONFIRMED if strong else FindingStatus.DISCOVERED
            out.append(self._local_model(lf, f"BUG-{i + 1:03d}", status, now, note))
        return out

    def stateless_ai(self, diagnosis: AIDiagnosis, local: list[Finding], *, now: datetime) -> tuple[list[Finding], dict[str, str]]:
        """Attach AI notes to local findings and convert the AI's own findings.

        Returns the AI findings and a map from the AI's ids to canonical ids
        (AI findings that duplicate a local finding map to that finding).
        """
        by_id = {f.id: f for f in local}
        for note in diagnosis.local_notes:
            target = by_id.get(canonical_bug_id(note.finding_id) or note.finding_id)
            if target is not None:
                target.ai_note = note.note
                target.ai_agrees = note.agrees
        out: list[Finding] = []
        id_map: dict[str, str] = {}
        next_number = len(local) + 1
        for af in diagnosis.findings:
            duplicate = next(
                (f for f in local if _same_issue(af, f.category.value, f.title, f.box, f.related_objects)), None
            )
            if duplicate is not None:
                if not duplicate.ai_note:
                    duplicate.ai_note = af.inference or af.evidence or af.title
                if af.id:
                    id_map[af.id] = duplicate.id
                continue
            canonical = f"BUG-{next_number:03d}"
            next_number += 1
            if af.id:
                id_map[af.id] = canonical
            status = FindingStatus.CONFIRMED if af.confidence >= STRONG_CONFIDENCE else FindingStatus.DISCOVERED
            out.append(
                Finding(
                    id=canonical,
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
                    source="ai",
                    first_seen_at=now,
                    last_seen_at=now,
                    history=[StatusChange(status=status, at=now)],
                )
            )
        return out, id_map

    # -- live lifecycle ----------------------------------------------------------------------

    def observe(self, session: ScanSession, scene: SceneModel, *, mode: AnalysisMode = AnalysisMode.LIVE) -> Observation:
        """Evaluate one scene model and advance the lifecycle of local findings."""
        now = utcnow()
        local, eval_ms = self.evaluate(scene, mode=mode, personality=session.personality, timers=session.timers)
        by_id = {o.id: o for o in scene.objects}
        at = scene.at_ms
        deep = mode == AnalysisMode.DEEP
        events: list[LifecycleEvent] = []
        claimed: set[str] = set()
        current: list[TrackedFinding] = []

        for lf in local:
            tf = self._match(session, lf, claimed)
            verified = self._verified(lf, by_id)
            if tf is None:
                tf = self._new_local(session, lf, scene, now)
                events.append(session.add_event("DISCOVERED", tf, now))
                if deep and verified:
                    self._transition(session, tf, FindingStatus.CONFIRMED, now, "Both detectors agree in a deep scan.", events)
            else:
                reopened = tf.status == FindingStatus.RESOLVED
                self._refresh(tf, lf, scene, by_id, now)
                if reopened:
                    tf.sightings = 1
                    tf.first_seen_ms = at
                    tf.transition(FindingStatus.DISCOVERED, now, "Condition measured again.")
                    events.append(session.add_event("REOPENED", tf, now, "Condition measured again."))
                else:
                    tf.sightings += 1
                    self._advance(session, tf, now, events, strong=deep and verified)
            claimed.add(tf.id)
            current.append(tf)

        for tf in list(session.findings.values()):
            if tf.source != "local" or tf.id in claimed or tf.status == FindingStatus.RESOLVED:
                continue
            if self._out_of_view(tf, scene, by_id):
                tf.out_of_view = True
                continue
            tf.out_of_view = False
            tf.misses += 1
            absent_ms = max(0.0, at - tf.last_seen_ms)
            if tf.misses >= self.resolve_observations and absent_ms >= self.resolve_after_ms:
                note = f"Condition no longer measured for {absent_ms / 1000:.1f} s ({tf.misses} observations)."
                self._transition(session, tf, FindingStatus.RESOLVED, now, note, events)

        session.observations += 1
        self.metrics.observations += 1
        self.metrics.record_events([e.type for e in events])
        session.updated_at = now
        session.view_id = scene.view_id
        self._refresh_session_scene(session, scene)
        return Observation(findings=[tf.to_model() for tf in current], events=events, local=local, eval_ms=eval_ms)

    def _refresh_session_scene(self, session: ScanSession, scene: SceneModel) -> None:
        tracked = list(session.findings.values())
        if session.ai_scene is not None and session.ai_scene_view == scene.view_id:
            session.scene = session.ai_scene
        else:
            open_local = [tf for tf in tracked if tf.source == "local" and tf.status != FindingStatus.RESOLVED and not tf.out_of_view]
            session.scene = self.local_scene_info(scene, len(open_local), self.score(None, tracked))
            session.final_diagnosis = final_diagnosis(
                session.personality,
                [tf for tf in sorted(open_local, key=_sort_key)],  # type: ignore[arg-type]
                len(scene.objects),
            )
        session.system_name = system_name(session.scene)

    def _match(self, session: ScanSession, lf: LocalFinding, claimed: set[str]) -> TrackedFinding | None:
        candidates = [tf for tf in session.findings.values() if tf.source == "local" and tf.id not in claimed and tf.rule == lf.rule]
        for tf in candidates:
            if tf.key == lf.key:
                return tf
        # Track ids change when a track is lost and re-acquired: fall back to the
        # same rule, the same labels and an overlapping location.
        best: tuple[float, TrackedFinding] | None = None
        for tf in candidates:
            if not lf.object_ids or sorted(tf.related_objects) != sorted(lf.related_objects):
                continue
            overlap = geo.iou(tf.box, lf.box) if tf.box is not None and lf.box is not None else 0.0
            if overlap >= FALLBACK_MATCH_IOU and (best is None or overlap > best[0]):
                best = (overlap, tf)
        return best[1] if best else None

    def _new_local(self, session: ScanSession, lf: LocalFinding, scene: SceneModel, now: datetime) -> TrackedFinding:
        by_id = {o.id: o for o in scene.objects}
        tf = TrackedFinding(
            id=session.allocate_id(),
            severity=lf.severity,
            category=lf.category,
            title=lf.title,
            evidence=lf.evidence,
            inference=lf.inference,
            impact=lf.impact,
            recommendation=lf.recommendation,
            confidence=lf.confidence,
            quip=lf.quip,
            box=lf.box,
            related_objects=list(lf.related_objects),
            status=FindingStatus.DISCOVERED,
            source="local",
            first_seen_at=now,
            last_seen_at=now,
            rule=lf.rule,
            key=lf.key,
            object_ids=list(lf.object_ids),
            object_boxes={i: by_id[i].box for i in lf.object_ids if i in by_id},
            measurements=dict(lf.measurements),
            first_seen_ms=scene.at_ms,
            last_seen_ms=scene.at_ms,
            view_id=scene.view_id,
        )
        tf.history.append(StatusChange(status=FindingStatus.DISCOVERED, at=now))
        session.findings[tf.id] = tf
        return tf

    @staticmethod
    def _refresh(tf: TrackedFinding, lf: LocalFinding, scene: SceneModel, by_id: dict[str, SceneObject], now: datetime) -> None:
        tf.key = lf.key
        tf.severity = lf.severity
        tf.category = lf.category
        tf.title = lf.title
        tf.evidence = lf.evidence
        tf.inference = lf.inference
        tf.impact = lf.impact
        tf.recommendation = lf.recommendation
        tf.quip = lf.quip
        # Smooth confidence so one shaky frame does not swing it.
        tf.confidence = round(0.6 * lf.confidence + 0.4 * tf.confidence, 3)
        tf.box = lf.box
        tf.related_objects = list(lf.related_objects)
        tf.object_ids = list(lf.object_ids)
        tf.object_boxes = {i: by_id[i].box for i in lf.object_ids if i in by_id}
        tf.measurements = dict(lf.measurements)
        tf.last_seen_ms = scene.at_ms
        tf.last_seen_at = now
        tf.misses = 0
        tf.out_of_view = False
        tf.view_id = scene.view_id

    def _advance(self, session: ScanSession, tf: TrackedFinding, now: datetime, events: list[LifecycleEvent], *, strong: bool) -> None:
        span = tf.last_seen_ms - tf.first_seen_ms
        if tf.status == FindingStatus.DISCOVERED:
            if strong:
                self._transition(session, tf, FindingStatus.CONFIRMED, now, "Both detectors agree in a deep scan.", events)
            elif tf.sightings >= self.confirm_observations and span >= self.confirm_span_ms:
                note = f"Measured in {tf.sightings} observations over {span / 1000:.1f} s."
                self._transition(session, tf, FindingStatus.CONFIRMED, now, note, events)
        elif tf.status == FindingStatus.CONFIRMED and tf.sightings >= self.tracking_observations:
            self._transition(session, tf, FindingStatus.TRACKING, now, "Persistent - now tracking.", events)

    def _out_of_view(self, tf: TrackedFinding, scene: SceneModel, by_id: dict[str, SceneObject]) -> bool:
        """Absent because the camera looks elsewhere (not because it was fixed)?"""
        if tf.view_id != scene.view_id:
            return True
        missing = [i for i in tf.object_ids if i not in by_id]
        return any(i in tf.object_boxes and geo.near_edge(tf.object_boxes[i], self.exit_edge_margin) for i in missing)

    def _transition(
        self, session: ScanSession, tf: TrackedFinding, status: FindingStatus, now: datetime, note: str | None, events: list[LifecycleEvent]
    ) -> None:
        tf.transition(status, now, note)
        events.append(session.add_event(status.value, tf, now, note))

    # -- AI merge (live) ---------------------------------------------------------------------

    def apply_ai(
        self, session: ScanSession, result: DiagnoseResult, *, mode: AnalysisMode, scene: SceneModel | None
    ) -> tuple[list[Finding], list[LifecycleEvent], list[str]]:
        """Merge one AI answer into a live session.

        Returns the AI findings of this answer, lifecycle events and the ids of
        local findings that received an AI note.
        """
        now = utcnow()
        diagnosis = result.diagnosis
        events: list[LifecycleEvent] = []
        explained: list[str] = []
        touched: set[str] = set()
        thorough = mode == AnalysisMode.DEEP

        for note in diagnosis.local_notes:
            tf = session.resolve_id(note.finding_id)
            if tf is not None and tf.source == "local":
                tf.ai_note = note.note
                tf.ai_agrees = note.agrees
                explained.append(tf.id)

        for update in diagnosis.status_updates:
            tf = session.resolve_id(update.id)
            if tf is None or tf.source != "ai" or tf.status == FindingStatus.RESOLVED or tf.id in touched:
                continue
            if update.state == "RESOLVED":
                note = update.observation or "No longer visible in an unobstructed view."
                tf.last_seen_at = now
                self._transition(session, tf, FindingStatus.RESOLVED, now, note, events)
                touched.add(tf.id)
            elif update.state == "PRESENT":
                self._sight_ai(session, tf, now, events, strong=thorough)
                touched.add(tf.id)
            else:
                tf.out_of_view = True

        current: list[TrackedFinding] = []
        for af in diagnosis.findings:
            tf = session.resolve_id(af.id)
            if tf is not None and tf.source == "local":
                # The AI reused a local id: keep the measurement, keep its words as a note.
                if not tf.ai_note:
                    tf.ai_note = af.inference or af.evidence or af.title
                    explained.append(tf.id)
                continue
            tf = tf or self._fuzzy_ai(session, af)
            if tf is not None and tf.id in {c.id for c in current}:
                tf = None  # two findings in one answer must not collapse into one
            if tf is None:
                duplicate = next(
                    (
                        t
                        for t in session.findings.values()
                        if t.source == "local"
                        and t.status != FindingStatus.RESOLVED
                        and _same_issue(af, t.category.value, t.title, t.box, t.related_objects)
                    ),
                    None,
                )
                if duplicate is not None:
                    if not duplicate.ai_note:
                        duplicate.ai_note = af.inference or af.evidence or af.title
                        explained.append(duplicate.id)
                    continue
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
                        self._sight_ai(session, tf, now, events, strong=thorough and af.confidence >= AI_CONFIRM_ON_FIRST_SIGHT)
            else:
                initial = (
                    FindingStatus.CONFIRMED
                    if thorough and af.confidence >= AI_CONFIRM_ON_FIRST_SIGHT
                    else FindingStatus.DISCOVERED
                )
                tf = TrackedFinding.from_ai(session.allocate_id(), af, status=initial, now=now)
                if scene is not None:
                    tf.view_id = scene.view_id
                session.findings[tf.id] = tf
                events.append(session.add_event("DISCOVERED", tf, now))
                if initial == FindingStatus.CONFIRMED:
                    events.append(session.add_event("CONFIRMED", tf, now, "High-confidence deep scan."))
            touched.add(tf.id)
            current.append(tf)

        session.analyses += 1
        session.updated_at = now
        session.ai_scene = ai_scene_info(diagnosis)
        session.ai_scene_view = scene.view_id if scene is not None else session.view_id
        session.scene = session.ai_scene
        session.system_name = system_name(session.scene)
        session.last_ai_score = diagnosis.system_score
        session.final_diagnosis = diagnosis.final_diagnosis or session.final_diagnosis
        session.optimizations = optimizations_from(diagnosis) or session.optimizations
        session.provider = result.provider
        session.model = result.model
        self.metrics.record_events([e.type for e in events])
        return [tf.to_model() for tf in current], events, explained

    def _sight_ai(self, session: ScanSession, tf: TrackedFinding, now: datetime, events: list[LifecycleEvent], *, strong: bool) -> None:
        tf.sightings += 1
        tf.last_seen_at = now
        tf.out_of_view = False
        if tf.status == FindingStatus.DISCOVERED and (tf.sightings >= self.confirm_observations or strong):
            self._transition(session, tf, FindingStatus.CONFIRMED, now, "Observed again - confirmed.", events)
        elif tf.status == FindingStatus.CONFIRMED and tf.sightings >= self.tracking_observations:
            self._transition(session, tf, FindingStatus.TRACKING, now, "Persistent - now tracking.", events)

    @staticmethod
    def _fuzzy_ai(session: ScanSession, af: AIFinding) -> TrackedFinding | None:
        best: tuple[float, TrackedFinding] | None = None
        for tf in session.findings.values():
            if tf.source != "ai" or tf.category != af.category:
                continue
            sim = coerce.similarity(tf.title, af.title)
            overlap = coerce.iou(tf.box, af.box)
            shared = bool(_lower(tf.related_objects) & _lower(af.related_objects))
            score = sim + 0.6 * overlap + (0.15 if shared else 0.0)
            if (sim >= 0.45 or (overlap >= 0.4 and sim >= 0.15) or (shared and overlap >= 0.25)) and (
                best is None or score > best[0]
            ):
                best = (score, tf)
        return best[1] if best else None

    # -- prompt briefs --------------------------------------------------------------------------

    @staticmethod
    def briefs(session: ScanSession) -> tuple[list[FindingBrief], list[FindingBrief]]:
        """Open local findings and open AI findings of a session, most severe first."""
        local: list[FindingBrief] = []
        ai: list[FindingBrief] = []
        for tf in sorted(session.findings.values(), key=_sort_key):
            if tf.status == FindingStatus.RESOLVED:
                continue
            brief = FindingBrief(
                id=tf.id,
                source=tf.source,
                title=tf.title,
                severity=tf.severity,
                category=tf.category,
                status="NOT_VISIBLE" if tf.out_of_view else tf.status.value,
                box=tf.box,
                related_objects=tf.related_objects,
                rule=tf.rule,
                object_ids=tf.object_ids,
                measurements=tf.measurements,
            )
            (local if tf.source == "local" else ai).append(brief)
        return local, ai

    @staticmethod
    def stateless_briefs(findings: Sequence[Finding], object_ids: dict[str, list[str]] | None = None) -> list[FindingBrief]:
        return [
            FindingBrief(
                id=f.id,
                source=f.source,
                title=f.title,
                severity=f.severity,
                category=f.category,
                status=f.status.value,
                box=f.box,
                related_objects=f.related_objects,
                rule=f.rule,
                object_ids=(object_ids or {}).get(f.id, []),
                measurements=f.measurements,
            )
            for f in findings
        ]

    # -- reports ------------------------------------------------------------------------------------

    def report(
        self,
        *,
        mode: AnalysisMode,
        personality: Personality,
        scene: SceneModel | None,
        findings: list[Finding],
        ai_run: AIRun,
        latency_ms: int,
        local: Sequence[LocalFinding] = (),
        ai: DiagnoseResult | None = None,
        score_findings: Sequence[Finding | TrackedFinding] | None = None,
        scene_info: SceneInfo | None = None,
        final: str | None = None,
        trigger: ScanTrigger | None = None,
        image: PreparedImage | None = None,
        warnings: Sequence[str] = (),
    ) -> DiagnosticReport:
        scored = score_findings if score_findings is not None else findings
        diagnosis = ai.diagnosis if ai is not None else None
        ai_score = diagnosis.system_score if diagnosis is not None else None
        score = self.score(ai_score, scored)
        if scene_info is None:
            if diagnosis is not None:
                scene_info = ai_scene_info(diagnosis)
            else:
                open_count = sum(1 for f in scored if f.status != FindingStatus.RESOLVED)
                scene_info = self.local_scene_info(scene, open_count, score)
        object_count = len(scene.objects) if scene is not None else 0
        final_text = final or (diagnosis.final_diagnosis if diagnosis is not None else "") or final_diagnosis(
            personality, list(local), object_count
        )
        optimizations = optimizations_from(diagnosis) if diagnosis is not None else []
        return DiagnosticReport(
            report_id=f"rpt_{secrets.token_hex(6)}",
            created_at=utcnow(),
            mode=mode,
            personality=personality,
            provider=ai.provider if ai is not None else "local",
            model=ai.model if ai is not None else ENGINE_VERSION,
            engine=ENGINE_VERSION,
            detectors=list(scene.stats.detectors) if scene is not None else [],
            ai=ai_run,
            latency_ms=latency_ms,
            system_name=system_name(scene_info),
            scene=scene_info,
            status=self.status_for(score, scored),
            system_score=score,
            ai_score=ai_score,
            counts=count(scored, optimizations),
            objects=objects_from_scene(scene) + (ai_objects(diagnosis) if diagnosis is not None else []),
            relationships=self.relations(scene) + (ai_relationships(diagnosis) if diagnosis is not None else []),
            findings=sorted(findings, key=_sort_key),
            optimizations=optimizations,
            final_diagnosis=final_text,
            trigger=trigger,
            image=(
                ImageMeta(
                    width=image.width,
                    height=image.height,
                    original_width=image.original_width,
                    original_height=image.original_height,
                    bytes_sent=len(image.jpeg),
                )
                if image is not None
                else None
            ),
            warnings=[*(ai.warnings if ai is not None else []), *warnings],
        )

    def scan_state(self, session: ScanSession) -> ScanState:
        tracked = sorted(session.findings.values(), key=_sort_key)
        if session.observations or session.analyses:
            ai_score = session.last_ai_score if session.ai_scene_view == session.view_id else None
            score: int | None = self.score(ai_score, tracked)
            status = self.status_for(score, tracked)
        else:
            score = None
            status = SystemStatus.STABLE
        return ScanState(
            scan_id=session.scan_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            personality=session.personality,
            analyses=session.analyses,
            observations=session.observations,
            status=status,
            system_score=score,
            system_name=session.system_name,
            scene=session.scene,
            final_diagnosis=session.final_diagnosis,
            findings=[tf.to_model() for tf in tracked],
            optimizations=session.optimizations,
            counts=count(tracked, session.optimizations),
            events=list(session.events)[-40:],
            ai=session.last_ai_run,
            engine=ENGINE_VERSION,
        )
