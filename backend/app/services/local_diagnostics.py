"""Local diagnostic engine: measurable findings from the scene model, no AI.

Every rule works on what the browser's vision pipeline actually measured -
detector boxes and confidences, tracker state (age, movement, occlusion) and
pixel statistics - plus the semantic attributes of each label from the
generated ontology. Rules never test label strings; thresholds come from
``config/diagnostics.json`` and ``config/temporal.json``.

Each finding records the measurements that triggered it, so the evidence
shown to the user is a statement about numbers, not an opinion.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from app.runtime_config import RuntimeConfig
from app.schemas.common import AnalysisMode, Box, Category, Personality, Severity
from app.schemas.scene import SceneModel, SceneObject
from app.services import geometry as geo
from app.services.ontology import Ontology

ENGINE_VERSION = "local-diagnostics/2"
# Modes whose scene objects carry tracker history (age, movement, occlusion).
# A deep scan is a frozen live frame, so the live tracks' history applies.
TEMPORAL_MODES = {AnalysisMode.LIVE, AnalysisMode.DEEP, AnalysisMode.VIDEO}
# Modes that observe a continuous stream (duration-based checks).
STREAM_MODES = {AnalysisMode.LIVE, AnalysisMode.VIDEO}

Measurements = dict[str, float | int | str]

# Semantic attributes each rule needs (alternatives; every attribute of one
# alternative must exist in the ontology). A rule whose attributes no bundled
# detector label has stays inactive until a model with such classes is added.
RULE_REQUIREMENTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "spill_risk": (("liquid_container", "electronic"),),
    "food_near_electronics": (("food", "electronic"),),
    "liquid_near_outlet": (("liquid_container", "outlet"),),
    "sharp_exposed": (("sharp",),),
    "edge_placement": (("liquid_container", "surface"),),
    "animal_near_equipment": (("animal", "electronic"),),
    "cable_congestion": (("cable",),),
    "surface_congestion": (("surface",),),
    "long_presence": (("food",), ("liquid_container",)),
}
LEFT_BEHIND_ATTRIBUTES = frozenset({"food", "liquid_container"})


@dataclass(slots=True)
class LocalRelation:
    """A spatial relation computed from two boxes (2D; no depth)."""

    subject: str
    subject_label: str
    relation: str  # on | overlaps | touching | near
    object: str
    object_label: str
    observation: str
    gap: float  # edge gap relative to the larger box (0 when touching)


@dataclass(slots=True)
class LocalFinding:
    key: str
    rule: str
    severity: Severity
    category: Category
    title: str
    evidence: str
    inference: str
    impact: str
    recommendation: str
    confidence: float
    quip: str
    box: Box | None
    object_ids: list[str]
    related_objects: list[str]
    measurements: Measurements = field(default_factory=dict)


def pct(value: float) -> str:
    return f"{round(value * 100)}%"


def _voice(personality: Personality, serious: str, brutal: str, unhinged: str) -> str:
    return {Personality.SERIOUS: serious, Personality.BRUTAL: brutal, Personality.UNHINGED: unhinged}[personality]


def _quip(personality: Personality, brutal: str, unhinged: str) -> str:
    return "" if personality == Personality.SERIOUS else _voice(personality, "", brutal, unhinged)


def _seconds(ms: float) -> str:
    return f"{ms / 1000:.1f} s" if ms < 60_000 else f"{ms / 60_000:.1f} min"


class LocalDiagnosticEngine:
    def __init__(self, config: RuntimeConfig, ontology: Ontology) -> None:
        self.cfg = config
        self.ontology = ontology
        self.touch_gap = float(config.get("diagnostics.proximity.touchGap"))
        self.near_gap = float(config.get("diagnostics.proximity.nearRelativeGap"))
        self.occluded_fraction = float(config.get("temporal.occlusion.occludedFraction"))
        self.persistence_full_ms = float(config.get("diagnostics.confidence.persistenceFullMs"))
        self.min_evidence = {"fast": float(config.get("diagnostics.confidence.minEvidenceFast")),
                             "deep": float(config.get("diagnostics.confidence.minEvidenceDeep"))}
        self._rules: list[tuple[str, Callable[..., list[LocalFinding]]]] = [
            ("spill_risk", self._spill_risk),
            ("food_near_electronics", self._food_near_electronics),
            ("liquid_near_outlet", self._liquid_near_outlet),
            ("sharp_exposed", self._sharp_exposed),
            ("edge_placement", self._edge_placement),
            ("animal_near_equipment", self._animal_near_equipment),
            ("cable_congestion", self._cable_congestion),
            ("dense_region", self._dense_region),
            ("surface_congestion", self._surface_congestion),
            ("overlap_cluster", self._overlap_cluster),
            ("keep_clear_zone", self._keep_clear_zone),
            ("persistent_occlusion", self._persistent_occlusion),
            ("view_obstruction", self._view_obstruction),
            ("lighting", self._lighting),
            ("blur", self._blur),
            ("long_presence", self._long_presence),
            ("repeated_movement", self._repeated_movement),
        ]

    # -- public ---------------------------------------------------------------

    @property
    def rule_names(self) -> list[str]:
        return [name for name, _ in self._rules]

    def armed_rules(self) -> list[str]:
        """Enabled rules whose required attributes exist among the known labels."""
        present: set[str] = set()
        for label in self.ontology.labels:
            present |= self.ontology.attributes(label)
        armed = []
        for name in self.rule_names:
            if not self._p(name, "enabled", True):
                continue
            needs = RULE_REQUIREMENTS.get(name, ((),))
            if any(all(a in present for a in alternative) for alternative in needs):
                armed.append(name)
        return armed

    def relations(self, scene: SceneModel, limit: int = 20) -> list[LocalRelation]:
        """Spatial relations between nearby objects, closest first."""
        out: list[LocalRelation] = []
        objects = scene.objects[:60]
        for i, a in enumerate(objects):
            for b in objects[i + 1:]:
                prox = geo.proximity(a.box, b.box, touch_gap=self.touch_gap, near_relative_gap=self.near_gap)
                if not prox.near:
                    continue
                relation = None
                sub, obj = (b, a) if geo.area(a.box) > geo.area(b.box) else (a, b)
                for x, y in ((a, b), (b, a)):
                    if (
                        self.ontology.has(y.label, "surface")
                        and not self.ontology.has(x.label, "surface")
                        and geo.containment(x.box, y.box) >= 0.5
                        and geo.base_inside(x.box, y.box)
                    ):
                        sub, obj, relation = x, y, "on"
                        break
                if relation is None:
                    relation = "overlaps" if prox.overlap_iou > 0 else ("touching" if prox.touching else "near")
                observation = (
                    f"{sub.label} is {geo.direction(sub.box, obj.box)} the {obj.label}; edge gap "
                    f"{prox.gap * 100:.1f}% of the frame, {prox.relative_gap:.2f}x the larger box, IoU {prox.overlap_iou:.2f}"
                )
                out.append(LocalRelation(
                    subject=sub.id, subject_label=sub.label, relation=relation, object=obj.id, object_label=obj.label,
                    observation=observation, gap=0.0 if prox.touching else prox.relative_gap,
                ))
        rank = {"on": 0, "overlaps": 1, "touching": 2, "near": 3}
        out.sort(key=lambda r: (rank[r.relation], r.gap))
        return out[:limit]

    def evaluate(
        self,
        scene: SceneModel,
        *,
        mode: AnalysisMode,
        personality: Personality,
        timers: dict[str, float] | None = None,
    ) -> list[LocalFinding]:
        """Run every enabled rule. ``timers`` (live sessions) remembers since when
        a duration-based condition has held; one-shot analyses pass ``None``."""
        scene, _ = self.evidence(scene, mode)
        findings: list[LocalFinding] = []
        for name, rule in self._rules:
            if not self._p(name, "enabled", True):
                continue
            findings.extend(rule(scene, mode=mode, personality=personality, timers=timers))
        findings.sort(key=lambda f: (-_severity_rank(f.severity), -f.confidence))
        return findings

    def evidence(self, scene: SceneModel, mode: AnalysisMode) -> tuple[SceneModel, list[SceneObject]]:
        """The scene the rules may use as evidence, and the objects set aside.

        A single photo has no temporal confirmation, so its objects support a finding only when their
        detections are more likely right than wrong on held-out data (config diagnostics.confidence
        minEvidence*). Tracked scenes (live, deep scan, video) keep every object: persistence is their evidence.
        """
        if mode in TEMPORAL_MODES:
            return scene, []
        weak = [o for o in scene.objects if o.confidence < self.min_evidence["fast" if o.source == "fast" else "deep"]]
        if not weak:
            return scene, []
        return scene.model_copy(update={"objects": [o for o in scene.objects if o not in weak]}), weak

    # -- helpers ----------------------------------------------------------------

    def _p(self, rule: str, name: str, default: object = ...) -> object:
        return self.cfg.get(f"diagnostics.rules.{rule}.{name}", default)

    def _sev(self, rule: str, name: str = "severity") -> Severity:
        return Severity(str(self._p(rule, name)))

    def _with(self, scene: SceneModel, attribute: str) -> list[SceneObject]:
        return [o for o in scene.objects if self.ontology.has(o.label, attribute)]

    def _confidence(self, objects: Sequence[SceneObject], mode: AnalysisMode) -> float:
        """Mean detection confidence x temporal-evidence factor (live/deep only)."""
        if not objects:
            return 0.5
        base = sum(o.confidence for o in objects) / len(objects)
        if mode not in TEMPORAL_MODES:
            return round(min(1.0, base), 3)
        persistence = min(o.age_ms for o in objects)
        factor = 0.7 + 0.3 * min(1.0, persistence / self.persistence_full_ms)
        return round(min(1.0, base * factor), 3)

    def _countable(self, scene: SceneModel) -> list[SceneObject]:
        excluded = set(self._p("dense_region", "excludeAttributes", []))  # type: ignore[arg-type]
        return [o for o in scene.objects if not (self.ontology.attributes(o.label) & excluded)]

    def _closest(self, obj: SceneObject, others: Sequence[SceneObject]) -> tuple[SceneObject, geo.Proximity] | None:
        """The nearest object among ``others`` that is touching or near ``obj``."""
        candidates = []
        for other in others:
            if other.id == obj.id:
                continue
            prox = geo.proximity(obj.box, other.box, touch_gap=self.touch_gap, near_relative_gap=self.near_gap)
            if prox.near:
                candidates.append(((0 if prox.touching else 1, prox.relative_gap, -prox.overlap_iou), other, prox))
        if not candidates:
            return None
        _, other, prox = min(candidates, key=lambda c: c[0])
        return other, prox

    @staticmethod
    def _held(timers: dict[str, float] | None, key: str, active: bool, now_ms: float) -> float:
        """Track how long a condition has held (ms). Without timers: 0 when active."""
        if timers is None:
            return 0.0 if active else -1.0
        if not active:
            timers.pop(key, None)
            return -1.0
        since = timers.setdefault(key, now_ms)
        return max(0.0, now_ms - since)

    def _proximity_words(self, prox: geo.Proximity) -> str:
        if prox.overlap_iou > 0:
            return f"overlapping (IoU {prox.overlap_iou:.2f})"
        if prox.touching:
            return "touching"
        return f"{prox.relative_gap:.2f}x its size away (edge gap {prox.gap * 100:.1f}% of the frame)"

    def _pair_measurements(self, a: SceneObject, b: SceneObject, prox: geo.Proximity) -> Measurements:
        return {
            "edge_gap": round(prox.gap, 4),
            "relative_gap": round(prox.relative_gap, 3),
            "overlap_iou": round(prox.overlap_iou, 3),
            f"{a.label}_confidence": round(a.confidence, 2),
            f"{b.label}_confidence": round(b.confidence, 2),
        }

    # -- proximity hazards ---------------------------------------------------------

    def _spill_risk(self, scene: SceneModel, *, mode: AnalysisMode, personality: Personality, **_: object) -> list[LocalFinding]:
        devices = self._with(scene, "electronic")
        out = []
        for c in self._with(scene, "liquid_container"):
            hit = self._closest(c, devices)
            if not hit:
                continue
            d, prox = hit
            severity = self._sev("spill_risk", "severityTouching" if prox.touching else "severityNear")
            out.append(LocalFinding(
                key=f"spill_risk:{c.id}+{d.id}",
                rule="spill_risk",
                severity=severity,
                category=Category.SAFETY,
                title=_voice(personality,
                             f"{c.label.capitalize()} within spill range of the {d.label}",
                             f"The {c.label} is one elbow away from a hardware incident",
                             f"ERROR 0xC0FFEE: {c.label} has gained root access to the {d.label}"),
                evidence=f"The local detector found a {c.label} ({pct(c.confidence)}) {self._proximity_words(prox)} "
                         f"next to a {d.label} ({pct(d.confidence)}).",
                inference="A container that can hold liquid sits within knock-over distance of an electronic device.",
                impact=f"A spill could damage the {d.label}.",
                recommendation=f"Move the {c.label} at least an arm's length from the {d.label}, or use a lidded cup.",
                confidence=self._confidence([c, d], mode),
                quip=_quip(personality, "Hydration and hardware: pick one.", "Liquid-to-silicon interface detected. Please don't."),
                box=geo.union_box([c.box, d.box]),
                object_ids=[c.id, d.id],
                related_objects=[c.label, d.label],
                measurements=self._pair_measurements(c, d, prox),
            ))
        return out

    def _food_near_electronics(self, scene: SceneModel, *, mode: AnalysisMode, personality: Personality, **_: object) -> list[LocalFinding]:
        devices = self._with(scene, "electronic")
        out = []
        for f in self._with(scene, "food"):
            hit = self._closest(f, devices)
            if not hit:
                continue
            d, prox = hit
            out.append(LocalFinding(
                key=f"food_near_electronics:{f.id}+{d.id}",
                rule="food_near_electronics",
                severity=self._sev("food_near_electronics", "severityTouching" if prox.touching else "severityNear"),
                category=Category.ORGANIZATION,
                title=_voice(personality,
                             f"Food next to the {d.label}",
                             f"The {d.label} is about to be seasoned",
                             f"SEGFAULT: {f.label} dereferenced near {d.label}"),
                evidence=f"The local detector found a {f.label} ({pct(f.confidence)}) {self._proximity_words(prox)} "
                         f"next to a {d.label} ({pct(d.confidence)}).",
                inference="Food is being kept or eaten right next to an electronic device.",
                impact=f"Crumbs and grease end up in or on the {d.label}.",
                recommendation=f"Eat away from the {d.label}, or keep the {f.label} on a plate on the far side of the surface.",
                confidence=self._confidence([f, d], mode),
                quip=_quip(personality, "Crumbs are not a feature.", f"Snack buffer overflow in sector {d.label}."),
                box=geo.union_box([f.box, d.box]),
                object_ids=[f.id, d.id],
                related_objects=[f.label, d.label],
                measurements=self._pair_measurements(f, d, prox),
            ))
        return out

    def _liquid_near_outlet(self, scene: SceneModel, *, mode: AnalysisMode, personality: Personality, **_: object) -> list[LocalFinding]:
        outlets = self._with(scene, "outlet")
        if not outlets:
            return []
        out = []
        for c in self._with(scene, "liquid_container"):
            hit = self._closest(c, outlets)
            if not hit:
                continue
            o, prox = hit
            out.append(LocalFinding(
                key=f"liquid_near_outlet:{c.id}+{o.id}",
                rule="liquid_near_outlet",
                severity=self._sev("liquid_near_outlet"),
                category=Category.SAFETY,
                title=_voice(personality, f"{c.label.capitalize()} near a power outlet", "Liquid, meet electricity", f"SHORT_CIRCUIT_PENDING: {c.label} x outlet"),
                evidence=f"A {c.label} ({pct(c.confidence)}) is {self._proximity_words(prox)} next to a {o.label} ({pct(o.confidence)}).",
                inference="A liquid container is within reach of a power outlet.",
                impact="A spill into an outlet is an electrical hazard.",
                recommendation=f"Move the {c.label} away from the outlet.",
                confidence=self._confidence([c, o], mode),
                quip=_quip(personality, "Bold choice.", "Sparks are not a UI animation."),
                box=geo.union_box([c.box, o.box]),
                object_ids=[c.id, o.id],
                related_objects=[c.label, o.label],
                measurements=self._pair_measurements(c, o, prox),
            ))
        return out

    def _animal_near_equipment(self, scene: SceneModel, *, mode: AnalysisMode, personality: Personality, **_: object) -> list[LocalFinding]:
        devices = self._with(scene, "electronic")
        out = []
        for a in self._with(scene, "animal"):
            hit = self._closest(a, devices)
            if not hit:
                continue
            d, prox = hit
            out.append(LocalFinding(
                key=f"animal_near_equipment:{a.id}+{d.id}",
                rule="animal_near_equipment",
                severity=self._sev("animal_near_equipment"),
                category=Category.SAFETY,
                title=_voice(personality, f"{a.label.capitalize()} next to the {d.label}", f"The {a.label} has claimed the {d.label}", f"UNAUTHORIZED PROCESS: {a.label}.exe attached to {d.label}"),
                evidence=f"A {a.label} ({pct(a.confidence)}) is {self._proximity_words(prox)} next to a {d.label} ({pct(d.confidence)}).",
                inference="An animal is in contact range of an electronic device.",
                impact="Knocked-over or chewed equipment and cables.",
                recommendation=f"Give the {a.label} its own spot away from the {d.label} and its cables.",
                confidence=self._confidence([a, d], mode),
                quip=_quip(personality, f"The {a.label} is supervising and is not impressed.", f"{a.label.capitalize()} privilege escalation in progress."),
                box=geo.union_box([a.box, d.box]),
                object_ids=[a.id, d.id],
                related_objects=[a.label, d.label],
                measurements=self._pair_measurements(a, d, prox),
            ))
        return out

    # -- single-object hazards -------------------------------------------------------

    def _sharp_exposed(self, scene: SceneModel, *, mode: AnalysisMode, personality: Personality, **_: object) -> list[LocalFinding]:
        min_ms = float(self._p("sharp_exposed", "minPersistenceMs"))  # type: ignore[arg-type]
        out = []
        for t in self._with(scene, "sharp"):
            if mode in STREAM_MODES and t.age_ms < min_ms:
                continue
            measurements: Measurements = {"confidence": round(t.confidence, 2)}
            if t.age_ms:
                measurements["visible_for_s"] = round(t.age_ms / 1000, 1)
            out.append(LocalFinding(
                key=f"sharp_exposed:{t.id}",
                rule="sharp_exposed",
                severity=self._sev("sharp_exposed"),
                category=Category.SAFETY,
                title=_voice(personality, f"{t.label.capitalize()} left out in the open", f"Loose {t.label}: a speedrun to the first-aid kit", f"WARNING: unsandboxed {t.label} running in user space"),
                evidence=f"A {t.label} ({pct(t.confidence)}) is visible"
                         + (f" and has been in view for {_seconds(t.age_ms)}." if t.age_ms else "."),
                inference="A tool with a cutting edge is lying out rather than stored.",
                impact="Easy to grab or brush against by accident.",
                recommendation=f"Put the {t.label} back in a drawer or holder when not in use.",
                confidence=self._confidence([t], mode),
                quip=_quip(personality, "Sharp objects love unattended surfaces.", "Edge case detected. Literally."),
                box=t.box,
                object_ids=[t.id],
                related_objects=[t.label],
                measurements=measurements,
            ))
        return out

    def _edge_placement(self, scene: SceneModel, *, mode: AnalysisMode, personality: Personality, **_: object) -> list[LocalFinding]:
        surfaces = self._with(scene, "surface")
        if not surfaces:
            return []
        margin = float(self._p("edge_placement", "edgeMargin"))  # type: ignore[arg-type]
        out = []
        for c in self._with(scene, "liquid_container"):
            base_x, base_y = c.box.x + c.box.w / 2, c.box.y + c.box.h
            for s in surfaces:
                inside_x = s.box.x <= base_x <= s.box.x + s.box.w
                inside_y = s.box.y <= base_y <= s.box.y + s.box.h + 0.02
                if not (inside_x and inside_y):
                    continue
                distances = {
                    "left": (base_x - s.box.x) / s.box.w,
                    "right": (s.box.x + s.box.w - base_x) / s.box.w,
                    "front": max(0.0, (s.box.y + s.box.h - base_y)) / s.box.w,
                }
                side, dist = min(distances.items(), key=lambda kv: kv[1])
                if dist > margin:
                    continue
                out.append(LocalFinding(
                    key=f"edge_placement:{c.id}+{s.id}",
                    rule="edge_placement",
                    severity=self._sev("edge_placement"),
                    category=Category.SPATIAL,
                    title=_voice(personality, f"{c.label.capitalize()} at the {side} edge of the {s.label}", f"The {c.label} is auditioning for gravity", f"EDGE_CASE: {c.label} at boundary of {s.label}"),
                    evidence=f"The base of the {c.label} ({pct(c.confidence)}) is {dist * 100:.0f}% of the {s.label}'s width "
                             f"from its {side} edge (threshold {margin * 100:.0f}%).",
                    inference="The container stands where a nudge would push it off the surface.",
                    impact="Spills and breakage.",
                    recommendation=f"Move the {c.label} towards the middle of the {s.label}.",
                    confidence=self._confidence([c, s], mode),
                    quip=_quip(personality, "Living on the edge, literally.", "Gravity patch pending."),
                    box=c.box,
                    object_ids=[c.id, s.id],
                    related_objects=[c.label, s.label],
                    measurements={"edge_distance_x_width": round(dist, 3), "side": side, "edge_margin": margin},
                ))
                break
        return out

    def _cable_congestion(self, scene: SceneModel, *, mode: AnalysisMode, personality: Personality, **_: object) -> list[LocalFinding]:
        cables = self._with(scene, "cable")
        need = int(self._p("cable_congestion", "minCables"))  # type: ignore[arg-type]
        if len(cables) < need:
            return []
        clustered = [c for c in cables if any(
            geo.proximity(c.box, o.box, touch_gap=self.touch_gap, near_relative_gap=self.near_gap).near
            for o in cables if o.id != c.id)]
        if len(clustered) < need:
            return []
        return [LocalFinding(
            key="cable_congestion:scene",
            rule="cable_congestion",
            severity=self._sev("cable_congestion"),
            category=Category.SAFETY,
            title=_voice(personality, "Cable congestion", "Cable spaghetti, al dente", "KERNEL PANIC: cable graph contains cycles"),
            evidence=f"{len(clustered)} cable-like objects are detected close together.",
            inference="Several cables cross or bunch up in one area.",
            impact="Snagging, tripping and hard-to-trace connections.",
            recommendation="Bundle the cables with ties or a sleeve and route them along an edge.",
            confidence=self._confidence(clustered, mode),
            quip=_quip(personality, "Who needs cable management anyway.", "Untangling requires root."),
            box=geo.union_box([c.box for c in clustered]),
            object_ids=[c.id for c in clustered],
            related_objects=sorted({c.label for c in clustered}),
            measurements={"cables": len(clustered)},
        )]

    # -- density / layout ----------------------------------------------------------------

    def _dense_region(self, scene: SceneModel, *, mode: AnalysisMode, personality: Personality, **_: object) -> list[LocalFinding]:
        """Several recognised loose objects within a small radius: a measured concentration, not a
        verdict on tidiness (a count alone says nothing about arrangement)."""
        items = self._countable(scene)
        need = int(self._p("dense_region", "minObjects"))  # type: ignore[arg-type]
        radius = float(self._p("dense_region", "radius"))  # type: ignore[arg-type]
        if len(items) < need:
            return []
        w, h = float(scene.width or 1), float(scene.height or 1)
        diag = (w * w + h * h) ** 0.5
        centres = [geo.center(o.box) for o in items]

        def dist(i: int, j: int) -> float:
            return (((centres[i][0] - centres[j][0]) * w) ** 2 + ((centres[i][1] - centres[j][1]) * h) ** 2) ** 0.5 / diag

        best: list[int] = []
        for i in range(len(items)):
            group = [j for j in range(len(items)) if dist(i, j) <= radius]
            if len(group) > len(best):
                best = group
        if len(best) < need:
            return []
        group = [items[j] for j in best]
        region = geo.union_box([o.box for o in group])
        share = geo.area(region) if region else 0.0
        labels = sorted({o.label for o in group})
        return [LocalFinding(
            key="dense_region:" + "+".join(sorted(o.id for o in group)),
            rule="dense_region",
            severity=self._sev("dense_region"),
            category=Category.ORGANIZATION,
            title=_voice(personality, f"{len(group)} recognised items concentrated in one area",
                         f"{len(group)} things fighting for the same patch of space", f"HOTSPOT: {len(group)} processes on one core"),
            evidence=f"{len(group)} of the {len(items)} recognised loose objects ({', '.join(labels[:6])}{'...' if len(labels) > 6 else ''}) "
                     f"have their centres within {radius * 100:.0f}% of the frame diagonal of each other; together they span "
                     f"{pct(share)} of the frame.",
            inference="Items are concentrated in this area. Whether that is a problem depends on how the space is used; "
                      "unrecognised items are not counted.",
            impact="Items in a crowded area are harder to reach and easier to knock over.",
            recommendation="If these items are not all in use, give some of them a place elsewhere.",
            confidence=self._confidence(group, mode),
            quip=_quip(personality, "Prime real estate, fully booked.", "Load balancer not found."),
            box=region,
            object_ids=[o.id for o in group],
            related_objects=labels[:8],
            measurements={"objects_in_region": len(group), "recognised_loose_objects": len(items),
                          "radius_of_diagonal": round(radius, 3), "region_share": round(share, 3)},
        )]

    def _surface_congestion(self, scene: SceneModel, *, mode: AnalysisMode, personality: Personality, **_: object) -> list[LocalFinding]:
        surfaces = self._with(scene, "surface")
        if not surfaces:
            return []
        threshold = float(self._p("surface_congestion", "minOccupancy"))  # type: ignore[arg-type]
        items = self._countable(scene)
        out = []
        for s in surfaces:
            on = [o for o in items if geo.containment(o.box, s.box) >= 0.5]
            if len(on) < 2:
                continue
            occ = geo.occupancy([o.box for o in on], within=s.box)
            if occ < threshold:
                continue
            out.append(LocalFinding(
                key=f"surface_congestion:{s.id}",
                rule="surface_congestion",
                severity=self._sev("surface_congestion"),
                category=Category.EFFICIENCY,
                title=_voice(personality, f"Little free space left on the {s.label}", f"The {s.label} is fully booked", f"DISK FULL: {s.label} at {pct(occ)} capacity"),
                evidence=f"{len(on)} objects cover {pct(occ)} of the {s.label} (threshold {pct(threshold)}).",
                inference="There is no clear work area left on the surface.",
                impact="Less room to work; items get pushed towards the edges.",
                recommendation=f"Free at least a third of the {s.label} by moving items that are not in use.",
                confidence=self._confidence([s, *on], mode),
                quip=_quip(personality, "Real estate is tight.", "Out of space, please delete something physical."),
                box=s.box,
                object_ids=[s.id, *[o.id for o in on]],
                related_objects=[s.label, *sorted({o.label for o in on})[:5]],
                measurements={"occupancy": round(occ, 3), "objects_on_surface": len(on)},
            ))
        return out

    def _overlap_cluster(self, scene: SceneModel, *, mode: AnalysisMode, personality: Personality, **_: object) -> list[LocalFinding]:
        """Groups of recognised objects whose boxes overlap each other (connected by intersection over
        the smaller box). In a 2-D photo, overlap can mean stacked, touching or merely in front of."""
        items = self._countable(scene)
        min_ios = float(self._p("overlap_cluster", "minIntersectionOverSmaller"))  # type: ignore[arg-type]
        need = int(self._p("overlap_cluster", "minObjects"))  # type: ignore[arg-type]
        parent = list(range(len(items)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        strongest: dict[int, float] = {}
        for i, a in enumerate(items):
            for j in range(i + 1, len(items)):
                b = items[j]
                ios = max(geo.containment(a.box, b.box), geo.containment(b.box, a.box))
                if ios >= min_ios:
                    parent[find(i)] = find(j)
                    strongest[i] = max(strongest.get(i, 0.0), ios)
                    strongest[j] = max(strongest.get(j, 0.0), ios)
        groups: dict[int, list[int]] = {}
        for i in range(len(items)):
            groups.setdefault(find(i), []).append(i)
        out = []
        for members in groups.values():
            if len(members) < need:
                continue
            group = [items[i] for i in members]
            region = geo.union_box([o.box for o in group])
            labels = sorted({o.label for o in group})
            out.append(LocalFinding(
                key="overlap_cluster:" + "+".join(sorted(o.id for o in group)),
                rule="overlap_cluster",
                severity=self._sev("overlap_cluster"),
                category=Category.ORGANIZATION,
                title=_voice(personality, f"{len(group)} recognised items overlap each other", "Stack overflow, physical edition",
                             "STACK OVERFLOW at 0xDESK"),
                evidence=f"{len(group)} recognised objects ({', '.join(labels[:6])}) form one group of overlapping boxes "
                         f"(each overlaps another by at least {min_ios * 100:.0f}% of the smaller box); the group spans "
                         f"{pct(geo.area(region) if region else 0.0)} of the frame.",
                inference="In a single photo, overlap can mean the items are stacked, touching or just in front of each "
                          "other; the image alone cannot tell which.",
                impact="If the items are piled up, things get knocked over or buried.",
                recommendation="If these items are stacked, give them separate places or a container.",
                confidence=self._confidence(group, mode),
                quip=_quip(personality, "Tetris, but nobody is winning.", "Recursion depth exceeded."),
                box=region,
                object_ids=[o.id for o in group],
                related_objects=labels[:6],
                measurements={"objects": len(group), "max_overlap_of_smaller": round(max(strongest[i] for i in members), 3)},
            ))
        return out

    def _keep_clear_zone(self, scene: SceneModel, *, mode: AnalysisMode, personality: Personality, **_: object) -> list[LocalFinding]:
        """Recognised objects inside an area the user asked to keep clear."""
        inside_min = float(self._p("keep_clear_zone", "minInsideFraction"))  # type: ignore[arg-type]
        out = []
        for zone in scene.zones:
            inside = [o for o in scene.objects if geo.containment(o.box, zone.box) >= inside_min]
            if not inside:
                continue
            covered = geo.occupancy([o.box for o in inside], within=zone.box)
            labels = sorted({o.label for o in inside})
            out.append(LocalFinding(
                key=f"keep_clear_zone:{zone.id}",
                rule="keep_clear_zone",
                severity=self._sev("keep_clear_zone"),
                category=Category.ORGANIZATION,
                title=_voice(personality, f"Items inside the keep-clear area \"{zone.name}\"",
                             f"\"{zone.name}\" was supposed to be empty", f"ACCESS VIOLATION in reserved region \"{zone.name}\""),
                evidence=f"{len(inside)} recognised object(s) ({', '.join(labels[:6])}) lie at least {inside_min * 100:.0f}% inside the area "
                         f"\"{zone.name}\" and cover {pct(covered)} of it.",
                inference="The area you marked to stay clear is occupied. Unrecognised items in it are not counted.",
                impact="The space you reserved is not available for its purpose.",
                recommendation=f"Move {', '.join(labels[:3])} out of \"{zone.name}\".",
                confidence=self._confidence(inside, mode),
                quip=_quip(personality, "Reserved means reserved.", "Segmentation fault: wrong memory region."),
                box=zone.box,
                object_ids=[o.id for o in inside],
                related_objects=labels[:6],
                measurements={"objects_inside": len(inside), "zone_covered": round(covered, 3)},
            ))
        return out

    # -- temporal / signal rules ------------------------------------------------------------

    def _persistent_occlusion(self, scene: SceneModel, *, mode: AnalysisMode, personality: Personality, **_: object) -> list[LocalFinding]:
        if mode not in TEMPORAL_MODES:
            return []
        min_ms = float(self._p("persistent_occlusion", "minDurationMs"))  # type: ignore[arg-type]
        out = []
        for o in scene.objects:
            if o.occlusion < self.occluded_fraction or o.occluded_ms < min_ms:
                continue
            out.append(LocalFinding(
                key=f"persistent_occlusion:{o.id}",
                rule="persistent_occlusion",
                severity=self._sev("persistent_occlusion"),
                category=Category.SPATIAL,
                title=_voice(personality, f"{o.label.capitalize()} keeps being covered", f"The {o.label} is playing hide and seek", f"Z-INDEX CONFLICT: {o.label} rendered behind other objects"),
                evidence=f"{pct(o.occlusion)} of the {o.label}'s box has overlapped other objects for {_seconds(o.occluded_ms)}.",
                inference="The object is partly hidden behind or under other items.",
                impact="Hard to reach and easy to knock over when reaching past.",
                recommendation=f"Move whatever is in front of the {o.label}, or give the {o.label} its own spot.",
                confidence=self._confidence([o], mode),
                quip=_quip(personality, "Out of sight, still in the way.", "Occlusion culling failed."),
                box=o.box,
                object_ids=[o.id],
                related_objects=[o.label],
                measurements={"occluded_fraction": round(o.occlusion, 2), "occluded_s": round(o.occluded_ms / 1000, 1)},
            ))
        return out

    def _view_obstruction(self, scene: SceneModel, *, mode: AnalysisMode, personality: Personality, timers: dict[str, float] | None = None, **_: object) -> list[LocalFinding]:
        if mode not in STREAM_MODES:
            return []
        min_area = float(self._p("view_obstruction", "minBoxArea"))  # type: ignore[arg-type]
        max_brightness = float(self._p("view_obstruction", "maxBrightness"))  # type: ignore[arg-type]
        min_ms = float(self._p("view_obstruction", "minDurationMs"))  # type: ignore[arg-type]
        brightness = scene.signals.brightness
        biggest = max(scene.objects, key=lambda o: geo.area(o.box), default=None)
        dark = brightness is not None and brightness <= max_brightness
        covering = biggest is not None and geo.area(biggest.box) >= min_area
        held = self._held(timers, "view_obstruction", dark or covering, scene.at_ms)
        if held < min_ms:
            return []
        if dark:
            evidence = f"Mean frame brightness has been {pct(brightness or 0)} (threshold {pct(max_brightness)}) for {_seconds(held)}."
            measurements: Measurements = {"brightness": round(brightness or 0, 3), "held_s": round(held / 1000, 1)}
        else:
            assert biggest is not None
            evidence = f"A {biggest.label} has covered {pct(geo.area(biggest.box))} of the view for {_seconds(held)}."
            measurements = {"covered_fraction": round(geo.area(biggest.box), 3), "held_s": round(held / 1000, 1)}
        return [LocalFinding(
            key="view_obstruction:scene",
            rule="view_obstruction",
            severity=self._sev("view_obstruction"),
            category=Category.WORKFLOW,
            title=_voice(personality, "Camera view is blocked", "Can't debug what I can't see", "VIEWPORT NULL: display buffer obstructed"),
            evidence=evidence,
            inference="Something is in front of the camera or the lens is covered.",
            impact="The scan cannot see the scene.",
            recommendation="Move the camera back or clear the lens.",
            confidence=0.9,
            quip=_quip(personality, "Bold strategy: debugging blindfolded.", "Screen saver activated in meatspace."),
            box=None,
            object_ids=[biggest.id] if covering and biggest else [],
            related_objects=[biggest.label] if covering and biggest else [],
            measurements=measurements,
        )]

    def _lighting(self, scene: SceneModel, *, mode: AnalysisMode, personality: Personality, **_: object) -> list[LocalFinding]:
        b = scene.signals.brightness
        if b is None:
            return []
        dark = float(self._p("lighting", "dark"))  # type: ignore[arg-type]
        bright = float(self._p("lighting", "bright"))  # type: ignore[arg-type]
        if dark < b < bright:
            return []
        low = b <= dark
        threshold = dark if low else bright
        margin = (dark - b) / dark if low else (b - bright) / (1 - bright)
        return [LocalFinding(
            key="lighting:scene",
            rule="lighting",
            severity=self._sev("lighting"),
            category=Category.ERGONOMICS,
            title=_voice(personality,
                         "Low light" if low else "Washed-out lighting",
                         "This room is running in dark mode" if low else "Overexposed and underthought",
                         "LOW_LUMINANCE_EXCEPTION" if low else "LUMINANCE_OVERFLOW"),
            evidence=f"Mean frame brightness is {pct(b)} ({'below' if low else 'above'} the {pct(threshold)} threshold).",
            inference="The light level is outside a comfortable range for working, and lowers detection reliability.",
            impact="Eye strain; fewer and less certain detections.",
            recommendation="Add a desk lamp or open the blinds." if low else "Reduce direct light or glare on the scene.",
            confidence=round(min(1.0, 0.6 + 0.4 * min(1.0, margin)), 3),
            quip=_quip(personality, "Mood lighting is not task lighting.", "Brightness slider not found."),
            box=None,
            object_ids=[],
            related_objects=[],
            measurements={"brightness": round(b, 3), "threshold": threshold},
        )]

    def _blur(self, scene: SceneModel, *, mode: AnalysisMode, personality: Personality, **_: object) -> list[LocalFinding]:
        s = scene.signals.sharpness
        threshold = float(self._p("blur", "minSharpness"))  # type: ignore[arg-type]
        if s is None or s >= threshold:
            return []
        return [LocalFinding(
            key="blur:scene",
            rule="blur",
            severity=self._sev("blur"),
            category=Category.WORKFLOW,
            title=_voice(personality, "Image too blurred to analyse reliably", "Did you take this mid-sprint?", "FOCUS_LOST: autofocus has left the chat"),
            evidence=f"Sharpness (log variance of the Laplacian) is {pct(s)} (threshold {pct(threshold)}).",
            inference="Camera motion or missing focus blurs the image.",
            impact="Small objects are missed and confidences drop.",
            recommendation="Hold the camera steady for a moment or tap to focus.",
            confidence=round(min(1.0, 0.6 + 0.4 * (threshold - s) / threshold), 3),
            quip=_quip(personality, "Impressionism is not a scan mode.", "Anti-aliasing applied to reality."),
            box=None,
            object_ids=[],
            related_objects=[],
            measurements={"sharpness": round(s, 3), "threshold": threshold},
        )]

    def _long_presence(self, scene: SceneModel, *, mode: AnalysisMode, personality: Personality, **_: object) -> list[LocalFinding]:
        if mode not in TEMPORAL_MODES:
            return []
        min_ms = float(self._p("long_presence", "minDurationMs"))  # type: ignore[arg-type]
        out = []
        for o in scene.objects:
            if o.static_ms < min_ms or not (self.ontology.attributes(o.label) & LEFT_BEHIND_ATTRIBUTES):
                continue
            out.append(LocalFinding(
                key=f"long_presence:{o.id}",
                rule="long_presence",
                severity=self._sev("long_presence"),
                category=Category.WORKFLOW,
                title=_voice(personality, f"{o.label.capitalize()} left in place for {_seconds(o.static_ms)}", f"The {o.label} has been abandoned", f"ZOMBIE PROCESS: {o.label} idle for {_seconds(o.static_ms)}"),
                evidence=f"The {o.label} has not moved for {_seconds(o.static_ms)}.",
                inference="It was probably left behind.",
                impact="Forgotten drinks and food attract spills and mess.",
                recommendation=f"Clear the {o.label} away if you are done with it.",
                confidence=self._confidence([o], mode),
                quip=_quip(personality, "It's not decor.", "Process has no parent; reaping recommended."),
                box=o.box,
                object_ids=[o.id],
                related_objects=[o.label],
                measurements={"static_s": round(o.static_ms / 1000, 1)},
            ))
        return out

    def _repeated_movement(self, scene: SceneModel, *, mode: AnalysisMode, personality: Personality, **_: object) -> list[LocalFinding]:
        if mode not in TEMPORAL_MODES:
            return []
        need = int(self._p("repeated_movement", "minReversals"))  # type: ignore[arg-type]
        out = []
        for o in scene.objects:
            if o.reversals < need:
                continue
            out.append(LocalFinding(
                key=f"repeated_movement:{o.id}",
                rule="repeated_movement",
                severity=self._sev("repeated_movement"),
                category=Category.WORKFLOW,
                title=_voice(personality, f"{o.label.capitalize()} moved back and forth repeatedly", f"The {o.label} can't decide where it lives", f"INFINITE LOOP: {o.label} oscillating"),
                evidence=f"The {o.label}'s tracked motion reversed direction {o.reversals} times recently.",
                inference="The object keeps being moved, possibly because it is in the way.",
                impact="Wasted motion; the item has no fixed place.",
                recommendation=f"Give the {o.label} a fixed spot out of the way.",
                confidence=self._confidence([o], mode),
                quip=_quip(personality, "Ping-pong is not a storage strategy.", "Thrashing detected."),
                box=o.box,
                object_ids=[o.id],
                related_objects=[o.label],
                measurements={"reversals": o.reversals},
            ))
        return out


def _severity_rank(severity: Severity) -> int:
    return {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "INFO": 0}[severity.value]


# --------------------------------------------------------------------------
# Scene-level summaries for local reports
# --------------------------------------------------------------------------


def scene_name(scene: SceneModel, ontology: Ontology) -> str:
    """Name the space from what was detected (attributes and COCO supercategories)."""
    attrs: dict[str, int] = {}
    supers: dict[str, int] = {}
    for o in scene.objects:
        for a in ontology.attributes(o.label):
            attrs[a] = attrs.get(a, 0) + 1
        sc = ontology.supercategory(o.label)
        if sc:
            supers[sc] = supers.get(sc, 0) + 1
    if attrs.get("electronic", 0) >= 2:
        return "WORKSPACE"
    if supers.get("kitchen", 0) + supers.get("appliance", 0) + attrs.get("food", 0) >= 2:
        return "KITCHEN"
    if attrs.get("animal", 0) >= 1:
        return "PET_ZONE"
    if attrs.get("furniture", 0) >= 2:
        return "ROOM"
    if attrs.get("electronic", 0) == 1:
        return "DEVICE_AREA"
    return "SCENE" if scene.objects else "UNKNOWN_SPACE"


def version_for(score: int | None) -> str:
    """Cosmetic 'system version' shown next to the scene name; '?' while the scene is unrated."""
    return "?" if score is None else f"{1 + score // 25}.{score % 10}"


def summary_text(scene: SceneModel, finding_count: int) -> str:
    labels: dict[str, int] = {}
    for o in scene.objects:
        labels[o.label] = labels.get(o.label, 0) + 1
    listed = ", ".join(f"{n}x {label}" if n > 1 else label for label, n in sorted(labels.items(), key=lambda kv: -kv[1])[:8])
    detectors = " + ".join(scene.stats.detectors) or "on-device detector"
    return (
        f"Local CV ({detectors}): {len(scene.objects)} object(s) in view"
        + (f" - {listed}" if listed else "")
        + f". {finding_count} measured finding(s)."
    )


_REASON_ORDER = ("detection_failed", "detection_not_run", "nothing_recognised", "detector_not_used", "fast_detector_only",
                 "runs_unreported", "too_dark", "too_blurred", "downscaled", "low_confidence", "unexplained_structure",
                 "structure_unmeasured", "closed_vocabulary")


def _main_reasons(inspection: object, limit: int = 2) -> str:
    reasons = sorted(getattr(inspection, "reasons", []), key=lambda r: _REASON_ORDER.index(r.code) if r.code in _REASON_ORDER else 99)
    return " ".join(r.message for r in reasons[:limit])


def final_diagnosis(personality: Personality, findings: Sequence[LocalFinding], object_count: int, inspection: object | None = None) -> str:
    """The one-paragraph conclusion. The facts (counts, coverage, reasons) are the same in every
    personality; only the opening line changes."""
    active = [f for f in findings if f.severity != Severity.INFO]
    status = getattr(inspection, "analysis_status", "complete") if inspection is not None else "complete"
    categories = len(getattr(inspection, "categories", []) or [])
    if status == "detection_failed":
        opener = _voice(personality, "Detection failed.", "The detectors never got a look in.", "SEGFAULT in the vision stack.")
        return f"{opener} {_main_reasons(inspection)} Nothing in this scene was examined, so no conclusion is possible."
    if status == "inconclusive":
        opener = _voice(personality, "Inconclusive.", "Can't call this one.", "TEST RESULT: UNDEFINED.")
        lead = f"{len(active)} issue(s) were found, but " if active else ""
        return f"{opener} {lead}{_main_reasons(inspection)} There is not enough evidence to judge the scene."
    if active:
        top = active[0]
        text = _voice(
            personality,
            f"{len(active)} open issue(s). Start with: {top.title[0].lower() + top.title[1:]}.",
            f"{len(active)} issue(s). Fix this first: {top.title[0].lower() + top.title[1:]}.",
            f"{len(active)} FAILING TEST(S). First failure: {top.title}.",
        )
        if status == "limited":
            text += f" The inspection is limited, so other issues may exist: {_main_reasons(inspection, 1)}"
        return text
    if status == "limited":
        opener = _voice(personality, "No issues found among the recognised objects - but this is not an all-clear.",
                        "Nothing to roast among what I could actually recognise. That's not much of a compliment.",
                        "0 FAILURES in a partial test run. Coverage report attached.")
        what = f"{object_count} object(s) in {categories} categor{'y' if categories == 1 else 'ies'} were recognised." if object_count else "No object was recognised."
        return f"{opener} {what} {_main_reasons(inspection)}"
    return _voice(
        personality,
        "No measurable issues found by the local checks." if object_count else "No objects detected - nothing to measure.",
        "Suspiciously clean. The local checks found nothing to roast." if object_count else "Nothing detected. Either it's spotless or the lens cap is on.",
        "ALL TESTS PASSED (local suite). Ship it." if object_count else "NULL SCENE: zero objects returned.",
    )
