"""DEMO MODE reasoner.

Produces *simulated* diagnoses without any AI provider so the product can be
explored without an API key. Everything it says is derived from real inputs -
the on-device detector's objects and basic pixel statistics - using fixed
rules, and every result is flagged ``simulated=True`` so the UI labels it as
DEMO MODE. It never pretends to be a vision model.

The output uses the same wire format as the vision model and goes through the
same validation path.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.schemas.analysis import LocalContext, LocalObject
from app.schemas.common import AnalysisMode, Box, Category, Personality, Severity
from app.services.prompts import ActiveFindingBrief
from app.services.vision_service import ImageStats

ELECTRONICS = {"laptop", "keyboard", "mouse", "cell phone", "tv", "remote", "microwave", "toaster", "oven"}
CONTAINERS = {"cup", "bottle", "wine glass", "bowl", "vase"}
FOOD = {"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"}
SHARP = {"knife", "scissors"}
ANIMALS = {"cat", "dog", "bird"}
FOCUS_ANCHORS = {"laptop", "keyboard", "book", "tv"}

SEVERITY_PENALTY = {
    Severity.CRITICAL: 30,
    Severity.HIGH: 16,
    Severity.MEDIUM: 8,
    Severity.LOW: 3,
    Severity.INFO: 0,
}

P = Personality


@dataclass(slots=True)
class RuleHit:
    key: str
    severity: Severity
    category: Category
    titles: dict[Personality, str]
    evidence: str
    inference: str
    impact: str
    recommendation: str
    confidence: float
    quips: dict[Personality, str] = field(default_factory=dict)
    box: Box | None = None
    related: list[str] = field(default_factory=list)

    def title(self, personality: Personality) -> str:
        return self.titles.get(personality) or self.titles[P.SERIOUS]


def _union(a: Box, b: Box) -> Box:
    x1, y1 = min(a.x, b.x), min(a.y, b.y)
    x2, y2 = max(a.x + a.w, b.x + b.w), max(a.y + a.h, b.y + b.h)
    return Box(x=x1, y=y1, w=max(0.01, x2 - x1), h=max(0.01, y2 - y1))


def _gap(a: Box, b: Box) -> float:
    """Distance between two boxes (0 when they touch or overlap)."""
    dx = max(0.0, max(a.x, b.x) - min(a.x + a.w, b.x + b.w))
    dy = max(0.0, max(a.y, b.y) - min(a.y + a.h, b.y + b.h))
    return (dx * dx + dy * dy) ** 0.5


def _labels(objects: Sequence[LocalObject]) -> Counter[str]:
    return Counter(o.label.lower() for o in objects)


def _first_pair(objects: Sequence[LocalObject], left: set[str], right: set[str], max_gap: float) -> tuple[LocalObject, LocalObject] | None:
    best: tuple[float, LocalObject, LocalObject] | None = None
    for a in objects:
        if a.label.lower() not in left or a.box is None:
            continue
        for b in objects:
            if b is a or b.label.lower() not in right or b.box is None:
                continue
            gap = _gap(a.box, b.box)
            if gap <= max_gap and (best is None or gap < best[0]):
                best = (gap, a, b)
    return (best[1], best[2]) if best else None


def evaluate_rules(objects: Sequence[LocalObject], stats: ImageStats | None, *, allow_nominal: bool = True) -> list[RuleHit]:
    """Run the fixed demo heuristics against one frame's local perception."""
    hits: list[RuleHit] = []
    counts = _labels(objects)
    total = sum(counts.values())

    pair = _first_pair(objects, CONTAINERS, ELECTRONICS, max_gap=0.06)
    if pair:
        liquid, device = pair
        touching = _gap(liquid.box, device.box) == 0  # type: ignore[arg-type]
        hits.append(
            RuleHit(
                key="spill_risk",
                severity=Severity.HIGH,
                category=Category.SAFETY,
                titles={
                    P.SERIOUS: f"{liquid.label.capitalize()} within spill range of the {device.label}",
                    P.BRUTAL: f"The {liquid.label} is one elbow away from a hardware incident",
                    P.UNHINGED: f"ERROR 0xC0FFEE: {liquid.label} has gained root access to the {device.label}",
                },
                evidence=(
                    f"The on-device detector placed a {liquid.label} "
                    f"{'overlapping' if touching else 'right next to'} a {device.label}."
                ),
                inference="A single knock could tip the container onto the device.",
                impact="Liquid damage to electronics and a likely repair bill.",
                recommendation=f"Move the {liquid.label} at least an arm's length from the {device.label}, or use a lidded cup.",
                confidence=round(min(0.95, 0.55 + 0.4 * min(liquid.confidence, device.confidence)), 2),
                quips={
                    P.BRUTAL: "Living dangerously, one sip at a time.",
                    P.UNHINGED: "The cup is drafting a hostile takeover of the keyboard.",
                },
                box=_union(liquid.box, device.box),  # type: ignore[arg-type]
                related=[liquid.label, device.label],
            )
        )

    containers = [o for o in objects if o.label.lower() in CONTAINERS]
    if len(containers) >= 2:
        names = ", ".join(f"{n}× {label}" for label, n in Counter(o.label for o in containers).items())
        box = containers[0].box
        for c in containers[1:]:
            if box and c.box:
                box = _union(box, c.box)
        hits.append(
            RuleHit(
                key="container_density",
                severity=Severity.MEDIUM,
                category=Category.ORGANIZATION,
                titles={
                    P.SERIOUS: "Several drink containers have accumulated on one surface",
                    P.BRUTAL: "This surface is running a cup museum, not a workspace",
                    P.UNHINGED: "ERROR 418: Surface has achieved maximum mug density",
                },
                evidence=f"{len(containers)} containers detected in frame ({names}).",
                inference="Containers are being left behind instead of returned.",
                impact="Takes up working area and raises the odds of a spill.",
                recommendation="Return the empty containers to the kitchen and keep one in use.",
                confidence=round(min(0.92, 0.5 + 0.1 * len(containers)), 2),
                quips={
                    P.BRUTAL: "Hydration strategy: hoarding.",
                    P.UNHINGED: "Mug count exceeds int8. Overflow imminent.",
                },
                box=box,
                related=[o.label for o in containers[:4]],
            )
        )

    sharp = [o for o in objects if o.label.lower() in SHARP]
    if sharp:
        tool = sharp[0]
        hits.append(
            RuleHit(
                key="sharp_tool",
                severity=Severity.MEDIUM,
                category=Category.SAFETY,
                titles={
                    P.SERIOUS: f"{tool.label.capitalize()} left out in the open",
                    P.BRUTAL: f"Loose {tool.label}: a speedrun to the first-aid kit",
                    P.UNHINGED: f"WARNING: unsandboxed {tool.label} running in user space",
                },
                evidence=f"A {tool.label} is visible in the frame.",
                inference="Sharp tools that are not put away get grabbed or knocked by accident.",
                impact="Risk of cuts, especially when reaching for something else.",
                recommendation=f"Put the {tool.label} back in a drawer or holder when not in use.",
                confidence=round(0.45 + 0.4 * tool.confidence, 2),
                box=tool.box,
                related=[tool.label],
            )
        )

    food_pair = _first_pair(objects, FOOD, ELECTRONICS, max_gap=0.08)
    if food_pair:
        food, device = food_pair
        hits.append(
            RuleHit(
                key="food_near_electronics",
                severity=Severity.MEDIUM,
                category=Category.SAFETY,
                titles={
                    P.SERIOUS: f"Food next to the {device.label}",
                    P.BRUTAL: f"The {device.label} is about to be seasoned",
                    P.UNHINGED: f"SEGFAULT: {food.label} dereferenced near {device.label}",
                },
                evidence=f"A {food.label} sits close to a {device.label}.",
                inference="Crumbs and sticky residue tend to migrate into devices.",
                impact="Dirty keys, sticky surfaces, possible damage.",
                recommendation=f"Eat away from the {device.label} or use a plate on the far side of the desk.",
                confidence=round(0.45 + 0.35 * min(food.confidence, device.confidence), 2),
                box=_union(food.box, device.box),  # type: ignore[arg-type]
                related=[food.label, device.label],
            )
        )

    if counts.get("cell phone") and any(counts.get(label) for label in FOCUS_ANCHORS):
        phone = next(o for o in objects if o.label.lower() == "cell phone")
        anchor = next(label for label in FOCUS_ANCHORS if counts.get(label))
        hits.append(
            RuleHit(
                key="phone_focus_zone",
                severity=Severity.LOW,
                category=Category.WORKFLOW,
                titles={
                    P.SERIOUS: "Phone parked inside the focus zone",
                    P.BRUTAL: "Your phone has a front-row seat to your productivity",
                    P.UNHINGED: "INTERRUPT HANDLER ATTACHED: phone polling attention at 60 Hz",
                },
                evidence=f"A phone is in frame together with a {anchor}.",
                inference="A visible phone invites context switches.",
                impact="More interruptions during focused work.",
                recommendation="Move the phone out of arm's reach or flip it face down while working.",
                confidence=round(0.4 + 0.4 * phone.confidence, 2),
                box=phone.box,
                related=["cell phone", anchor],
            )
        )

    if total >= 7:
        severity = Severity.HIGH if total >= 12 else Severity.MEDIUM
        hits.append(
            RuleHit(
                key="object_density",
                severity=severity,
                category=Category.ORGANIZATION,
                titles={
                    P.SERIOUS: f"High object density ({total} items in view)",
                    P.BRUTAL: f"{total} objects in one frame. Minimalism has left the chat",
                    P.UNHINGED: f"STACK OVERFLOW: {total} objects pushed, zero popped",
                },
                evidence=f"The detector counted {total} recognizable objects in a single view.",
                inference="Surfaces are being used as storage.",
                impact="Harder to find things and less usable working space.",
                recommendation="Spend five minutes clearing everything that is not used daily.",
                confidence=0.7,
                related=[label for label, _ in counts.most_common(4)],
            )
        )

    if counts.get("book", 0) >= 3:
        hits.append(
            RuleHit(
                key="book_pile",
                severity=Severity.LOW,
                category=Category.ORGANIZATION,
                titles={
                    P.SERIOUS: "Loose stack of books",
                    P.BRUTAL: "A library with no index and no plan",
                    P.UNHINGED: "UNINDEXED DATABASE: books stored in heap memory",
                },
                evidence=f"{counts['book']} books detected outside a shelf structure.",
                inference="Books pile up where they were last used.",
                impact="Clutter and harder retrieval.",
                recommendation="Shelve the books you are not currently reading.",
                confidence=0.6,
                related=["book"],
            )
        )

    animals = [o for o in objects if o.label.lower() in ANIMALS]
    if animals:
        pet = animals[0]
        hits.append(
            RuleHit(
                key="unauthorized_process",
                severity=Severity.LOW,
                category=Category.ABSURD,
                titles={
                    P.SERIOUS: f"{pet.label.capitalize()} present in the work area",
                    P.BRUTAL: f"A {pet.label} is supervising this setup and is not impressed",
                    P.UNHINGED: f"UNAUTHORIZED PROCESS: {pet.label}.exe running with elevated privileges",
                },
                evidence=f"A {pet.label} is visible in the frame.",
                inference="Animals tend to walk across keyboards and knock items over.",
                impact="Occasional chaos; stray hair near equipment.",
                recommendation=f"Give the {pet.label} its own spot away from cables and drinks.",
                confidence=round(0.4 + 0.5 * pet.confidence, 2),
                quips={P.BRUTAL: "Management has arrived.", P.UNHINGED: "Cannot kill process: permission denied (it's a cat)."},
                box=pet.box,
                related=[pet.label],
            )
        )

    if counts.get("potted plant"):
        hits.append(
            RuleHit(
                key="plant_online",
                severity=Severity.INFO,
                category=Category.AESTHETICS,
                titles={
                    P.SERIOUS: "Plant present - a good addition",
                    P.BRUTAL: "One plant. Credit where it is due",
                    P.UNHINGED: "BIOLOGICAL SUBSYSTEM ONLINE: morale +2",
                },
                evidence="A potted plant is visible.",
                inference="Greenery generally makes a space feel calmer.",
                impact="Positive.",
                recommendation="Keep it - just make sure it is not dripping onto anything electric.",
                confidence=0.6,
                related=["potted plant"],
            )
        )

    if counts.get("person"):
        hits.append(
            RuleHit(
                key="operator_present",
                severity=Severity.INFO,
                category=Category.WORKFLOW,
                titles={
                    P.SERIOUS: "Person in frame - diagnostics limited to the environment",
                    P.BRUTAL: "A human is in the frame. They are not being graded",
                    P.UNHINGED: "OPERATOR DETECTED: user-space only, kernel untouched",
                },
                evidence="The detector reported a person.",
                inference="People are never assessed; only the surroundings are.",
                impact="None.",
                recommendation="No action needed.",
                confidence=0.5,
                related=["person"],
            )
        )

    if stats is not None:
        if stats.brightness < 0.28:
            hits.append(
                RuleHit(
                    key="low_light",
                    severity=Severity.MEDIUM,
                    category=Category.ERGONOMICS,
                    titles={
                        P.SERIOUS: "Low ambient light",
                        P.BRUTAL: "It is too dark in here to judge properly - which is a judgement",
                        P.UNHINGED: "LIGHTING DRIVER NOT FOUND: running in dark mode IRL",
                    },
                    evidence=f"Average frame brightness is {stats.brightness:.0%}.",
                    inference="The space is dim for detailed tasks.",
                    impact="Eye strain and slower work.",
                    recommendation="Add a desk lamp or open the curtains.",
                    confidence=0.65,
                )
            )
        elif stats.brightness > 0.86:
            hits.append(
                RuleHit(
                    key="glare",
                    severity=Severity.LOW,
                    category=Category.AESTHETICS,
                    titles={
                        P.SERIOUS: "Very bright / washed-out lighting",
                        P.BRUTAL: "Lit like an interrogation room",
                        P.UNHINGED: "PHOTON FLOOD: exposure buffer overflow",
                    },
                    evidence=f"Average frame brightness is {stats.brightness:.0%}.",
                    inference="Strong direct light or glare.",
                    impact="Glare on screens and harsh shadows.",
                    recommendation="Diffuse or redirect the main light source.",
                    confidence=0.55,
                )
            )

    if not hits and allow_nominal:
        hits.append(
            RuleHit(
                key="nominal",
                severity=Severity.INFO,
                category=Category.CONSISTENCY,
                titles={
                    P.SERIOUS: "No issues found by the demo heuristics",
                    P.BRUTAL: "Suspiciously clean. Demo heuristics found nothing to roast",
                    P.UNHINGED: "200 OK: reality compiled without warnings (for now)",
                },
                evidence=(
                    f"The detector recognised {total} object(s); none matched a demo rule."
                    if total
                    else "The on-device detector did not recognise any of its 80 object classes."
                ),
                inference="Demo mode only checks a handful of fixed rules; a vision model would look deeper.",
                impact="None detected.",
                recommendation="Add an API key in .env for real AI diagnostics, or point the camera at a desk or kitchen.",
                confidence=0.5,
            )
        )
    return hits


def _scene_name(counts: Counter[str]) -> str:
    def has(*labels: str) -> bool:
        return any(counts.get(label) for label in labels)

    if has("oven", "microwave", "refrigerator", "sink", "toaster"):
        return "KITCHEN"
    if has("laptop", "keyboard", "mouse"):
        return "WORKSPACE"
    if has("couch", "tv", "remote"):
        return "LIVING_ROOM"
    if has("bed"):
        return "BEDROOM"
    if has(*FOOD, "fork", "spoon", "dining table"):
        return "MEAL"
    if has("car", "bus", "truck", "bicycle", "traffic light", "motorcycle"):
        return "STREET"
    if has(*ANIMALS):
        return "PET_ZONE"
    return "UNKNOWN_SPACE"


def penalty_score(hits: Sequence[RuleHit]) -> int:
    penalty = sum(SEVERITY_PENALTY[h.severity] * (0.5 + 0.5 * h.confidence) for h in hits)
    return int(round(max(5.0, 100.0 - penalty)))


def _version(score: int) -> str:
    if score >= 88:
        return "3.2"
    if score >= 72:
        return "2.4"
    if score >= 55:
        return "1.3-beta"
    return "0.9-alpha"


def _final(personality: Personality, hits: Sequence[RuleHit]) -> str:
    bugs = [h for h in hits if h.severity != Severity.INFO]
    if not bugs:
        return {
            P.SERIOUS: "System stable. The demo heuristics found nothing that needs attention.",
            P.BRUTAL: "Nothing to roast. Enjoy it while it lasts.",
            P.UNHINGED: "All daemons sleeping. Reality is running suspiciously smoothly.",
        }[personality]
    top = max(bugs, key=lambda h: SEVERITY_PENALTY[h.severity]).title(P.SERIOUS).lower()
    return {
        P.SERIOUS: f"{len(bugs)} issue(s) detected; start with: {top}.",
        P.BRUTAL: f"Functional, but suffering from technical debt. Start with: {top}.",
        P.UNHINGED: f"Kernel panic narrowly avoided. Root cause: {top}.",
    }[personality]


def _finding_dict(hit: RuleHit, personality: Personality, frames: list[int] | None = None) -> dict:
    return {
        "id": f"demo_{hit.key}",
        "severity": hit.severity.value,
        "category": hit.category.value,
        "title": hit.title(personality),
        "evidence": hit.evidence,
        "inference": hit.inference,
        "impact": hit.impact,
        "recommendation": hit.recommendation,
        "confidence": hit.confidence,
        "quip": "" if personality == P.SERIOUS else hit.quips.get(personality, ""),
        "box": hit.box.model_dump() if hit.box else None,
        "related_objects": hit.related,
        "frames": frames or [],
    }


def _objects(objects: Sequence[LocalObject]) -> list[dict]:
    return [
        {
            "id": o.track_id or f"obj_{i + 1:02d}",
            "label": o.label,
            "confidence": o.confidence,
            "box": o.box.model_dump() if o.box else None,
        }
        for i, o in enumerate(objects[:25])
    ]


def diagnose_frame(
    *,
    mode: AnalysisMode,
    personality: Personality,
    context: LocalContext | None,
    stats: ImageStats | None,
    active: Sequence[ActiveFindingBrief] = (),
    active_keys: dict[str, str] | None = None,
) -> dict:
    """Simulated diagnosis for a single frame (live, deep or image mode).

    ``active_keys`` maps canonical finding ids to their demo rule key so the
    simulated lifecycle can confirm or resolve them.
    """
    objects = list(context.objects) if context else []
    # A clean live/deep frame simply has no findings (an "all clear" note must not
    # become a tracked bug); only one-shot image reports get a nominal entry.
    hits = evaluate_rules(objects, stats, allow_nominal=mode == AnalysisMode.IMAGE)
    hit_keys = {h.key for h in hits}
    counts = _labels(objects)
    score = penalty_score(hits)
    status_updates = []
    scene_moved = bool(context and context.scene_change is not None and context.scene_change > 0.5)
    for brief in active:
        key = (active_keys or {}).get(brief.id)
        if key is None:
            continue
        if key in hit_keys:
            state, note = "PRESENT", "Rule condition still met in the local detections (simulated)."
        elif scene_moved:
            state, note = "NOT_VISIBLE", "Camera moved to a different view (simulated)."
        else:
            state, note = "RESOLVED", "Rule condition no longer met in the local detections (simulated)."
        status_updates.append({"id": brief.id, "state": state, "observation": note})

    name = _scene_name(counts)
    summary = (
        f"Simulated diagnosis from {sum(counts.values())} locally detected object(s): "
        + (", ".join(f"{n}× {label}" for label, n in counts.most_common(6)) or "none")
        + "."
    )
    return {
        "scene": {"name": name, "version": _version(score), "summary": summary, "confidence": 0.5},
        "objects": _objects(objects),
        "relationships": [],
        "findings": [_finding_dict(h, personality) for h in sorted(hits, key=lambda h: -SEVERITY_PENALTY[h.severity])[:7]],
        "status_updates": status_updates if mode in (AnalysisMode.LIVE, AnalysisMode.DEEP) else [],
        "optimizations": _optimizations(hits),
        "timeline": [],
        "system_score": score,
        "final_diagnosis": _final(personality, hits),
    }


def _optimizations(hits: Sequence[RuleHit]) -> list[dict]:
    keys = {h.key for h in hits}
    out = []
    if "spill_risk" in keys or "container_density" in keys:
        out.append(
            {
                "id": "opt_01",
                "title": "Designate a drinks zone",
                "description": "Keep drinks on one fixed coaster on the side away from devices.",
                "effort": "LOW",
                "impact": "Removes the most common spill path.",
            }
        )
    if "object_density" in keys or "book_pile" in keys:
        out.append(
            {
                "id": f"opt_{len(out) + 1:02d}",
                "title": "Daily 5-minute reset",
                "description": "Clear the surface back to its default state at the end of each day.",
                "effort": "LOW",
                "impact": "Keeps clutter from compounding.",
            }
        )
    if "low_light" in keys:
        out.append(
            {
                "id": f"opt_{len(out) + 1:02d}",
                "title": "Add task lighting",
                "description": "A small lamp aimed at the work area, not at the screen.",
                "effort": "MEDIUM",
                "impact": "Less eye strain.",
            }
        )
    return out


def diagnose_video(
    *,
    personality: Personality,
    frames: Sequence[tuple[int, float, list[LocalObject]]],
    stats: Sequence[ImageStats | None],
) -> dict:
    """Simulated chronological diagnosis from per-keyframe local detections.

    ``frames`` holds (frame_number, timestamp_s, objects) in chronological order.
    """
    # INFO-level observations (a person or a plant in view) would flap on and
    # off between keyframes; only real issues get a narrated lifecycle.
    per_frame: list[dict[str, RuleHit]] = []
    for (_, _, objects), frame_stats in zip(frames, stats, strict=True):
        per_frame.append(
            {h.key: h for h in evaluate_rules(objects, frame_stats, allow_nominal=False) if h.severity != Severity.INFO}
        )

    timeline: list[dict] = []
    findings: dict[str, tuple[RuleHit, list[int]]] = {}
    active: dict[str, int] = {}  # rule key -> consecutive sightings
    for (frame_no, _, _), hits in zip(frames, per_frame, strict=True):
        for key, hit in hits.items():
            entry = findings.setdefault(key, (hit, []))
            entry[1].append(frame_no)
            streak = active.get(key, 0) + 1
            active[key] = streak
            if streak == 1:
                timeline.append(
                    {"frame": frame_no, "finding_id": f"demo_{key}", "kind": "DISCOVERED", "severity": hit.severity.value, "text": hit.title(personality)}
                )
            elif streak == 2:
                timeline.append(
                    {"frame": frame_no, "finding_id": f"demo_{key}", "kind": "CONFIRMED", "severity": hit.severity.value, "text": f"Confirmed across frames: {hit.title(P.SERIOUS).lower()}"}
                )
        for key in list(active):
            if key not in hits:
                hit = findings[key][0]
                timeline.append(
                    {"frame": frame_no, "finding_id": f"demo_{key}", "kind": "RESOLVED", "severity": "INFO", "text": f"No longer detected: {hit.title(P.SERIOUS).lower()}"}
                )
                del active[key]

    hits_all = [hit for hit, _ in findings.values()]
    all_objects = [o for _, _, objs in frames for o in objs]
    counts = _labels(all_objects)
    still_active = [findings[k][0] for k in active]
    score = penalty_score(still_active or hits_all[:2])
    final_hits = hits_all or evaluate_rules([], None)
    return {
        "scene": {
            "name": _scene_name(counts),
            "version": _version(score),
            "summary": f"Simulated timeline built from {len(frames)} keyframes and local detections.",
            "confidence": 0.5,
        },
        "objects": _objects(list({o.label: o for o in all_objects}.values())),
        "relationships": [],
        "findings": [_finding_dict(hit, personality, frames=frames_seen) for hit, frames_seen in findings.values()][:7]
        or [_finding_dict(final_hits[0], personality)],
        "status_updates": [],
        "optimizations": _optimizations(hits_all),
        "timeline": timeline,
        "system_score": score,
        "final_diagnosis": _final(personality, still_active or hits_all),
    }
