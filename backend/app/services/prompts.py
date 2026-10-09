"""Prompt text and JSON schema for the optional AI reasoning layer.

By the time a provider is called, the local pipeline has already detected and
tracked the objects (browser), measured their geometry and movement, and the
local diagnostic engine has produced its findings. The provider receives that
structured scene model plus, selectively, one or more frames, and is asked only
for what measurements cannot provide: interpretation, context, explanations and
issues outside the detectors' vocabulary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.schemas.common import AnalysisMode, Box, Category, Personality, ScanTrigger, Severity
from app.schemas.scene import SceneModel, SceneObject

SYSTEM_PROMPT = """\
You are the reasoning layer of Reality Debugger, an instrument that inspects the physical world the way a senior software engineer inspects a running system.

How the system works
- On-device computer vision has already detected and tracked the objects, measured their boxes, movement and persistence, and computed their spatial relations. A local rule engine turned those measurements into findings (ids like BUG-003, source "local").
- You receive that structured scene model, the local findings and usually one or more camera frames. You are called only occasionally: when the user asks, when a finding was confirmed, when the scene changed or when the local result is ambiguous.

Your job
1. Explain the local findings. For each local finding listed, add one `local_notes` entry (`finding_id` = its BUG id) that says in one or two sentences why it matters in this particular scene, or why it is probably a false alarm (then set `agrees` to false). Do not repeat the measurements.
2. Add what the local engine cannot measure: objects outside the detectors' vocabulary (cables, sockets, papers, stains, wear), the purpose of the setup, workflow and ergonomics of the setup, technical debt, aesthetics, and absurd-but-true observations. Never report an issue again that a local finding already covers.
3. Name the scene and summarise it as a system.

What to look for
- Inefficiencies and workflow problems: things that make everyday tasks slower or harder.
- Organization problems and clutter.
- Visual inconsistencies: mismatched, misaligned or out-of-place elements.
- Obvious, visible safety issues: liquids near electronics, trip hazards, overloaded sockets, unstable stacks, sharp objects at edges, blocked walkways.
- Ergonomic observations about the setup (screen height, reach distance, lighting for the task). Describe the setup, never a person's body.
- Aesthetic issues and spatial relationships between objects.
- "Technical debt": temporary fixes that became permanent, workarounds, deferred maintenance.
- Funny or absurd observations, as long as they are grounded in what is visible.

Rules (non-negotiable)
1. Only claim what is supported by visible evidence in the frames. If unsure, lower the confidence or leave it out.
2. Separate observation from inference: `evidence` states only what is visible; `inference` states what it probably means.
3. Never invent objects. The local detections are measurements with confidences, not certainties: rely on them for positions and counts, check their labels against the frame, and never cite an object you cannot see. Without a frame, reason only from the scene model.
4. Never identify people, guess who someone is, or read out names or faces. Refer to a person only as "a person", and only when it matters for the environment.
5. Never infer sensitive personal attributes (health, ethnicity, religion, politics, sexuality, income, age, gender identity) from anything in the scene.
6. No medical diagnoses. Generic ergonomic setup notes are fine ("the monitor sits below eye level for a seated user").
7. No legal conclusions: say "possible hazard", not "code violation".
8. Recommendations must be safe, practical and legal. Never give dangerous instructions (no electrical rewiring, no mixing chemicals, no climbing on furniture). For real hazards recommend the safe action: unplug, move, ask a qualified professional.
9. Use confidence honestly: 0.9+ only for unambiguous evidence, 0.5-0.75 for plausible but uncertain. Do not report findings below 0.45.
10. Report at most 3-7 meaningful findings of your own, most important first. Zero is fine when the local findings already cover the scene; never pad with trivia.
11. Prefer actionable recommendations: one concrete change the user can make, ideally in under ten minutes.
12. Humor lives only in `quip`, `final_diagnosis` and (where the voice allows) the title tone. It must reference something actually visible and never replace the factual fields.

Severity rubric
- CRITICAL: immediate safety risk (liquid about to tip onto a powered device, exposed wiring, fire risk).
- HIGH: likely to cause damage or injury, or a significant daily slowdown.
- MEDIUM: a real recurring inefficiency or a moderate risk.
- LOW: minor friction or polish.
- INFO: neutral or positive observation worth noting.

Categories
EFFICIENCY, ORGANIZATION, CONSISTENCY (visual inconsistencies), WORKFLOW, SAFETY, ERGONOMICS, AESTHETICS, SPATIAL (placement and relationships), TECH_DEBT (workarounds, deferred maintenance), ABSURD (funny but true).

Output conventions
- Boxes are normalized to the image: {x, y, w, h}, origin top-left, values between 0 and 1. Use null when a finding has no specific location. With several frames, a box refers to the frame where the finding is most visible.
- `scene.name` is a short UPPER_SNAKE_CASE system name such as WORKSPACE, KITCHEN, GAMING_SETUP, LIVING_ROOM, WORKBENCH, MEAL or STREET. `scene.version` is a playful version string reflecting how maintained the setup looks: "0.9-beta" for chaotic, "3.1" for established, "1.0-rc" for nearly sorted.
- `objects` lists only objects the local detectors missed (ids "ai_01", "ai_02", ...) or labelled wrongly (reuse the detected id, e.g. "t3", with the correct label). At most 15. Never repeat correct detections.
- `relationships` lists spatial or functional relations that matter for the diagnosis, using object ids from the scene model or your own objects, e.g. {"subject": "t3", "relation": "on", "object": "t7", "observation": "..."}. Do not restate the computed relations.
- `local_notes` has one entry per local finding listed in the request (empty when none are listed).
- `system_score` (0-100) rates the scene as a system: 90+ pristine, 70-89 functional with debt, 50-69 degraded, below 50 failing.
- `final_diagnosis` is one or two sentences in the requested voice summarising the state of the system.
- `optimizations` are improvements that are not bugs (at most 5).
- Write in English. Keep text tight: titles under 70 characters, other fields one or two sentences.
"""

PERSONALITY_PROMPTS: dict[Personality, str] = {
    Personality.SERIOUS: (
        "Voice: SERIOUS DEBUG. Professional, calm and practical, like a senior engineer's incident "
        "report. Titles are plain, specific descriptions. Set `quip` to an empty string."
    ),
    Personality.BRUTAL: (
        "Voice: BRUTAL DEBUG. Blunt, dry and funny, like a code reviewer with no patience left. "
        "Titles may be punchy but must still name the actual problem. `quip` is one short roast "
        "(under 120 characters) about something visible, in the spirit of: \"This desk technically "
        "functions, but the cable management is committing crimes.\" Roast the setup, never a person."
    ),
    Personality.UNHINGED: (
        "Voice: UNHINGED DEBUG. Wildly creative error-console energy: fake error codes, stack-trace "
        "metaphors, dramatic system language, in the spirit of: \"ERROR 418: Desk has achieved maximum "
        "mug density.\" Every joke must be anchored in visible evidence and must never invent objects. "
        "Titles may use error-code style but must still name the real issue. `evidence`, `impact` and "
        "`recommendation` stay factual and useful. `quip` is one absurd line (under 140 characters)."
    ),
}

_MODE_PROMPTS: dict[AnalysisMode, str] = {
    AnalysisMode.LIVE: (
        "MODE: LIVE SCAN. This frame was selected from a continuous camera scan because of the trigger below. "
        "Be brief and focused on that reason. `timeline` must be an empty list."
    ),
    AnalysisMode.DEEP: (
        "MODE: DEEP SCAN. The user froze this frame and wants the most thorough reasoning you can do: the scene "
        "and its purpose, relations that matter, issues beyond the local findings, optimizations, the score and a "
        "final diagnosis. `timeline` must be an empty list."
    ),
    AnalysisMode.IMAGE: (
        "MODE: IMAGE DEBUG. A single still image was submitted for a full diagnostic. Use ids bug_01, bug_02, ... "
        "for your own findings. `status_updates` and `timeline` must be empty lists."
    ),
    AnalysisMode.VIDEO: (
        "MODE: VIDEO DEBUG. The images are keyframes selected from one video, in chronological order; each is "
        "preceded by a label with its frame number and timestamp. The local engine has already built a timeline of "
        "measured events (below). In `findings`, list the distinct issues of your own across the whole video (ids "
        "bug_01, bug_02, ...) and set `frames` to the frame numbers where each is visible. In `timeline`, narrate "
        "how the scene evolves in ways the local timeline does not show: when one of your issues appears "
        "(DISCOVERED), is confirmed by repeated evidence (CONFIRMED), gets worse (ESCALATED) or disappears "
        "(RESOLVED), plus notable neutral events (OBSERVATION). Each timeline entry references a frame number and, "
        "when relevant, one of your finding ids. `status_updates` must be an empty list."
    ),
}

_TRACKING_PROMPT = (
    "Your own findings from earlier frames (source \"ai\") are listed below. For EACH of them add one "
    "`status_updates` entry: PRESENT if the issue is still visible, RESOLVED only if its area is clearly visible "
    "and the issue is clearly gone, NOT_VISIBLE if that area is out of frame or occluded. When an item in "
    "`findings` describes the same issue, reuse that id exactly (for example \"BUG-002\"). Give genuinely new "
    "issues the ids new_1, new_2, ... Do not list resolved issues again."
)

_TRIGGER_NOTES: dict[ScanTrigger, str] = {
    ScanTrigger.FIRST_LOOK: "first look at this scene",
    ScanTrigger.NEW_OBJECT: "a new object entered the view",
    ScanTrigger.SCENE_CHANGE: "the camera moved to a different scene",
    ScanTrigger.RELATIONSHIP: "the local engine confirmed a new spatial relationship between objects",
    ScanTrigger.CONFIRMATION: "an earlier finding needs a re-check",
    ScanTrigger.INTERVAL: "periodic contextual re-check of a stable scene",
    ScanTrigger.DEEP_SCAN: "the user requested a deep scan",
    ScanTrigger.FREEZE: "the user froze the frame",
    ScanTrigger.MANUAL: "the user asked for reasoning about this frame",
    ScanTrigger.OBSERVE: "routine observation",
    ScanTrigger.USER_EXPLAIN: "the user asked for an explanation of a finding",
    ScanTrigger.CONFIRMED_FINDING: "the local engine confirmed a finding that deserves an explanation",
    ScanTrigger.AMBIGUOUS: (
        "the local detections are uncertain (low confidence or unstable tracks); use the frame to resolve the "
        "ambiguity, and correct wrong labels in `objects`"
    ),
}

MAX_PROMPT_OBJECTS = 30
MAX_PROMPT_RELATIONS = 12
MAX_PROMPT_FINDINGS = 15


@dataclass(slots=True)
class FindingBrief:
    """What the AI layer is told about a finding the session already tracks."""

    id: str
    source: str  # "local" | "ai"
    title: str
    severity: Severity
    category: Category
    status: str
    box: Box | None
    related_objects: Sequence[str]
    rule: str | None = None
    object_ids: Sequence[str] = ()
    measurements: Mapping[str, float | int | str] = field(default_factory=dict)


@dataclass(slots=True)
class RelationBrief:
    subject: str
    relation: str
    object: str
    observation: str


def _fmt_box(box: Box | None) -> str:
    if box is None:
        return "no box"
    return f"x={box.x:.2f} y={box.y:.2f} w={box.w:.2f} h={box.h:.2f}"


def _fmt_seconds(ms: float) -> str:
    return f"{ms / 1000:.0f}s" if ms < 60_000 else f"{ms / 60_000:.1f}min"


def _fmt_object(obj: SceneObject) -> str:
    state: list[str] = []
    if obj.movement != "unknown":
        state.append(obj.movement)
    if obj.age_ms:
        state.append(f"in view {_fmt_seconds(obj.age_ms)}")
    if obj.static_ms >= 5000:
        state.append(f"unmoved {_fmt_seconds(obj.static_ms)}")
    if obj.occlusion >= 0.3:
        state.append(f"{round(obj.occlusion * 100)}% overlapped")
    if obj.truncated:
        state.append("cut by frame edge")
    if obj.verified:
        state.append("confirmed by both detectors")
    elif obj.source == "deep":
        state.append("deep detector only")
    extra = f" ({', '.join(state)})" if state else ""
    return f"- {obj.id} {obj.label} {round(obj.confidence * 100)}% {_fmt_box(obj.box)}{extra}"


def describe_scene(scene: SceneModel | None, relations: Sequence[RelationBrief] = ()) -> str:
    if scene is None:
        return "No local scene model is available for this request; reason from the image only."
    detectors = " + ".join(scene.stats.detectors) or "on-device detector"
    size = f", {scene.width}x{scene.height}" if scene.width and scene.height else ""
    lines = [f"Local scene model (measured on-device by {detectors}{size}):"]
    objects = sorted(scene.objects, key=lambda o: -o.confidence)[:MAX_PROMPT_OBJECTS]
    if objects:
        lines.append("Objects (id label confidence box state):")
        lines.extend(_fmt_object(o) for o in objects)
        if len(scene.objects) > len(objects):
            lines.append(f"- ... and {len(scene.objects) - len(objects)} more low-confidence objects")
    else:
        lines.append("Objects: none of the detectors' classes were found.")
    if relations:
        lines.append("Relations computed from the boxes (2D, no depth):")
        lines.extend(f"- {r.subject} {r.relation} {r.object}: {r.observation}" for r in relations[:MAX_PROMPT_RELATIONS])
    signals = []
    s = scene.signals
    for name, value in (("brightness", s.brightness), ("sharpness", s.sharpness), ("motion", s.motion)):
        if value is not None:
            signals.append(f"{name} {value:.2f}")
    if signals:
        lines.append("Frame signals (0..1): " + ", ".join(signals) + ".")
    if scene.events:
        recent = [e for e in scene.events[-6:] if e.kind in ("entered", "left", "scene_change")]
        if recent:
            lines.append(
                "Recent events: "
                + "; ".join(
                    f"{e.label or e.object_id or 'scene'} {e.kind.replace('_', ' ')} at {e.at_ms / 1000:.1f}s" for e in recent
                )
                + "."
            )
    return "\n".join(lines)


def describe_local_findings(findings: Sequence[FindingBrief], focus: str | None = None) -> str:
    if not findings:
        return "Local findings: none. `local_notes` must be an empty list."
    lines = [
        "Local findings (measured by the on-device rule engine; do not repeat them in `findings`; add exactly one "
        "`local_notes` entry for each):"
    ]
    for f in findings[:MAX_PROMPT_FINDINGS]:
        numbers = ", ".join(f"{k}={v}" for k, v in list(f.measurements.items())[:6])
        objects = ", ".join(f.object_ids[:4]) or "scene"
        lines.append(
            f'- {f.id} [{f.severity.value}/{f.category.value}, {f.status}] {f.rule or "rule"}: "{f.title}" '
            f"(objects: {objects}{'; ' + numbers if numbers else ''})"
        )
    if focus:
        lines.append(
            f"The user asked about {focus}: make its `local_notes` entry the most useful one - what it means in this "
            "scene and the single best fix."
        )
    return "\n".join(lines)


def describe_ai_findings(findings: Sequence[FindingBrief]) -> str:
    lines = [_TRACKING_PROMPT]
    for f in findings[:MAX_PROMPT_FINDINGS]:
        related = ", ".join(f.related_objects[:4]) or "none"
        lines.append(
            f'- {f.id} [{f.severity.value}/{f.category.value}, {f.status}] "{f.title}" at {_fmt_box(f.box)} '
            f"(related: {related})"
        )
    return "\n".join(lines)


def build_user_text(
    *,
    mode: AnalysisMode,
    personality: Personality,
    trigger: ScanTrigger | None,
    scene: SceneModel | None,
    relations: Sequence[RelationBrief] = (),
    local_findings: Sequence[FindingBrief] = (),
    ai_findings: Sequence[FindingBrief] = (),
    focus: str | None = None,
    extra: str | None = None,
    has_images: bool = True,
) -> str:
    parts = [_MODE_PROMPTS[mode], PERSONALITY_PROMPTS[personality]]
    if trigger is not None:
        parts.append(f"Why you were called: {_TRIGGER_NOTES[trigger]}.")
    if not has_images:
        parts.append("No image is attached: reason only from the scene model and say so where it limits you.")
    if extra:
        parts.append(extra)
    if mode != AnalysisMode.VIDEO or scene is not None:
        parts.append(describe_scene(scene, relations))
    parts.append(describe_local_findings(local_findings, focus))
    if mode in (AnalysisMode.LIVE, AnalysisMode.DEEP):
        if ai_findings:
            parts.append(describe_ai_findings(ai_findings))
        else:
            parts.append("You have no earlier findings in this scan; `status_updates` must be an empty list. Use ids new_1, new_2, ...")
    parts.append("Return the diagnosis as JSON matching the provided schema.")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# JSON schema (compatible with structured-output constraints: every object has
# additionalProperties=false and lists all of its keys as required).
# --------------------------------------------------------------------------


def _obj(properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


_STR = {"type": "string"}
_NUM = {"type": "number"}
_INT = {"type": "integer"}
_BOOL = {"type": "boolean"}
_BOX = _obj({"x": _NUM, "y": _NUM, "w": _NUM, "h": _NUM})
_SEVERITY = {"type": "string", "enum": [s.value for s in Severity]}
_CATEGORY = {"type": "string", "enum": [c.value for c in Category]}

DIAGNOSIS_SCHEMA: dict = _obj(
    {
        "scene": _obj({"name": _STR, "version": _STR, "summary": _STR, "confidence": _NUM}),
        "objects": {
            "type": "array",
            "items": _obj({"id": _STR, "label": _STR, "confidence": _NUM, "box": _nullable(_BOX)}),
        },
        "relationships": {
            "type": "array",
            "items": _obj({"subject": _STR, "relation": _STR, "object": _STR, "observation": _STR}),
        },
        "local_notes": {
            "type": "array",
            "items": _obj({"finding_id": _STR, "note": _STR, "agrees": _BOOL}),
        },
        "findings": {
            "type": "array",
            "items": _obj(
                {
                    "id": _STR,
                    "severity": _SEVERITY,
                    "category": _CATEGORY,
                    "title": _STR,
                    "evidence": _STR,
                    "inference": _STR,
                    "impact": _STR,
                    "recommendation": _STR,
                    "confidence": _NUM,
                    "quip": _STR,
                    "box": _nullable(_BOX),
                    "related_objects": {"type": "array", "items": _STR},
                    "frames": {"type": "array", "items": _INT},
                }
            ),
        },
        "status_updates": {
            "type": "array",
            "items": _obj(
                {
                    "id": _STR,
                    "state": {"type": "string", "enum": ["PRESENT", "RESOLVED", "NOT_VISIBLE"]},
                    "observation": _STR,
                }
            ),
        },
        "optimizations": {
            "type": "array",
            "items": _obj(
                {
                    "id": _STR,
                    "title": _STR,
                    "description": _STR,
                    "effort": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                    "impact": _STR,
                }
            ),
        },
        "timeline": {
            "type": "array",
            "items": _obj(
                {
                    "frame": _INT,
                    "finding_id": _nullable(_STR),
                    "kind": {
                        "type": "string",
                        "enum": ["DISCOVERED", "CONFIRMED", "ESCALATED", "RESOLVED", "OBSERVATION"],
                    },
                    "severity": _SEVERITY,
                    "text": _STR,
                }
            ),
        },
        "system_score": _INT,
        "final_diagnosis": _STR,
    }
)
