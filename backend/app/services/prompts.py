"""Prompt text and the JSON schema the vision model must follow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas.analysis import LocalContext, LocalObject
from app.schemas.common import AnalysisMode, Box, Category, Personality, ScanTrigger, Severity

SYSTEM_PROMPT = """\
You are the diagnostic core of Reality Debugger, an instrument that inspects camera frames of the physical world the way a senior software engineer inspects a running system. You receive one or more frames plus hints from an on-device object detector, and you return a structured diagnosis.

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
3. Never invent objects. Every object you list or cite must be visible. Detector hints can be wrong: confirm them visually, ignore the ones you cannot see, and add objects the detector missed.
4. Never identify people, guess who someone is, or read out names or faces. Refer to a person only as "a person", and only when it matters for the environment.
5. Never infer sensitive personal attributes (health, ethnicity, religion, politics, sexuality, income, age, gender identity) from anything in the scene.
6. No medical diagnoses. Generic ergonomic setup notes are fine ("the monitor sits below eye level for a seated user").
7. No legal conclusions: say "possible hazard", not "code violation".
8. Recommendations must be safe, practical and legal. Never give dangerous instructions (no electrical rewiring, no mixing chemicals, no climbing on furniture). For real hazards recommend the safe action: unplug, move, ask a qualified professional.
9. Use confidence honestly: 0.9+ only for unambiguous evidence, 0.5-0.75 for plausible but uncertain. Do not report findings below 0.45.
10. Report 3-7 meaningful findings, most important first. Fewer is fine for a genuinely clean scene; never pad with trivia.
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
- `objects` lists the notable visible objects (at most 25) with ids like "obj_01".
- `relationships` lists spatial relations that matter for the diagnosis, e.g. {"subject": "obj_03", "relation": "overlaps", "object": "obj_01", "observation": "..."}.
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
        "MODE: LIVE SCAN. This frame was selected from a continuous camera scan. Be quick and focused: "
        "report only what is clearly visible in this frame. `timeline` must be an empty list."
    ),
    AnalysisMode.DEEP: (
        "MODE: DEEP SCAN. The user froze this frame and wants the most thorough analysis you can do: the "
        "scene, every notable object, the spatial relationships, all active bugs, optimizations, the score "
        "and a final diagnosis. `timeline` must be an empty list."
    ),
    AnalysisMode.IMAGE: (
        "MODE: IMAGE DEBUG. A single still image was submitted for a full diagnostic. Use ids bug_01, "
        "bug_02, ... for findings. `status_updates` and `timeline` must be empty lists."
    ),
    AnalysisMode.VIDEO: (
        "MODE: VIDEO DEBUG. The images are keyframes selected from one video, in chronological order; each "
        "is preceded by a label with its frame number and timestamp. Treat them as a timeline of an evolving "
        "scene. In `findings`, list the distinct issues across the whole video (ids bug_01, bug_02, ...) and "
        "set `frames` to the frame numbers where each issue is visible. In `timeline`, narrate in "
        "chronological order how things evolve: when an issue first appears (DISCOVERED), when repeated "
        "evidence confirms it (CONFIRMED), when it gets worse (ESCALATED), when it disappears or is fixed "
        "(RESOLVED), plus notable neutral events (OBSERVATION). Each timeline entry references a frame "
        "number and, when relevant, a finding id. `status_updates` must be an empty list."
    ),
}

_TRACKING_PROMPT = (
    "Active findings from earlier frames are listed below. For EACH of them add one `status_updates` entry: "
    "PRESENT if the issue is still visible, RESOLVED only if its area is clearly visible and the issue is "
    "clearly gone, NOT_VISIBLE if that area is out of frame or occluded. When an item in `findings` "
    "describes the same issue as an active finding, reuse that id exactly (for example \"BUG-002\"). Give "
    "genuinely new issues the ids new_1, new_2, ... Do not list resolved issues again."
)

_TRIGGER_NOTES: dict[ScanTrigger, str] = {
    ScanTrigger.FIRST_LOOK: "first look at this scene",
    ScanTrigger.NEW_OBJECT: "a significant new object entered the frame",
    ScanTrigger.SCENE_CHANGE: "the scene changed substantially",
    ScanTrigger.RELATIONSHIP: "the on-device engine spotted a potentially interesting spatial relationship",
    ScanTrigger.CONFIRMATION: "an earlier finding needs confirmation or a resolution check",
    ScanTrigger.INTERVAL: "periodic re-check of a stable scene",
    ScanTrigger.DEEP_SCAN: "the user requested a deep scan",
    ScanTrigger.FREEZE: "the user froze the frame",
    ScanTrigger.MANUAL: "manual request",
}


@dataclass(slots=True)
class ActiveFindingBrief:
    id: str
    title: str
    severity: Severity
    category: Category
    status: str
    box: Box | None
    related_objects: Sequence[str]


def _fmt_box(box: Box | None) -> str:
    if box is None:
        return "no box"
    return f"x={box.x:.2f} y={box.y:.2f} w={box.w:.2f} h={box.h:.2f}"


def _fmt_object(obj: LocalObject) -> str:
    name = obj.track_id or obj.label
    return f"- {name} [{obj.label}] conf {obj.confidence:.2f}, {_fmt_box(obj.box)}"


def describe_context(context: LocalContext | None) -> str:
    if context is None:
        return "No on-device detector hints were provided."
    lines: list[str] = []
    detector = context.detector or "on-device detector"
    if context.objects:
        lines.append(f"On-device detector ({detector}) hints. They may be incomplete or wrong:")
        lines.extend(_fmt_object(obj) for obj in context.objects[:30])
    else:
        lines.append(f"The on-device detector ({detector}) reported no known object classes.")
    signals = []
    if context.motion is not None:
        signals.append(f"motion {context.motion:.2f}")
    if context.scene_change is not None:
        signals.append(f"scene change {context.scene_change:.2f}")
    if context.brightness is not None:
        signals.append(f"brightness {context.brightness:.2f}")
    if context.sharpness is not None:
        signals.append(f"sharpness {context.sharpness:.2f}")
    if signals:
        lines.append("Local signals (0..1): " + ", ".join(signals) + ".")
    if context.relationships:
        lines.append("Local spatial heuristics: " + "; ".join(context.relationships[:8]) + ".")
    if context.trigger_detail:
        lines.append(f"Client note: {context.trigger_detail}")
    return "\n".join(lines)


def describe_active(findings: Sequence[ActiveFindingBrief]) -> str:
    lines = [_TRACKING_PROMPT]
    for f in findings[:15]:
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
    context: LocalContext | None,
    active: Sequence[ActiveFindingBrief] = (),
    extra: str | None = None,
) -> str:
    parts = [_MODE_PROMPTS[mode], PERSONALITY_PROMPTS[personality]]
    if trigger is not None:
        parts.append(f"Why this frame was sent: {_TRIGGER_NOTES[trigger]}.")
    if extra:
        parts.append(extra)
    if context is not None or mode != AnalysisMode.VIDEO:
        parts.append(describe_context(context))
    if mode in (AnalysisMode.LIVE, AnalysisMode.DEEP):
        if active:
            parts.append(describe_active(active))
        else:
            parts.append("There are no active findings yet; `status_updates` must be an empty list. Use ids new_1, new_2, ...")
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
