"""Request-side models: video manifests from the browser and the structured
diagnosis returned by the optional AI reasoning layer ("wire" format).

The wire models are deliberately lenient: every field is coerced and clipped,
and :func:`parse_diagnosis` validates list items one by one so a malformed
entry is dropped (with a warning) instead of failing the whole analysis.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

from app.schemas.common import Box, Category, Severity
from app.schemas.scene import DetectorRun, OptUnit, SceneCoverage, SceneObject
from app.utils import coerce


def _text(max_len: int) -> Any:
    return BeforeValidator(lambda v: coerce.text(v, max_len))


Confidence = Annotated[float, BeforeValidator(coerce.unit_float)]
OptionalBox = Annotated[Box | None, BeforeValidator(coerce.box)]
SeverityField = Annotated[Severity, BeforeValidator(coerce.severity)]
CategoryField = Annotated[Category, BeforeValidator(coerce.category)]


# --------------------------------------------------------------------------
# Video manifest (browser → backend)
# --------------------------------------------------------------------------


class LocalObject(BaseModel):
    model_config = ConfigDict(extra="ignore")

    track_id: Annotated[str | None, _text(32)] = None
    label: Annotated[str, _text(40)]
    confidence: Confidence = 0.5
    box: OptionalBox = None


class VideoFrameManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int = Field(ge=0, le=999)
    t: float = Field(ge=0)
    scene: int = Field(default=0, ge=0)
    reason: Annotated[str, _text(60)] = "representative"
    objects: list[LocalObject] = Field(default_factory=list, max_length=60)
    sharpness: float | None = None
    brightness: float | None = None
    motion: float | None = None
    detectors: list[Annotated[str, _text(48)]] = Field(default_factory=list, max_length=4)


class VideoSceneManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int = Field(ge=0)
    start_t: float = Field(ge=0)
    end_t: float = Field(ge=0)


class LocalVideoEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    t: float = Field(ge=0)
    kind: Literal["SCENE_CHANGE", "OBJECT_ENTERED", "OBJECT_LEFT"]
    text: Annotated[str, _text(160)]


class VideoSample(BaseModel):
    """One sampled video frame as seen by the browser's detector and tracker."""

    model_config = ConfigDict(extra="ignore")

    t: float = Field(ge=0)
    scene: int = Field(default=0, ge=0)
    objects: list[SceneObject] = Field(default_factory=list, max_length=40)
    brightness: OptUnit = None
    sharpness: OptUnit = None
    motion: OptUnit = None
    detectors: list[Annotated[str, _text(48)]] = Field(default_factory=list, max_length=4)
    runs: list[DetectorRun] = Field(default_factory=list, max_length=8)
    coverage: SceneCoverage | None = None

    @model_validator(mode="before")
    @classmethod
    def _drop_invalid_objects(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("objects"), list):
            data = dict(data)
            kept = []
            for item in data["objects"][:40]:
                try:
                    kept.append(SceneObject.model_validate(item))
                except ValidationError:
                    continue
            data["objects"] = kept
        return data


class VideoTrackSummary(BaseModel):
    """One object followed across the sampled frames by the browser's tracker."""

    model_config = ConfigDict(extra="ignore")

    id: Annotated[str, _text(40)]
    label: Annotated[str, _text(48)]
    first_t: float = Field(ge=0)
    last_t: float = Field(ge=0)
    samples: int = Field(default=1, ge=1, le=100000)
    mean_confidence: Confidence = 0.5


class VideoManifest(BaseModel):
    """Describes keyframes the browser selected from a local video."""

    model_config = ConfigDict(extra="ignore")

    duration_s: float = Field(ge=0, le=6 * 60 * 60)
    width: int | None = Field(default=None, ge=1, le=16384)
    height: int | None = Field(default=None, ge=1, le=16384)
    name: Annotated[str | None, _text(120)] = None
    size_bytes: int | None = Field(default=None, ge=0)
    detector: Annotated[str | None, _text(80)] = None
    sampled_frames: int = Field(default=0, ge=0, le=100000)
    redundant_removed: int = Field(default=0, ge=0, le=100000)
    scenes: list[VideoSceneManifest] = Field(default_factory=list, max_length=500)
    # Keyframes (images are uploaded only when AI reasoning is on).
    frames: list[VideoFrameManifest] = Field(default_factory=list, max_length=64)
    # Every sampled frame with tracked objects: the local engine's input.
    samples: list[VideoSample] = Field(default_factory=list, max_length=240)
    events: list[LocalVideoEvent] = Field(default_factory=list, max_length=200)
    tracks: list[VideoTrackSummary] = Field(default_factory=list, max_length=300)

    @model_validator(mode="after")
    def _needs_frames(self) -> "VideoManifest":
        if not self.frames and not self.samples:
            raise ValueError("the manifest lists neither samples nor keyframes")
        return self


# --------------------------------------------------------------------------
# Vision-model output ("wire" format)
# --------------------------------------------------------------------------


class AIScene(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: Annotated[str, _text(40)] = "UNKNOWN_SPACE"
    version: Annotated[str, _text(16)] = "1.0"
    summary: Annotated[str, _text(600)] = ""
    confidence: Confidence = 0.5


class AIObject(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: Annotated[str, _text(48)]
    label: Annotated[str, _text(60)]
    confidence: Confidence = 0.5
    box: OptionalBox = None

    @model_validator(mode="after")
    def _needs_label(self) -> "AIObject":
        if not self.label:
            raise ValueError("object without label")
        return self


class AIRelationship(BaseModel):
    model_config = ConfigDict(extra="ignore")

    subject: Annotated[str, _text(60)]
    relation: Annotated[str, _text(60)]
    object: Annotated[str, _text(60)]
    observation: Annotated[str, _text(300)] = ""


class AIFinding(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: Annotated[str, _text(48)] = ""
    severity: SeverityField = Severity.MEDIUM
    category: CategoryField = Category.WORKFLOW
    title: Annotated[str, _text(140)]
    evidence: Annotated[str, _text(600)] = ""
    inference: Annotated[str, _text(600)] = ""
    impact: Annotated[str, _text(400)] = ""
    recommendation: Annotated[str, _text(500)] = ""
    confidence: Confidence = 0.5
    quip: Annotated[str, _text(240)] = ""
    box: OptionalBox = None
    related_objects: list[Annotated[str, _text(60)]] = Field(default_factory=list)
    frames: list[int] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        # Accept the field names from the original product brief too.
        if isinstance(data, dict):
            data = dict(data)
            if not data.get("evidence") and data.get("observation"):
                data["evidence"] = data["observation"]
            if not data.get("inference") and data.get("description"):
                data["inference"] = data["description"]
            if isinstance(data.get("related_objects"), list):
                data["related_objects"] = data["related_objects"][:8]
            frames = data.get("frames")
            if isinstance(frames, list):
                data["frames"] = [f for f in frames if isinstance(f, int) and 0 <= f < 1000][:32]
            else:
                data["frames"] = []
        return data

    @model_validator(mode="after")
    def _needs_title(self) -> "AIFinding":
        if not self.title:
            raise ValueError("finding without title")
        return self


class AIStatusUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: Annotated[str, _text(48)]
    state: Literal["PRESENT", "RESOLVED", "NOT_VISIBLE"]
    observation: Annotated[str, _text(300)] = ""

    @model_validator(mode="before")
    @classmethod
    def _normalise_state(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("state"), str):
            data = dict(data)
            state = data["state"].strip().upper().replace(" ", "_")
            data["state"] = {"FIXED": "RESOLVED", "GONE": "RESOLVED", "STILL_PRESENT": "PRESENT", "OUT_OF_VIEW": "NOT_VISIBLE"}.get(state, state)
        return data


class AIOptimization(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: Annotated[str, _text(48)] = ""
    title: Annotated[str, _text(140)]
    description: Annotated[str, _text(500)] = ""
    effort: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    impact: Annotated[str, _text(300)] = ""

    @model_validator(mode="before")
    @classmethod
    def _normalise_effort(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            effort = str(data.get("effort") or "MEDIUM").strip().upper()
            data["effort"] = effort if effort in {"LOW", "MEDIUM", "HIGH"} else "MEDIUM"
        return data

    @model_validator(mode="after")
    def _needs_title(self) -> "AIOptimization":
        if not self.title:
            raise ValueError("optimization without title")
        return self


TimelineKind = Literal["DISCOVERED", "CONFIRMED", "ESCALATED", "RESOLVED", "OBSERVATION"]


class AITimelineEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    frame: int = Field(ge=0, le=999)
    finding_id: Annotated[str | None, _text(48)] = None
    kind: TimelineKind = "OBSERVATION"
    severity: SeverityField = Severity.INFO
    text: Annotated[str, _text(240)]

    @model_validator(mode="before")
    @classmethod
    def _normalise_kind(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            kind = str(data.get("kind") or "OBSERVATION").strip().upper()
            data["kind"] = kind if kind in {"DISCOVERED", "CONFIRMED", "ESCALATED", "RESOLVED", "OBSERVATION"} else "OBSERVATION"
            if data.get("finding_id") in ("", "null", "none"):
                data["finding_id"] = None
        return data


class AILocalNote(BaseModel):
    """The AI layer's explanation of one finding measured by the local engine."""

    model_config = ConfigDict(extra="ignore")

    finding_id: Annotated[str, _text(48)]
    note: Annotated[str, _text(400)]
    agrees: bool = True

    @model_validator(mode="after")
    def _needs_note(self) -> "AILocalNote":
        if not self.finding_id or not self.note:
            raise ValueError("note without id or text")
        return self


class AIDiagnosis(BaseModel):
    scene: AIScene = Field(default_factory=AIScene)
    objects: list[AIObject] = Field(default_factory=list)
    relationships: list[AIRelationship] = Field(default_factory=list)
    findings: list[AIFinding] = Field(default_factory=list)
    status_updates: list[AIStatusUpdate] = Field(default_factory=list)
    optimizations: list[AIOptimization] = Field(default_factory=list)
    timeline: list[AITimelineEvent] = Field(default_factory=list)
    local_notes: list[AILocalNote] = Field(default_factory=list)
    system_score: int | None = None
    final_diagnosis: str = ""


class MalformedDiagnosisError(ValueError):
    """The model output could not be interpreted as a diagnosis at all."""


_LIMITS: dict[str, tuple[type[BaseModel], int]] = {
    "objects": (AIObject, 40),
    "relationships": (AIRelationship, 20),
    "findings": (AIFinding, 10),
    "status_updates": (AIStatusUpdate, 30),
    "optimizations": (AIOptimization, 8),
    "timeline": (AITimelineEvent, 60),
    "local_notes": (AILocalNote, 30),
}


def parse_diagnosis(data: Any) -> tuple[AIDiagnosis, list[str]]:
    """Validate raw model JSON into an :class:`AIDiagnosis`.

    Raises :class:`MalformedDiagnosisError` only when the payload is unusable
    (not an object, or no scene/finding information at all).
    """
    if not isinstance(data, dict):
        raise MalformedDiagnosisError("Model output is not a JSON object.")

    warnings: list[str] = []
    parsed: dict[str, Any] = {}

    for key, (model, limit) in _LIMITS.items():
        raw = data.get(key)
        if raw is None:
            parsed[key] = []
            continue
        if not isinstance(raw, list):
            warnings.append(f"'{key}' was not a list and was ignored.")
            parsed[key] = []
            continue
        items: list[BaseModel] = []
        dropped = 0
        for item in raw[: limit * 2]:
            try:
                items.append(model.model_validate(item))
            except ValidationError:
                dropped += 1
        if dropped:
            warnings.append(f"Dropped {dropped} malformed '{key}' entr{'y' if dropped == 1 else 'ies'}.")
        if len(items) > limit:
            warnings.append(f"'{key}' truncated to {limit} entries.")
            items = items[:limit]
        parsed[key] = items

    scene_raw = data.get("scene")
    try:
        scene = AIScene.model_validate(scene_raw if isinstance(scene_raw, dict) else {})
    except ValidationError:
        warnings.append("Scene block was malformed; using defaults.")
        scene = AIScene()

    if scene_raw is None and not parsed["findings"] and not parsed["objects"] and not parsed["local_notes"]:
        raise MalformedDiagnosisError("Model output contained no scene, objects, findings or notes.")

    diagnosis = AIDiagnosis(
        scene=scene,
        system_score=coerce.score(data.get("system_score")),
        final_diagnosis=coerce.text(data.get("final_diagnosis"), 500),
        **parsed,
    )
    return diagnosis, warnings
