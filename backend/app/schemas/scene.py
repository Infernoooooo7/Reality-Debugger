"""The structured scene model the browser's vision pipeline sends.

Everything in it is computed on the device from detector outputs, the
tracker and pixel statistics - no AI. The backend treats it as measured data:
the local diagnostic engine evaluates its rules on it, and the optional AI
layer receives it as context.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, field_validator

from app.schemas.common import Box
from app.utils import coerce


def _text(max_len: int) -> Any:
    return BeforeValidator(lambda v: coerce.text(v, max_len))


def _unit(default: float) -> Any:
    return BeforeValidator(lambda v: coerce.unit_float(v, default))


Unit0 = Annotated[float, _unit(0.0)]
OptUnit = Annotated[float | None, BeforeValidator(lambda v: None if v is None else coerce.unit_float(v, 0.0))]


def _box(value: object) -> object:
    return coerce.box(value)


class SceneObject(BaseModel):
    """One object in the scene: a confirmed track (fast detector), possibly
    verified or added by the deep detector."""

    model_config = ConfigDict(extra="ignore")

    id: Annotated[str, _text(40)]
    label: Annotated[str, _text(48)]
    confidence: Unit0 = 0.0
    box: Annotated[Box, BeforeValidator(_box)]
    source: Literal["fast", "deep", "fused"] = "fast"
    verified: bool = False
    age_ms: float = Field(default=0.0, ge=0)
    persistent: bool = False
    movement: Literal["static", "moving", "unknown"] = "unknown"
    speed: float | None = Field(default=None, ge=0, description="Kalman centre speed in frame widths per second.")
    static_ms: float = Field(default=0.0, ge=0, description="Time since the object last moved.")
    reversals: int = Field(default=0, ge=0, le=1000)
    occlusion: Unit0 = 0.0
    occluded_ms: float = Field(default=0.0, ge=0)
    truncated: bool = False


class SceneEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["entered", "left", "moved", "scene_change"]
    at_ms: float = Field(ge=0, description="Milliseconds since the start of the session (or video time).")
    object_id: Annotated[str | None, _text(40)] = None
    label: Annotated[str | None, _text(48)] = None


class SceneSignals(BaseModel):
    model_config = ConfigDict(extra="ignore")

    motion: OptUnit = None
    brightness: OptUnit = None
    sharpness: OptUnit = None
    scene_change: OptUnit = None
    motion_box: Annotated[Box | None, BeforeValidator(coerce.box)] = None


class SceneStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    fps: float | None = Field(default=None, ge=0)
    tentative_tracks: int = Field(default=0, ge=0)
    mean_confidence: OptUnit = None
    detectors: list[Annotated[str, _text(48)]] = Field(default_factory=list, max_length=4)


class SceneModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: int = 2
    at_ms: float = Field(default=0.0, ge=0, description="Session clock (ms) when the scene was captured.")
    view_id: int = Field(default=0, ge=0, description="Increments whenever the camera cuts to a different view.")
    width: int | None = Field(default=None, ge=1, le=16384)
    height: int | None = Field(default=None, ge=1, le=16384)
    objects: list[SceneObject] = Field(default_factory=list, max_length=80)
    signals: SceneSignals = Field(default_factory=SceneSignals)
    events: list[SceneEvent] = Field(default_factory=list, max_length=40)
    stats: SceneStats = Field(default_factory=SceneStats)

    @field_validator("objects", mode="before")
    @classmethod
    def _drop_invalid_objects(cls, value: object) -> object:
        """Keep every object that validates on its own; drop the rest instead of
        rejecting the whole scene (one bad box must not cost the analysis)."""
        if not isinstance(value, list):
            return value
        kept = []
        for item in value[:80]:
            try:
                kept.append(SceneObject.model_validate(item))
            except ValidationError:
                continue
        return kept
