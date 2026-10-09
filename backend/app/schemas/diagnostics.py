"""Response models returned to the frontend."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import (
    AnalysisMode,
    Box,
    Category,
    FindingStatus,
    Personality,
    ScanTrigger,
    Severity,
    SystemStatus,
)
from app.schemas.scene import SceneModel

Source = Literal["local", "ai"]
LifecycleEventType = Literal["DISCOVERED", "CONFIRMED", "TRACKING", "RESOLVED", "REOPENED"]
MeasurementValue = float | int | str


class StatusChange(BaseModel):
    status: FindingStatus
    at: datetime
    note: str | None = None
    video_t: float | None = None


class Finding(BaseModel):
    id: str = Field(description="Stable id, e.g. BUG-003")
    severity: Severity
    category: Category
    title: str
    evidence: str = Field(description="What was measured or observed.")
    inference: str = Field(description="What the evidence suggests (interpretation).")
    impact: str
    recommendation: str
    confidence: float = Field(ge=0, le=1)
    quip: str = Field(default="", description="Personality line; never replaces the facts above.")
    status: FindingStatus
    box: Box | None = None
    related_objects: list[str] = Field(default_factory=list)
    object_ids: list[str] = Field(default_factory=list, description="Ids of the scene objects (tracks) the finding involves.")
    source: Source = Field(default="local", description="local = measured by the local CV engine; ai = AI reasoning.")
    rule: str | None = Field(default=None, description="Local rule that produced the finding.")
    measurements: dict[str, MeasurementValue] = Field(default_factory=dict, description="Numbers behind a local finding.")
    ai_note: str | None = Field(default=None, description="AI explanation attached to a local finding (separate from the measurement).")
    ai_agrees: bool | None = Field(default=None, description="False when the AI layer thinks a local finding is a false alarm.")
    sightings: int = 1
    out_of_view: bool = False
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    first_seen_s: float | None = Field(default=None, description="Video only: first timestamp (s).")
    last_seen_s: float | None = Field(default=None, description="Video only: last timestamp (s).")
    resolved_note: str | None = None
    history: list[StatusChange] = Field(default_factory=list)


class DetectedObject(BaseModel):
    id: str
    label: str
    confidence: float = Field(ge=0, le=1)
    box: Box | None = None
    source: Literal["fast", "deep", "fused", "ai"] = "fast"
    verified: bool = False


class Relationship(BaseModel):
    subject: str
    relation: str
    object: str
    observation: str = ""
    source: Source = "local"


class Optimization(BaseModel):
    id: str
    title: str
    description: str = ""
    effort: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    impact: str = ""


class SceneInfo(BaseModel):
    name: str
    version: str
    summary: str
    confidence: float = Field(ge=0, le=1)


class Counts(BaseModel):
    active_bugs: int = 0
    high_priority: int = 0
    optimizations: int = 0
    resolved: int = 0


class ImageMeta(BaseModel):
    width: int
    height: int
    original_width: int | None = None
    original_height: int | None = None
    bytes_sent: int | None = None


class ErrorInfo(BaseModel):
    code: str
    message: str
    hint: str | None = None
    retryable: bool = False


AIRunStatus = Literal["off", "ok", "cached", "skipped", "unavailable"]


class AIRun(BaseModel):
    """What the optional AI reasoning layer did for one request."""

    status: AIRunStatus = Field(description="off = no provider; ok/cached = reasoning attached; skipped = not needed or throttled; unavailable = provider failed (local results still valid).")
    provider: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    trigger: str | None = None
    reason: str | None = Field(default=None, description="Why reasoning was skipped, if it was.")
    error: ErrorInfo | None = None


class DiagnosticReport(BaseModel):
    report_id: str
    created_at: datetime
    mode: AnalysisMode
    personality: Personality
    provider: str = Field(description="'local' when only the local engine produced the report, else the AI provider.")
    model: str
    engine: str = Field(description="Version of the local diagnostic engine.")
    detectors: list[str] = Field(default_factory=list, description="Local detectors whose output was used.")
    ai: AIRun
    latency_ms: int
    system_name: str = Field(description="e.g. WORKSPACE_v3.1")
    scene: SceneInfo
    status: SystemStatus
    system_score: int = Field(ge=0, le=100)
    ai_score: int | None = None
    counts: Counts
    objects: list[DetectedObject] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    optimizations: list[Optimization] = Field(default_factory=list)
    final_diagnosis: str
    trigger: ScanTrigger | None = None
    image: ImageMeta | None = None
    warnings: list[str] = Field(default_factory=list)


class LifecycleEvent(BaseModel):
    id: str
    type: LifecycleEventType
    finding_id: str
    title: str
    severity: Severity
    at: datetime
    note: str | None = None


class ScanState(BaseModel):
    scan_id: str
    created_at: datetime
    updated_at: datetime
    personality: Personality
    analyses: int = Field(description="AI reasoning passes in this session.")
    observations: int = Field(default=0, description="Local observations evaluated in this session.")
    status: SystemStatus
    system_score: int | None = None
    system_name: str | None = None
    scene: SceneInfo | None = None
    final_diagnosis: str | None = None
    findings: list[Finding] = Field(default_factory=list)
    optimizations: list[Optimization] = Field(default_factory=list)
    counts: Counts
    events: list[LifecycleEvent] = Field(default_factory=list)
    ai: AIRun | None = Field(default=None, description="Most recent AI reasoning attempt in this session.")
    engine: str | None = None


class AISuggestion(BaseModel):
    """The backend's hint that sending the current frame for AI reasoning is worthwhile now."""

    trigger: ScanTrigger
    reason: str
    finding_ids: list[str] = Field(default_factory=list)


class ScanAnalysisResponse(BaseModel):
    """Returned by /api/scan/observe, /api/analyze/frame and /api/scan/deep."""

    scan: ScanState
    report: DiagnosticReport
    events: list[LifecycleEvent] = Field(description="Lifecycle transitions caused by this request.")
    ai_suggestion: AISuggestion | None = Field(
        default=None, description="Observe only: send a frame to /api/analyze/frame with this trigger (AI configured and allowed now)."
    )


class CreateScanRequest(BaseModel):
    personality: Personality = Personality.SERIOUS


class ObserveRequest(BaseModel):
    """A local observation: the scene model only, no image."""

    scan_id: str | None = None
    personality: Personality | None = None
    scene: SceneModel


class SceneAnalysisRequest(BaseModel):
    """One-shot local analysis of a scene model (Image Debug without AI)."""

    personality: Personality = Personality.SERIOUS
    scene: SceneModel


TimelineKind = Literal[
    "DISCOVERED",
    "CONFIRMED",
    "ESCALATED",
    "RESOLVED",
    "OBSERVATION",
    "SCENE_CHANGE",
    "OBJECT_ENTERED",
    "OBJECT_LEFT",
]


class TimelineEvent(BaseModel):
    t: float = Field(ge=0, description="Seconds from the start of the video.")
    frame_index: int | None = None
    kind: TimelineKind
    severity: Severity
    finding_id: str | None = None
    text: str
    source: Source


class VideoKeyframe(BaseModel):
    index: int
    t: float
    scene: int
    reason: str


class VideoMeta(BaseModel):
    duration_s: float
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    name: str | None = None
    size_bytes: int | None = None


class VideoSamplingStats(BaseModel):
    processed_on: Literal["browser", "server"]
    sampled_frames: int
    scenes: int
    redundant_removed: int
    keyframes: int
    tracked_objects: int = 0


class VideoReport(BaseModel):
    report: DiagnosticReport
    timeline: list[TimelineEvent]
    video: VideoMeta
    sampling: VideoSamplingStats
    keyframes: list[VideoKeyframe]


AIState = Literal["off", "ready", "unverified", "unavailable"]


class AIStatus(BaseModel):
    provider: str = Field(description="none, gemini, claude or openai.")
    model: str | None
    configured: bool
    state: AIState = Field(description="off = local-only; unverified = configured, not called yet; ready = last call worked; unavailable = last call failed.")
    detail: str
    fallback: str | None = Field(default=None, description="Explicitly configured fallback provider, if any.")
    last_error: ErrorInfo | None = None


class LocalStatus(BaseModel):
    engine: str
    rules: list[str]
    labels: int = Field(description="Object classes known to the ontology.")
    attributes: list[str]
    detectors: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    time: datetime
    ai: AIStatus
    local: LocalStatus
    limits: dict[str, int]
    features: dict[str, bool]


class AICheckResponse(BaseModel):
    ok: bool
    provider: str
    model: str | None
    latency_ms: int
    error: ErrorInfo | None = None


class MetricsResponse(BaseModel):
    uptime_s: int
    sessions: int
    local: dict[str, object]
    ai: dict[str, object]
