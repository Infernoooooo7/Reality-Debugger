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

Source = Literal["ai", "demo"]
LifecycleEventType = Literal["DISCOVERED", "CONFIRMED", "TRACKING", "RESOLVED", "REOPENED"]


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
    evidence: str = Field(description="What is directly visible (observation).")
    inference: str = Field(description="What the evidence suggests (interpretation).")
    impact: str
    recommendation: str
    confidence: float = Field(ge=0, le=1)
    quip: str = Field(default="", description="Personality line; never replaces the facts above.")
    status: FindingStatus
    box: Box | None = None
    related_objects: list[str] = Field(default_factory=list)
    source: Source = "ai"
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
    source: Literal["ai", "local"] = "ai"


class Relationship(BaseModel):
    subject: str
    relation: str
    object: str
    observation: str = ""


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


class DiagnosticReport(BaseModel):
    report_id: str
    created_at: datetime
    mode: AnalysisMode
    personality: Personality
    provider: str
    model: str
    simulated: bool = Field(description="True when produced by DEMO MODE, not a vision model.")
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
    analyses: int
    status: SystemStatus
    system_score: int | None = None
    system_name: str | None = None
    scene: SceneInfo | None = None
    final_diagnosis: str | None = None
    findings: list[Finding] = Field(default_factory=list)
    optimizations: list[Optimization] = Field(default_factory=list)
    counts: Counts
    events: list[LifecycleEvent] = Field(default_factory=list)
    simulated: bool = False
    provider: str | None = None
    model: str | None = None


class ScanAnalysisResponse(BaseModel):
    """Returned by /api/analyze/frame and /api/scan/deep."""

    scan: ScanState
    report: DiagnosticReport
    events: list[LifecycleEvent] = Field(description="Lifecycle transitions caused by this analysis.")


class CreateScanRequest(BaseModel):
    personality: Personality = Personality.SERIOUS


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
    source: Literal["ai", "demo", "local"]


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


class VideoReport(BaseModel):
    report: DiagnosticReport
    timeline: list[TimelineEvent]
    video: VideoMeta
    sampling: VideoSamplingStats
    keyframes: list[VideoKeyframe]


class AIStatus(BaseModel):
    provider: str
    model: str | None
    configured: bool
    simulated: bool
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    time: datetime
    ai: AIStatus
    limits: dict[str, int]
    features: dict[str, bool]


class AICheckResponse(BaseModel):
    ok: bool
    provider: str
    model: str | None
    latency_ms: int
    simulated: bool
    error: dict | None = None
