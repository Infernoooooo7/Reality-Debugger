"""Enums and small value types shared by request and response schemas."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


SEVERITY_RANK: dict[Severity, int] = {
    Severity.CRITICAL: 4,
    Severity.HIGH: 3,
    Severity.MEDIUM: 2,
    Severity.LOW: 1,
    Severity.INFO: 0,
}


class Category(str, Enum):
    EFFICIENCY = "EFFICIENCY"
    ORGANIZATION = "ORGANIZATION"
    CONSISTENCY = "CONSISTENCY"
    WORKFLOW = "WORKFLOW"
    SAFETY = "SAFETY"
    ERGONOMICS = "ERGONOMICS"
    AESTHETICS = "AESTHETICS"
    SPATIAL = "SPATIAL"
    TECH_DEBT = "TECH_DEBT"
    ABSURD = "ABSURD"


class FindingStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    CONFIRMED = "CONFIRMED"
    TRACKING = "TRACKING"
    RESOLVED = "RESOLVED"


class SystemStatus(str, Enum):
    STABLE = "STABLE"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


class Personality(str, Enum):
    SERIOUS = "serious"
    BRUTAL = "brutal"
    UNHINGED = "unhinged"


class AnalysisMode(str, Enum):
    LIVE = "live"
    DEEP = "deep"
    IMAGE = "image"
    VIDEO = "video"


class ScanTrigger(str, Enum):
    """Why the client sent an observation or asked for AI reasoning."""

    FIRST_LOOK = "first_look"
    NEW_OBJECT = "new_object"
    SCENE_CHANGE = "scene_change"
    RELATIONSHIP = "relationship"
    CONFIRMATION = "confirmation"
    INTERVAL = "interval"
    DEEP_SCAN = "deep_scan"
    FREEZE = "freeze"
    MANUAL = "manual"
    # Local-first triggers (v2): the local engine runs on every observation;
    # these say why the optional AI layer was asked for reasoning.
    OBSERVE = "observe"
    USER_EXPLAIN = "user_explain"
    CONFIRMED_FINDING = "confirmed_finding"
    AMBIGUOUS = "ambiguous"


class Box(BaseModel):
    """Axis-aligned box in normalized image coordinates (0..1, top-left origin)."""

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)
