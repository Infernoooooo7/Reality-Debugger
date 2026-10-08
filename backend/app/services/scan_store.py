"""In-memory live-scan sessions.

Only findings and metadata are kept - never images. Sessions expire after
``SCAN_TTL_SECONDS`` of inactivity and can be deleted explicitly by the user.
"""

from __future__ import annotations

import asyncio
import re
import secrets
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.errors import NotFoundError, RateLimitedError
from app.schemas.analysis import AIFinding
from app.schemas.common import Box, Category, FindingStatus, Personality, Severity
from app.schemas.diagnostics import Finding, LifecycleEvent, Optimization, SceneInfo, StatusChange

SCAN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,48}$")
_BUG_RE = re.compile(r"^bug[\s_-]*0*(\d{1,4})$", re.IGNORECASE)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def canonical_bug_id(raw: str) -> str | None:
    match = _BUG_RE.match(raw.strip())
    return f"BUG-{int(match.group(1)):03d}" if match else None


@dataclass(slots=True)
class TrackedFinding:
    id: str
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
    related_objects: list[str]
    status: FindingStatus
    source: str
    first_seen_at: datetime
    last_seen_at: datetime
    sightings: int = 1
    out_of_view: bool = False
    resolved_note: str | None = None
    demo_key: str | None = None
    history: list[StatusChange] = field(default_factory=list)

    @classmethod
    def from_ai(cls, finding_id: str, af: AIFinding, *, source: str, status: FindingStatus, now: datetime) -> "TrackedFinding":
        tf = cls(
            id=finding_id,
            severity=af.severity,
            category=af.category,
            title=af.title,
            evidence=af.evidence,
            inference=af.inference,
            impact=af.impact,
            recommendation=af.recommendation,
            confidence=af.confidence,
            quip=af.quip,
            box=af.box,
            related_objects=list(af.related_objects),
            status=status,
            source=source,
            first_seen_at=now,
            last_seen_at=now,
            demo_key=af.id[5:] if af.id.startswith("demo_") else None,
        )
        tf.history.append(StatusChange(status=status, at=now))
        return tf

    def update_from(self, af: AIFinding) -> None:
        self.severity = af.severity
        self.category = af.category
        self.title = af.title or self.title
        self.evidence = af.evidence or self.evidence
        self.inference = af.inference or self.inference
        self.impact = af.impact or self.impact
        self.recommendation = af.recommendation or self.recommendation
        # Smooth confidence so one shaky frame does not swing it wildly.
        self.confidence = round(0.6 * af.confidence + 0.4 * self.confidence, 3)
        self.quip = af.quip or self.quip
        self.box = af.box or self.box
        if af.related_objects:
            self.related_objects = list(af.related_objects)

    def transition(self, status: FindingStatus, now: datetime, note: str | None = None) -> None:
        self.status = status
        self.history.append(StatusChange(status=status, at=now, note=note))
        if status == FindingStatus.RESOLVED:
            self.resolved_note = note
        else:
            self.resolved_note = None

    def to_model(self) -> Finding:
        return Finding(
            id=self.id,
            severity=self.severity,
            category=self.category,
            title=self.title,
            evidence=self.evidence,
            inference=self.inference,
            impact=self.impact,
            recommendation=self.recommendation,
            confidence=self.confidence,
            quip=self.quip,
            status=self.status,
            box=self.box,
            related_objects=self.related_objects,
            source="demo" if self.source == "demo" else "ai",
            sightings=self.sightings,
            out_of_view=self.out_of_view,
            first_seen_at=self.first_seen_at,
            last_seen_at=self.last_seen_at,
            resolved_note=self.resolved_note,
            history=list(self.history[-12:]),
        )


@dataclass(slots=True)
class ScanSession:
    scan_id: str
    personality: Personality
    created_at: datetime
    updated_at: datetime
    findings: dict[str, TrackedFinding] = field(default_factory=dict)
    events: deque[LifecycleEvent] = field(default_factory=lambda: deque(maxlen=80))
    next_bug: int = 1
    next_event: int = 1
    analyses: int = 0
    scene: SceneInfo | None = None
    system_name: str | None = None
    last_ai_score: int | None = None
    final_diagnosis: str | None = None
    optimizations: list[Optimization] = field(default_factory=list)
    simulated: bool = False
    provider: str | None = None
    model: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    call_times: deque[float] = field(default_factory=lambda: deque(maxlen=64))
    last_access: float = field(default_factory=time.monotonic)

    def allocate_id(self) -> str:
        finding_id = f"BUG-{self.next_bug:03d}"
        self.next_bug += 1
        return finding_id

    def resolve_id(self, raw: str) -> TrackedFinding | None:
        """Find a finding by canonical id (BUG-002, bug_2, ...) or demo rule id."""
        if not raw:
            return None
        canonical = canonical_bug_id(raw)
        if canonical and canonical in self.findings:
            return self.findings[canonical]
        if raw.startswith("demo_"):
            key = raw[5:]
            for tf in self.findings.values():
                if tf.demo_key == key:
                    return tf
        return None

    def add_event(self, kind: str, tf: TrackedFinding, now: datetime, note: str | None = None) -> LifecycleEvent:
        event = LifecycleEvent(
            id=f"EVT-{self.next_event:04d}",
            type=kind,  # type: ignore[arg-type]
            finding_id=tf.id,
            title=tf.title,
            severity=tf.severity,
            at=now,
            note=note,
        )
        self.next_event += 1
        self.events.append(event)
        return event

    def check_rate(self, per_minute: int) -> None:
        now = time.monotonic()
        while self.call_times and now - self.call_times[0] > 60:
            self.call_times.popleft()
        if len(self.call_times) >= per_minute:
            retry = max(1, int(60 - (now - self.call_times[0])) + 1)
            raise RateLimitedError(
                f"Analysis budget reached ({per_minute} AI analyses per minute for one scan).",
                hint="Local tracking keeps running; the next analysis will start automatically.",
                code="SCAN_RATE_LIMITED",
                headers={"Retry-After": str(retry)},
            )
        self.call_times.append(now)


class ScanStore:
    def __init__(self, *, ttl_seconds: int, max_scans: int) -> None:
        self._ttl = ttl_seconds
        self._max = max_scans
        self._scans: OrderedDict[str, ScanSession] = OrderedDict()

    def __len__(self) -> int:
        return len(self._scans)

    def _evict(self) -> None:
        now = time.monotonic()
        for scan_id in [sid for sid, s in self._scans.items() if now - s.last_access > self._ttl]:
            del self._scans[scan_id]
        while len(self._scans) > self._max:
            self._scans.popitem(last=False)

    def create(self, personality: Personality) -> ScanSession:
        self._evict()
        now = utcnow()
        session = ScanSession(
            scan_id=f"scan_{secrets.token_urlsafe(9)}",
            personality=personality,
            created_at=now,
            updated_at=now,
        )
        self._scans[session.scan_id] = session
        return session

    def get(self, scan_id: str) -> ScanSession:
        self._evict()
        if not SCAN_ID_RE.match(scan_id or "") or scan_id not in self._scans:
            raise NotFoundError(
                "Scan session not found (it may have expired or been cleared).",
                code="SCAN_NOT_FOUND",
                hint="Start a new scan.",
            )
        session = self._scans[scan_id]
        session.last_access = time.monotonic()
        self._scans.move_to_end(scan_id)
        return session

    def get_or_create(self, scan_id: str | None, personality: Personality) -> ScanSession:
        if scan_id:
            try:
                session = self.get(scan_id)
                session.personality = personality
                return session
            except NotFoundError:
                pass
        return self.create(personality)

    def delete(self, scan_id: str) -> bool:
        return self._scans.pop(scan_id, None) is not None
