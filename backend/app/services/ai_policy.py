"""When the optional AI layer may be called (``config/ai.json``).

The local pipeline runs on every observation; a provider is called only for a
configured trigger, never more often than the cooldown and per-minute budget
allow, never twice for a near-identical frame (average-hash de-duplication
with a short-lived cache), and not at all for a while after it failed.
User-requested calls (Deep Scan, Explain) skip the cooldown but still count
against the per-minute budget.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.runtime_config import RuntimeConfig
from app.schemas.common import SEVERITY_RANK, AnalysisMode, ScanTrigger, Severity
from app.schemas.diagnostics import AISuggestion
from app.schemas.scene import SceneModel
from app.services.vision_service import hamming

USER_TRIGGERS = frozenset({ScanTrigger.DEEP_SCAN, ScanTrigger.FREEZE, ScanTrigger.USER_EXPLAIN, ScanTrigger.MANUAL})

# Trigger -> switch in config/ai.json "triggers". INTERVAL depends on
# periodicMs; triggers missing here (NEW_OBJECT, OBSERVE) never call the AI:
# new objects and routine observations are handled by local CV.
_SWITCHES: dict[ScanTrigger, str] = {
    ScanTrigger.DEEP_SCAN: "deepScan",
    ScanTrigger.FREEZE: "deepScan",
    ScanTrigger.USER_EXPLAIN: "userExplain",
    ScanTrigger.MANUAL: "userExplain",
    ScanTrigger.CONFIRMED_FINDING: "confirmedFinding",
    ScanTrigger.CONFIRMATION: "confirmedFinding",
    ScanTrigger.RELATIONSHIP: "newRelation",
    ScanTrigger.SCENE_CHANGE: "sceneChange",
    ScanTrigger.FIRST_LOOK: "sceneChange",
    ScanTrigger.AMBIGUOUS: "ambiguousScene",
}

# Failures that need a configuration change (long back-off) vs. transient ones.
HARD_FAILURES = frozenset({"AI_AUTH_FAILED", "AI_MODEL_NOT_FOUND", "AI_REQUEST_REJECTED", "AI_NOT_CONFIGURED"})
CACHE_ENTRIES = 8


def monotonic_ms() -> float:
    return time.monotonic() * 1000.0


@dataclass(slots=True)
class CacheEntry:
    ahash: int | None
    at_ms: float
    result: Any  # DiagnoseResult


@dataclass(slots=True)
class AIBudget:
    """AI usage state of one live scan (or of all one-shot requests)."""

    calls: deque[float] = field(default_factory=lambda: deque(maxlen=256))  # server ms of provider calls
    last_call_ms: float | None = None
    cache: OrderedDict[tuple[str, str | None], CacheEntry] = field(default_factory=OrderedDict)
    explained: set[str] = field(default_factory=set)  # local finding ids that already have an AI note
    last_view_id: int | None = None  # view the AI last reasoned about
    ambiguous_views: set[int] = field(default_factory=set)


@dataclass(slots=True)
class Decision:
    call: bool
    reason: str | None = None
    code: str | None = None  # trigger_off | cached | paused | budget | cooldown
    cached: Any = None  # DiagnoseResult reused for a duplicate frame
    retry_after_ms: float | None = None


class AIPolicy:
    def __init__(self, config: RuntimeConfig, settings: Settings) -> None:
        self.cooldown_ms = float(config.get("ai.budget.cooldownMs"))
        self.per_minute = int(settings.scan_ai_calls_per_minute or config.get("ai.budget.maxCallsPerMinute"))
        self.dedupe_bits = int(config.get("ai.budget.dedupeHamming"))
        self.cache_ttl_ms = float(config.get("ai.budget.cacheTtlMs"))
        self.failure_backoff_ms = float(config.get("ai.budget.failureBackoffMs"))
        self.hard_failure_backoff_ms = float(config.get("ai.budget.hardFailureBackoffMs"))
        self.periodic_ms = float(config.get("ai.triggers.periodicMs"))
        self.min_severity = Severity(str(config.get("ai.triggers.minSeverity")))
        self.ambiguous_confidence = float(config.get("ai.triggers.ambiguousMeanConfidence"))
        self._switches = {name: bool(config.get(f"ai.triggers.{name}")) for name in set(_SWITCHES.values())}
        # One-shot requests (Image / Video Debug) share one budget.
        self.stateless = AIBudget()
        self.blocked_until_ms = 0.0
        self.blocked_reason: str | None = None

    # -- trigger switches ---------------------------------------------------------

    def enabled_for(self, trigger: ScanTrigger) -> bool:
        if trigger == ScanTrigger.INTERVAL:
            return self.periodic_ms > 0
        switch = _SWITCHES.get(trigger)
        return bool(switch and self._switches[switch])

    # -- decisions ------------------------------------------------------------------

    def decide(
        self,
        budget: AIBudget,
        *,
        trigger: ScanTrigger,
        mode: AnalysisMode,
        ahash: int | None,
        focus: str | None = None,
        now_ms: float | None = None,
    ) -> Decision:
        now = monotonic_ms() if now_ms is None else now_ms
        user = trigger in USER_TRIGGERS
        if not self.enabled_for(trigger):
            return Decision(False, f"trigger '{trigger.value}' is off in config/ai.json", code="trigger_off")
        cached = self._lookup(budget, mode, focus, ahash, now)
        if cached is not None:
            return Decision(False, "same frame as a recent AI answer (cached)", code="cached", cached=cached)
        if not user and now < self.blocked_until_ms:
            return Decision(
                False, self.blocked_reason or "provider failing; paused", code="paused", retry_after_ms=self.blocked_until_ms - now
            )
        while budget.calls and now - budget.calls[0] > 60_000:
            budget.calls.popleft()
        if len(budget.calls) >= self.per_minute:
            return Decision(
                False,
                f"AI budget reached ({self.per_minute} calls per minute)",
                code="budget",
                retry_after_ms=60_000 - (now - budget.calls[0]),
            )
        if not user and budget.last_call_ms is not None and now - budget.last_call_ms < self.cooldown_ms:
            return Decision(
                False,
                f"cooldown ({self.cooldown_ms / 1000:.0f} s between automatic AI calls)",
                code="cooldown",
                retry_after_ms=self.cooldown_ms - (now - budget.last_call_ms),
            )
        return Decision(True)

    def record_call(self, budget: AIBudget, now_ms: float | None = None) -> None:
        """Count a provider call (successful or not) against the budget."""
        now = monotonic_ms() if now_ms is None else now_ms
        budget.calls.append(now)
        budget.last_call_ms = now

    def store(
        self, budget: AIBudget, *, mode: AnalysisMode, focus: str | None, ahash: int | None, result: Any, now_ms: float | None = None
    ) -> None:
        if ahash is None:
            return
        key = (mode.value, focus)
        budget.cache[key] = CacheEntry(ahash=ahash, at_ms=monotonic_ms() if now_ms is None else now_ms, result=result)
        budget.cache.move_to_end(key)
        while len(budget.cache) > CACHE_ENTRIES:
            budget.cache.popitem(last=False)

    def _lookup(self, budget: AIBudget, mode: AnalysisMode, focus: str | None, ahash: int | None, now: float) -> Any:
        if ahash is None:
            return None
        entry = budget.cache.get((mode.value, focus))
        if entry is None or entry.ahash is None:
            return None
        if now - entry.at_ms > self.cache_ttl_ms or hamming(entry.ahash, ahash) > self.dedupe_bits:
            return None
        return entry.result

    # -- provider health ----------------------------------------------------------------

    def on_failure(self, code: str, now_ms: float | None = None) -> None:
        now = monotonic_ms() if now_ms is None else now_ms
        hard = code in HARD_FAILURES
        self.blocked_until_ms = now + (self.hard_failure_backoff_ms if hard else self.failure_backoff_ms)
        self.blocked_reason = "provider failed recently; automatic AI calls paused" + (
            " until the configuration is fixed" if hard else ""
        )

    def on_success(self) -> None:
        self.blocked_until_ms = 0.0
        self.blocked_reason = None

    # -- live-scan suggestions ------------------------------------------------------------

    def suggest(
        self,
        budget: AIBudget,
        *,
        scene: SceneModel,
        pending: list[tuple[str, Severity, bool]],
        now_ms: float | None = None,
    ) -> AISuggestion | None:
        """Whether sending the current frame for AI reasoning is worthwhile now.

        ``pending`` lists the open, confirmed local findings in view as
        (id, severity, involves two or more objects); those without an AI note
        yet are the "confirmed finding" / "new relation" triggers.
        """
        candidates: list[tuple[ScanTrigger, str, list[str]]] = []
        waiting = [
            (fid, relation)
            for fid, severity, relation in pending
            if fid not in budget.explained and SEVERITY_RANK[severity] >= SEVERITY_RANK[self.min_severity]
        ]
        relations = [fid for fid, relation in waiting if relation]
        singles = [fid for fid, relation in waiting if not relation]
        if relations:
            candidates.append((ScanTrigger.RELATIONSHIP, "a relation between objects was confirmed", relations))
        if singles:
            candidates.append((ScanTrigger.CONFIRMED_FINDING, "a finding was confirmed", singles))
        if scene.objects and scene.view_id != budget.last_view_id:
            first = budget.last_view_id is None
            candidates.append(
                (ScanTrigger.FIRST_LOOK if first else ScanTrigger.SCENE_CHANGE, "first view" if first else "new view", [])
            )
        tentative = scene.stats.tentative_tracks
        mean = scene.stats.mean_confidence
        if mean is None and scene.objects:
            mean = sum(o.confidence for o in scene.objects) / len(scene.objects)
        uncertain = (mean is not None and mean < self.ambiguous_confidence) or (
            tentative >= 2 and tentative > len(scene.objects)
        )
        if uncertain and scene.view_id not in budget.ambiguous_views:
            candidates.append((ScanTrigger.AMBIGUOUS, "uncertain detections", []))
        now = monotonic_ms() if now_ms is None else now_ms
        if self.periodic_ms > 0 and budget.last_call_ms is not None and now - budget.last_call_ms >= self.periodic_ms:
            candidates.append((ScanTrigger.INTERVAL, "periodic re-check", []))

        for trigger, reason, ids in candidates:
            decision = self.decide(budget, trigger=trigger, mode=AnalysisMode.LIVE, ahash=None, now_ms=now)
            if decision.call:
                return AISuggestion(trigger=trigger, reason=reason, finding_ids=ids)
        return None

    def after_call(self, budget: AIBudget, *, trigger: ScanTrigger, scene: SceneModel | None, explained: list[str]) -> None:
        """Remember what a successful call covered so it is not requested again."""
        budget.explained.update(explained)
        if scene is not None:
            budget.last_view_id = scene.view_id
            if trigger == ScanTrigger.AMBIGUOUS:
                budget.ambiguous_views.add(scene.view_id)
