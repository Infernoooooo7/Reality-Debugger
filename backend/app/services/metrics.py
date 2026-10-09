"""In-process counters for the developer metrics panel (GET /api/metrics).

Only counts and timings are kept - never images, scene contents or keys.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class Timing:
    """Running mean plus a small window for percentiles."""

    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def add(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        self.max_ms = max(self.max_ms, ms)
        self.recent.append(ms)

    def snapshot(self) -> dict[str, float | int]:
        ordered = sorted(self.recent)

        def pct(p: float) -> float:
            if not ordered:
                return 0.0
            return round(ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1) + 0.5))], 2)

        return {
            "count": self.count,
            "mean_ms": round(self.total_ms / self.count, 2) if self.count else 0.0,
            "p50_ms": pct(0.5),
            "p95_ms": pct(0.95),
            "max_ms": round(self.max_ms, 2),
        }


class Metrics:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.local_eval = Timing()
        self.observations = 0
        self.local_findings = Counter[str]()  # findings emitted per rule
        self.lifecycle = Counter[str]()  # DISCOVERED / CONFIRMED / ... events
        self.ai_latency = Timing()
        self.ai_calls = 0
        self.ai_ok = 0
        self.ai_errors = Counter[str]()  # error code -> count
        self.ai_skipped = Counter[str]()  # reason -> count
        self.ai_cached = 0
        self.ai_fallback_used = 0
        self.ai_tokens_in = 0
        self.ai_tokens_out = 0
        self.ai_by_trigger = Counter[str]()

    # -- local ----------------------------------------------------------------

    def record_local(self, elapsed_ms: float, rules: list[str]) -> None:
        self.local_eval.add(elapsed_ms)
        self.local_findings.update(rules)

    def record_events(self, kinds: list[str]) -> None:
        self.lifecycle.update(kinds)

    # -- AI ---------------------------------------------------------------------

    def record_ai_call(
        self,
        *,
        ok: bool,
        latency_ms: float,
        trigger: str | None,
        error_code: str | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
    ) -> None:
        self.ai_calls += 1
        if trigger:
            self.ai_by_trigger[trigger] += 1
        if ok:
            self.ai_ok += 1
            self.ai_latency.add(latency_ms)
        elif error_code:
            self.ai_errors[error_code] += 1
        self.ai_tokens_in += tokens_in or 0
        self.ai_tokens_out += tokens_out or 0

    def record_ai_skip(self, reason: str) -> None:
        self.ai_skipped[reason] += 1

    def snapshot(self) -> dict[str, dict]:
        return {
            "local": {
                "observations": self.observations,
                "evaluations": self.local_eval.snapshot(),
                "findings_by_rule": dict(self.local_findings.most_common()),
                "lifecycle_events": dict(self.lifecycle),
            },
            "ai": {
                "calls": self.ai_calls,
                "ok": self.ai_ok,
                "cached": self.ai_cached,
                "fallback_used": self.ai_fallback_used,
                "errors": dict(self.ai_errors),
                "skipped": dict(self.ai_skipped),
                "by_trigger": dict(self.ai_by_trigger),
                "latency": self.ai_latency.snapshot(),
                "tokens_in": self.ai_tokens_in,
                "tokens_out": self.ai_tokens_out,
            },
        }

    @property
    def uptime_s(self) -> int:
        return int(time.monotonic() - self.started)
