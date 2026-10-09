"""Optional AI reasoning layer: provider selection, prompting and validation.

Nothing in the application depends on this layer. Provider selection
(``AI_PROVIDER``):

* ``none`` - local only; no provider is ever called.
* ``gemini`` / ``claude`` / ``openai`` - that provider only.
* ``auto`` (default) - Gemini when ``GEMINI_API_KEY`` is set, otherwise none.
  A paid provider is never selected implicitly.

``AI_FALLBACK_PROVIDER`` (default ``none``) names a second provider that is
tried only when the primary fails. It is never used unless configured
explicitly, so a Gemini failure can never cause an unexpected Claude bill.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.config import Settings
from app.runtime_config import RuntimeConfig
from app.schemas.analysis import AIDiagnosis, MalformedDiagnosisError, parse_diagnosis
from app.schemas.common import AnalysisMode, Personality, ScanTrigger
from app.schemas.diagnostics import AIState, AIStatus, ErrorInfo
from app.schemas.scene import SceneModel
from app.services.ai_providers import (
    AIError,
    AIMalformedError,
    AINotConfiguredError,
    AnthropicProvider,
    GeminiProvider,
    OpenAICompatibleProvider,
    ProviderImage,
    VisionProvider,
)
from app.services.metrics import Metrics
from app.services.prompts import DIAGNOSIS_SCHEMA, SYSTEM_PROMPT, FindingBrief, RelationBrief, build_user_text
from app.services.vision_service import PreparedImage

log = logging.getLogger("reality.ai")

PROVIDERS = ("gemini", "claude", "openai")
_ALIASES = {"anthropic": "claude", "demo": "none"}
_LABELS = {"gemini": "Gemini", "claude": "Claude", "openai": "OpenAI-compatible model"}
LIVE_TIMEOUT_CAP_S = 60.0

_REPAIR_NOTE = (
    "IMPORTANT: your previous answer could not be parsed. Respond with a single JSON object that matches the "
    "schema exactly - no prose, no markdown fences."
)


def resolve_primary(settings: Settings) -> str:
    choice = _ALIASES.get(settings.ai_provider, settings.ai_provider)
    if choice == "auto":
        # Only the provider the user explicitly gave a key for in this release
        # (Gemini) is picked automatically; Claude/OpenAI need AI_PROVIDER.
        return "gemini" if settings.gemini_api_key is not None else "none"
    return choice


def resolve_fallback(settings: Settings, primary: str) -> str | None:
    choice = _ALIASES.get(settings.ai_fallback_provider, settings.ai_fallback_provider)
    if choice == "none" or choice == primary:
        return None
    return choice


@dataclass(slots=True)
class DiagnoseRequest:
    mode: AnalysisMode
    personality: Personality
    images: list[PreparedImage]
    scene: SceneModel | None = None
    relations: Sequence[RelationBrief] = ()
    trigger: ScanTrigger | None = None
    local_findings: Sequence[FindingBrief] = ()
    ai_findings: Sequence[FindingBrief] = ()  # earlier AI findings of a live scan (status updates)
    focus: str | None = None  # finding id the user asked about
    extra_prompt: str | None = None  # video: keyframe hints and the local timeline


@dataclass(slots=True)
class DiagnoseResult:
    diagnosis: AIDiagnosis
    provider: str
    model: str
    latency_ms: int
    warnings: list[str]
    input_tokens: int | None = None
    output_tokens: int | None = None
    fallback_used: bool = False


def extract_json(text: str) -> object:
    """Parse JSON from model text, tolerating code fences or surrounding prose."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("empty response")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError("no JSON object found")


def error_info(exc: AIError) -> ErrorInfo:
    return ErrorInfo(code=exc.code, message=exc.message, hint=exc.hint, retryable=exc.retryable)


class AIService:
    def __init__(
        self,
        settings: Settings,
        config: RuntimeConfig,
        metrics: Metrics | None = None,
        *,
        providers: Mapping[str, VisionProvider] | None = None,
    ) -> None:
        self.settings = settings
        self.config = config
        self.metrics = metrics or Metrics()
        self._semaphore = asyncio.Semaphore(max(1, settings.ai_max_concurrency))
        self.primary_name = resolve_primary(settings)
        self.fallback_name = resolve_fallback(settings, self.primary_name)
        self._providers: dict[str, VisionProvider] = {}
        self._problems: dict[str, str] = {}
        for name in (self.primary_name, self.fallback_name):
            if not name or name == "none":
                continue
            if providers is not None and name in providers:  # injected by tests
                self._providers[name] = providers[name]
                continue
            try:
                self._providers[name] = self._build(name)
            except AINotConfiguredError as exc:
                self._problems[name] = exc.message
            except Exception as exc:  # pragma: no cover - defensive
                log.error("Could not initialise AI provider %s: %s", name, type(exc).__name__)
                self._problems[name] = "The provider could not be initialised."
        self._state: AIState = "unverified" if self._providers else "off"
        self._last_error: ErrorInfo | None = None
        self._served_by_fallback = False

    # -- providers ---------------------------------------------------------------

    def _build(self, name: str) -> VisionProvider:
        s = self.settings
        if name == "gemini":
            if s.gemini_api_key is None:
                raise AINotConfiguredError("GEMINI_API_KEY is not set.")
            return GeminiProvider(
                s,
                default_model=str(self.config.get("ai.gemini.defaultModel")),
                temperature=float(self.config.get("ai.gemini.temperature")),
                max_output_tokens=int(self.config.get("ai.gemini.maxOutputTokens")),
            )
        if name == "claude":
            if s.anthropic_api_key is None:
                raise AINotConfiguredError("ANTHROPIC_API_KEY is not set.")
            return AnthropicProvider(s)
        if name == "openai":
            if not s.openai_model:
                raise AINotConfiguredError("OPENAI_MODEL is not set.")
            if s.openai_api_key is None and not s.openai_base_url:
                raise AINotConfiguredError("OPENAI_API_KEY or OPENAI_BASE_URL is not set.")
            return OpenAICompatibleProvider(s)
        raise AINotConfiguredError(f"Unknown AI provider '{name}'.")

    def _order(self) -> list[tuple[str, VisionProvider]]:
        return [(n, self._providers[n]) for n in (self.primary_name, self.fallback_name) if n in self._providers]

    @property
    def enabled(self) -> bool:
        """True when at least one provider could be called."""
        return bool(self._providers)

    @property
    def status(self) -> AIStatus:
        primary = self.primary_name
        fallback = self.fallback_name if self.fallback_name in self._providers else None
        if not self._providers:
            if primary == "none":
                detail = (
                    "Local-only mode (AI_PROVIDER=none). All detection, tracking and diagnostics run locally."
                    if _ALIASES.get(self.settings.ai_provider, self.settings.ai_provider) == "none"
                    else "Local-only mode: no GEMINI_API_KEY is configured. AI reasoning is optional."
                )
            else:
                problem = self._problems.get(primary, "not configured")
                detail = f"{_LABELS.get(primary, primary)} selected, but {problem.rstrip('.')}. Running local-only."
            return AIStatus(provider=primary, model=None, configured=False, state="off", detail=detail, fallback=None)

        name = primary if primary in self._providers else str(fallback)
        model = self._providers[name].model or None
        label = _LABELS.get(name, name)
        if self._state == "unavailable":
            detail = f"{label} reasoning unavailable: {self._last_error.message if self._last_error else 'last call failed'} Local CV keeps running."
        elif self._state == "ready":
            detail = f"{label} reasoning active ({model})." + (" Served by the fallback provider." if self._served_by_fallback else "")
        else:
            detail = f"{label} reasoning configured ({model}); not called yet."
        if primary not in self._providers and primary != "none":
            detail += f" Primary provider {primary} is not configured ({self._problems.get(primary, 'missing settings')})."
        return AIStatus(
            provider=name,
            model=model,
            configured=True,
            state=self._state,
            detail=detail,
            fallback=fallback if name != fallback else None,
            last_error=self._last_error,
        )

    def _mark_ok(self, *, fallback: bool) -> None:
        self._state = "ready"
        self._served_by_fallback = fallback
        if not fallback:
            self._last_error = None

    def _mark_failed(self, exc: AIError) -> None:
        self._state = "unavailable"
        self._last_error = error_info(exc)

    # -- calls -------------------------------------------------------------------------

    async def check(self) -> str:
        """Verify credentials/model of the first configured provider; returns its name."""
        order = self._order()
        if not order:
            raise AINotConfiguredError(
                "AI reasoning is off (local-only mode).",
                code="AI_OFF",
                hint="Optional: set GEMINI_API_KEY in .env (or AI_PROVIDER=claude with ANTHROPIC_API_KEY) and restart.",
            )
        name, provider = order[0]
        try:
            await provider.check()
        except AIError as exc:
            self._mark_failed(exc)
            raise
        self._mark_ok(fallback=name != self.primary_name)
        return name

    async def diagnose(self, request: DiagnoseRequest) -> DiagnoseResult:
        order = self._order()
        if not order:
            raise AINotConfiguredError("AI reasoning is off (local-only mode).", code="AI_OFF")
        last: AIError | None = None
        for index, (name, provider) in enumerate(order):
            started = time.perf_counter()
            try:
                result = await self._diagnose_with(name, provider, request)
            except AIError as exc:
                last = exc
                self.metrics.record_ai_call(
                    ok=False,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    trigger=request.trigger.value if request.trigger else request.mode.value,
                    error_code=exc.code,
                )
                log.warning("AI provider %s failed: %s (%s)", name, exc.code, exc.message)
                continue
            self.metrics.record_ai_call(
                ok=True,
                latency_ms=result.latency_ms,
                trigger=request.trigger.value if request.trigger else request.mode.value,
                tokens_in=result.input_tokens,
                tokens_out=result.output_tokens,
            )
            if index > 0:
                result.fallback_used = True
                self.metrics.ai_fallback_used += 1
                if last is not None:
                    self._last_error = error_info(last)
            self._mark_ok(fallback=index > 0)
            return result
        assert last is not None
        self._mark_failed(last)
        raise last

    async def _diagnose_with(self, name: str, provider: VisionProvider, request: DiagnoseRequest) -> DiagnoseResult:
        s = self.settings
        user_text = build_user_text(
            mode=request.mode,
            personality=request.personality,
            trigger=request.trigger,
            scene=request.scene,
            relations=request.relations,
            local_findings=request.local_findings,
            ai_findings=request.ai_findings,
            focus=request.focus,
            extra=request.extra_prompt,
            has_images=bool(request.images),
        )
        images = [ProviderImage(jpeg=img.jpeg, label=img.label) for img in request.images]
        effort = s.ai_effort_live if request.mode == AnalysisMode.LIVE else s.ai_effort_deep
        timeout = s.ai_timeout_seconds if request.mode != AnalysisMode.LIVE else min(s.ai_timeout_seconds, LIVE_TIMEOUT_CAP_S)

        started = time.perf_counter()
        attempts = 1 + max(0, s.ai_malformed_retries)
        tokens_in = tokens_out = 0
        async with self._semaphore:
            for attempt in range(attempts):
                completion = await provider.complete(
                    system=SYSTEM_PROMPT,
                    user_text=user_text if attempt == 0 else f"{user_text}\n\n{_REPAIR_NOTE}",
                    images=images,
                    schema=DIAGNOSIS_SCHEMA,
                    effort=effort or None,
                    timeout=timeout,
                    max_tokens=s.ai_max_tokens,
                )
                tokens_in += completion.input_tokens or 0
                tokens_out += completion.output_tokens or 0
                try:
                    data = extract_json(completion.text)
                    diagnosis, warnings = parse_diagnosis(data)
                except (ValueError, MalformedDiagnosisError) as exc:
                    log.warning(
                        "Malformed AI output from %s (attempt %d/%d, %d chars, stop=%s): %s",
                        name,
                        attempt + 1,
                        attempts,
                        len(completion.text),
                        completion.stop_reason,
                        exc,
                    )
                    continue
                if completion.stop_reason in ("max_tokens", "length"):
                    warnings.append("The AI model hit its output limit; its reasoning may be incomplete.")
                log.info(
                    "AI reasoning ok: provider=%s model=%s mode=%s trigger=%s images=%d tokens_in=%s tokens_out=%s",
                    name,
                    completion.model,
                    request.mode.value,
                    request.trigger.value if request.trigger else "-",
                    len(images),
                    completion.input_tokens,
                    completion.output_tokens,
                )
                return DiagnoseResult(
                    diagnosis=diagnosis,
                    provider=name,
                    model=completion.model,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    warnings=warnings,
                    input_tokens=tokens_in or None,
                    output_tokens=tokens_out or None,
                )
        raise AIMalformedError(
            "The AI model returned an answer that could not be validated.",
            hint="Local results are unaffected. Try again, or set a different model.",
        )

    async def aclose(self) -> None:
        for provider in self._providers.values():
            try:
                await provider.aclose()
            except Exception:  # pragma: no cover - shutdown best effort
                pass


__all__ = [
    "AIService",
    "AIError",
    "DiagnoseRequest",
    "DiagnoseResult",
    "PROVIDERS",
    "error_info",
    "extract_json",
    "resolve_fallback",
    "resolve_primary",
]
