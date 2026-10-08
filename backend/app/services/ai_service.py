"""Vision-AI orchestration: provider selection, prompting, validation, retries."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.config import Settings
from app.schemas.analysis import AIDiagnosis, LocalContext, MalformedDiagnosisError, parse_diagnosis
from app.schemas.common import AnalysisMode, Personality, ScanTrigger
from app.schemas.diagnostics import AIStatus
from app.services import demo_reasoner
from app.services.ai_providers import (
    AIError,
    AIMalformedError,
    AINotConfiguredError,
    AnthropicProvider,
    OpenAICompatibleProvider,
    ProviderImage,
    VisionProvider,
)
from app.services.prompts import DIAGNOSIS_SCHEMA, SYSTEM_PROMPT, ActiveFindingBrief, build_user_text
from app.services.vision_service import PreparedImage

log = logging.getLogger("reality.ai")

DEMO_MODEL = "demo-heuristics-v1"

_REPAIR_NOTE = (
    "IMPORTANT: your previous answer could not be parsed. Respond with a single JSON object that matches the "
    "schema exactly - no prose, no markdown fences."
)


@dataclass(slots=True)
class DiagnoseRequest:
    mode: AnalysisMode
    personality: Personality
    images: list[PreparedImage]
    context: LocalContext | None = None
    trigger: ScanTrigger | None = None
    active: Sequence[ActiveFindingBrief] = ()
    # canonical finding id -> demo rule key (demo lifecycle only)
    active_demo_keys: dict[str, str] = field(default_factory=dict)
    # video only: (frame_number, t, local context) per image, chronological
    video_frames: Sequence[tuple[int, float, LocalContext | None]] = ()
    extra_prompt: str | None = None
    force_demo: bool = False


@dataclass(slots=True)
class DiagnoseResult:
    diagnosis: AIDiagnosis
    provider: str
    model: str
    simulated: bool
    latency_ms: int
    warnings: list[str]


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


class AIService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._semaphore = asyncio.Semaphore(max(1, settings.ai_max_concurrency))
        self._provider: VisionProvider | None = None
        self._status = self._init_provider()

    # -- provider selection --------------------------------------------------

    def _init_provider(self) -> AIStatus:
        s = self.settings
        choice = s.ai_provider
        if choice == "auto":
            if s.anthropic_api_key is not None:
                choice = "anthropic"
            elif s.openai_api_key is not None or s.openai_base_url:
                choice = "openai"
            else:
                choice = "demo"

        if choice == "demo":
            detail = (
                "DEMO MODE: no AI provider configured. Diagnoses are simulated from on-device detections."
                if s.ai_provider == "auto"
                else "DEMO MODE forced by AI_PROVIDER=demo."
            )
            return AIStatus(provider="demo", model=DEMO_MODEL, configured=False, simulated=True, detail=detail)

        try:
            if choice == "anthropic":
                self._provider = AnthropicProvider(s)
            else:
                self._provider = OpenAICompatibleProvider(s)
        except Exception as exc:  # pragma: no cover - defensive
            log.error("Could not initialise AI provider %s: %s", choice, type(exc).__name__)
            return AIStatus(
                provider=choice, model=None, configured=False, simulated=True, detail="AI provider failed to initialise."
            )

        model = self._provider.model or None
        if choice == "openai" and not model:
            return AIStatus(
                provider=choice, model=None, configured=False, simulated=False, detail="OPENAI_MODEL is not set."
            )
        return AIStatus(
            provider=choice, model=model, configured=True, simulated=False, detail=f"Vision model ready ({model})."
        )

    @property
    def status(self) -> AIStatus:
        return self._status

    @property
    def demo_only(self) -> bool:
        return self._provider is None

    async def check(self) -> None:
        """Verify credentials / model with a lightweight provider call."""
        if self._provider is None:
            raise AINotConfiguredError(
                "No AI provider is configured - running in DEMO MODE.",
                hint="Add ANTHROPIC_API_KEY (or OPENAI_* settings) to .env and restart the backend.",
            )
        await self._provider.check()

    async def aclose(self) -> None:
        if self._provider is not None:
            await self._provider.aclose()

    # -- diagnosis -----------------------------------------------------------

    async def diagnose(self, request: DiagnoseRequest) -> DiagnoseResult:
        if request.force_demo or self._provider is None:
            return await self._diagnose_demo(request)
        return await self._diagnose_ai(request, self._provider)

    async def _diagnose_demo(self, request: DiagnoseRequest) -> DiagnoseResult:
        started = time.perf_counter()
        if request.mode == AnalysisMode.VIDEO:
            raw = demo_reasoner.diagnose_video(
                personality=request.personality,
                frames=[(n, t, list(ctx.objects) if ctx else []) for n, t, ctx in request.video_frames],
                stats=[img.stats for img in request.images],
            )
        else:
            raw = demo_reasoner.diagnose_frame(
                mode=request.mode,
                personality=request.personality,
                context=request.context,
                stats=request.images[0].stats if request.images else None,
                active=request.active,
                active_keys=request.active_demo_keys,
            )
        diagnosis, warnings = parse_diagnosis(raw)
        if self.settings.demo_latency_ms > 0:
            await asyncio.sleep(self.settings.demo_latency_ms / 1000.0)
        return DiagnoseResult(
            diagnosis=diagnosis,
            provider="demo",
            model=DEMO_MODEL,
            simulated=True,
            latency_ms=int((time.perf_counter() - started) * 1000),
            warnings=warnings,
        )

    async def _diagnose_ai(self, request: DiagnoseRequest, provider: VisionProvider) -> DiagnoseResult:
        s = self.settings
        user_text = build_user_text(
            mode=request.mode,
            personality=request.personality,
            trigger=request.trigger,
            context=request.context,
            active=request.active,
            extra=request.extra_prompt,
        )
        images = [ProviderImage(jpeg=img.jpeg, label=img.label) for img in request.images]
        effort = s.ai_effort_live if request.mode == AnalysisMode.LIVE else s.ai_effort_deep
        timeout = s.ai_timeout_seconds if request.mode != AnalysisMode.LIVE else min(s.ai_timeout_seconds, 90.0)

        started = time.perf_counter()
        attempts = 1 + max(0, s.ai_malformed_retries)
        last_problem = "unknown"
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
                try:
                    data = extract_json(completion.text)
                    diagnosis, warnings = parse_diagnosis(data)
                except (ValueError, MalformedDiagnosisError) as exc:
                    last_problem = str(exc)
                    log.warning(
                        "Malformed AI output (attempt %d/%d, %d chars, stop=%s): %s",
                        attempt + 1,
                        attempts,
                        len(completion.text),
                        completion.stop_reason,
                        last_problem,
                    )
                    continue
                if completion.stop_reason in ("max_tokens", "length"):
                    warnings.append("The model hit its output limit; the diagnosis may be incomplete.")
                log.info(
                    "AI diagnosis ok: provider=%s model=%s mode=%s images=%d tokens_in=%s tokens_out=%s",
                    provider.name,
                    completion.model,
                    request.mode.value,
                    len(images),
                    completion.input_tokens,
                    completion.output_tokens,
                )
                return DiagnoseResult(
                    diagnosis=diagnosis,
                    provider=provider.name,
                    model=completion.model,
                    simulated=False,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    warnings=warnings,
                )
        raise AIMalformedError(
            "The vision model returned an answer that could not be validated.",
            hint="Try again. If it keeps happening, try a different model.",
        )


__all__ = ["AIService", "AIError", "DiagnoseRequest", "DiagnoseResult", "extract_json"]
