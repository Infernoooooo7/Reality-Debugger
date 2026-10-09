"""GET /api/health, GET /api/health/ai and GET /api/metrics."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from app import __version__
from app.schemas.diagnostics import AICheckResponse, HealthResponse, LocalStatus, MetricsResponse
from app.services.ai_providers import AIError
from app.services.ai_service import error_info
from app.services.local_diagnostics import ENGINE_VERSION
from app.services.scan_store import utcnow
from app.services.video_service import HAS_OPENCV

router = APIRouter(tags=["health"])

_AI_CHECK_TTL = 60.0


def local_status(request: Request) -> LocalStatus:
    state = request.app.state
    return LocalStatus(
        engine=ENGINE_VERSION,
        rules=state.engine.armed_rules(),
        labels=len(state.ontology.labels),
        attributes=state.ontology.attribute_names,
        detectors=state.ontology.models,
    )


@router.get("/health", response_model=HealthResponse, summary="Backend status, local engine, AI layer and limits")
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    ai = request.app.state.ai
    policy = request.app.state.policy
    return HealthResponse(
        status="ok",
        version=__version__,
        time=utcnow(),
        ai=ai.status,
        local=local_status(request),
        limits={
            "max_image_bytes": settings.max_image_bytes,
            "max_frame_bytes": settings.max_frame_bytes,
            "max_video_bytes": settings.max_video_bytes,
            "max_video_keyframes": settings.max_video_keyframes,
            "max_scene_chars": settings.max_scene_chars,
            "ai_calls_per_minute": policy.per_minute,
            "ai_cooldown_ms": int(policy.cooldown_ms),
        },
        features={
            "local_cv": True,
            "ai_reasoning": ai.enabled,
            "server_video_fallback": HAS_OPENCV,
        },
    )


@router.get(
    "/health/ai",
    response_model=AICheckResponse,
    summary="Verify the AI provider credentials and model (cached for 60 s)",
)
async def ai_check(request: Request) -> AICheckResponse:
    ai = request.app.state.ai
    status = ai.status
    if not ai.enabled:
        # Local-only mode is a valid configuration, not a failure: no provider call.
        return AICheckResponse(
            ok=False,
            provider=status.provider,
            model=None,
            latency_ms=0,
            error={"code": "AI_OFF", "message": "AI reasoning is off (local-only mode).", "hint": status.detail, "retryable": False},
        )
    cache = getattr(request.app.state, "ai_check_cache", None)
    if cache and time.monotonic() - cache[0] < _AI_CHECK_TTL:
        return cache[1]
    started = time.perf_counter()
    try:
        name = await ai.check()
        response = AICheckResponse(
            ok=True, provider=name, model=ai.status.model, latency_ms=int((time.perf_counter() - started) * 1000)
        )
    except AIError as exc:
        response = AICheckResponse(
            ok=False,
            provider=status.provider,
            model=status.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            error=error_info(exc),
        )
    request.app.state.ai_check_cache = (time.monotonic(), response)
    return response


@router.get("/metrics", response_model=MetricsResponse, summary="Developer metrics: local engine and AI usage counters")
async def metrics(request: Request) -> MetricsResponse:
    state = request.app.state
    snapshot = state.metrics.snapshot()
    ai_status = state.ai.status
    snapshot["ai"].update(
        {
            "provider": ai_status.provider,
            "model": ai_status.model,
            "state": ai_status.state,
            "paused_for_ms": max(0, int(state.policy.blocked_until_ms - time.monotonic() * 1000)),
        }
    )
    return MetricsResponse(uptime_s=state.metrics.uptime_s, sessions=len(state.scans), local=snapshot["local"], ai=snapshot["ai"])
