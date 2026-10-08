"""GET /api/health and GET /api/health/ai."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from app import __version__
from app.schemas.diagnostics import AICheckResponse, HealthResponse
from app.services.ai_providers import AIError
from app.services.scan_store import utcnow
from app.services.video_service import HAS_OPENCV

router = APIRouter(tags=["health"])

_AI_CHECK_TTL = 60.0


@router.get("/health", response_model=HealthResponse, summary="Backend status, AI provider and limits")
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    ai = request.app.state.ai
    return HealthResponse(
        status="ok",
        version=__version__,
        time=utcnow(),
        ai=ai.status,
        limits={
            "max_image_bytes": settings.max_image_bytes,
            "max_frame_bytes": settings.max_frame_bytes,
            "max_video_bytes": settings.max_video_bytes,
            "max_video_keyframes": settings.max_video_keyframes,
            "scan_ai_calls_per_minute": settings.scan_ai_calls_per_minute,
        },
        features={
            "live_ai": not ai.demo_only,
            "demo_mode": True,
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
    cache = getattr(request.app.state, "ai_check_cache", None)
    if cache and time.monotonic() - cache[0] < _AI_CHECK_TTL:
        return cache[1]

    started = time.perf_counter()
    status = ai.status
    try:
        await ai.check()
        response = AICheckResponse(
            ok=True,
            provider=status.provider,
            model=status.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            simulated=False,
        )
    except AIError as exc:
        response = AICheckResponse(
            ok=False,
            provider=status.provider,
            model=status.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            simulated=status.simulated,
            error=exc.to_body()["error"],
        )
    request.app.state.ai_check_cache = (time.monotonic(), response)
    return response
