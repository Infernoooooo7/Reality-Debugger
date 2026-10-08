"""FastAPI application entrypoint.

Run with:  uvicorn app.main:app --host 0.0.0.0 --port 8000
Interactive API docs: http://localhost:8000/api/docs
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import analyze, health, image, scan, video
from app.config import LAN_ORIGIN_REGEX, Settings, get_settings
from app.errors import register_exception_handlers
from app.middleware import AccessLogMiddleware, BodySizeLimitMiddleware
from app.services.ai_service import AIService
from app.services.scan_store import ScanStore

_SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|sk-ant-[A-Za-z0-9_\-]{8,})")


class _RedactSecrets(logging.Filter):
    """Belt and braces: mask anything that looks like an API key in log output."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if _SECRET_RE.search(message):
            record.msg = _SECRET_RE.sub("sk-***redacted***", message)
            record.args = ()
        return True


def _configure_logging(level: str) -> None:
    root = logging.getLogger()
    if not any(getattr(h, "_reality", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
        handler.addFilter(_RedactSecrets())
        handler._reality = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    logging.getLogger("reality").setLevel(level.upper())
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    # The SDK/httpx can log full request URLs at INFO; keep them quiet.
    for noisy in ("httpx", "httpx2", "anthropic", "python_multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings.log_level)
    log = logging.getLogger("reality.app")

    ai = AIService(settings)
    scans = ScanStore(ttl_seconds=settings.scan_ttl_seconds, max_scans=settings.max_scans)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        status = ai.status
        log.info("Reality Debugger API %s starting", __version__)
        log.info("AI provider: %s (%s) - %s", status.provider, status.model or "no model", status.detail)
        yield
        await ai.aclose()

    app = FastAPI(
        title="Reality Debugger API",
        version=__version__,
        description="Backend for Reality Debugger: validates uploads, calls the vision model, "
        "validates its structured output and maintains live finding lifecycles.",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.state.ai = ai
    app.state.scans = scans

    register_exception_handlers(app)

    # Order matters: the last middleware added is the outermost one, so CORS
    # headers are present even on 413 responses from the body limiter.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=LAN_ORIGIN_REGEX if settings.cors_allow_lan else None,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        max_age=600,
    )

    for router in (health.router, analyze.router, image.router, video.router, scan.router):
        app.include_router(router, prefix="/api")

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "name": "Reality Debugger API",
            "docs": "/api/docs",
            "health": "/api/health",
            "frontend": "Run the frontend (npm run dev) and open http://localhost:5173",
        }

    return app


app = create_app()
