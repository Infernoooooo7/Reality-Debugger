"""FastAPI application entrypoint.

Run with:  uvicorn app.main:app --host 0.0.0.0 --port 8000
Interactive API docs: http://localhost:8000/api/docs
"""

from __future__ import annotations

import logging
import mimetypes
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import analyze, health, image, scan, video
from app.config import LAN_ORIGIN_REGEX, Settings, get_settings
from app.errors import register_exception_handlers
from app.middleware import AccessLogMiddleware, BodySizeLimitMiddleware
from app.runtime_config import load_config
from app.services.ai_policy import AIPolicy
from app.services.ai_providers import VisionProvider
from app.services.ai_service import AIService
from app.services.diagnostic_service import DiagnosticService
from app.services.local_diagnostics import ENGINE_VERSION, LocalDiagnosticEngine
from app.services.metrics import Metrics
from app.services.ontology import load_ontology
from app.services.pipeline import Pipeline
from app.services.scan_store import ScanStore

# Slim container images ship without /etc/mime.types; browsers need these to
# compile WebAssembly while streaming and to run module workers.
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".js")

# Anthropic/OpenAI style keys (sk-...) and Google API keys (AIza...).
_SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|sk-ant-[A-Za-z0-9_\-]{8,}|AIza[0-9A-Za-z_\-]{20,})")


class _RedactSecrets(logging.Filter):
    """Belt and braces: mask anything that looks like an API key in log output."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if _SECRET_RE.search(message):
            record.msg = _SECRET_RE.sub("***redacted***", message)
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
    for noisy in ("httpx", "httpx2", "anthropic", "google_genai", "google.genai", "python_multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def create_app(settings: Settings | None = None, *, providers: Mapping[str, VisionProvider] | None = None) -> FastAPI:
    """Build the app. ``providers`` injects AI providers by name (tests only)."""
    settings = settings or get_settings()
    _configure_logging(settings.log_level)
    log = logging.getLogger("reality.app")

    config = load_config(settings.config_dir)
    ontology = load_ontology(settings.config_dir)
    metrics = Metrics()
    engine = LocalDiagnosticEngine(config, ontology)
    diagnostics = DiagnosticService(config, ontology, engine, metrics)
    ai = AIService(settings, config, metrics, providers=providers)
    policy = AIPolicy(config, settings)
    pipeline = Pipeline(settings=settings, config=config, diagnostics=diagnostics, ai=ai, policy=policy, metrics=metrics)
    scans = ScanStore(ttl_seconds=settings.scan_ttl_seconds, max_scans=settings.max_scans)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        status = ai.status
        log.info("Reality Debugger API %s starting", __version__)
        log.info(
            "Local CV: %s, %d labels, %d/%d rules active",
            ENGINE_VERSION,
            len(ontology.labels),
            len(engine.armed_rules()),
            len(engine.rule_names),
        )
        log.info("AI reasoning: %s (%s) - %s", status.provider, status.model or "no model", status.detail)
        yield
        await ai.aclose()

    app = FastAPI(
        title="Reality Debugger API",
        version=__version__,
        description="Backend for Reality Debugger: a local-first diagnostic engine over scene models measured "
        "on-device, finding lifecycles, and an optional AI reasoning layer (Gemini or Claude).",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.state.config = config
    app.state.ontology = ontology
    app.state.engine = engine
    app.state.metrics = metrics
    app.state.diagnostics = diagnostics
    app.state.ai = ai
    app.state.policy = policy
    app.state.pipeline = pipeline
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

    dist = settings.frontend_dist
    if dist is not None and (dist / "index.html").is_file():
        # Single-service deployment: the built frontend shares this origin, so
        # /api needs no CORS and the camera works on the same HTTPS URL.
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
        log.info("Serving the frontend from %s", dist)
    else:

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
