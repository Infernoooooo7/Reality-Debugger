"""Shared request helpers for the API routers."""

from __future__ import annotations

from fastapi import Request, UploadFile
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.errors import AppError, UnsupportedMediaError
from app.schemas.analysis import LocalContext
from app.schemas.common import Personality
from app.services.ai_service import AIService
from app.services.scan_store import ScanStore
from app.services.vision_service import PreparedImage, prepare_image, read_upload


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_ai(request: Request) -> AIService:
    return request.app.state.ai


def get_scans(request: Request) -> ScanStore:
    return request.app.state.scans


def parse_personality(value: str | None) -> Personality:
    if not value:
        return Personality.SERIOUS
    try:
        return Personality(value.strip().lower())
    except ValueError as exc:
        raise AppError(
            f"Unknown personality '{value}'.",
            code="INVALID_PERSONALITY",
            hint="Use serious, brutal or unhinged.",
            status_code=422,
        ) from exc


def parse_flag(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def parse_context(raw: str | None, settings: Settings) -> LocalContext | None:
    if raw is None or not raw.strip():
        return None
    if len(raw) > settings.max_context_chars:
        raise AppError("The local context payload is too large.", code="INVALID_CONTEXT", status_code=413)
    try:
        return LocalContext.model_validate_json(raw)
    except ValidationError as exc:
        raise AppError(
            "The local context payload is malformed.",
            code="INVALID_CONTEXT",
            hint=str(exc.errors()[0].get("msg")) if exc.errors() else None,
            status_code=422,
        ) from exc


async def load_image(upload: UploadFile | None, *, limit: int, max_edge: int, settings: Settings, what: str = "image") -> PreparedImage:
    if upload is None:
        raise AppError(f"No {what} was uploaded.", code="MISSING_FILE", status_code=422, hint="Attach an image file.")
    declared = (upload.content_type or "").lower()
    if declared and not (declared.startswith("image/") or declared == "application/octet-stream"):
        raise UnsupportedMediaError(f"Expected an image but received '{declared}'.", hint="Use a JPG, PNG or WEBP image.")
    data = await read_upload(upload, limit, what)
    return await run_in_threadpool(prepare_image, data, max_edge=max_edge, max_pixels=settings.max_image_pixels)
