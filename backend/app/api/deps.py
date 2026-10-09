"""Shared request helpers for the API routers."""

from __future__ import annotations

from fastapi import Request, UploadFile
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.errors import AppError, UnsupportedMediaError
from app.schemas.common import Personality
from app.schemas.scene import SceneModel
from app.services.pipeline import Pipeline
from app.services.vision_service import PreparedImage, prepare_image, read_upload


def get_pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


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


def parse_scene(raw: str | None, settings: Settings) -> SceneModel | None:
    """Validate the JSON scene model sent as a multipart form field."""
    if raw is None or not raw.strip():
        return None
    if len(raw) > settings.max_scene_chars:
        raise AppError("The scene model is too large.", code="INVALID_SCENE", status_code=413)
    try:
        return SceneModel.model_validate_json(raw)
    except ValidationError as exc:
        raise AppError(
            "The scene model is malformed.",
            code="INVALID_SCENE",
            hint=str(exc.errors()[0].get("msg")) if exc.errors() else None,
            status_code=422,
        ) from exc


def image_edge(request: Request, kind: str) -> tuple[int, int]:
    """(max edge, JPEG quality) for images forwarded to the AI layer (config/ai.json)."""
    config = request.app.state.config
    return int(config.get(f"ai.images.{kind}MaxEdge")), int(config.get("ai.images.jpegQuality"))


async def load_image(
    upload: UploadFile | None, *, limit: int, max_edge: int, quality: int, settings: Settings, what: str = "image"
) -> PreparedImage:
    if upload is None:
        raise AppError(f"No {what} was uploaded.", code="MISSING_FILE", status_code=422, hint="Attach an image file.")
    declared = (upload.content_type or "").lower()
    if declared and not (declared.startswith("image/") or declared == "application/octet-stream"):
        raise UnsupportedMediaError(f"Expected an image but received '{declared}'.", hint="Use a JPG, PNG or WEBP image.")
    data = await read_upload(upload, limit, what)
    return await run_in_threadpool(
        prepare_image, data, max_edge=max_edge, max_pixels=settings.max_image_pixels, quality=quality
    )
