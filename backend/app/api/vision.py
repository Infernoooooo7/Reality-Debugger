"""Server-side vision: /api/vision/status, /models, /profiles, /detect and /compare.

Uploaded images are decoded in memory, analysed and dropped; nothing is
stored or logged. A model that cannot be loaded returns HTTP 503
MODEL_UNAVAILABLE with the real reason.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.errors import AppError, UnsupportedMediaError
from app.services.inference_service import VisionInputError, VisionService
from app.services.vision_service import decode_bgr, read_upload
from app.vision.registry import ModelUnavailable

router = APIRouter(prefix="/vision", tags=["vision"])


def _service(request: Request) -> VisionService:
    service: VisionService | None = getattr(request.app.state, "vision", None)
    if service is None:
        raise AppError("Server-side vision is turned off on this server.", code="VISION_OFF", status_code=503,
                       hint="Set VISION_BACKEND=true to enable /api/vision.")
    return service


async def _image(request: Request, upload: UploadFile, what: str, limit: int | None = None) -> Any:
    settings = request.app.state.settings
    declared = (upload.content_type or "").lower()
    if declared and not (declared.startswith("image/") or declared == "application/octet-stream"):
        raise UnsupportedMediaError(f"Expected an image but received '{declared}'.", hint="Use a JPG, PNG or WEBP image.")
    data = await read_upload(upload, limit or settings.max_image_bytes, what)
    return await run_in_threadpool(decode_bgr, data, max_pixels=settings.max_image_pixels)


async def _run(fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return await run_in_threadpool(fn, *args, **kwargs)
    except ModelUnavailable as exc:
        raise AppError(f"The model '{exc.model_id}' is not available on this server.", code="MODEL_UNAVAILABLE",
                       status_code=503, hint=exc.reason) from exc
    except VisionInputError as exc:
        raise AppError(str(exc), code="INVALID_VISION_REQUEST", status_code=422) from exc


@router.get("/status", summary="Server-side models, availability and memory budget")
async def status(request: Request) -> dict[str, Any]:
    service = _service(request)
    return {**service.status(), "profiles": {pid: {"name": p["name"], "status": p["status"]} for pid, p in service.profiles.items()}}


@router.get("/models", summary="The model registry: every researched model with its status, licence and limitations")
async def models(request: Request) -> list[dict[str, Any]]:
    return _service(request).registry_summary()


@router.get("/profiles", summary="Domain profiles: supported categories, engines, limitations and safety notices")
async def profiles(request: Request) -> dict[str, Any]:
    return _service(request).profiles


@router.post("/detect", summary="Server-side detection (standard or tiled precision mode) with category queries")
async def detect(
    request: Request,
    image: UploadFile = File(..., description="JPG, PNG or WEBP."),
    mode: str = Form("standard", description="standard (one pass) or precision (tiles + full image, for small objects)"),
    profile: str = Form("general"),
    queries: str = Form("", description="Comma-separated categories to look for, e.g. 'bottle, forklift'."),
) -> dict[str, Any]:
    service = _service(request)
    bgr = await _image(request, image, "image")
    asked = [q.strip() for q in queries.split(",") if q.strip()][:10]
    return await _run(service.detect, bgr, mode=mode, profile=profile, queries=asked)


@router.post("/compare", summary="Reference comparison: anomaly map of an image against known-good references")
async def compare(
    request: Request,
    image: UploadFile = File(..., description="The part to check."),
    references: list[UploadFile] = File(..., description="2-10 images of known-good parts, photographed the same way."),
) -> dict[str, Any]:
    service = _service(request)
    limit = request.app.state.settings.max_reference_bytes
    cap = int(service.config.get("inference.compare.maxReferences"))
    if len(references) > cap:
        raise AppError(f"At most {cap} reference images are accepted.", code="INVALID_VISION_REQUEST", status_code=422)
    refs = [await _image(request, r, f"reference image {i + 1}", limit) for i, r in enumerate(references)]
    query = await _image(request, image, "image")
    return await _run(service.compare, refs, query)
