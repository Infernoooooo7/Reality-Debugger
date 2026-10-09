"""POST /api/analyze/image - Image Debug (local diagnostics + optional AI reasoning)."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile

from app.api.deps import get_pipeline, image_edge, load_image, parse_personality, parse_scene
from app.config import Settings
from app.schemas.diagnostics import DiagnosticReport

router = APIRouter(tags=["analyze"])


@router.post("/analyze/image", response_model=DiagnosticReport, summary="Full diagnostic of a single image")
async def analyze_image(
    request: Request,
    image: UploadFile = File(..., description="JPG, PNG or WEBP."),
    personality: str | None = Form(None),
    scene: str | None = Form(None, description="JSON SceneModel measured on-device for this image."),
) -> DiagnosticReport:
    settings: Settings = request.app.state.settings
    pers = parse_personality(personality)
    scene_model = parse_scene(scene, settings)
    edge, quality = image_edge(request, "deep")
    prepared = await load_image(image, limit=settings.max_image_bytes, max_edge=edge, quality=quality, settings=settings)
    return await get_pipeline(request).analyze_image(personality=pers, scene=scene_model, image=prepared)
