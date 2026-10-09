"""POST /api/analyze/frame (live frame + optional AI reasoning) and
POST /api/analyze/scene (one-shot local analysis of a scene model)."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile

from app.api.deps import get_pipeline, image_edge, load_image, parse_personality, parse_scene
from app.config import Settings
from app.schemas.common import AnalysisMode, ScanTrigger
from app.schemas.diagnostics import DiagnosticReport, ScanAnalysisResponse, SceneAnalysisRequest

router = APIRouter(tags=["analyze"])


@router.post(
    "/analyze/frame",
    response_model=ScanAnalysisResponse,
    summary="Live frame: local observation plus optional AI reasoning (cooldown, budget and de-duplication apply)",
)
async def analyze_frame(
    request: Request,
    image: UploadFile | None = File(None, description="The selected frame (JPG/PNG/WEBP); needed for AI reasoning."),
    scan_id: str | None = Form(None, description="Existing scan id; a new scan is created if missing/expired."),
    personality: str | None = Form(None),
    trigger: ScanTrigger = Form(ScanTrigger.MANUAL),
    scene: str | None = Form(None, description="JSON SceneModel measured on-device."),
    focus: str | None = Form(None, description="Finding id to explain (trigger user_explain)."),
) -> ScanAnalysisResponse:
    settings: Settings = request.app.state.settings
    pers = parse_personality(personality) if personality else None
    scene_model = parse_scene(scene, settings)
    prepared = None
    if image is not None and image.filename is not None:
        edge, quality = image_edge(request, "live")
        prepared = await load_image(image, limit=settings.max_frame_bytes, max_edge=edge, quality=quality, settings=settings, what="frame")
    session = request.app.state.scans.get_or_create(scan_id, pers)
    return await get_pipeline(request).live_frame(
        session, scene=scene_model, image=prepared, trigger=trigger, mode=AnalysisMode.LIVE, focus=focus
    )


@router.post(
    "/analyze/scene",
    response_model=DiagnosticReport,
    summary="Local-only diagnostic of a scene model (no image is uploaded)",
)
async def analyze_scene(request: Request, body: SceneAnalysisRequest) -> DiagnosticReport:
    return await get_pipeline(request).analyze_image(personality=body.personality, scene=body.scene, image=None)
