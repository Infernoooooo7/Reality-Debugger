"""POST /api/analyze/frame - a live-scan frame chosen by the on-device engine."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile

from app.api.deps import load_image, parse_context, parse_flag, parse_personality
from app.api.scan import run_scan_analysis
from app.config import Settings
from app.schemas.common import AnalysisMode, ScanTrigger
from app.schemas.diagnostics import ScanAnalysisResponse

router = APIRouter(tags=["analyze"])


@router.post(
    "/analyze/frame",
    response_model=ScanAnalysisResponse,
    summary="Analyse one live frame and update the scan's finding lifecycle",
)
async def analyze_frame(
    request: Request,
    image: UploadFile = File(..., description="The selected frame (JPG/PNG/WEBP)."),
    scan_id: str | None = Form(None, description="Existing scan id; a new scan is created if missing/expired."),
    personality: str | None = Form(None),
    trigger: ScanTrigger = Form(ScanTrigger.MANUAL),
    context: str | None = Form(None, description="JSON LocalContext from the on-device engine."),
    demo: str | None = Form(None, description="'true' forces DEMO MODE for this request."),
) -> ScanAnalysisResponse:
    settings: Settings = request.app.state.settings
    pers = parse_personality(personality)
    ctx = parse_context(context, settings)
    prepared = await load_image(
        image, limit=settings.max_frame_bytes, max_edge=settings.ai_image_max_edge_live, settings=settings, what="frame"
    )
    session = request.app.state.scans.get_or_create(scan_id, pers)
    return await run_scan_analysis(
        session=session,
        image=prepared,
        context=ctx,
        trigger=trigger,
        mode=AnalysisMode.LIVE,
        ai=request.app.state.ai,
        settings=settings,
        force_demo=parse_flag(demo),
    )
