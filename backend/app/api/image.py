"""POST /api/analyze/image - Image Debug."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile

from app.api.deps import load_image, parse_context, parse_flag, parse_personality
from app.config import Settings
from app.schemas.common import AnalysisMode
from app.schemas.diagnostics import DiagnosticReport
from app.services.ai_service import DiagnoseRequest
from app.services.diagnostic_service import build_report, stateless_findings
from app.services.scan_store import utcnow

router = APIRouter(tags=["analyze"])


@router.post("/analyze/image", response_model=DiagnosticReport, summary="Full diagnostic of a single image")
async def analyze_image(
    request: Request,
    image: UploadFile = File(..., description="JPG, PNG or WEBP."),
    personality: str | None = Form(None),
    context: str | None = Form(None, description="JSON LocalContext from the on-device engine."),
    demo: str | None = Form(None, description="'true' forces DEMO MODE for this request."),
) -> DiagnosticReport:
    settings: Settings = request.app.state.settings
    pers = parse_personality(personality)
    ctx = parse_context(context, settings)
    prepared = await load_image(
        image, limit=settings.max_image_bytes, max_edge=settings.ai_image_max_edge_deep, settings=settings
    )
    result = await request.app.state.ai.diagnose(
        DiagnoseRequest(
            mode=AnalysisMode.IMAGE,
            personality=pers,
            images=[prepared],
            context=ctx,
            force_demo=parse_flag(demo),
        )
    )
    findings, _ = stateless_findings(result.diagnosis, simulated=result.simulated, now=utcnow())
    return build_report(result, mode=AnalysisMode.IMAGE, personality=pers, findings=findings, image=prepared)
