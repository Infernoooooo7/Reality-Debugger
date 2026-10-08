"""Live-scan sessions: create, deep scan, read, clear."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, Response, UploadFile

from app.api.deps import load_image, parse_context, parse_flag, parse_personality
from app.config import Settings
from app.schemas.analysis import LocalContext
from app.schemas.common import AnalysisMode, ScanTrigger
from app.schemas.diagnostics import CreateScanRequest, ScanAnalysisResponse, ScanState
from app.services.ai_service import AIService, DiagnoseRequest
from app.services.diagnostic_service import active_briefs, build_report, merge_into_scan, scan_state
from app.services.scan_store import ScanSession
from app.services.vision_service import PreparedImage

router = APIRouter(tags=["scan"])


async def run_scan_analysis(
    *,
    session: ScanSession,
    image: PreparedImage,
    context: LocalContext | None,
    trigger: ScanTrigger,
    mode: AnalysisMode,
    ai: AIService,
    settings: Settings,
    force_demo: bool,
) -> ScanAnalysisResponse:
    async with session.lock:
        if not (force_demo or ai.demo_only):
            session.check_rate(settings.scan_ai_calls_per_minute)
        briefs, demo_keys = active_briefs(session)
        result = await ai.diagnose(
            DiagnoseRequest(
                mode=mode,
                personality=session.personality,
                images=[image],
                context=context,
                trigger=trigger,
                active=briefs,
                active_demo_keys=demo_keys,
                force_demo=force_demo,
            )
        )
        findings, events = merge_into_scan(session, result, mode=mode)
        state = scan_state(session)
        report = build_report(
            result,
            mode=mode,
            personality=session.personality,
            findings=findings,
            trigger=trigger,
            image=image,
        )
        return ScanAnalysisResponse(scan=state, report=report, events=events)


@router.post("/scan", response_model=ScanState, status_code=201, summary="Start a live-scan session")
async def create_scan(request: Request, body: CreateScanRequest | None = None) -> ScanState:
    session = request.app.state.scans.create((body or CreateScanRequest()).personality)
    return scan_state(session)


@router.post(
    "/scan/deep",
    response_model=ScanAnalysisResponse,
    summary="Deep Scan: thorough analysis of a frozen live frame",
)
async def deep_scan(
    request: Request,
    image: UploadFile = File(..., description="The frozen frame (JPG/PNG/WEBP)."),
    scan_id: str | None = Form(None),
    personality: str | None = Form(None),
    context: str | None = Form(None, description="JSON LocalContext from the on-device engine."),
    trigger: ScanTrigger = Form(ScanTrigger.DEEP_SCAN),
    demo: str | None = Form(None, description="'true' forces DEMO MODE for this request."),
) -> ScanAnalysisResponse:
    settings: Settings = request.app.state.settings
    pers = parse_personality(personality)
    ctx = parse_context(context, settings)
    prepared = await load_image(
        image, limit=settings.max_frame_bytes, max_edge=settings.ai_image_max_edge_deep, settings=settings, what="frame"
    )
    session = request.app.state.scans.get_or_create(scan_id, pers)
    return await run_scan_analysis(
        session=session,
        image=prepared,
        context=ctx,
        trigger=trigger if trigger in (ScanTrigger.DEEP_SCAN, ScanTrigger.FREEZE) else ScanTrigger.DEEP_SCAN,
        mode=AnalysisMode.DEEP,
        ai=request.app.state.ai,
        settings=settings,
        force_demo=parse_flag(demo),
    )


@router.get("/scan/{scan_id}", response_model=ScanState, summary="Current state of a live-scan session")
async def get_scan(request: Request, scan_id: str) -> ScanState:
    return scan_state(request.app.state.scans.get(scan_id))


@router.delete("/scan/{scan_id}", status_code=204, summary="Clear a scan session (privacy)")
async def delete_scan(request: Request, scan_id: str) -> Response:
    request.app.state.scans.delete(scan_id)
    return Response(status_code=204)
