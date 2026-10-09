"""Live-scan sessions: create, observe (local), deep scan, read, clear."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, Response, UploadFile

from app.api.deps import get_pipeline, image_edge, load_image, parse_personality, parse_scene
from app.config import Settings
from app.schemas.common import AnalysisMode, ScanTrigger
from app.schemas.diagnostics import CreateScanRequest, ObserveRequest, ScanAnalysisResponse, ScanState

router = APIRouter(tags=["scan"])


@router.post("/scan", response_model=ScanState, status_code=201, summary="Start a live-scan session")
async def create_scan(request: Request, body: CreateScanRequest | None = None) -> ScanState:
    session = request.app.state.scans.create((body or CreateScanRequest()).personality)
    return get_pipeline(request).diagnostics.scan_state(session)


@router.post(
    "/scan/observe",
    response_model=ScanAnalysisResponse,
    summary="Local observation: evaluate a scene model (no image, no AI call)",
)
async def observe(request: Request, body: ObserveRequest) -> ScanAnalysisResponse:
    session = request.app.state.scans.get_or_create(body.scan_id, body.personality)
    return await get_pipeline(request).observe(session, body.scene)


@router.post(
    "/scan/deep",
    response_model=ScanAnalysisResponse,
    summary="Deep Scan: local deep detection results plus optional AI reasoning on the frozen frame",
)
async def deep_scan(
    request: Request,
    image: UploadFile | None = File(None, description="The frozen frame (only needed when AI reasoning is on)."),
    scan_id: str | None = Form(None),
    personality: str | None = Form(None),
    scene: str | None = Form(None, description="JSON SceneModel measured on-device (fast + deep detector)."),
    trigger: ScanTrigger = Form(ScanTrigger.DEEP_SCAN),
) -> ScanAnalysisResponse:
    settings: Settings = request.app.state.settings
    pers = parse_personality(personality)
    scene_model = parse_scene(scene, settings)
    prepared = None
    if image is not None and image.filename is not None:
        edge, quality = image_edge(request, "deep")
        prepared = await load_image(image, limit=settings.max_frame_bytes, max_edge=edge, quality=quality, settings=settings, what="frame")
    session = request.app.state.scans.get_or_create(scan_id, pers)
    return await get_pipeline(request).live_frame(
        session,
        scene=scene_model,
        image=prepared,
        trigger=trigger if trigger in (ScanTrigger.DEEP_SCAN, ScanTrigger.FREEZE) else ScanTrigger.DEEP_SCAN,
        mode=AnalysisMode.DEEP,
    )


@router.get("/scan/{scan_id}", response_model=ScanState, summary="Current state of a live-scan session")
async def get_scan(request: Request, scan_id: str) -> ScanState:
    return get_pipeline(request).diagnostics.scan_state(request.app.state.scans.get(scan_id))


@router.delete("/scan/{scan_id}", status_code=204, summary="Clear a scan session (privacy)")
async def delete_scan(request: Request, scan_id: str) -> Response:
    request.app.state.scans.delete(scan_id)
    return Response(status_code=204)
