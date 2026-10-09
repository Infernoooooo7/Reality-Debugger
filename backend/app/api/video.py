"""POST /api/analyze/video - Video Debug.

Two input forms:

* **Manifest (default, privacy-preserving):** ``manifest`` (JSON) with the
  scene model of every sampled frame, produced by the browser's detectors and
  tracker. The local engine replays it with video time as the clock. When AI
  reasoning is on, the browser also uploads a few keyframes (``frames``).
* **Whole video (fallback):** ``video`` - only used when the browser cannot
  decode the file. It is streamed to a temporary file, sampled with OpenCV and
  deleted immediately. No object detector runs on the server, so local
  diagnostics cover frame signals (lighting, sharpness) and scene changes.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import PurePath

from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_pipeline, image_edge, load_image, parse_personality
from app.config import Settings
from app.errors import AppError, InvalidUploadError, PayloadTooLargeError, UnsupportedMediaError
from app.schemas.analysis import VideoManifest
from app.schemas.common import AnalysisMode, Personality, ScanTrigger, Severity
from app.schemas.diagnostics import Finding, TimelineEvent, VideoMeta, VideoReport, VideoSamplingStats
from app.services.ai_service import DiagnoseRequest
from app.services.pipeline import Pipeline, Reasoning
from app.services.video_service import (
    KeyframeInfo,
    LocalVideoResult,
    VideoParams,
    ai_video_findings,
    analyze_scenes,
    finish_timeline,
    fmt_time,
    keyframe_label,
    keyframe_models,
    local_timeline_summary,
    manifest_scenes,
    sample_video_file,
    scene_change_events,
)
from app.services.vision_service import PreparedImage, prepare_image

router = APIRouter(tags=["analyze"])

_VIDEO_TYPES_PREFIX = ("video/",)
SERVER_FALLBACK_NOTE = (
    "This video was decoded on the server because the browser could not play it. No object detector runs on the "
    "server, so the local diagnostics cover lighting, sharpness and scene changes only."
)


def _safe_name(name: str | None) -> str | None:
    """Keep only the file name (no client directory structure), clipped."""
    if not name:
        return None
    base = name.replace("\\", "/").rsplit("/", 1)[-1].strip()
    return base[:120] or None


def _keyframe_hints(manifest: VideoManifest, keyframes: list[KeyframeInfo], order: list[int]) -> str:
    lines = [f"On-device detections per keyframe ({manifest.detector or 'browser models'}):"]
    for info, idx in zip(keyframes, order, strict=True):
        frame = manifest.frames[idx]
        labels: dict[str, int] = {}
        for obj in frame.objects:
            labels[obj.label] = labels.get(obj.label, 0) + 1
        summary = ", ".join(f"{n}x {label}" for label, n in sorted(labels.items(), key=lambda kv: -kv[1])) or "nothing recognised"
        lines.append(f"- Frame {info.number} ({fmt_time(info.t)}): {summary}")
    return "\n".join(lines)


@router.post("/analyze/video", response_model=VideoReport, summary="Chronological diagnostic of a video")
async def analyze_video(
    request: Request,
    frames: list[UploadFile] | None = File(None, description="Keyframes (JPEG); only needed for AI reasoning."),
    manifest: str | None = Form(None, description="JSON VideoManifest: sampled scene models and keyframes."),
    video: UploadFile | None = File(None, description="Fallback: the whole video, sampled on the server."),
    personality: str | None = Form(None),
) -> VideoReport:
    settings: Settings = request.app.state.settings
    pers = parse_personality(personality)
    if video is not None and video.filename:
        return await _server_side(request, video, pers, settings)
    if not manifest:
        raise AppError(
            "No video manifest or video was uploaded.", code="MISSING_FILE", status_code=422, hint="Select a video again."
        )
    return await _from_manifest(request, frames or [], manifest, pers, settings)


async def _reason(
    pipeline: Pipeline,
    *,
    personality: Personality,
    images: list[PreparedImage],
    local: LocalVideoResult,
    extra: str,
) -> Reasoning:
    if not images:
        if not pipeline.ai.enabled:
            return Reasoning(None, pipeline.off_run(ScanTrigger.MANUAL))
        return Reasoning(None, pipeline._run("skipped", trigger=ScanTrigger.MANUAL, reason="No keyframes were uploaded."))
    request = DiagnoseRequest(
        mode=AnalysisMode.VIDEO,
        personality=personality,
        images=images,
        trigger=ScanTrigger.MANUAL,
        local_findings=pipeline.diagnostics.stateless_briefs(local.findings, local.object_ids),
        extra_prompt=extra,
    )
    return await pipeline.reason(pipeline.policy.stateless, request, ahash=None)


def _report(
    pipeline: Pipeline,
    *,
    personality: Personality,
    local: LocalVideoResult,
    reasoning: Reasoning,
    keyframes: list[KeyframeInfo],
    local_events: list[TimelineEvent],
    meta: VideoMeta,
    sampling: VideoSamplingStats,
    started: float,
    warnings: list[str],
) -> VideoReport:
    ai_findings: list[Finding] = []
    timeline = list(local.timeline) + local_events
    if reasoning.result is not None:
        ai_findings, ai_timeline = ai_video_findings(
            reasoning.result.diagnosis, local=local.findings, keyframes=keyframes, diagnostics=pipeline.diagnostics
        )
        timeline += ai_timeline
    report = pipeline.diagnostics.report(
        mode=AnalysisMode.VIDEO,
        personality=personality,
        scene=local.representative,
        findings=local.findings + ai_findings,
        ai_run=reasoning.run,
        latency_ms=int((time.perf_counter() - started) * 1000),
        local=local.local,
        ai=reasoning.result,
        warnings=warnings,
    )
    return VideoReport(
        report=report,
        timeline=finish_timeline(timeline),
        video=meta,
        sampling=sampling,
        keyframes=keyframe_models(keyframes),
    )


async def _from_manifest(
    request: Request, frames: list[UploadFile], manifest_raw: str, personality: Personality, settings: Settings
) -> VideoReport:
    started = time.perf_counter()
    pipeline = get_pipeline(request)
    if len(manifest_raw) > settings.max_scene_chars * 16:
        raise AppError("The video manifest is too large.", code="INVALID_MANIFEST", status_code=413)
    try:
        manifest = VideoManifest.model_validate_json(manifest_raw)
    except ValidationError as exc:
        raise AppError(
            "The video manifest is malformed.",
            code="INVALID_MANIFEST",
            status_code=422,
            hint=str(exc.errors()[0].get("msg")) if exc.errors() else None,
        ) from exc
    if len(frames) > settings.max_video_keyframes:
        raise AppError(
            f"Too many keyframes ({len(frames)}); the limit is {settings.max_video_keyframes}.",
            code="TOO_MANY_FRAMES",
            status_code=422,
        )
    if frames and len(frames) != len(manifest.frames):
        raise AppError("The number of keyframes does not match the manifest.", code="INVALID_MANIFEST", status_code=422)

    order = sorted(range(len(manifest.frames)), key=lambda i: manifest.frames[i].t)
    keyframes = [
        KeyframeInfo(number=n, t=round(manifest.frames[i].t, 2), scene=manifest.frames[i].scene, reason=manifest.frames[i].reason)
        for n, i in enumerate(order, start=1)
    ]
    images: list[PreparedImage] = []
    if frames and pipeline.ai.enabled:
        edge, quality = image_edge(request, "video")
        for info, idx in zip(keyframes, order, strict=True):
            prepared = await load_image(
                frames[idx], limit=settings.max_frame_bytes, max_edge=edge, quality=quality, settings=settings, what="keyframe"
            )
            prepared.label = keyframe_label(info, len(keyframes), manifest.duration_s)
            images.append(prepared)

    scenes = manifest_scenes(manifest)
    local = await run_in_threadpool(analyze_scenes, pipeline.diagnostics, scenes, personality)
    local_events = [
        TimelineEvent(t=round(e.t, 2), frame_index=None, kind=e.kind, severity=Severity.INFO, finding_id=None, text=e.text, source="local")
        for e in manifest.events
    ]
    extra = "\n\n".join(
        part for part in (_keyframe_hints(manifest, keyframes, order) if manifest.frames else "", local_timeline_summary(local)) if part
    )
    reasoning = await _reason(pipeline, personality=personality, images=images, local=local, extra=extra)

    track_ids = {o.id for s in scenes for o in s.objects}
    meta = VideoMeta(
        duration_s=round(manifest.duration_s, 2),
        width=manifest.width,
        height=manifest.height,
        name=_safe_name(manifest.name),
        size_bytes=manifest.size_bytes,
    )
    sampling = VideoSamplingStats(
        processed_on="browser",
        sampled_frames=max(manifest.sampled_frames, len(scenes)),
        scenes=len(manifest.scenes) or len({s.view_id for s in scenes}),
        redundant_removed=manifest.redundant_removed,
        keyframes=len(keyframes),
        tracked_objects=len(manifest.tracks) or len(track_ids),
    )
    return _report(
        pipeline,
        personality=personality,
        local=local,
        reasoning=reasoning,
        keyframes=keyframes,
        local_events=local_events,
        meta=meta,
        sampling=sampling,
        started=started,
        warnings=[],
    )


async def _server_side(request: Request, video: UploadFile, personality: Personality, settings: Settings) -> VideoReport:
    started = time.perf_counter()
    pipeline = get_pipeline(request)
    declared = (video.content_type or "").lower()
    if declared and not (declared.startswith(_VIDEO_TYPES_PREFIX) or declared == "application/octet-stream"):
        raise UnsupportedMediaError(f"Expected a video but received '{declared}'.", hint="Use MP4, MOV or WEBM.")

    edge, quality = image_edge(request, "video")
    params = VideoParams.from_config(request.app.state.config)
    max_keyframes = min(int(request.app.state.config.get("temporal.video.maxKeyframes")), settings.max_video_keyframes)
    suffix = PurePath(video.filename or "upload").suffix[:8] or ".bin"
    fd, path = tempfile.mkstemp(prefix="rd_video_", suffix=suffix)
    total = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await video.read(1 << 20):
                total += len(chunk)
                if total > settings.max_video_bytes:
                    raise PayloadTooLargeError(
                        f"The video is larger than {settings.max_video_bytes // (1024 * 1024)} MB.",
                        hint="Trim the video or let the browser sample it locally.",
                    )
                out.write(chunk)
        if total == 0:
            raise InvalidUploadError("The video is empty.", hint="Pick the file again.")
        sampling = await run_in_threadpool(sample_video_file, path, max_keyframes=max_keyframes, edge=edge, params=params)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    keyframes: list[KeyframeInfo] = []
    images: list[PreparedImage] = []
    for number, (t, scene, reason, jpeg) in enumerate(sampling.keyframes, start=1):
        info = KeyframeInfo(number=number, t=round(t, 2), scene=scene, reason=reason)
        keyframes.append(info)
        if pipeline.ai.enabled:
            prepared = await run_in_threadpool(
                prepare_image, jpeg, max_edge=edge, max_pixels=settings.max_image_pixels, quality=quality
            )
            prepared.label = keyframe_label(info, len(sampling.keyframes), sampling.meta.duration_s)
            images.append(prepared)

    local = await run_in_threadpool(analyze_scenes, pipeline.diagnostics, sampling.scenes, personality)
    reasoning = await _reason(pipeline, personality=personality, images=images, local=local, extra=local_timeline_summary(local))
    meta = sampling.meta.model_copy(update={"name": _safe_name(video.filename), "size_bytes": total})
    return _report(
        pipeline,
        personality=personality,
        local=local,
        reasoning=reasoning,
        keyframes=keyframes,
        local_events=scene_change_events(sampling.scene_changes),
        meta=meta,
        sampling=sampling.stats,
        started=started,
        warnings=[SERVER_FALLBACK_NOTE],
    )
