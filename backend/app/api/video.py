"""POST /api/analyze/video - Video Debug.

Two input forms:

* **Keyframes (default, privacy-preserving):** ``frames`` (several images) +
  ``manifest`` (JSON) produced by the browser's local sampling pipeline.
* **Whole video (fallback):** ``video`` - only used when the browser cannot
  decode the file. It is streamed to a temporary file, sampled with OpenCV and
  deleted immediately.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import PurePath

from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.api.deps import load_image, parse_flag, parse_personality
from app.config import Settings
from app.errors import AppError, InvalidUploadError, PayloadTooLargeError, UnsupportedMediaError
from app.schemas.analysis import LocalContext, VideoManifest
from app.schemas.common import AnalysisMode, Personality, Severity
from app.schemas.diagnostics import TimelineEvent, VideoMeta, VideoReport, VideoSamplingStats
from app.services.ai_service import DiagnoseRequest
from app.services.video_service import (
    KeyframeInfo,
    build_video_report,
    fmt_time,
    keyframe_label,
    sample_video_file,
    scene_change_events,
)
from app.services.vision_service import prepare_image

router = APIRouter(tags=["analyze"])

_VIDEO_TYPES_PREFIX = ("video/",)


def _safe_name(name: str | None) -> str | None:
    """Keep only the file name (no client directory structure), clipped."""
    if not name:
        return None
    base = name.replace("\\", "/").rsplit("/", 1)[-1].strip()
    return base[:120] or None


def _hints(manifest: VideoManifest, order: list[int]) -> str:
    lines = [
        f"On-device detector ({manifest.detector or 'browser model'}) hints per keyframe. They may be incomplete or wrong:"
    ]
    for number, idx in enumerate(order, start=1):
        frame = manifest.frames[idx]
        labels: dict[str, int] = {}
        for obj in frame.objects:
            labels[obj.label] = labels.get(obj.label, 0) + 1
        summary = ", ".join(f"{n}x {label}" for label, n in sorted(labels.items(), key=lambda kv: -kv[1])) or "nothing recognised"
        lines.append(f"- Frame {number} ({fmt_time(frame.t)}): {summary}")
    return "\n".join(lines)


@router.post("/analyze/video", response_model=VideoReport, summary="Chronological diagnostic of a video")
async def analyze_video(
    request: Request,
    frames: list[UploadFile] | None = File(None, description="Keyframes selected by the browser (JPEG)."),
    manifest: str | None = Form(None, description="JSON VideoManifest describing the keyframes."),
    video: UploadFile | None = File(None, description="Fallback: the whole video, sampled on the server."),
    personality: str | None = Form(None),
    demo: str | None = Form(None, description="'true' forces DEMO MODE for this request."),
) -> VideoReport:
    settings: Settings = request.app.state.settings
    pers = parse_personality(personality)
    force_demo = parse_flag(demo)
    if video is not None and video.filename:
        return await _server_side(request, video, pers, force_demo, settings)
    if not frames:
        raise AppError(
            "No keyframes or video were uploaded.",
            code="MISSING_FILE",
            status_code=422,
            hint="Select a video again.",
        )
    return await _keyframes(request, frames, manifest, pers, force_demo, settings)


async def _keyframes(
    request: Request,
    frames: list[UploadFile],
    manifest_raw: str | None,
    personality: Personality,
    force_demo: bool,
    settings: Settings,
) -> VideoReport:
    if not manifest_raw:
        raise AppError("The keyframe manifest is missing.", code="INVALID_MANIFEST", status_code=422)
    if len(manifest_raw) > settings.max_context_chars * 4:
        raise AppError("The keyframe manifest is too large.", code="INVALID_MANIFEST", status_code=413)
    try:
        manifest = VideoManifest.model_validate_json(manifest_raw)
    except ValidationError as exc:
        raise AppError(
            "The keyframe manifest is malformed.",
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
    if len(frames) != len(manifest.frames):
        raise AppError(
            "The number of keyframes does not match the manifest.", code="INVALID_MANIFEST", status_code=422
        )

    order = sorted(range(len(frames)), key=lambda i: manifest.frames[i].t)
    images = []
    keyframes: list[KeyframeInfo] = []
    video_frames: list[tuple[int, float, LocalContext | None]] = []
    for number, idx in enumerate(order, start=1):
        entry = manifest.frames[idx]
        prepared = await load_image(
            frames[idx],
            limit=settings.max_frame_bytes,
            max_edge=settings.ai_image_max_edge_video,
            settings=settings,
            what="keyframe",
        )
        info = KeyframeInfo(number=number, t=round(entry.t, 2), scene=entry.scene, reason=entry.reason)
        prepared.label = keyframe_label(info, len(order), manifest.duration_s)
        images.append(prepared)
        keyframes.append(info)
        video_frames.append(
            (number, info.t, LocalContext(detector=manifest.detector, objects=entry.objects, sharpness=entry.sharpness))
        )

    meta = VideoMeta(
        duration_s=round(manifest.duration_s, 2),
        width=manifest.width,
        height=manifest.height,
        name=_safe_name(manifest.name),
        size_bytes=manifest.size_bytes,
    )
    sampling = VideoSamplingStats(
        processed_on="browser",
        sampled_frames=max(manifest.sampled_frames, len(frames)),
        scenes=len(manifest.scenes) or len({f.scene for f in manifest.frames}),
        redundant_removed=manifest.redundant_removed,
        keyframes=len(frames),
    )
    local_events = [
        TimelineEvent(
            t=round(e.t, 2), frame_index=None, kind=e.kind, severity=Severity.INFO, finding_id=None, text=e.text, source="local"
        )
        for e in manifest.events
    ]
    result = await request.app.state.ai.diagnose(
        DiagnoseRequest(
            mode=AnalysisMode.VIDEO,
            personality=personality,
            images=images,
            video_frames=video_frames,
            extra_prompt=_hints(manifest, order),
            force_demo=force_demo,
        )
    )
    return build_video_report(
        result, personality=personality, keyframes=keyframes, meta=meta, sampling=sampling, local_events=local_events
    )


async def _server_side(
    request: Request, video: UploadFile, personality: Personality, force_demo: bool, settings: Settings
) -> VideoReport:
    declared = (video.content_type or "").lower()
    if declared and not (declared.startswith(_VIDEO_TYPES_PREFIX) or declared == "application/octet-stream"):
        raise UnsupportedMediaError(f"Expected a video but received '{declared}'.", hint="Use MP4, MOV or WEBM.")

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
        sampling = await run_in_threadpool(
            sample_video_file,
            path,
            max_keyframes=min(8, settings.max_video_keyframes),
            edge=settings.ai_image_max_edge_video,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    images = []
    keyframes: list[KeyframeInfo] = []
    for number, (t, scene, reason, jpeg) in enumerate(sampling.keyframes, start=1):
        prepared = await run_in_threadpool(
            prepare_image, jpeg, max_edge=settings.ai_image_max_edge_video, max_pixels=settings.max_image_pixels
        )
        info = KeyframeInfo(number=number, t=round(t, 2), scene=scene, reason=reason)
        prepared.label = keyframe_label(info, len(sampling.keyframes), sampling.meta.duration_s)
        images.append(prepared)
        keyframes.append(info)

    meta = sampling.meta.model_copy(update={"name": _safe_name(video.filename), "size_bytes": total})
    result = await request.app.state.ai.diagnose(
        DiagnoseRequest(
            mode=AnalysisMode.VIDEO,
            personality=personality,
            images=images,
            video_frames=[(k.number, k.t, None) for k in keyframes],
            force_demo=force_demo,
        )
    )
    return build_video_report(
        result,
        personality=personality,
        keyframes=keyframes,
        meta=meta,
        sampling=sampling.stats,
        local_events=scene_change_events(sampling.scene_changes),
    )
