"""Video Debug: keyframe handling, timeline building and the server-side
sampling fallback (used only when the browser cannot decode a video).

The primary path never uploads the video: the browser samples frames,
detects scene changes, removes redundant frames and uploads only a handful
of representative keyframes plus its local detections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from app.errors import InvalidUploadError
from app.schemas.common import FindingStatus, Personality, Severity
from app.schemas.diagnostics import (
    Finding,
    StatusChange,
    TimelineEvent,
    VideoKeyframe,
    VideoMeta,
    VideoReport,
    VideoSamplingStats,
)
from app.services.ai_service import DiagnoseResult
from app.services.diagnostic_service import build_report
from app.services.scan_store import canonical_bug_id, utcnow
from app.schemas.common import AnalysisMode

log = logging.getLogger("reality.video")

try:  # OpenCV is only needed for the server-side fallback.
    import cv2

    HAS_OPENCV = True
except Exception:  # pragma: no cover - import guard
    cv2 = None  # type: ignore[assignment]
    HAS_OPENCV = False

SCENE_THRESHOLD = 0.30
DEDUPE_THRESHOLD = 0.06

_KIND_ORDER = {
    "SCENE_CHANGE": 0,
    "OBJECT_ENTERED": 1,
    "DISCOVERED": 2,
    "CONFIRMED": 3,
    "ESCALATED": 4,
    "OBSERVATION": 5,
    "OBJECT_LEFT": 6,
    "RESOLVED": 7,
}


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:04.1f}"


@dataclass(slots=True)
class KeyframeInfo:
    number: int  # 1-based frame number shown to the model
    t: float
    scene: int
    reason: str


# --------------------------------------------------------------------------
# Timeline / report
# --------------------------------------------------------------------------


def build_video_report(
    result: DiagnoseResult,
    *,
    personality: Personality,
    keyframes: list[KeyframeInfo],
    meta: VideoMeta,
    sampling: VideoSamplingStats,
    local_events: list[TimelineEvent],
) -> VideoReport:
    now = utcnow()
    diagnosis = result.diagnosis
    source = "demo" if result.simulated else "ai"
    times = {k.number: k.t for k in keyframes}

    def time_for(frame_no: int) -> float:
        if frame_no in times:
            return times[frame_no]
        nearest = min(keyframes, key=lambda k: abs(k.number - frame_no))
        return nearest.t

    id_map: dict[str, str] = {}
    planned: list[tuple[str, list[float]]] = []
    for i, af in enumerate(diagnosis.findings):
        fid = f"BUG-{i + 1:03d}"
        if af.id:
            id_map[af.id] = fid
        seen = sorted(times[n] for n in af.frames if n in times)
        planned.append((fid, seen))

    def map_id(raw: str | None) -> str | None:
        if not raw:
            return None
        return id_map.get(raw) or (canonical_bug_id(raw) if canonical_bug_id(raw) in {p[0] for p in planned} else None)

    timeline: list[TimelineEvent] = [
        TimelineEvent(
            t=round(time_for(ev.frame), 2),
            frame_index=ev.frame,
            kind=ev.kind,
            severity=ev.severity,
            finding_id=map_id(ev.finding_id),
            text=ev.text,
            source=source,
        )
        for ev in diagnosis.timeline
        if ev.text
    ]

    findings: list[Finding] = []
    for af, (fid, seen) in zip(diagnosis.findings, planned, strict=True):
        events = sorted((e for e in timeline if e.finding_id == fid), key=lambda e: e.t)
        history: list[StatusChange] = []
        status = FindingStatus.DISCOVERED
        for e in events:
            if e.kind == "DISCOVERED":
                new = FindingStatus.DISCOVERED
            elif e.kind in ("CONFIRMED", "ESCALATED"):
                new = FindingStatus.CONFIRMED
            elif e.kind == "RESOLVED":
                new = FindingStatus.RESOLVED
            else:
                continue
            history.append(StatusChange(status=new, at=now, video_t=e.t, note=e.text))
            status = new
        if not events:
            # No narrated events for this finding: synthesise a minimal history.
            first_t = seen[0] if seen else None
            history.append(StatusChange(status=FindingStatus.DISCOVERED, at=now, video_t=first_t))
            timeline.append(
                TimelineEvent(
                    t=round(first_t or 0.0, 2),
                    frame_index=None,
                    kind="DISCOVERED",
                    severity=af.severity,
                    finding_id=fid,
                    text=af.title,
                    source=source,
                )
            )
            if len(seen) >= 2:
                status = FindingStatus.CONFIRMED
                history.append(StatusChange(status=status, at=now, video_t=seen[1]))
                timeline.append(
                    TimelineEvent(
                        t=round(seen[1], 2),
                        frame_index=None,
                        kind="CONFIRMED",
                        severity=af.severity,
                        finding_id=fid,
                        text=f"Seen again: {af.title}",
                        source=source,
                    )
                )
        elif status == FindingStatus.DISCOVERED and len(seen) >= 2:
            status = FindingStatus.CONFIRMED
            history.append(StatusChange(status=status, at=now, video_t=seen[1]))

        event_times = [e.t for e in events if e.kind != "RESOLVED"]
        all_times = sorted(seen + event_times)
        findings.append(
            Finding(
                id=fid,
                severity=af.severity,
                category=af.category,
                title=af.title,
                evidence=af.evidence,
                inference=af.inference,
                impact=af.impact,
                recommendation=af.recommendation,
                confidence=af.confidence,
                quip=af.quip,
                status=status,
                box=af.box,
                related_objects=list(af.related_objects),
                source=source,
                sightings=max(1, len(seen)),
                first_seen_at=now,
                last_seen_at=now,
                first_seen_s=all_times[0] if all_times else None,
                last_seen_s=all_times[-1] if all_times else None,
                resolved_note=history[-1].note if status == FindingStatus.RESOLVED else None,
                history=history,
            )
        )

    timeline.extend(local_events)
    timeline.sort(key=lambda e: (e.t, _KIND_ORDER.get(e.kind, 9)))

    report = build_report(result, mode=AnalysisMode.VIDEO, personality=personality, findings=findings)
    return VideoReport(
        report=report,
        timeline=timeline[:120],
        video=meta,
        sampling=sampling,
        keyframes=[VideoKeyframe(index=k.number, t=k.t, scene=k.scene, reason=k.reason) for k in keyframes],
    )


def keyframe_label(frame: KeyframeInfo, total: int, duration: float) -> str:
    return (
        f"Frame {frame.number} of {total} - t={fmt_time(frame.t)} of {fmt_time(duration)} "
        f"(scene {frame.scene + 1}, {frame.reason})"
    )


# --------------------------------------------------------------------------
# Keyframe selection (shared by the server-side fallback)
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Sample:
    t: float
    hist: np.ndarray
    grid: np.ndarray
    sharpness: float


@dataclass(slots=True)
class Selection:
    keyframes: list[tuple[float, int, str]]  # (t, scene, reason)
    scenes: list[tuple[float, float]]
    redundant: int
    scene_changes: list[float] = field(default_factory=list)


def signature_distance(a: Sample, b: Sample) -> float:
    hist = 0.5 * float(np.abs(a.hist - b.hist).sum())
    grid = float(np.abs(a.grid - b.grid).mean())
    return 0.55 * hist + 0.45 * min(1.0, grid * 3.0)


def select_keyframes(samples: list[Sample], max_keyframes: int) -> Selection:
    if not samples:
        return Selection(keyframes=[], scenes=[], redundant=0)
    scenes: list[list[int]] = [[0]]
    changes: list[float] = []
    for i in range(1, len(samples)):
        step = signature_distance(samples[i], samples[i - 1])
        drift = signature_distance(samples[i], samples[scenes[-1][0]])
        if step > SCENE_THRESHOLD or drift > SCENE_THRESHOLD * 1.6:
            scenes.append([i])
            changes.append(samples[i].t)
        else:
            scenes[-1].append(i)

    redundant = 0
    for scene in scenes:
        kept = scene[0]
        for i in scene[1:]:
            if signature_distance(samples[i], samples[kept]) < DEDUPE_THRESHOLD:
                redundant += 1
            else:
                kept = i

    def representative(scene: list[int]) -> int:
        lo = int(len(scene) * 0.2)
        hi = max(lo + 1, int(len(scene) * 0.8))
        core = scene[lo:hi] or scene
        return max(core, key=lambda i: samples[i].sharpness)

    picks: list[tuple[int, int, str]] = [(representative(s), si, "scene representative") for si, s in enumerate(scenes)]
    if len(picks) > max_keyframes:
        # Keep the longest scenes, always including the first one.
        lengths = sorted(range(len(scenes)), key=lambda si: -len(scenes[si]))
        keep = set(lengths[: max_keyframes - 1]) | {0}
        picks = [p for p in picks if p[1] in keep][:max_keyframes]
    else:
        # Spend the remaining budget on the frames most unlike their scene representative.
        extra: list[tuple[float, int, int]] = []
        for si, scene in enumerate(scenes):
            rep = representative(scene)
            for i in scene:
                if i != rep:
                    extra.append((signature_distance(samples[i], samples[rep]), i, si))
        extra.sort(reverse=True)
        chosen = {p[0] for p in picks}
        for dist, i, si in extra:
            if len(picks) >= max_keyframes or dist < DEDUPE_THRESHOLD * 2:
                break
            if all(abs(samples[i].t - samples[c].t) > 0.75 for c in chosen):
                picks.append((i, si, "state change within scene"))
                chosen.add(i)

    picks.sort(key=lambda p: samples[p[0]].t)
    scene_ranges = [(samples[s[0]].t, samples[s[-1]].t) for s in scenes]
    return Selection(
        keyframes=[(samples[i].t, si, reason) for i, si, reason in picks],
        scenes=scene_ranges,
        redundant=redundant,
        scene_changes=changes,
    )


# --------------------------------------------------------------------------
# Server-side fallback (OpenCV)
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ServerSampling:
    keyframes: list[tuple[float, int, str, bytes]]  # t, scene, reason, jpeg
    meta: VideoMeta
    stats: VideoSamplingStats
    scene_changes: list[float]


def _sample_signature(frame: np.ndarray, t: float) -> Sample:
    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist / max(1.0, float(hist.sum()))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    grid = cv2.resize(gray, (16, 9), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mid = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    sharp = float(cv2.Laplacian(cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    return Sample(t=t, hist=hist.astype(np.float32), grid=grid, sharpness=sharp)


def sample_video_file(path: str, *, max_keyframes: int, edge: int, max_samples: int = 72) -> ServerSampling:
    if not HAS_OPENCV:
        raise InvalidUploadError(
            "Server-side video decoding is not available.",
            code="VIDEO_DECODER_MISSING",
            hint="Install opencv-python-headless or use a browser that can play this video.",
        )
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise InvalidUploadError(
            "The video could not be opened. It may be corrupted or use an unsupported codec.",
            code="VIDEO_UNREADABLE",
            hint="Try exporting it as MP4 (H.264).",
        )
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

        samples: list[Sample] = []
        if duration > 0:
            step = max(duration / max_samples, 0.25)
            t = step / 2
            while t < duration and len(samples) < max_samples:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, frame = cap.read()
                if ok and frame is not None:
                    samples.append(_sample_signature(frame, t))
                t += step
        else:
            # Unknown duration: read sequentially, keeping every Nth frame.
            index = 0
            guess_fps = fps if fps > 0 else 30.0
            while len(samples) < max_samples:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if index % int(guess_fps // 2 or 1) == 0:
                    samples.append(_sample_signature(frame, index / guess_fps))
                index += 1
            duration = index / guess_fps if index else 0.0

        if not samples:
            raise InvalidUploadError(
                "No frames could be decoded from this video.",
                code="VIDEO_UNREADABLE",
                hint="The file may be corrupted. Try exporting it again as MP4 (H.264).",
            )

        selection = select_keyframes(samples, max_keyframes)
        keyframes: list[tuple[float, int, str, bytes]] = []
        for t, scene, reason in selection.keyframes:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            scale = min(1.0, edge / max(h, w))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                keyframes.append((t, scene, reason, buf.tobytes()))
        if not keyframes:
            raise InvalidUploadError("Keyframes could not be extracted.", code="VIDEO_UNREADABLE")

        return ServerSampling(
            keyframes=keyframes,
            meta=VideoMeta(duration_s=round(duration, 2), width=width, height=height, fps=round(fps, 2) or None),
            stats=VideoSamplingStats(
                processed_on="server",
                sampled_frames=len(samples),
                scenes=len(selection.scenes),
                redundant_removed=selection.redundant,
                keyframes=len(keyframes),
            ),
            scene_changes=selection.scene_changes,
        )
    finally:
        cap.release()


def scene_change_events(changes: list[float]) -> list[TimelineEvent]:
    return [
        TimelineEvent(
            t=round(t, 2),
            frame_index=None,
            kind="SCENE_CHANGE",
            severity=Severity.INFO,
            finding_id=None,
            text="Scene change detected (local frame comparison).",
            source="local",
        )
        for t in changes
    ]
