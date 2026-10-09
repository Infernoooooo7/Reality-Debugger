"""Video Debug: local temporal analysis, optional AI narration, and the
server-side sampling fallback (used only when the browser cannot decode a video).

The primary path never uploads the video. The browser samples frames, runs the
detectors and the tracker on every sample and sends the resulting scene models
(``manifest.samples``). The backend replays them through the same local engine
and finding lifecycle as a live scan, using video time as the clock, so a video
gets DISCOVERED / CONFIRMED / RESOLVED events without any AI. Keyframe images
are uploaded only when AI reasoning is enabled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from app.errors import InvalidUploadError
from app.runtime_config import RuntimeConfig
from app.schemas.analysis import AIDiagnosis, VideoManifest
from app.schemas.common import AnalysisMode, FindingStatus, Personality, Severity
from app.schemas.diagnostics import (
    Finding,
    StatusChange,
    TimelineEvent,
    TimelineKind,
    VideoKeyframe,
    VideoMeta,
    VideoSamplingStats,
)
from app.schemas.scene import SceneModel, SceneObject, SceneSignals, SceneStats
from app.services.diagnostic_service import DiagnosticService
from app.services.local_diagnostics import LocalFinding
from app.services.scan_store import ScanSession, canonical_bug_id, utcnow
from app.services.vision_service import gray_signals

log = logging.getLogger("reality.video")

try:  # OpenCV is only needed for the server-side fallback.
    import cv2

    HAS_OPENCV = True
except Exception:  # pragma: no cover - import guard
    cv2 = None  # type: ignore[assignment]
    HAS_OPENCV = False

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
_LIFECYCLE_TO_TIMELINE: dict[str, TimelineKind] = {
    "DISCOVERED": "DISCOVERED",
    "REOPENED": "DISCOVERED",
    "CONFIRMED": "CONFIRMED",
    "RESOLVED": "RESOLVED",
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


def keyframe_label(frame: KeyframeInfo, total: int, duration: float) -> str:
    return (
        f"Frame {frame.number} of {total} - t={fmt_time(frame.t)} of {fmt_time(duration)} "
        f"(scene {frame.scene + 1}, {frame.reason})"
    )


# --------------------------------------------------------------------------
# Local temporal analysis (no AI)
# --------------------------------------------------------------------------


def manifest_scenes(manifest: VideoManifest) -> list[SceneModel]:
    """Scene models for every sample (older clients: one per keyframe)."""
    scenes: list[SceneModel] = []
    if manifest.samples:
        for sample in sorted(manifest.samples, key=lambda s: s.t):
            scenes.append(
                SceneModel(
                    at_ms=round(sample.t * 1000, 1),
                    view_id=sample.scene,
                    width=manifest.width,
                    height=manifest.height,
                    objects=sample.objects,
                    signals=SceneSignals(brightness=sample.brightness, sharpness=sample.sharpness, motion=sample.motion),
                    stats=SceneStats(detectors=sample.detectors or ([manifest.detector] if manifest.detector else [])),
                )
            )
        return scenes
    for frame in sorted(manifest.frames, key=lambda f: f.t):
        objects = [
            SceneObject(
                id=o.track_id or f"k{frame.index}_{i}",
                label=o.label,
                confidence=o.confidence,
                box=o.box,
            )
            for i, o in enumerate(frame.objects)
            if o.box is not None
        ]
        scenes.append(
            SceneModel(
                at_ms=round(frame.t * 1000, 1),
                view_id=frame.scene,
                width=manifest.width,
                height=manifest.height,
                objects=objects,
                signals=SceneSignals(brightness=frame.brightness, sharpness=frame.sharpness, motion=frame.motion),
                stats=SceneStats(detectors=frame.detectors or ([manifest.detector] if manifest.detector else [])),
            )
        )
    return scenes


@dataclass(slots=True)
class LocalVideoResult:
    findings: list[Finding]
    timeline: list[TimelineEvent]
    local: list[LocalFinding]  # every local finding of the richest sample (report summary)
    representative: SceneModel | None
    object_ids: dict[str, list[str]] = field(default_factory=dict)


def analyze_scenes(diagnostics: DiagnosticService, scenes: list[SceneModel], personality: Personality) -> LocalVideoResult:
    """Replay the samples through the live lifecycle with video time as the clock."""
    now = utcnow()
    session = ScanSession(scan_id="video_local", personality=personality, created_at=now, updated_at=now)
    timeline: list[TimelineEvent] = []
    richest: tuple[int, SceneModel, list[LocalFinding]] | None = None
    for scene in scenes:
        obs = diagnostics.observe(session, scene, mode=AnalysisMode.VIDEO)
        t = round(scene.at_ms / 1000, 2)
        for tf in session.findings.values():
            for change in tf.history:
                if change.video_t is None:
                    change.video_t = t
        for event in obs.events:
            kind = _LIFECYCLE_TO_TIMELINE.get(event.type)
            if kind is None:
                continue
            text = event.title if event.type == "DISCOVERED" else f"{event.title}: {event.note or event.type.lower()}"
            if event.type == "REOPENED":
                text = f"Reappeared: {event.title}"
            timeline.append(
                TimelineEvent(t=t, frame_index=None, kind=kind, severity=event.severity, finding_id=event.finding_id, text=text, source="local")
            )
        weight = len(scene.objects) * 10 + len(obs.local)
        if richest is None or weight > richest[0]:
            richest = (weight, scene, obs.local)

    findings: list[Finding] = []
    object_ids: dict[str, list[str]] = {}
    for tf in session.findings.values():
        model = tf.to_model()
        seen = [h.video_t for h in tf.history if h.video_t is not None]
        model.first_seen_s = seen[0] if seen else round(tf.first_seen_ms / 1000, 2)
        model.last_seen_s = round(tf.last_seen_ms / 1000, 2)
        model.history = list(tf.history[-12:])
        findings.append(model)
        object_ids[tf.id] = list(tf.object_ids)
    return LocalVideoResult(
        findings=findings,
        timeline=timeline,
        local=richest[2] if richest else [],
        representative=richest[1] if richest else None,
        object_ids=object_ids,
    )


def local_timeline_summary(result: LocalVideoResult, limit: int = 30) -> str:
    """The local timeline as prompt text for the optional AI narration."""
    if not result.timeline:
        return "Local timeline: no measured events."
    lines = ["Local timeline (measured on-device; do not repeat these events in `timeline`):"]
    for event in sorted(result.timeline, key=lambda e: e.t)[:limit]:
        lines.append(f"- {fmt_time(event.t)} {event.kind} {event.finding_id or ''} {event.text}".rstrip())
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Optional AI narration
# --------------------------------------------------------------------------


def ai_video_findings(
    diagnosis: AIDiagnosis,
    *,
    local: list[Finding],
    keyframes: list[KeyframeInfo],
    diagnostics: DiagnosticService,
) -> tuple[list[Finding], list[TimelineEvent]]:
    """AI findings (ids continuing after the local ones) and the AI timeline."""
    now = utcnow()
    ai_findings, id_map = diagnostics.stateless_ai(diagnosis, local, now=now)
    times = {k.number: k.t for k in keyframes}

    def time_for(frame_no: int) -> float:
        if frame_no in times:
            return times[frame_no]
        return min(keyframes, key=lambda k: abs(k.number - frame_no)).t if keyframes else 0.0

    known = {f.id for f in ai_findings} | {f.id for f in local}

    def map_id(raw: str | None) -> str | None:
        if not raw:
            return None
        if raw in id_map:
            return id_map[raw]
        canonical = canonical_bug_id(raw)
        return canonical if canonical in known else None

    timeline = [
        TimelineEvent(
            t=round(time_for(ev.frame), 2),
            frame_index=ev.frame,
            kind=ev.kind,
            severity=ev.severity,
            finding_id=map_id(ev.finding_id),
            text=ev.text,
            source="ai",
        )
        for ev in diagnosis.timeline
        if ev.text
    ]
    for finding in ai_findings:
        af = next((a for a in diagnosis.findings if a.id and id_map.get(a.id) == finding.id), None)
        seen = sorted(times[n] for n in (af.frames if af else []) if n in times)
        events = sorted((e for e in timeline if e.finding_id == finding.id), key=lambda e: e.t)
        history: list[StatusChange] = []
        status = FindingStatus.DISCOVERED
        for e in events:
            new = {"DISCOVERED": FindingStatus.DISCOVERED, "CONFIRMED": FindingStatus.CONFIRMED,
                   "ESCALATED": FindingStatus.CONFIRMED, "RESOLVED": FindingStatus.RESOLVED}.get(e.kind)
            if new is None:
                continue
            history.append(StatusChange(status=new, at=now, video_t=e.t, note=e.text))
            status = new
        if not events:
            first_t = seen[0] if seen else 0.0
            history.append(StatusChange(status=FindingStatus.DISCOVERED, at=now, video_t=first_t))
            timeline.append(TimelineEvent(t=round(first_t, 2), frame_index=None, kind="DISCOVERED", severity=finding.severity,
                                          finding_id=finding.id, text=finding.title, source="ai"))
            if len(seen) >= 2:
                status = FindingStatus.CONFIRMED
                history.append(StatusChange(status=status, at=now, video_t=seen[1]))
                timeline.append(TimelineEvent(t=round(seen[1], 2), frame_index=None, kind="CONFIRMED", severity=finding.severity,
                                              finding_id=finding.id, text=f"Seen again: {finding.title}", source="ai"))
        elif status == FindingStatus.DISCOVERED and len(seen) >= 2:
            status = FindingStatus.CONFIRMED
            history.append(StatusChange(status=status, at=now, video_t=seen[1]))
        all_times = sorted(seen + [e.t for e in events if e.kind != "RESOLVED"])
        finding.status = status
        finding.history = history
        finding.sightings = max(1, len(seen))
        finding.first_seen_s = all_times[0] if all_times else None
        finding.last_seen_s = all_times[-1] if all_times else None
        finding.resolved_note = history[-1].note if status == FindingStatus.RESOLVED else None
    return ai_findings, timeline


def finish_timeline(events: list[TimelineEvent], limit: int = 160) -> list[TimelineEvent]:
    events.sort(key=lambda e: (e.t, _KIND_ORDER.get(e.kind, 9)))
    return events[:limit]


def keyframe_models(keyframes: list[KeyframeInfo]) -> list[VideoKeyframe]:
    return [VideoKeyframe(index=k.number, t=k.t, scene=k.scene, reason=k.reason) for k in keyframes]


# --------------------------------------------------------------------------
# Keyframe selection (shared by the server-side fallback)
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Sample:
    t: float
    hist: np.ndarray
    grid: np.ndarray
    sharpness: float  # 0..1, browser-compatible
    brightness: float
    motion: float


@dataclass(slots=True)
class Selection:
    keyframes: list[tuple[float, int, str]]  # (t, scene, reason)
    scenes: list[tuple[float, float]]
    scene_of: list[int]  # scene index of every sample
    redundant: int
    scene_changes: list[float] = field(default_factory=list)


@dataclass(slots=True)
class VideoParams:
    boundary: float
    drift: float
    dedupe: float
    window: tuple[float, float]
    min_gap_s: float
    histogram_weight: float
    grid_gain: float
    motion_gain: float
    log_offset: float
    log_span: float
    max_samples: int
    interval_s: float

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> "VideoParams":
        window = config.get("temporal.video.representativeWindow")
        return cls(
            boundary=float(config.get("temporal.video.sceneBoundary")),
            drift=float(config.get("temporal.video.sceneDrift")),
            dedupe=float(config.get("temporal.video.dedupe")),
            window=(float(window[0]), float(window[1])),
            min_gap_s=float(config.get("temporal.video.keyframeMinGapS")),
            histogram_weight=float(config.get("vision.signals.histogramWeight")),
            grid_gain=float(config.get("vision.signals.gridGain")),
            motion_gain=float(config.get("vision.signals.motionGain")),
            log_offset=float(config.get("vision.signals.sharpnessLogOffset")),
            log_span=float(config.get("vision.signals.sharpnessLogSpan")),
            max_samples=int(config.get("temporal.video.maxSamples")),
            interval_s=float(config.get("temporal.video.sampleIntervalS")),
        )


def signature_distance(a: Sample, b: Sample, p: VideoParams) -> float:
    hist = 0.5 * float(np.abs(a.hist - b.hist).sum())
    grid = float(np.abs(a.grid - b.grid).mean())
    return p.histogram_weight * hist + (1 - p.histogram_weight) * min(1.0, grid * p.grid_gain)


def select_keyframes(samples: list[Sample], max_keyframes: int, p: VideoParams) -> Selection:
    if not samples:
        return Selection(keyframes=[], scenes=[], scene_of=[], redundant=0)
    scenes: list[list[int]] = [[0]]
    changes: list[float] = []
    for i in range(1, len(samples)):
        step = signature_distance(samples[i], samples[i - 1], p)
        drift = signature_distance(samples[i], samples[scenes[-1][0]], p)
        if step > p.boundary or drift > p.drift:
            scenes.append([i])
            changes.append(samples[i].t)
        else:
            scenes[-1].append(i)

    redundant = 0
    for scene in scenes:
        kept = scene[0]
        for i in scene[1:]:
            if signature_distance(samples[i], samples[kept], p) < p.dedupe:
                redundant += 1
            else:
                kept = i

    def representative(scene: list[int]) -> int:
        lo = int(len(scene) * p.window[0])
        hi = max(lo + 1, int(len(scene) * p.window[1]))
        core = scene[lo:hi] or scene
        return max(core, key=lambda i: samples[i].sharpness)

    picks: list[tuple[int, int, str]] = [(representative(s), si, "scene representative") for si, s in enumerate(scenes)]
    if len(picks) > max_keyframes:
        lengths = sorted(range(len(scenes)), key=lambda si: -len(scenes[si]))
        keep = set(lengths[: max_keyframes - 1]) | {0}
        picks = [pk for pk in picks if pk[1] in keep][:max_keyframes]
    else:
        extra: list[tuple[float, int, int]] = []
        for si, scene in enumerate(scenes):
            rep = representative(scene)
            for i in scene:
                if i != rep:
                    extra.append((signature_distance(samples[i], samples[rep], p), i, si))
        extra.sort(reverse=True)
        chosen = {pk[0] for pk in picks}
        for dist, i, si in extra:
            if len(picks) >= max_keyframes or dist < p.dedupe * 2:
                break
            if all(abs(samples[i].t - samples[c].t) > p.min_gap_s for c in chosen):
                picks.append((i, si, "state change within scene"))
                chosen.add(i)

    picks.sort(key=lambda pk: samples[pk[0]].t)
    scene_of = [0] * len(samples)
    for si, scene in enumerate(scenes):
        for i in scene:
            scene_of[i] = si
    return Selection(
        keyframes=[(samples[i].t, si, reason) for i, si, reason in picks],
        scenes=[(samples[s[0]].t, samples[s[-1]].t) for s in scenes],
        scene_of=scene_of,
        redundant=redundant,
        scene_changes=changes,
    )


# --------------------------------------------------------------------------
# Server-side fallback (OpenCV)
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ServerSampling:
    keyframes: list[tuple[float, int, str, bytes]]  # t, scene, reason, jpeg
    scenes: list[SceneModel]  # signal-only scene model per sample (no detector on the server)
    meta: VideoMeta
    stats: VideoSamplingStats
    scene_changes: list[float]


def _sample_signature(frame: np.ndarray, t: float, prev_gray: np.ndarray | None, p: VideoParams) -> tuple[Sample, np.ndarray]:
    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist / max(1.0, float(hist.sum()))
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    grid = cv2.resize(gray_small, (16, 9), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    thumb = cv2.resize(frame, (128, 96), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    brightness, sharpness = gray_signals(gray, log_offset=p.log_offset, log_span=p.log_span)
    motion = 0.0 if prev_gray is None else min(1.0, float(np.abs(gray - prev_gray).mean()) * p.motion_gain)
    sample = Sample(t=t, hist=hist.astype(np.float32), grid=grid, sharpness=sharpness, brightness=brightness, motion=round(motion, 4))
    return sample, gray


def sample_video_file(path: str, *, max_keyframes: int, edge: int, params: VideoParams) -> ServerSampling:
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
        prev: np.ndarray | None = None
        if duration > 0:
            step = max(duration / params.max_samples, params.interval_s / 2)
            t = step / 2
            while t < duration and len(samples) < params.max_samples:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, frame = cap.read()
                if ok and frame is not None:
                    sample, prev = _sample_signature(frame, t, prev, params)
                    samples.append(sample)
                t += step
        else:
            # Unknown duration: read sequentially, keeping every Nth frame.
            index = 0
            guess_fps = fps if fps > 0 else 30.0
            every = max(1, int(guess_fps * params.interval_s))
            while len(samples) < params.max_samples:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if index % every == 0:
                    sample, prev = _sample_signature(frame, index / guess_fps, prev, params)
                    samples.append(sample)
                index += 1
            duration = index / guess_fps if index else 0.0

        if not samples:
            raise InvalidUploadError(
                "No frames could be decoded from this video.",
                code="VIDEO_UNREADABLE",
                hint="The file may be corrupted. Try exporting it again as MP4 (H.264).",
            )

        selection = select_keyframes(samples, max_keyframes, params)
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

        scenes = [
            SceneModel(
                at_ms=round(s.t * 1000, 1),
                view_id=selection.scene_of[i],
                width=width,
                height=height,
                signals=SceneSignals(brightness=s.brightness, sharpness=s.sharpness, motion=s.motion),
                stats=SceneStats(detectors=[]),
            )
            for i, s in enumerate(samples)
        ]
        return ServerSampling(
            keyframes=keyframes,
            scenes=scenes,
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
