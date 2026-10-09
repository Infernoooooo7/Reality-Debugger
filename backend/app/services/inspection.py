"""Inspection evidence: what a scan actually examined, and whether that supports a conclusion.

An empty findings list only means that no check fired. Whether that says
anything about the scene depends on what was inspected:
- which detectors ran, and whether any failed;
- at what effective resolution they saw the image;
- how much of the image's visible structure lies outside every recognised object;
- image quality, and how confident the recognitions are;
- what the detectors can recognise at all (a closed vocabulary of 80 COCO categories).

``assess`` turns these measurements into a coverage level with explicit
reasons. The diagnostic service uses it to choose the headline status and to
decide whether the scene score is rated. Thresholds are in
config/diagnostics.json "inspection"; their evidence is in docs/SCORING.md.

Coverage levels:
- **sufficient**: nothing below suggests that the scan missed a meaningful part of the scene;
- **limited**: the scan ran, but at least one measured limitation means absent findings are not an all-clear;
- **insufficient**: the evidence cannot support any conclusion about the scene (detection failed, or nothing
  was recognised in an image full of structure).

Every reason is a measurement or a reported failure, never a guess about how
many objects "should" be there.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any

from app.runtime_config import RuntimeConfig
from app.schemas.common import AnalysisMode
from app.schemas.diagnostics import DetectorReport, InspectionReason, InspectionReport, SkippedAnalysis
from app.schemas.scene import DetectorRun, SceneModel

STILL_MODES = {AnalysisMode.IMAGE, AnalysisMode.DEEP, AnalysisMode.VIDEO}

#: Checks that need a category no loaded model can output, with what would be needed.
UNAVAILABLE_CHECKS = {
    "cable_congestion": "cable and power-strip analysis (no cable/outlet category in the loaded models; needs a detector or segmenter trained on them)",
    "liquid_near_outlet": "liquid-near-outlet check (no wall-outlet category in the loaded models)",
}


def _effective_scale(run: DetectorRun, width: int | None, height: int | None) -> float | None:
    """Network pixels per source pixel for the best pass of this run (capped at 1 = full detail)."""
    if run.status != "ok" or not run.input_size or not width or not height:
        return None
    source = run.tile_px if run.passes > 1 and run.tile_px else max(width, height)
    return round(min(1.0, run.input_size / source), 3)


def _runs(scene: SceneModel) -> list[DetectorRun]:
    if scene.stats.runs:
        return list(scene.stats.runs)
    # Older clients report only the names of detectors whose output was used.
    return [DetectorRun(model=name, role="deep" if "yolox" in name else "fast", status="ok", note="reported by name only")
            for name in scene.stats.detectors]


def assess(scene: SceneModel | None, *, mode: AnalysisMode, config: RuntimeConfig, findings: int,
           vocabulary_size: int, armed_rules: Sequence[str], all_rules: Sequence[str], set_aside: int = 0) -> InspectionReport:
    get = config.get
    min_scale = float(get("diagnostics.inspection.minEffectiveScale"))
    max_unexplained = float(get("diagnostics.inspection.maxUnexplainedShare"))
    min_edges = float(get("diagnostics.inspection.minEdgeDensityForContent"))
    low_conf = float(get("diagnostics.inspection.lowConfidence"))
    max_low_share = float(get("diagnostics.inspection.maxLowConfidenceShare"))
    unrecognised_median = float(get("diagnostics.inspection.closedVocabularyUnrecognisedMedian"))
    min_brightness = float(get("inference.quality.minBrightness"))
    min_sharpness = float(get("inference.quality.minSharpness"))

    reasons: list[InspectionReason] = []
    skipped = [SkippedAnalysis(analysis=text.split(" (")[0], reason=text) for rule, text in UNAVAILABLE_CHECKS.items()
               if rule in all_rules and rule not in armed_rules]
    skipped += [
        SkippedAnalysis(analysis="segmentation", reason="no segmentation model is loaded: object outlines and pixel areas are not measured, boxes only"),
        SkippedAnalysis(analysis="open-vocabulary recognition", reason="no open-vocabulary model is available: only the detectors' fixed categories can be recognised"),
    ]
    vocabulary = (f"The detectors recognise {vocabulary_size} object categories (COCO). Objects of other kinds are not inspected "
                  "and are never reported as checked.")

    if scene is None:
        reasons.append(InspectionReason(code="detection_not_run", level="insufficient",
                                        message="No detector output was supplied, so no object in the scene was examined."))
        return InspectionReport(analysis_status="detection_failed", coverage="insufficient", detection="not_run",
                                findings="findings" if findings else "no_findings", reasons=reasons, vocabulary=vocabulary,
                                checks_run=list(armed_rules), skipped=skipped)

    width, height = scene.width, scene.height
    runs = _runs(scene)
    reports = [DetectorReport(model=r.model, role=r.role, status=r.status, boxes=r.boxes, ms=r.ms, input_size=r.input_size,
                              passes=r.passes, tile_px=r.tile_px, incomplete=r.incomplete,
                              effective_scale=_effective_scale(r, width, height),
                              vocabulary=r.vocabulary, note=r.note) for r in runs]
    ok = [r for r in reports if r.status == "ok"]
    objects = scene.objects
    stats = scene.stats.coverage

    # -- detection -----------------------------------------------------------------------------
    if not runs:
        detection = "not_run"
        reasons.append(InspectionReason(code="detection_not_run", level="insufficient",
                                        message="No detector reported running on this scene, so its objects were not examined."))
    elif not ok:
        detection = "failed"
        notes = "; ".join(f"{r.model}: {r.status}{f' ({r.note})' if r.note else ''}" for r in reports)
        reasons.append(InspectionReason(code="detection_failed", level="insufficient",
                                        message=f"No detector produced a result ({notes}).", evidence={"detectors": len(reports)}))
    else:
        failed = [r for r in reports if r.status != "ok"]
        detection = "partial" if failed else "ok"
        for r in failed:
            reasons.append(InspectionReason(
                code="detector_not_used", level="limited",
                message=f"The {r.role} detector ({r.model}) {('failed' if r.status == 'failed' else 'was ' + r.status)}"
                        f"{f': {r.note}' if r.note else ''}. Only the other detector's results were used.",
                evidence={"model": r.model, "status": r.status}))
        if mode in STILL_MODES and not any(r.role in ("deep", "server") for r in ok):
            reasons.append(InspectionReason(
                code="fast_detector_only", level="limited",
                message="Only the fast detector ran. On 118 held-out COCO desk scenes it finds about a third of the "
                        "annotated objects it knows (recall 0.34, against 0.52 for the deep detector).",
                evidence={"recall_fast": 0.34, "recall_deep": 0.52}))
        if any(r.note == "reported by name only" for r in reports):
            reasons.append(InspectionReason(code="runs_unreported", level="limited",
                                            message="The client did not report how its detectors ran (resolution, failures); coverage cannot be verified."))

    # -- resolution ------------------------------------------------------------------------------
    scales = [r.effective_scale for r in ok if r.effective_scale is not None]
    best_scale = max(scales) if scales else None
    if best_scale is not None and best_scale < min_scale:
        reasons.append(InspectionReason(
            code="downscaled", level="limited",
            message=f"The image ({width}x{height}) was analysed at {best_scale * 100:.0f}% of its resolution at best, so small "
                    "objects are likely to be missed (on 118 held-out COCO desk scenes the deep detector finds 27% of small "
                    "objects against 76% of large ones, even at full detail).",
            evidence={"effective_scale": best_scale, "min_effective_scale": min_scale}))

    for r in ok:
        if r.incomplete:
            reasons.append(InspectionReason(
                code="tiling_incomplete", level="limited",
                message=f"The high-resolution pass of {r.model} stopped early ({r.passes - 1} tile(s) analysed{f'; {r.note}' if r.note else ''}), "
                        "so parts of the image were only seen at reduced resolution.",
                evidence={"model": r.model, "passes": r.passes}))

    # -- structure the recognised objects do not explain -----------------------------------------------
    unexplained = stats.unexplained_share if stats else None
    edge_density = stats.edge_density if stats else None
    if stats is None:
        if detection in ("ok", "partial"):
            reasons.append(InspectionReason(code="structure_unmeasured", level="limited",
                                            message="The client did not measure how much of the image the recognised objects explain."))
    elif not objects and detection in ("ok", "partial") and edge_density is not None and edge_density >= min_edges:
        reasons.append(InspectionReason(
            code="nothing_recognised", level="insufficient",
            message=f"No object was recognised, although {edge_density * 100:.0f}% of the image is detailed structure: "
                    "whatever is there is either of an unsupported kind or was missed.",
            evidence={"edge_density": edge_density, "min_edge_density": min_edges}))
    elif unexplained is not None and objects and unexplained > max_unexplained:
        reasons.append(InspectionReason(
            code="unexplained_structure", level="limited",
            message=f"{unexplained * 100:.0f}% of the image's visible detail lies outside every recognised object. That detail "
                    "may be background texture or objects the detectors cannot recognise; it was not inspected.",
            evidence={"unexplained_share": unexplained, "max_unexplained_share": max_unexplained}))

    total = scene.stats.objects_total
    if total is not None and total > len(objects):
        reasons.append(InspectionReason(
            code="objects_capped", level="limited",
            message=f"{total} objects were recognised but only the {len(objects)} most confident were sent for analysis "
                    "(payload limit), so the spatial checks did not see the rest.",
            evidence={"recognised": total, "analysed": len(objects)}))

    # -- vocabulary ---------------------------------------------------------------------------------
    # Every loaded detector has a closed vocabulary, which limits every open-world scan
    # (revisit when an open-vocabulary detector is added to the registry).
    if detection in ("ok", "partial"):
        reasons.append(InspectionReason(
            code="closed_vocabulary", level="limited",
            message=f"Only {vocabulary_size} object categories can be recognised. On 4978 held-out COCO images the median image "
                    f"has {unrecognised_median * 100:.0f}% of its annotated objects outside what these detectors recognise, so the "
                    "absence of findings is not an all-clear.",
            evidence={"categories": vocabulary_size, "median_unrecognised_share": unrecognised_median}))

    # -- confidence ----------------------------------------------------------------------------------
    confs = [o.confidence for o in objects]
    confidence: dict[str, Any] | None = None
    if confs:
        low = sum(1 for c in confs if c < low_conf)
        confidence = {"min": round(min(confs), 3), "median": round(statistics.median(confs), 3), "max": round(max(confs), 3),
                      "below_threshold": low, "threshold": low_conf}
        if low / len(confs) > max_low_share:
            reasons.append(InspectionReason(
                code="low_confidence", level="limited",
                message=f"{low} of {len(confs)} recognised objects have confidence below {low_conf:.1f}; on held-out COCO "
                        "images 58-68% of deep-detector detections scored 0.3-0.5 are wrong.",
                evidence={"below_threshold": low, "objects": len(confs)}))

    if set_aside:
        fast_floor = float(get("diagnostics.confidence.minEvidenceFast"))
        deep_floor = float(get("diagnostics.confidence.minEvidenceDeep"))
        reasons.append(InspectionReason(
            code="weak_evidence", level="limited",
            message=f"{set_aside} of {len(objects)} recognised objects scored below {fast_floor:.1f} (fast detector) or {deep_floor:.1f} "
                    "(deep detector) and were not used as evidence for findings: on held-out COCO images most detections at "
                    "those scores are wrong (precision 0.30-0.47).",
            evidence={"set_aside": set_aside, "objects": len(objects)}))

    # -- image quality -----------------------------------------------------------------------------
    quality = {"brightness": scene.signals.brightness, "sharpness": scene.signals.sharpness}
    if scene.signals.brightness is not None and scene.signals.brightness < min_brightness:
        reasons.append(InspectionReason(code="too_dark", level="limited",
                                        message=f"The image is dark (mean brightness {scene.signals.brightness:.2f} < {min_brightness:.2f}); detection degrades sharply.",
                                        evidence={"brightness": scene.signals.brightness}))
    if scene.signals.sharpness is not None and scene.signals.sharpness < min_sharpness:
        reasons.append(InspectionReason(code="too_blurred", level="limited",
                                        message=f"The image is blurred (sharpness {scene.signals.sharpness:.2f} < {min_sharpness:.2f}); detail is lost.",
                                        evidence={"sharpness": scene.signals.sharpness}))

    tiled = any(r.passes > 1 for r in ok)
    max_input = max((r.input_size or 0 for r in reports), default=0)
    if width and height and max_input and max(width, height) > 2 * max_input and not tiled:
        skipped.append(SkippedAnalysis(analysis="tiled high-resolution pass",
                                       reason="the image is larger than twice the detector input and was not analysed in tiles"))

    levels = {r.level for r in reasons}
    coverage = "insufficient" if "insufficient" in levels else "limited" if "limited" in levels else "sufficient"
    if detection in ("failed", "not_run"):
        analysis_status = "detection_failed"
    else:
        analysis_status = {"insufficient": "inconclusive", "limited": "limited", "sufficient": "complete"}[coverage]
    return InspectionReport(
        analysis_status=analysis_status,
        coverage=coverage,
        detection=detection,
        findings="findings" if findings else "no_findings",
        reasons=reasons,
        detectors=reports,
        objects=len(objects),
        categories=sorted({o.label for o in objects}),
        confidence=confidence,
        image={"width": width, "height": height} if width and height else None,
        structure=stats.model_dump() if stats else None,
        quality=quality,
        vocabulary=vocabulary,
        checks_run=[r for r in armed_rules if r != "keep_clear_zone" or scene.zones],
        skipped=skipped,
        # A score needs something to rate: sufficient coverage and at least one recognised object.
        score_rated=coverage == "sufficient" and bool(objects),
    )
