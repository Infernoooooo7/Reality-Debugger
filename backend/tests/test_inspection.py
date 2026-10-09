"""Inspection coverage, scores and status: an empty findings list is never an all-clear on its own.

Regression for the "false 100/100": a cluttered desk photo with four recognised
objects and no findings used to be reported as STABLE, 100/100, "No measurable
issues found". docs/SCORING.md describes the rules tested here.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.runtime_config import load_config
from app.schemas.common import AnalysisMode, SystemStatus
from app.schemas.scene import SceneModel
from app.services import inspection
from tests.conftest import CUP, LAPTOP, jpeg_bytes, obj, scene

CONFIG = load_config()
RULES = ["spill_risk", "dense_region", "overlap_cluster", "keep_clear_zone", "cable_congestion"]
ARMED = ["spill_risk", "dense_region", "overlap_cluster", "keep_clear_zone"]


def run(model: str = "yolox_s", role: str = "deep", status: str = "ok", **extra: Any) -> dict[str, Any]:
    return {"model": model, "role": role, "status": status, "boxes": 4, "ms": 500, "input_size": 640 if role == "deep" else 320,
            "passes": 1, "tile_px": None, "vocabulary": 80, "note": None, **extra}


FAST_OK = run("efficientdet_lite0", "fast")
DEEP_OK = run()
COVERED = {"edge_density": 0.2, "unexplained_share": 0.3, "box_coverage": 0.5}


def still(*objects: dict[str, Any], runs: list[dict[str, Any]] | None = None, coverage: dict[str, Any] | None = COVERED,
          width: int = 640, height: int = 480, **stats: Any) -> dict[str, Any]:
    runs = [FAST_OK, DEEP_OK] if runs is None else runs
    body = scene(*objects, width=width, height=height)
    body["stats"] = {"detectors": [r["model"] for r in runs if r["status"] == "ok"], "runs": runs, "coverage": coverage, **stats}
    return body


def assess(body: dict[str, Any] | None, findings: int = 0, mode: AnalysisMode = AnalysisMode.IMAGE):
    model = SceneModel.model_validate(body) if body is not None else None
    return inspection.assess(model, mode=mode, config=CONFIG, findings=findings, vocabulary_size=80, armed_rules=ARMED, all_rules=RULES)


def codes(report) -> set[str]:
    return {r.code for r in report.reasons}


DESK = [LAPTOP, obj("k", "keyboard", 0.3, 0.75, 0.3, 0.1), obj("m", "mouse", 0.65, 0.78, 0.05, 0.05), obj("b", "bottle", 0.8, 0.3, 0.05, 0.2)]


# -- assess(): what was examined -----------------------------------------------------------


def test_no_scene_means_nothing_was_examined() -> None:
    report = assess(None)
    assert report.analysis_status == "detection_failed" and report.coverage == "insufficient"
    assert report.detection == "not_run" and codes(report) == {"detection_not_run"}
    assert not report.score_rated


def test_failed_detectors_are_never_a_clean_result() -> None:
    body = still(runs=[run("efficientdet_lite0", "fast", "failed", note="WebAssembly unavailable"), run(status="unavailable", note="model download failed")])
    report = assess(body)
    assert report.detection == "failed" and report.analysis_status == "detection_failed" and report.coverage == "insufficient"
    message = next(r.message for r in report.reasons if r.code == "detection_failed")
    assert "WebAssembly unavailable" in message and "model download failed" in message
    assert not report.score_rated


def test_one_failed_detector_limits_the_inspection() -> None:
    report = assess(still(*DESK, runs=[FAST_OK, run(status="failed", note="out of memory")]))
    assert report.detection == "partial" and report.coverage == "limited"
    assert {"detector_not_used", "fast_detector_only"} <= codes(report)


def test_downscaled_large_photo_is_reported_with_the_scale() -> None:
    # A 12-megapixel phone photo letterboxed into the 640 px deep input: 16% of its resolution.
    report = assess(still(*DESK, width=4032, height=3024))
    reason = next(r for r in report.reasons if r.code == "downscaled")
    assert reason.evidence["effective_scale"] == 0.159
    # Tiles help, but 3 x 3 tiles of 1551 px are still below half resolution.
    tiled = assess(still(*DESK, width=4032, height=3024, runs=[FAST_OK, run(passes=10, tile_px=1551)]))
    assert next(r for r in tiled.reasons if r.code == "downscaled").evidence["effective_scale"] == 0.413
    # A 1280 px image in 640 px tiles is seen at full detail.
    full = assess(still(*DESK, width=1280, height=960, runs=[FAST_OK, run(passes=7, tile_px=640)]))
    assert "downscaled" not in codes(full)
    assert "tiled high-resolution pass" not in [s.analysis for s in full.skipped]
    assert "tiled high-resolution pass" in [s.analysis for s in report.skipped]


def test_nothing_recognised_in_a_detailed_image_is_inconclusive() -> None:
    report = assess(still(coverage={"edge_density": 0.25, "unexplained_share": 1.0, "box_coverage": 0.0}))
    assert report.coverage == "insufficient" and report.analysis_status == "inconclusive"
    assert "nothing_recognised" in codes(report)
    # A blank wall: nothing recognised and nothing there - limited (closed vocabulary), but not inconclusive.
    blank = assess(still(coverage={"edge_density": 0.004, "unexplained_share": 1.0, "box_coverage": 0.0}))
    assert blank.coverage == "limited" and "nothing_recognised" not in codes(blank)
    assert not blank.score_rated  # nothing to rate


def test_unexplained_structure_and_low_confidence() -> None:
    busy = assess(still(*DESK, coverage={"edge_density": 0.2, "unexplained_share": 0.71, "box_coverage": 0.3}))
    assert next(r for r in busy.reasons if r.code == "unexplained_structure").evidence["unexplained_share"] == 0.71
    weak = assess(still(*[{**o, "confidence": 0.35} for o in DESK[:3]], DESK[3]))
    assert "low_confidence" in codes(weak) and weak.confidence["below_threshold"] == 3
    assert "low_confidence" not in codes(assess(still(*DESK)))


def test_closed_vocabulary_always_limits_a_detection_scan() -> None:
    report = assess(still(*DESK))
    assert "closed_vocabulary" in codes(report) and report.coverage == "limited"
    assert report.categories == ["bottle", "keyboard", "laptop", "mouse"] and report.objects == 4
    assert "80 object categories" in report.vocabulary
    assert not report.score_rated


def test_capped_objects_incomplete_tiles_old_clients_and_dark_images() -> None:
    assert "objects_capped" in codes(assess(still(*DESK, objects_total=112)))
    incomplete = assess(still(*DESK, width=1920, height=1440, runs=[FAST_OK, run(passes=6, tile_px=739, incomplete=True, note="time budget reached")]))
    assert "tiling_incomplete" in codes(incomplete)
    old = scene(*DESK)  # names only, no run reports, no structure measure
    assert {"runs_unreported", "structure_unmeasured"} <= codes(assess(old))
    dark = still(*DESK)
    dark["signals"] = {"brightness": 0.05, "sharpness": 0.5}
    assert "too_dark" in codes(assess(dark))


def test_checks_and_skipped_analyses_are_listed() -> None:
    report = assess(still(*DESK))
    assert "keep_clear_zone" not in report.checks_run  # no zone was defined
    skipped = {s.analysis for s in report.skipped}
    assert {"cable and power-strip analysis", "segmentation", "open-vocabulary recognition"} <= skipped
    zoned = still(*DESK)
    zoned["zones"] = [{"id": "z1", "box": {"x": 0.0, "y": 0.0, "w": 0.2, "h": 0.2}}]
    assert "keep_clear_zone" in assess(zoned).checks_run


# -- scores and status ---------------------------------------------------------------------


def test_empty_findings_never_produce_a_perfect_score(client: TestClient) -> None:
    # The reported failure: a cluttered desk photo, four objects recognised, no findings.
    body = still(*DESK, width=4032, height=3024, coverage={"edge_density": 0.18, "unexplained_share": 0.71, "box_coverage": 0.2})
    res = client.post("/api/analyze/scene", json={"personality": "serious", "scene": body})
    assert res.status_code == 200, res.text
    report = res.json()
    assert report["findings"] == []
    assert report["system_score"] is None and report["issue_score"] is None
    assert report["status"] == "LIMITED"
    assert report["inspection"]["findings"] == "no_findings" and report["inspection"]["analysis_status"] == "limited"
    assert {"downscaled", "unexplained_structure", "closed_vocabulary"} <= {r["code"] for r in report["inspection"]["reasons"]}
    assert "not an all-clear" in report["final_diagnosis"]
    assert "No measurable issues" not in report["final_diagnosis"]


def test_detector_failure_reports_detection_failed(client: TestClient) -> None:
    body = still(runs=[run("efficientdet_lite0", "fast", "failed", note="model failed to load"), run(status="failed", note="worker crashed")],
                 coverage=None)
    report = client.post("/api/analyze/scene", json={"personality": "brutal", "scene": body}).json()
    assert report["status"] == "INCONCLUSIVE" and report["system_score"] is None and report["issue_score"] is None
    assert report["inspection"]["analysis_status"] == "detection_failed"
    assert "No detector produced a result" in report["final_diagnosis"] and "no conclusion is possible" in report["final_diagnosis"]
    assert report["counts"]["active_bugs"] == 0


def test_image_without_detections_is_inconclusive(client: TestClient) -> None:
    res = client.post("/api/analyze/image", files={"image": ("desk.jpg", jpeg_bytes(), "image/jpeg")}, data={"personality": "serious"})
    report = res.json()
    assert report["status"] == "INCONCLUSIVE" and report["system_score"] is None
    assert report["inspection"]["detection"] == "not_run"


def test_findings_still_set_the_status_when_coverage_is_limited(client: TestClient) -> None:
    report = client.post("/api/analyze/scene", json={"personality": "serious", "scene": still(LAPTOP, CUP)}).json()
    assert report["status"] == "DEGRADED"  # the HIGH spill risk is reported whatever the coverage
    assert report["system_score"] is None and 0 <= report["issue_score"] < 100
    assert "other issues may exist" in report["final_diagnosis"]


def test_status_rules(client: TestClient) -> None:
    service = client.app.state.pipeline.diagnostics
    assert service.status_for(None, [], "sufficient") == SystemStatus.STABLE
    assert service.status_for(None, [], "limited") == SystemStatus.LIMITED
    assert service.status_for(None, [], "insufficient") == SystemStatus.INCONCLUSIVE
    assert service.issue_score(None, []) is None


def test_live_scan_without_observations_is_inconclusive(client: TestClient) -> None:
    created = client.post("/api/scan", json={}).json()
    assert created["status"] == "INCONCLUSIVE" and created["system_score"] is None


def test_video_samples_carry_their_detector_runs(client: TestClient) -> None:
    samples = [{"t": t, "objects": DESK, "detectors": ["efficientdet_lite0"], "runs": [FAST_OK], "coverage": COVERED} for t in (0.5, 1.5, 2.5)]
    manifest = {"duration_s": 3, "width": 1280, "height": 720, "samples": samples, "scenes": [{"index": 0, "start_t": 0, "end_t": 3}]}
    res = client.post("/api/analyze/video", data={"personality": "serious", "manifest": __import__("json").dumps(manifest)})
    assert res.status_code == 200, res.text
    inspected = res.json()["report"]["inspection"]
    assert [d["model"] for d in inspected["detectors"]] == ["efficientdet_lite0"]
    assert "fast_detector_only" in {r["code"] for r in inspected["reasons"]}
    assert inspected["structure"]["unexplained_share"] == 0.3


def test_single_photo_findings_need_detections_that_are_more_likely_right_than_wrong(client: TestClient) -> None:
    def analyse(cup: dict[str, Any]) -> dict[str, Any]:
        return client.post("/api/analyze/scene", json={"personality": "serious", "scene": still(LAPTOP, cup)}).json()

    weak = analyse({**CUP, "confidence": 0.35, "source": "deep"})  # deep detections at 0.3-0.4: precision 0.32
    assert not any(f["rule"] == "spill_risk" for f in weak["findings"])
    reason = next(r for r in weak["inspection"]["reasons"] if r["code"] == "weak_evidence")
    assert reason["evidence"] == {"set_aside": 1, "objects": 2}
    assert any(o["label"] == "cup" for o in weak["objects"])  # still shown, just not used as evidence
    assert any(f["rule"] == "spill_risk" for f in analyse({**CUP, "confidence": 0.55, "source": "deep"})["findings"])
    assert any(f["rule"] == "spill_risk" for f in analyse({**CUP, "confidence": 0.45, "source": "fast"})["findings"])  # fast: 0.61


def test_tracked_scenes_keep_low_confidence_objects(client: TestClient) -> None:
    body = {"scan_id": None, "scene": scene(LAPTOP, {**CUP, "confidence": 0.35}, at_ms=0)}
    res = client.post("/api/scan/observe", json=body).json()
    assert any(f["rule"] == "spill_risk" for f in res["scan"]["findings"])  # persistence is the evidence in live scans
