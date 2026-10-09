"""Vision core and evaluation metrics, checked against analytic cases and the TypeScript twin.

Tests that need optional evaluation packages (scipy, motmetrics) are skipped
when those are not installed (they are in tools/requirements-eval.txt).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from pytest import approx

from app.evaluation import anomaly as anomaly_eval
from app.vision import motion, nms, tiling
from app.vision.anomaly import _upsample, anomaly_regions, greedy_coreset, nearest_distance
from app.vision.types import AnalysisState, Detection, ImageInfo, InferenceResult, mask_to_rle, rle_to_mask

ROOT = Path(__file__).resolve().parents[2]

# -- camera motion: parity with frontend/src/vision/__tests__/motion.test.ts ----------------------

#: shift in thumbnail pixels -> (dx, dy) in frame fractions, as computed by both implementations.
MOTION_PARITY = {
    (3, -2): (0.02343547491914607, -0.02084305726970272),
    (-7, 4): (-0.054689174747940314, 0.04166071550341955),
    (10, 0): (0.07812596782956774, 4.818206770395917e-06),
    (-12, -9): (-0.09376030495359981, -0.09376069019985761),
}


def test_motion_test_pattern_matches_typescript() -> None:
    p = motion.test_pattern()
    assert p.shape == (96, 128)
    assert p[10, 10:14].tolist() == approx([0.450545, 0.432244, 0.51024, 0.484096], abs=1e-6)
    assert float(p.mean()) == approx(0.498205, abs=1e-6)


@pytest.mark.parametrize(("shift", "expected"), list(MOTION_PARITY.items()))
def test_motion_estimate_matches_typescript(shift: tuple[int, int], expected: tuple[float, float]) -> None:
    dx, dy, confidence = motion.estimate_shift(motion.test_pattern(), motion.test_pattern(shift_x=shift[0], shift_y=shift[1]))
    assert (dx, dy) == approx(expected, abs=1e-9)
    assert dx * 128 == approx(shift[0], abs=0.05) and dy * 96 == approx(shift[1], abs=0.05)
    assert confidence > 0.5


def test_motion_flat_image_has_no_confidence() -> None:
    flat = np.full((96, 128), 0.5)
    assert motion.estimate_shift(flat, flat)[2] == 0.0


# -- suppression and merging ----------------------------------------------------------------------


def test_nms_keeps_best_of_overlapping_and_disjoint_boxes() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], float)
    keep = nms.nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert keep.tolist() == [0, 2]


def test_batched_nms_is_class_aware_unless_agnostic() -> None:
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float)
    scores, classes = np.array([0.9, 0.8]), np.array([0, 1])
    assert len(nms.batched_nms(boxes, scores, classes, 0.5, class_agnostic=False)) == 2
    assert len(nms.batched_nms(boxes, scores, classes, 0.5, class_agnostic=True)) == 1


def test_greedy_nmm_reassembles_an_object_cut_at_a_tile_border() -> None:
    # Left and right halves of one object (different tiles) plus the full-image detection.
    boxes = np.array([[100, 50, 160, 120], [150, 50, 200, 120], [102, 52, 198, 118]], float)
    b, s, c = nms.greedy_nmm(boxes, np.array([0.6, 0.7, 0.9]), np.array([2, 2, 2]), metric="ios", threshold=0.5)
    assert len(b) == 1
    assert b[0].tolist() == [100, 50, 200, 120]
    assert s[0] == 0.9


# -- sliced inference ------------------------------------------------------------------------------


def test_tile_windows_cover_the_image_with_the_requested_overlap() -> None:
    windows = tiling.tile_windows(1500, 900, 640, 0.2)
    covered = np.zeros((900, 1500), bool)
    for x1, y1, x2, y2 in windows:
        assert x2 - x1 <= 640 and y2 - y1 <= 640
        covered[y1:y2, x1:x2] = True
    assert covered.all()
    xs = sorted({w[0] for w in windows})
    assert xs[-1] + 640 == 1500  # last column aligned to the border
    assert all(b - a <= 640 * 0.8 + 1 for a, b in zip(xs, xs[1:], strict=False))
    assert tiling.tile_windows(500, 400, 640, 0.2) == [(0, 0, 500, 400)]


def _fake_detector(object_box: tuple[int, int, int, int]):
    """Detects the visible part of one object (class 0) in whatever crop it is given."""

    def detect(crop: np.ndarray) -> InferenceResult:
        # The crop's origin is recovered from a coordinate image (channel 0 = x, channel 1 = y).
        ox, oy = int(crop[0, 0, 0]), int(crop[0, 0, 1])
        h, w = crop.shape[:2]
        x1, y1, x2, y2 = object_box
        vx1, vy1, vx2, vy2 = max(x1, ox), max(y1, oy), min(x2, ox + w), min(y2, oy + h)
        dets = []
        if vx2 > vx1 and vy2 > vy1:
            dets.append(Detection(0, "thing", 0.8, (vx1 - ox, vy1 - oy, vx2 - ox, vy2 - oy)))
        return InferenceResult("fake", "1", ImageInfo(w, h), dets, config={}, timings_ms={})

    return detect


def _coordinate_image(w: int, h: int) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w]
    return np.dstack([xs, ys, np.zeros_like(xs)]).astype(np.int32)


def test_sliced_detect_merges_partial_detections_from_neighbouring_tiles() -> None:
    image = _coordinate_image(1200, 600)
    obj = (500, 200, 700, 300)  # straddles the first tile border (tile 640)
    cfg = tiling.TilingConfig(tile=640, overlap=0.2, full_image=False)
    result = tiling.sliced_detect(_fake_detector(obj), image, cfg, model_id="fake", model_version="1")
    assert result.state == AnalysisState.COMPLETE
    assert [d.box for d in result.detections] == [obj]


def test_sliced_detect_reports_incomplete_analysis_when_out_of_time() -> None:
    image = _coordinate_image(2000, 2000)
    cfg = tiling.TilingConfig(tile=320, overlap=0.2, full_image=True, time_budget_ms=0.0)
    result = tiling.sliced_detect(_fake_detector((10, 10, 50, 50)), image, cfg, model_id="fake", model_version="1")
    assert result.state == AnalysisState.INCOMPLETE
    assert result.notes and "time budget" in result.notes[0]


def test_mask_rle_round_trip() -> None:
    rng = np.random.default_rng(1)
    for first in (0, 1):
        mask = rng.random((7, 9)) > 0.5
        mask[0, 0] = bool(first)
        assert (rle_to_mask(mask_to_rle(mask)) == mask).all()


# -- anomaly detection ----------------------------------------------------------------------------


def test_upsample_matches_opencv_bilinear() -> None:
    f = np.random.default_rng(2).random((1, 3, 15, 21)).astype(np.float32)
    ours = _upsample(f, 29, 42)[0].transpose(1, 2, 0)
    ref = cv2.resize(np.ascontiguousarray(f[0].transpose(1, 2, 0)), (42, 29), interpolation=cv2.INTER_LINEAR)
    assert np.abs(ours - ref).max() < 1e-5


def test_greedy_coreset_covers_every_cluster() -> None:
    rng = np.random.default_rng(3)
    centres = rng.normal(0, 50, (8, 32)).astype(np.float32)
    points = np.concatenate([c + rng.normal(0, 0.5, (200, 32)).astype(np.float32) for c in centres])
    idx = greedy_coreset(points, 8, projection_dim=16)
    assert len(set(idx.tolist())) == 8
    assert sorted({int(i) // 200 for i in idx}) == list(range(8))


def test_nearest_distance_matches_brute_force() -> None:
    rng = np.random.default_rng(4)
    q, bank = rng.random((50, 16)).astype(np.float32), rng.random((70, 16)).astype(np.float32)
    brute = np.sqrt(((q[:, None] - bank[None]) ** 2).sum(-1)).min(1)
    assert nearest_distance(q, bank, chunk=7) == approx(brute, abs=1e-4)


def test_auroc_and_average_precision_on_analytic_cases() -> None:
    labels = np.array([0, 0, 1, 1], bool)
    assert anomaly_eval.auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert anomaly_eval.auroc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
    assert anomaly_eval.auroc(np.ones(4), labels) == 0.5
    # ranking: pos, neg, pos, neg -> precision 1 at recall 0.5, 2/3 at recall 1
    ap = anomaly_eval.average_precision(np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 0, 1, 0], bool))
    assert ap == approx(0.5 * 1 + 0.5 * 2 / 3)


def test_aupro_on_analytic_cases() -> None:
    masks = np.zeros((2, 20, 20), bool)
    masks[0, 2:6, 2:6] = True
    masks[0, 10:12, 10:18] = True
    perfect = masks.astype(np.float64)
    assert anomaly_eval.aupro(perfect, masks) == approx(1.0)
    # A constant map: PRO rises with FPR along the diagonal -> area 0.3^2/2, normalised by 0.3.
    assert anomaly_eval.aupro(np.zeros_like(perfect), masks) == approx(0.15)


def test_anomaly_regions_finds_the_hot_spot() -> None:
    heat = np.zeros((100, 100), np.float32)
    heat[40:50, 60:75] = 5.0
    regions = anomaly_regions(heat, threshold=1.0)
    assert regions == [{"box": (60, 40, 75, 50), "area_px": 150, "peak_score": 5.0}]


# -- tracking metrics (optional packages) ---------------------------------------------------------


def _frames(n: int, swap_at: int | None = None):
    from app.evaluation.tracking import Frame

    frames = []
    for t in range(n):
        a, b = (0, 0, 10, 10), (50, 0, 60, 10)
        ids = (1, 2) if swap_at is None or t < swap_at else (2, 1)
        frames.append(Frame(gt=[(1, a), (2, b)], tracks=[(ids[0], a), (ids[1], b)], ignore=[]))
    return frames


def test_tracking_metrics_perfect_and_swapped_identities() -> None:
    pytest.importorskip("scipy")
    pytest.importorskip("motmetrics")
    from app.evaluation.tracking import evaluate

    perfect = evaluate(_frames(10))
    assert perfect["HOTA"] == 100.0 and perfect["IDF1"] == 100.0 and perfect["MOTA"] == 100.0 and perfect["num_switches"] == 0
    swapped = evaluate(_frames(10, swap_at=5))
    assert swapped["DetA"] == 100.0  # detection is perfect, association is not
    assert swapped["AssA"] == approx(100 / 3, abs=0.01)  # each matched pair: 5 TPA / (10 + 10 - 5)
    assert swapped["num_switches"] == 2 and swapped["IDF1"] == 50.0


# -- model registry integrity ---------------------------------------------------------------------

STATUSES = {"bundled", "fetchable", "buildable", "deferred", "unavailable", "rejected"}


def test_model_registry_entries_are_complete() -> None:
    entries = sorted((ROOT / "models" / "registry").glob("*.json"))
    assert len(entries) >= 10
    for path in entries:
        e = json.loads(path.read_text())
        assert e["id"] == path.stem
        assert e["status"] in STATUSES, path.name
        assert e["licence"]["weights"] and e.get("tasks"), path.name
        if e["status"] in ("unavailable", "deferred", "rejected"):
            assert e.get("unavailable_reason"), f"{path.name}: say why it is not used"
        if e["status"] in ("bundled", "fetchable"):
            assert e["weights"]["url"].startswith("https://"), path.name
            sha = e["weights"].get("sha256")
            assert sha is None or (len(sha) == 64 and all(ch in "0123456789abcdef" for ch in sha)), path.name
