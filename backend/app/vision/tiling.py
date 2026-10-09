"""Sliced (tiled) inference for small objects, after SAHI (Akyon et al., ICIP 2022).

A detector with a fixed input size (640 for YOLOX-S) shrinks a 1920x1080 image
by 3x, so a 20 px object becomes ~7 px and is lost. Sliced inference runs the
detector on overlapping tiles at (close to) native resolution, maps every box
back to the original image, optionally adds a full-image pass for large
objects, and merges duplicates across tile borders with greedy non-maximum
merging (IoS) or NMS.

What it does NOT do: invent detail. Tiling only avoids throwing away pixels
the camera captured; objects below the sensor's effective resolution stay
undetectable (docs/LIMITATIONS_AND_RISKS.md).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from app.vision.nms import batched_nms, greedy_nmm
from app.vision.types import AnalysisState, Detection, ImageInfo, InferenceResult


@dataclass(slots=True)
class TilingConfig:
    tile: int = 640  # tile side in source pixels
    overlap: float = 0.2  # fraction of the tile shared with its neighbour
    full_image: bool = True  # also run the detector on the whole (downscaled) image
    merge: str = "nmm"  # "nmm" (greedy merging, SAHI default) or "nms"
    match_metric: str = "ios"  # "ios" (intersection over smaller box) or "iou"
    match_threshold: float = 0.5
    class_agnostic: bool = False
    min_image_side_for_tiling: int = 0  # images with both sides <= tile are not tiled when 0 (no gain)
    max_tiles: int = 64  # time/compute guard; if exceeded the tile size grows to fit
    time_budget_ms: float | None = None  # stop early (state: analysis_incomplete) when exceeded


def tile_windows(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping windows (x1, y1, x2, y2) covering the image; the last row/column is aligned to the border."""
    if width <= tile and height <= tile:
        return [(0, 0, width, height)]
    step = max(1, int(round(tile * (1 - overlap))))

    def starts(size: int) -> list[int]:
        if size <= tile:
            return [0]
        s = list(range(0, size - tile, step))
        s.append(size - tile)  # align the last window to the border
        return sorted(set(s))

    return [(x, y, min(width, x + tile), min(height, y + tile)) for y in starts(height) for x in starts(width)]


def sliced_detect(detect: Callable[[np.ndarray], InferenceResult], image_bgr: np.ndarray, config: TilingConfig,
                  *, model_id: str, model_version: str) -> InferenceResult:
    h, w = image_bgr.shape[:2]
    t0 = time.perf_counter()
    tile = config.tile
    windows = tile_windows(w, h, tile, config.overlap)
    while len(windows) > config.max_tiles:
        tile = int(tile * 1.25)
        windows = tile_windows(w, h, tile, config.overlap)
    tiled = len(windows) > 1 and max(w, h) > config.min_image_side_for_tiling
    if not tiled:
        # Nothing to slice: the single pass is already non-maximum-suppressed; merging it again
        # would fuse distinct neighbouring objects of the same class.
        result = detect(image_bgr)
        for d in result.detections:
            d.extra.setdefault("source", "full")
        result.config = {**result.config, "tiling": {**asdict(config), "effective_tile": tile, "windows": 0, "passes": 1}}
        return result
    boxes, scores, classes, names, origins, sources = [], [], [], [], [], []
    state = AnalysisState.COMPLETE
    notes: list[str] = []
    passes = 0

    def collect(result: InferenceResult, ox: int, oy: int, origin: str) -> None:
        for d in result.detections:
            x1, y1, x2, y2 = d.box
            boxes.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy))
            scores.append(d.confidence)
            classes.append(d.class_id)
            names.append(d.class_name)
            origins.append(origin)
            sources.append(d)

    if config.full_image or not tiled:
        collect(detect(image_bgr), 0, 0, "full")
        passes += 1
    if tiled:
        for i, (x1, y1, x2, y2) in enumerate(windows):
            if config.time_budget_ms is not None and (time.perf_counter() - t0) * 1000 > config.time_budget_ms:
                state = AnalysisState.INCOMPLETE
                notes.append(f"time budget reached after {i} of {len(windows)} tiles; uncovered regions were not analysed at full resolution")
                break
            collect(detect(image_bgr[y1:y2, x1:x2]), x1, y1, f"tile{i}")
            passes += 1
    t1 = time.perf_counter()
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    s = np.asarray(scores, dtype=np.float64)
    c = np.asarray(classes, dtype=np.int64)
    if config.merge == "nms":
        lead = batched_nms(b, s, c, config.match_threshold, config.class_agnostic)
        mb, ms, mc = b[lead], s[lead], c[lead]
    else:
        mb, ms, mc, lead = greedy_nmm(b, s, c, metric=config.match_metric, threshold=config.match_threshold,
                                      class_agnostic=config.class_agnostic, return_index=True)
    name_of = dict(zip(classes, names, strict=True))
    dets = [Detection(class_id=int(ci), class_name=name_of[int(ci)], confidence=float(si),
                      box=(max(0.0, float(bb[0])), max(0.0, float(bb[1])), min(float(w), float(bb[2])), min(float(h), float(bb[3]))),
                      uncertainty=sources[int(li)].uncertainty,
                      extra={**sources[int(li)].extra, "source": origins[int(li)]})
            for bb, si, ci, li in zip(mb, ms, mc, lead, strict=True)]
    t2 = time.perf_counter()
    return InferenceResult(
        model_id=model_id, model_version=model_version, image=ImageInfo(w, h), detections=dets,
        config={**asdict(config), "effective_tile": tile, "windows": len(windows) if tiled else 0, "passes": passes},
        timings_ms={"detector_passes": (t1 - t0) * 1000, "merge": (t2 - t1) * 1000, "total": (t2 - t0) * 1000},
        state=state, notes=notes,
    )
