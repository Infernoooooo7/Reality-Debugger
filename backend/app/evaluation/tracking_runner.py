"""Tracking benchmark on KITTI tracking sequences, run through the production tracker.

The tracker is evaluated independently of the detector by feeding it
different inputs with identical tracker settings:
  * oracle  - ground-truth boxes without identities: isolates association;
  * a real detector (the app's fast detector, or YOLOX-S): the full pipeline.
and comparing tracker variants on identical inputs (camera-motion
compensation on/off, ByteTrack's low-score association on/off).

KITTI-style preprocessing (approximates, but is not, the official devkit):
Car is evaluated against tracks labelled "car"; Van boxes are ignore regions.
Pedestrian is evaluated against tracks labelled "person"; Person_sitting and
Cyclist boxes are ignore regions (COCO detectors call a cyclist's rider
"person" and have no cyclist class). DontCare boxes are ignore regions for
both. Tracker boxes mostly inside an ignore region are not counted as false
positives. No minimum-height filter is applied.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.datasets.adapters import KittiTrackingAdapter
from app.evaluation import records
from app.evaluation.tracking import Frame, evaluate
from app.vision.motion import estimate_shift, thumbnail_gray

ROOT_DIR = Path(__file__).resolve().parents[3]
RUNNER = ROOT_DIR / "frontend" / ".tools" / "track-runner.js"
CLASSES = {
    "car": {"gt": {"Car"}, "ignore": {"Van", "DontCare"}, "track_labels": {"car"}},
    "pedestrian": {"gt": {"Pedestrian"}, "ignore": {"Person_sitting", "Cyclist", "DontCare"}, "track_labels": {"person"}},
}
#: KITTI GT classes given to the oracle input, with the COCO label the tracker sees.
ORACLE_LABELS = {"Car": "car", "Pedestrian": "person"}

Detector = Callable[[np.ndarray], list[dict[str, Any]]]  # bgr -> [{label, score, box: (x1,y1,x2,y2) px}]


def ensure_runner() -> Path:
    if not RUNNER.exists():
        subprocess.run(["npm", "run", "build:tools"], cwd=ROOT_DIR / "frontend", check=True, capture_output=True)
    return RUNNER


def camera_motion(frames: list[Path]) -> list[dict[str, float] | None]:
    out: list[dict[str, float] | None] = [None]
    prev = thumbnail_gray(cv2.imread(str(frames[0])))
    for p in frames[1:]:
        cur = thumbnail_gray(cv2.imread(str(p)))
        dx, dy, conf = estimate_shift(prev, cur)
        out.append({"dx": dx, "dy": dy, "confidence": conf})
        prev = cur
    return out


def run_tracker(sequences: list[dict[str, Any]], options: dict[str, Any], work: Path, tag: str) -> list[dict[str, Any]]:
    runner = ensure_runner()
    work.mkdir(parents=True, exist_ok=True)
    inp, out = work / f"{tag}.in.json", work / f"{tag}.out.json"
    inp.write_text(json.dumps({"options": options, "sequences": sequences}))
    subprocess.run(["node", str(runner), str(inp), str(out)], check=True)
    return json.loads(out.read_text())["sequences"]


def frame_size(adapter: KittiTrackingAdapter, seq: str) -> tuple[int, int]:
    h, w = cv2.imread(str(adapter.frames(seq)[0])).shape[:2]
    return w, h


def score(adapter: KittiTrackingAdapter, seq_ids: list[str], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    by_seq = {s["id"]: s for s in outputs}
    result: dict[str, Any] = {}
    for cname, spec in CLASSES.items():
        all_frames: list[Frame] = []
        per_seq = {}
        for k, seq in enumerate(seq_ids):
            labels = adapter.labels(seq)
            n = len(adapter.frames(seq))
            width, height = frame_size(adapter, seq)
            frames = []
            for i in range(n):
                anns = labels.get(i, [])
                gt = [(a["track_id"] + 100000 * k, tuple(a["box"])) for a in anns if a["category"] in spec["gt"]]
                ignore = [tuple(a["box"]) for a in anns if a["category"] in spec["ignore"]]
                tracks = []
                for t in by_seq[seq]["frames"][i]["tracks"]:
                    if t["label"] not in spec["track_labels"]:
                        continue
                    b = t["box"]
                    tracks.append((t["id"] + 100000 * k, (b["x"] * width, b["y"] * height, (b["x"] + b["w"]) * width, (b["y"] + b["h"]) * height)))
                frames.append(Frame(gt, tracks, ignore))
            per_seq[seq] = {k2: v for k2, v in evaluate(frames).items() if k2 in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "num_switches")}
            all_frames.extend(frames)
        result[cname] = {"combined": evaluate(all_frames), "per_sequence": per_seq}
    return result


def detections_for(adapter: KittiTrackingAdapter, seq: str, detector: Detector | None, cache: Path, tag: str) -> list[list[dict[str, Any]]]:
    """Per-frame detections in normalised boxes, cached on disk."""
    path = cache / f"{tag}__{seq}.json"
    if path.exists():
        return json.loads(path.read_text())
    frames = adapter.frames(seq)
    out = []
    if detector is None:  # oracle
        labels = adapter.labels(seq)
        w, h = frame_size(adapter, seq)
        for i in range(len(frames)):
            dets = []
            for a in labels.get(i, []):
                if a["category"] in ORACLE_LABELS:
                    x1, y1, x2, y2 = a["box"]
                    dets.append({"label": ORACLE_LABELS[a["category"]], "score": 0.9,
                                 "box": {"x": x1 / w, "y": y1 / h, "w": (x2 - x1) / w, "h": (y2 - y1) / h}})
            out.append(dets)
    else:
        for p in frames:
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            out.append([{"label": d["label"], "score": round(d["score"], 4),
                         "box": {"x": d["box"][0] / w, "y": d["box"][1] / h, "w": (d["box"][2] - d["box"][0]) / w, "h": (d["box"][3] - d["box"][1]) / h}}
                        for d in detector(img)])
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def run_kitti(detectors: dict[str, Detector | None], variants: dict[str, dict[str, Any]], work: Path) -> list[Path]:
    adapter = KittiTrackingAdapter()
    seq_ids = adapter.sequences()
    t0 = time.perf_counter()
    motion = {seq: camera_motion(adapter.frames(seq)) for seq in seq_ids}
    motion_s = time.perf_counter() - t0
    conf = [m["confidence"] for seq in seq_ids for m in motion[seq] if m]
    written = []
    for det_name, detector in detectors.items():
        t1 = time.perf_counter()
        dets = {seq: detections_for(adapter, seq, detector, work / "detections", det_name) for seq in seq_ids}
        det_s = time.perf_counter() - t1
        for var_name, options in variants.items():
            sequences = [{"id": seq, "frames": [{"t_ms": i * 1000.0 / adapter.fps, "dets": dets[seq][i], "camera": motion[seq][i]}
                                                  for i in range(len(dets[seq]))]} for seq in seq_ids]
            t2 = time.perf_counter()
            outputs = run_tracker(sequences, options, work / "runs", f"{det_name}__{var_name}")
            track_s = time.perf_counter() - t2
            metrics = score(adapter, seq_ids, outputs)
            n_frames = sum(len(s["frames"]) for s in sequences)
            record = {
                "task": "tracking",
                "date": records.now(),
                "model": {"id": f"bytetrack-ts+{det_name}", "name": f"Reality Debugger ByteTrack tracker ({var_name}) on {det_name} detections",
                          "version": records.git_commit()},
                "dataset": {"id": "kitti_tracking", "split": "training (selected sequences)", "sequences": seq_ids, "frames": n_frames,
                            "fps": adapter.fps, "image_sizes": {seq: list(frame_size(adapter, seq)) for seq in seq_ids}},
                "config_name": f"{det_name}-{var_name}",
                "config": {"tracker_options": options, "input": det_name},
                "environment": records.environment(),
                "performance": {"tracker_ms_per_frame": round(1000 * track_s / max(1, n_frames), 3),
                                "detector_s_total": round(det_s, 1), "camera_motion_ms_per_frame": round(1000 * motion_s / max(1, n_frames), 2),
                                "note": "tracker time includes Node start-up and JSON I/O"},
                "metrics": metrics,
                "camera_motion": {"frames": len(conf), "median_confidence": round(float(np.median(conf)), 3) if conf else None,
                                  "share_used": round(float(np.mean(np.asarray(conf) >= options.get("cmcMinConfidence", 0.25))), 3) if conf else None},
                "notes": [__doc__.split("\n\n", 2)[2].strip()],
            }
            written.append(records.write(record))
            c, p = metrics["car"]["combined"], metrics["pedestrian"]["combined"]
            print(f"  {det_name:10} {var_name:18} car HOTA {c['HOTA']:5.1f} IDF1 {c['IDF1']:5.1f} MOTA {c['MOTA']:6.1f} IDSW {c['num_switches']:3d} | "
                  f"ped HOTA {p['HOTA']:5.1f} IDF1 {p['IDF1']:5.1f} MOTA {p['MOTA']:6.1f} IDSW {p['num_switches']:3d}", flush=True)
    return written
