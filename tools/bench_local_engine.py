#!/usr/bin/env python3
"""Measure the backend's local diagnostic pipeline (no AI, no network).

Runs the FastAPI app in-process with AI_PROVIDER=none and times:

  * the rule engine alone (the backend's own metrics, GET /api/metrics);
  * POST /api/analyze/scene (validation + relations + rules + report);
  * POST /api/scan/observe (the same plus the finding lifecycle of a session);

for synthetic scene models with 2, 10 and 25 objects (random boxes, labels
from the COCO vocabulary). Also records the scene-model payload size, which is
what Live Scan sends every 1.5 s instead of an image.

Usage (from the repository root, with the backend's virtualenv):
    backend/venv/bin/python tools/bench_local_engine.py --out docs/benchmarks/backend_local_engine.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

LABELS = ["cup", "laptop", "keyboard", "mouse", "cell phone", "book", "bottle", "scissors", "tv", "remote",
          "bowl", "banana", "chair", "dining table", "potted plant"]


def scene(n: int, rng: random.Random) -> dict:
    objects = []
    for i in range(n):
        w, h = rng.uniform(0.05, 0.25), rng.uniform(0.05, 0.25)
        objects.append({
            "id": f"t{i}", "label": LABELS[i % len(LABELS)], "confidence": round(rng.uniform(0.35, 0.95), 2),
            "box": {"x": round(rng.uniform(0, 1 - w), 3), "y": round(rng.uniform(0, 1 - h), 3), "w": round(w, 3), "h": round(h, 3)},
            "source": "fast", "age_ms": 5000, "persistent": True, "movement": "static", "speed": 0.01, "static_ms": 5000,
        })
    return {"version": 2, "at_ms": 0, "view_id": 0, "width": 1280, "height": 720, "objects": objects,
            "signals": {"motion": 0.01, "brightness": 0.5, "sharpness": 0.6, "scene_change": 0.02},
            "stats": {"fps": 10, "tentative_tracks": 0, "mean_confidence": 0.7, "detectors": ["efficientdet_lite0"]}}


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return round(s[min(len(s) - 1, round(p * (len(s) - 1)))], 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    settings = Settings(_env_file=None, ai_provider="none", log_level="WARNING")  # type: ignore[call-arg]
    rng = random.Random(1)
    results = []
    for n in (2, 10, 25):
        sc = scene(n, rng)
        with TestClient(create_app(settings)) as client:
            body = json.dumps({"scene": sc})
            analyze = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                res = client.post("/api/analyze/scene", content=body, headers={"content-type": "application/json"})
                analyze.append((time.perf_counter() - t0) * 1000)
                res.raise_for_status()
            engine = client.get("/api/metrics").json()["local"]["evaluations"]
            scan_id = client.post("/api/scan", json={}).json()["scan_id"]
            observe = []
            for k in range(args.runs // 2):
                sc["at_ms"] = k * 1500
                payload = json.dumps({"scan_id": scan_id, "scene": sc})
                t0 = time.perf_counter()
                res = client.post("/api/scan/observe", content=payload, headers={"content-type": "application/json"})
                observe.append((time.perf_counter() - t0) * 1000)
                res.raise_for_status()
            findings = len(res.json()["scan"]["findings"])
        row = {
            "objects": n,
            "scene_json_bytes": len(json.dumps(sc)),
            "rule_engine_ms": {"p50": engine["p50_ms"], "p95": engine["p95_ms"]},
            "analyze_scene_ms": {"p50": round(statistics.median(analyze), 2), "p95": pct(analyze, 0.95)},
            "observe_ms": {"p50": round(statistics.median(observe), 2), "p95": pct(observe, 0.95)},
            "findings": findings,
        }
        results.append(row)
        print(json.dumps(row))

    report = {
        "description": "Backend local diagnostic pipeline, in-process (FastAPI TestClient), AI_PROVIDER=none. Synthetic scenes (random boxes, COCO labels). Request timings include JSON parsing, validation and response serialisation but no network.",
        "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs, Python {platform.python_version()}",
        "runs_per_size": args.runs,
        "results": results,
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
