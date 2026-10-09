#!/usr/bin/env python3
"""Promotion gate: may a candidate model (e.g. a fine-tuned detector) replace the current one?

Compares two benchmark records (docs/benchmarks/records/*.json) produced by
tools/evaluate.py on the SAME held-out dataset, split and configuration
name, and refuses promotion unless every criterion holds:

  1. same dataset id, split, image count and evaluated categories;
  2. overall AP improves by at least --min-ap-gain points;
  3. no evaluated class loses more than --max-class-drop AP points
     (an average gain must not hide a regression on one class);
  4. calibration does not get worse by more than --max-ece-increase;
  5. median latency stays within --max-latency-ratio of the baseline's.

    python tools/promote.py docs/benchmarks/records/<baseline>.json <candidate>.json

Exit status 0 = promote, 1 = keep the baseline. The verdict and every check
are printed; nothing is changed automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check(baseline: dict, candidate: dict, *, min_ap_gain: float, max_class_drop: float, max_ece_increase: float,
          max_latency_ratio: float) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    bd, cd = baseline["dataset"], candidate["dataset"]
    same = (bd["id"], bd.get("split"), bd.get("images"), bd.get("evaluated_categories")) == \
           (cd["id"], cd.get("split"), cd.get("images"), cd.get("evaluated_categories"))
    out.append(("same held-out data", same, f"{bd['id']}/{bd.get('split')} ({bd.get('images')} images) vs {cd['id']}/{cd.get('split')} ({cd.get('images')})"))
    b_ap, c_ap = baseline["metrics"]["coco"]["AP"], candidate["metrics"]["coco"]["AP"]
    out.append(("AP gain", c_ap - b_ap >= min_ap_gain, f"{b_ap} -> {c_ap} (need +{min_ap_gain})"))
    drops = []
    for name, stats in baseline.get("per_class", {}).items():
        b, c = stats.get("AP"), candidate.get("per_class", {}).get(name, {}).get("AP")
        if b is not None and c is not None and b - c > max_class_drop:
            drops.append(f"{name} {b}->{c}")
    out.append(("no per-class regression", not drops, ", ".join(drops[:8]) or f"no class drops more than {max_class_drop} AP"))
    b_ece = baseline["metrics"].get("calibration", {}).get("ece")
    c_ece = candidate["metrics"].get("calibration", {}).get("ece")
    if b_ece is not None and c_ece is not None:
        out.append(("calibration", c_ece - b_ece <= max_ece_increase, f"ECE {b_ece} -> {c_ece} (max +{max_ece_increase})"))
    b_ms = (baseline.get("performance", {}).get("model_ms") or {}).get("median")
    c_ms = (candidate.get("performance", {}).get("model_ms") or {}).get("median")
    if b_ms and c_ms:
        out.append(("latency", c_ms <= max_latency_ratio * b_ms, f"median {b_ms} ms -> {c_ms} ms (max x{max_latency_ratio})"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--min-ap-gain", type=float, default=1.0)
    ap.add_argument("--max-class-drop", type=float, default=2.0)
    ap.add_argument("--max-ece-increase", type=float, default=0.02)
    ap.add_argument("--max-latency-ratio", type=float, default=1.5)
    args = ap.parse_args()
    baseline, candidate = (json.loads(Path(p).read_text()) for p in (args.baseline, args.candidate))
    results = check(baseline, candidate, min_ap_gain=args.min_ap_gain, max_class_drop=args.max_class_drop,
                    max_ece_increase=args.max_ece_increase, max_latency_ratio=args.max_latency_ratio)
    for name, ok, detail in results:
        print(f"  [{'pass' if ok else 'FAIL'}] {name}: {detail}")
    verdict = all(ok for _, ok, _ in results)
    print("PROMOTE" if verdict else "KEEP BASELINE")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
