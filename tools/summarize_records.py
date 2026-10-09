#!/usr/bin/env python3
"""Write docs/benchmarks/SUMMARY.md from the benchmark records (never by hand).

Every number in the summary comes from a record in docs/benchmarks/records/
with its dataset, configuration, environment and commit; evaluations without
a record are listed as "not measured".

    python tools/summarize_records.py           (re)write docs/benchmarks/SUMMARY.md
    python tools/summarize_records.py --stdout  print instead
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "docs" / "benchmarks" / "records"
OUT = ROOT / "docs" / "benchmarks" / "SUMMARY.md"

MODEL_ORDER = ["efficientdet_lite0", "efficientdet_lite2", "yolox_nano", "yolox_tiny", "yolox_s", "yolox_m", "yolox_l"]


def load() -> list[tuple[str, dict[str, Any]]]:
    return [(p.name, json.loads(p.read_text())) for p in sorted(RECORDS.glob("*.json"))]


def table(header: list[str], rows: list[list[Any]]) -> str:
    def fmt(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.2f}" if abs(v) < 1000 else f"{v:.0f}"
        return str(v)

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def detection(records: list[tuple[str, dict]]) -> str:
    rows = []
    recs = [(n, r) for n, r in records if r["task"] == "detection" and r["dataset"]["id"] == "coco_val2017"]
    recs.sort(key=lambda x: (MODEL_ORDER.index(x[1]["model"]["id"]) if x[1]["model"]["id"] in MODEL_ORDER else 99, x[1]["config_name"]))
    for name, r in recs:
        c, op = r["metrics"]["coco"], r["metrics"].get("at_operating_threshold") or {}
        perf = r.get("performance", {})
        rows.append([r["model"]["id"], r["config_name"], c["AP"], c["AP50"], c["APs"], c["APm"], c["APl"],
                     op.get("precision"), op.get("recall"), r["metrics"]["calibration"].get("ece"),
                     (perf.get("model_ms") or {}).get("median"), f"[record](records/{name})"])
    if not rows:
        return "Not measured.\n"
    out = table(["model", "config", "AP", "AP50", "APs", "APm", "APl", "P@thr", "R@thr", "ECE", "ms (median)", "source"], rows)
    return out + ("\n\nAP is COCO mAP@[.5:.95] in points. P/R at the operating threshold (0.3 for the deep detector and "
                  "EfficientDet). Latency: model forward pass on the evaluation machine's CPU while other jobs shared it - "
                  "indicative only; see the latency section.\n")


def miss_causes(records: list[tuple[str, dict]]) -> str:
    rows = []
    for _, r in records:
        if r["task"] != "detection" or r["dataset"]["id"] != "coco_val2017" or r["config_name"] != "deployed":
            continue
        ea = r.get("error_analysis", {})
        share = ea.get("miss_causes_share", {})
        rows.append([r["model"]["id"], ea.get("missed_total"), share.get("undetected"), share.get("below_threshold"),
                     share.get("poor_localization"), share.get("wrong_class"), share.get("suppressed_by_nms"), share.get("lost_to_neighbour")])
    if not rows:
        return "Not measured.\n"
    rows.sort(key=lambda x: MODEL_ORDER.index(x[0]) if x[0] in MODEL_ORDER else 99)
    return table(["model (deployed config)", "missed objects", "undetected", "below threshold", "poor localisation", "wrong class",
                  "suppressed by NMS", "lost to neighbour"], rows) + "\n"


def tiling(records: list[tuple[str, dict]]) -> str:
    rows = []
    for name, r in records:
        if r["task"] != "detection-tiling":
            continue
        c, op = r["metrics"]["coco"], r["metrics"].get("at_operating_threshold") or {}
        rows.append([r["dataset"]["id"], r["dataset"]["split"], r["config_name"], c["AP"], c["AP50"], c["APs"], c["APm"], c["APl"],
                     op.get("recall"), op.get("precision"), (r.get("performance", {}).get("model_ms") or {}).get("median"),
                     f"[record](records/{name})"])
    if not rows:
        return "Not measured.\n"
    return table(["dataset", "split", "config", "AP", "AP50", "APs", "APm", "APl", "R@thr", "P@thr", "ms (median)", "source"], rows) + "\n"


def robustness(records: list[tuple[str, dict]]) -> str:
    by_model: dict[str, dict[str, float]] = defaultdict(dict)
    for _, r in records:
        if r["task"] == "detection-robustness":
            by_model[r["model"]["id"]][r["config_name"]] = r["metrics"]["coco"]["AP"]
    if not by_model:
        return "Not measured.\n"
    conditions = sorted({c for m in by_model.values() for c in m}, key=lambda c: (c != "clean", c))
    rows = []
    for cond in conditions:
        row: list[Any] = [cond]
        for model in sorted(by_model):
            ap, clean = by_model[model].get(cond), by_model[model].get("clean")
            row += [ap, None if ap is None or not clean else round(100 * ap / clean, 1)]
        rows.append(row)
    header = ["condition"] + [h for m in sorted(by_model) for h in (f"{m} AP", f"{m} % of clean")]
    return table(header, rows) + "\n"


def tracking(records: list[tuple[str, dict]]) -> str:
    rows = []
    for name, r in records:
        if r["task"] != "tracking":
            continue
        for cls in ("car", "pedestrian"):
            m = r["metrics"][cls]["combined"]
            rows.append([r["config"]["input"], r["config_name"].split("-", 1)[-1], cls, m["HOTA"], m["DetA"], m["AssA"], m["IDF1"], m["MOTA"],
                         m["num_switches"], f"[record](records/{name})"])
    if not rows:
        return "Not measured.\n"
    rows.sort(key=lambda x: (x[0] != "oracle", x[0], x[2], x[1]))
    return table(["input", "tracker variant", "class", "HOTA", "DetA", "AssA", "IDF1", "MOTA", "ID switches", "source"], rows) + "\n"


def anomaly(records: list[tuple[str, dict]]) -> str:
    rows, per_cat = [], None
    for name, r in records:
        if r["task"] != "anomaly-detection":
            continue
        m = r["metrics"]["mean_over_categories"]
        rows.append([r["config_name"], r["config"].get("k"), m["image_auroc"], m["pixel_auroc"], m["aupro_0.3"],
                     m["defect_recall_at_threshold"], m["false_alarm_rate_at_threshold"],
                     m.get("defect_recall_at_loo_threshold"), m.get("false_alarm_rate_at_loo_threshold"),
                     r.get("performance", {}).get("score_ms_per_image"), f"[record](records/{name})"])
        if r["config_name"] == "patchcore-k200-coreset1":
            per_cat = r["metrics"]["per_category"]
    if not rows:
        return "Not measured.\n"
    rows.sort(key=lambda x: (not str(x[0]).startswith("patchcore"), x[1] or 0))
    out = table(["config", "references", "image AUROC", "pixel AUROC", "AUPRO@0.3", "recall @ held-out thr", "false alarms @ held-out thr",
                 "recall @ LOO thr", "false alarms @ LOO thr", "ms / image", "source"], rows)
    if per_cat:
        out += "\n\nPer category (PatchCore, 200 references, 1% coreset):\n\n" + table(
            ["category", "image AUROC", "pixel AUROC", "AUPRO@0.3", "test images", "anomalous"],
            [[c, v["image_auroc"], v["pixel_auroc"], v["aupro_0.3"], v["n_test"], v["n_anomalous"]] for c, v in sorted(per_cat.items())])
    return out + "\n"


def vocabulary(records: list[tuple[str, dict]]) -> str:
    for name, r in records:
        if r["task"] == "vocabulary-coverage":
            lv, oi = r["metrics"]["lvis"], r["metrics"]["open_images"]
            return (f"- LVIS v1 minival: {lv['categories']['total']} categories, {lv['categories'].get('outside', 0)} outside COCO-80; "
                    f"{lv['instance_share']['outside'] * 100:.1f}% of {lv['instances_total']} instances are outside; "
                    f"localised by the COCO detector (any label, IoU >= 0.5): same {lv['localized_rate_by_relation']['same'] * 100:.1f}%, "
                    f"subtype {lv['localized_rate_by_relation']['subtype'] * 100:.1f}%, outside {lv['localized_rate_by_relation']['outside'] * 100:.1f}%.\n"
                    f"- Open Images V5 validation: {oi['classes_same_as_coco80'] + oi['classes_subtype_of_coco80']} of {oi['classes_with_boxes']} classes map onto COCO-80 "
                    f"({oi['box_share_mapped'] * 100:.1f}% of {oi['boxes_total']} boxes).\n- Source: [record](records/{name})\n")
    return "Not measured.\n"


def build() -> str:
    records = load()
    parts = [
        "# Benchmark summary\n",
        "Generated by `python tools/summarize_records.py` from the records in [records/](records/). Each record holds the dataset, "
        "split, configuration, environment, git commit and full metrics; this page only tabulates them. Anything without a record "
        "is **not measured**.\n",
        "## Detection - COCO val2017 (5000 held-out images)\n", detection(records),
        "## Why objects are missed (COCO val2017, deployed configurations)\n",
        "Each missed ground-truth object is assigned the first matching cause: lost to a neighbour (its only candidate matched another "
        "object), suppressed by NMS, wrong class, poor localisation (IoU 0.1-0.5), below the display threshold, undetected (no candidate at all).\n",
        miss_causes(records),
        "## Small objects - sliced inference\n", tiling(records),
        "## Robustness - synthetic corruptions (first 500 COCO val2017 images, protocol configuration)\n", robustness(records),
        "## Tracking - KITTI tracking training sequences (independent of detection with oracle inputs)\n", tracking(records),
        "## Anomaly detection - VisA (official one-class split, 12 categories)\n", anomaly(records),
        "## Vocabulary gap\n", vocabulary(records),
    ]
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    text = build()
    if args.stdout:
        print(text)
    else:
        OUT.write_text(text)
        print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
