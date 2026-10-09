"""Server-side vision: models too heavy for every browser, run on demand.

* ``detect`` - YOLOX on the server: one full-image pass ("standard") or
  sliced inference over overlapping tiles plus the full image ("precision",
  for small objects; SAHI-style merging) under a time budget.
* ``compare`` - reference comparison for inspection: a PatchCore-style
  memory bank of patch features from 2-10 known-good images; the decision
  threshold comes from leave-one-out over those references, never from a
  guess.
* queries - "is there an X?" answered as detected / tentative / ambiguous /
  not detected / unsupported category, and "not detected" is downgraded when
  the image is too poor or the analysis incomplete to rule X out.

Images are decoded in memory and dropped with the response. Models load
lazily through ``ModelProvider`` (checksum-verified, memory-bounded); a model
that cannot be loaded raises ``ModelUnavailable`` - a real error, never a
fallback to anything simulated.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import ROOT_DIR, Settings
from app.runtime_config import RuntimeConfig
from app.services.ontology import Ontology
from app.vision import quality as quality_gate
from app.vision.registry import ModelProvider, ModelRegistry, ModelUnavailable
from app.vision.types import COORDINATES, AnalysisState, Detection, ObjectState, QueryState

log = logging.getLogger("reality.vision")

FEATURES_ID = "resnet50_v1_patch_features"
MODES = ("standard", "precision")


class VisionInputError(ValueError):
    """The request cannot be served as asked (unknown profile, wrong number of references...)."""


def load_profiles(config_dir: Path | None, labels: set[str]) -> dict[str, dict[str, Any]]:
    path = (config_dir or ROOT_DIR / "config") / "profiles.json"
    profiles = json.loads(path.read_text(encoding="utf-8"))["profiles"]
    for pid, p in profiles.items():
        cats = p.get("categories")
        if isinstance(cats, list):
            unknown = sorted(set(cats) - labels)
            if unknown:  # a profile must never promise a category no model can output
                raise ValueError(f"profile '{pid}' lists categories the detector cannot output: {unknown}")
    return profiles


class VisionService:
    def __init__(self, settings: Settings, config: RuntimeConfig, ontology: Ontology, *,
                 registry: ModelRegistry | None = None, loaders: dict[str, Any] | None = None) -> None:
        self.settings = settings
        self.config = config
        self.ontology = ontology
        self.registry = registry or ModelRegistry()
        self.detector_id = str(config.get("inference.detector.model"))
        from app.vision.runtime import default_threads

        self.threads = settings.vision_threads or default_threads()
        self.provider = ModelProvider(self.registry, loaders or {"yolox": self._load_yolox, "resnet": self._load_features},
                                      memory_budget_mb=settings.vision_memory_budget_mb)
        self._gate = threading.BoundedSemaphore(max(1, settings.vision_max_concurrency))
        try:
            self.labels: list[str] = [lb for lb in self.registry.labels(self.detector_id) if lb]
        except ModelUnavailable:
            self.labels = []
        self.profiles = load_profiles(settings.config_dir, set(self.labels))

    # -- model loading ---------------------------------------------------------------------------
    def _detector_config(self) -> Any:
        from app.vision.detectors.yolox import YoloxConfig

        return YoloxConfig(
            score_floor=float(self.config.get("detection.deep.preNmsScore")),
            nms_iou=float(self.config.get("detection.deep.nmsIou")),
            class_agnostic=bool(self.config.get("detection.deep.classAgnosticNms")),
        )

    def _load_yolox(self, entry: dict[str, Any], path: Path) -> Any:
        from app.vision.detectors.yolox import YoloxDetector

        return YoloxDetector(entry, path, self.registry.labels(entry["id"]), config=self._detector_config(), threads=self.threads)

    def _load_features(self, entry: dict[str, Any], path: Path) -> Any:
        from app.vision.anomaly import PatchFeatureExtractor

        if entry["id"] != FEATURES_ID:
            raise ValueError(f"{entry['id']} is not a patch-feature extractor")
        return PatchFeatureExtractor(path, size=int(self.config.get("inference.compare.inputSize")), threads=self.threads)

    # -- status ----------------------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        models = []
        for model_id in (self.detector_id, FEATURES_ID):
            if model_id not in self.registry.entries:
                models.append({"id": model_id, "name": None, "tasks": [], "available": False, "reason": "not in the model registry",
                               "loaded": False, "load_error": None, "licence": None})
                continue
            avail = self.registry.availability(model_id)
            entry = self.registry.get(model_id)
            models.append({
                "id": model_id, "name": entry.get("name"), "tasks": entry.get("tasks", []),
                "available": avail.available, "reason": avail.reason, "loaded": model_id in self.provider.loaded(),
                "load_error": self.provider.errors.get(model_id), "licence": entry.get("licence", {}).get("weights"),
            })
        return {
            "enabled": self.settings.vision_backend,
            "models": models,
            "memory_budget_mb": self.settings.vision_memory_budget_mb,
            "threads": self.threads,
            "detector_vocabulary": {"type": "closed", "size": len(self.labels), "dataset": "COCO 2017"},
        }

    def registry_summary(self) -> list[dict[str, Any]]:
        out = []
        for model_id, e in self.registry.entries.items():
            avail = self.registry.availability(model_id)
            out.append({
                "id": model_id, "name": e.get("name"), "tasks": e.get("tasks", []), "status": e.get("status"),
                "available_here": avail.available, "reason": avail.reason,
                "vocabulary": e.get("vocabulary"), "licence": e.get("licence"), "paper": e.get("paper"),
                "limitations": e.get("limitations", []), "unavailable_reason": e.get("unavailable_reason"),
            })
        return out

    def profile(self, profile_id: str) -> dict[str, Any]:
        if profile_id not in self.profiles:
            raise VisionInputError(f"unknown profile '{profile_id}' (known: {', '.join(self.profiles)})")
        return self.profiles[profile_id]

    def _model(self, model_id: str) -> Any:
        return self.provider.get(model_id)  # raises ModelUnavailable with the real reason

    # -- detection -------------------------------------------------------------------------------
    def tiling_config(self) -> Any:
        from app.vision.tiling import TilingConfig

        get = self.config.get
        return TilingConfig(
            tile=int(get("inference.precision.tile")), overlap=float(get("inference.precision.overlap")),
            full_image=bool(get("inference.precision.fullImage")), merge=str(get("inference.precision.merge")),
            match_metric=str(get("inference.precision.matchMetric")), match_threshold=float(get("inference.precision.matchThreshold")),
            max_tiles=int(get("inference.precision.maxTiles")), time_budget_ms=float(get("inference.precision.timeBudgetMs")),
        )

    def _object_state(self, d: Detection, threshold: float) -> tuple[ObjectState, str | None]:
        ratio = float(self.config.get("inference.ambiguity.secondClassRatio"))
        second = (d.uncertainty or {}).get("second_class_score")
        if second is not None and d.confidence > 0 and second / d.confidence >= ratio:
            return ObjectState.AMBIGUOUS, d.extra.get("second_class")
        return (ObjectState.DETECTED if d.confidence >= threshold else ObjectState.TENTATIVE), None

    def detect(self, image_bgr: np.ndarray, *, mode: str = "standard", profile: str = "general",
               queries: list[str] | None = None) -> dict[str, Any]:
        if mode not in MODES:
            raise VisionInputError(f"mode must be one of {MODES}")
        prof = self.profile(profile)
        if prof.get("status") == "unavailable":
            raise VisionInputError(f"the '{prof['name']}' profile has no validated model in this build: {prof['summary']}")
        report = quality_gate.assess(image_bgr, self.config)
        detector = self._model(self.detector_id)
        threshold = float(self.config.get("detection.deep.scoreThreshold"))
        t0 = time.perf_counter()
        with self._gate, self.provider.run_lock(self.detector_id):
            if mode == "precision":
                from app.vision.tiling import sliced_detect

                result = sliced_detect(detector.detect, image_bgr, self.tiling_config(),
                                       model_id=detector.model_id, model_version=detector.model_version)
            else:
                result = detector.detect(image_bgr)
        elapsed = (time.perf_counter() - t0) * 1000
        allowed = prof.get("categories")
        objects = []
        for i, d in enumerate(sorted(result.detections, key=lambda x: -x.confidence)):
            state, alternative = self._object_state(d, threshold)
            objects.append({
                "id": i, "label": d.class_name, "class_id": d.class_id, "confidence": round(d.confidence, 4),
                "box": [round(v, 1) for v in d.box], "state": state.value, "alternative_label": alternative,
                "in_profile": allowed == "all" or (isinstance(allowed, list) and d.class_name in allowed),
                "uncertainty": d.uncertainty, "source": d.extra.get("source", "full"),
            })
        state = result.state
        if state == AnalysisState.COMPLETE and not report.ok:
            state = AnalysisState.INSUFFICIENT_IMAGE_QUALITY
        answers = [self.answer_query(q, objects, state) for q in (queries or []) if q.strip()]
        notes = list(result.notes)
        if mode == "standard" and max(image_bgr.shape[:2]) > 2 * (result.config.get("input_size") or 640):
            notes.append("Large image analysed in one downscaled pass; use precision mode to look for small objects.")
        return {
            "state": state.value,
            "mode": mode,
            "profile": profile,
            "model": {"id": result.model_id, "version": result.model_version, "vocabulary": "COCO-80 (closed set)",
                      "operating_threshold": threshold},
            "image": {"width": result.image.width, "height": result.image.height},
            "coordinate_system": COORDINATES,
            "objects": objects,
            "queries": answers,
            "quality": report.to_dict(),
            "config": {k: v for k, v in result.config.items() if not k.startswith("_")},
            "timings_ms": {**{k: round(v, 1) for k, v in result.timings_ms.items()}, "request": round(elapsed, 1)},
            "notes": notes,
        }

    def resolve_query(self, query: str) -> tuple[list[str], str | None]:
        """Model labels that answer a free-text category, and how they were matched."""
        q = " ".join(query.lower().split())
        if q in self.labels:
            return [q], "exact"
        labels, _ = self.ontology.match_text(q)
        labels &= set(self.labels)
        return (sorted(labels), "wordnet") if labels else ([], None)

    def answer_query(self, query: str, objects: list[dict[str, Any]], state: AnalysisState) -> dict[str, Any]:
        labels, match = self.resolve_query(query)
        if not labels:
            return {"query": query, "state": QueryState.UNSUPPORTED_CATEGORY.value, "labels": [], "match": None, "count": 0,
                    "explanation": f"None of the loaded models can name '{query}': the detector knows {len(self.labels)} COCO "
                                   "categories. It is not reported as absent - it cannot be looked for."}
        hits = [o for o in objects if o["label"] in labels]
        by_state = {s: [o for o in hits if o["state"] == s] for s in ("detected", "ambiguous", "tentative")}
        coarse = "" if match == "exact" else f" (matched to the model label{'s' if len(labels) > 1 else ''} {', '.join(labels)}; the model cannot tell '{query}' from other kinds)"
        if by_state["detected"]:
            qs, text = QueryState.DETECTED, f"{len(by_state['detected'])} found{coarse}."
        elif by_state["ambiguous"]:
            qs, text = QueryState.AMBIGUOUS, f"Possible match, but the model nearly as strongly suggests another class{coarse}."
        elif by_state["tentative"]:
            qs, text = QueryState.TENTATIVE, f"Only low-confidence candidates (below the operating threshold){coarse}."
        elif state == AnalysisState.INSUFFICIENT_IMAGE_QUALITY:
            qs, text = QueryState.INSUFFICIENT_IMAGE_QUALITY, "Not found, but the image quality is too low to rule it out."
        elif state == AnalysisState.INCOMPLETE:
            qs, text = QueryState.ANALYSIS_INCOMPLETE, "Not found in the analysed part; part of the image was not analysed at full resolution."
        else:
            qs, text = QueryState.NOT_DETECTED, f"Not detected{coarse}. This is not proof of absence (small, occluded or unusual instances are missed)."
        return {"query": query, "state": qs.value, "labels": labels, "match": match,
                "count": len(by_state["detected"]), "object_ids": [o["id"] for o in hits], "explanation": text}

    # -- reference comparison --------------------------------------------------------------------
    def compare(self, references: list[np.ndarray], image_bgr: np.ndarray) -> dict[str, Any]:
        from app.vision.anomaly import anomaly_regions, leave_one_out_scores, nearest_distance

        get = self.config.get
        lo, hi = int(get("inference.compare.minReferences")), int(get("inference.compare.maxReferences"))
        if not lo <= len(references) <= hi:
            raise VisionInputError(f"give between {lo} and {hi} reference images of known-good parts (got {len(references)})")
        extractor = self._model(FEATURES_ID)
        t0 = time.perf_counter()
        with self._gate, self.provider.run_lock(FEATURES_ID):
            ref_feats = extractor(references)
            query_feats = extractor([image_bgr])[0]
        t1 = time.perf_counter()
        h, w, d = query_feats.shape
        # Leave-one-out: how anomalous does each known-good reference look against the others?
        loo_max, loo_median = leave_one_out_scores(ref_feats)
        threshold = float(get("inference.compare.thresholdMargin")) * max(loo_max)
        bank = ref_feats.reshape(-1, d)
        patch = nearest_distance(query_feats.reshape(-1, d), bank).reshape(h, w)
        score = float(patch.max())
        size = extractor.size
        amap = cv2.GaussianBlur(cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR), (0, 0), 4.0)
        regions = anomaly_regions(amap, threshold, float(get("inference.compare.regionMinAreaFraction")))
        ih, iw = image_bgr.shape[:2]
        sx, sy = iw / size, ih / size
        for r in regions:
            x1, y1, x2, y2 = r["box"]
            r["box"] = [round(x1 * sx, 1), round(y1 * sy, 1), round(x2 * sx, 1), round(y2 * sy, 1)]
        notes = []
        median_ratio = float(np.median(patch)) / max(1e-9, float(np.median(loo_median)))
        if median_ratio > 1.5:
            notes.append(f"Most of the image differs from the references (median patch distance {median_ratio:.1f}x theirs): "
                         "check that the same part, pose, background and lighting were used; the result is unreliable otherwise.")
        report = quality_gate.assess(image_bgr, self.config)
        state = AnalysisState.COMPLETE if report.ok else AnalysisState.INSUFFICIENT_IMAGE_QUALITY
        heat = np.clip(amap / max(threshold, 1e-9) * 127.5, 0, 255).astype(np.uint8)
        ok, png = cv2.imencode(".png", heat)
        return {
            "state": state.value,
            "verdict": "anomalous" if score > threshold else "within_reference_variation",
            "score": round(score, 4),
            "threshold": round(threshold, 4),
            "score_to_threshold": round(score / max(threshold, 1e-9), 3),
            "regions": regions,
            "references": len(references),
            "method": {
                "name": "PatchCore-style nearest-neighbour patch distance (Roth et al., CVPR 2022)",
                "features": f"{FEATURES_ID} (ResNet-50 v1 stages 2+3, {size}x{size} input)",
                "memory_bank_patches": int(len(bank)),
                "threshold_rule": "highest leave-one-out score among the references (x thresholdMargin)",
                "leave_one_out_scores": [round(v, 4) for v in loo_max],
            },
            "heatmap": {"width": size, "height": size, "encoding": "png-base64",
                        "scale": "pixel = anomaly / threshold x 127.5 (128 = threshold), clipped at 255",
                        "data": base64.b64encode(png.tobytes()).decode("ascii") if ok else None},
            "quality": report.to_dict(),
            "coordinate_system": COORDINATES,
            "image": {"width": iw, "height": ih},
            "timings_ms": {"features": round((t1 - t0) * 1000, 1), "total": round((time.perf_counter() - t0) * 1000, 1)},
            "notes": notes,
            "evidence": "VisA (12 categories, official split): see docs/benchmarks/records/anomaly-detection-*.json",
        }
