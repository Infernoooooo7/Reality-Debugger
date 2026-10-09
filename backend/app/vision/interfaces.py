"""Stable interfaces between the platform and individual models.

Applications and the evaluation engine depend on these protocols, never on a
specific model's code, so compatible models can be swapped by configuration
(models/registry/*.json). Implementations live in app.vision.detectors,
.segmentation, .anomaly, .motion and app.datasets / app.evaluation.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from app.vision.types import Detection, InferenceResult


@runtime_checkable
class Detector(Protocol):
    model_id: str
    model_version: str

    @property
    def labels(self) -> Sequence[str | None]:
        """The model's own vocabulary (index = class id; None for unused slots)."""
        ...

    def detect(self, image_bgr: np.ndarray, *, frame_id: int | None = None, timestamp_ms: float | None = None) -> InferenceResult: ...


@runtime_checkable
class Classifier(Protocol):
    model_id: str

    def classify(self, image_bgr: np.ndarray, boxes: Sequence[tuple[float, float, float, float]], top_k: int = 5) -> list[list[tuple[str, float]]]: ...


@runtime_checkable
class Segmenter(Protocol):
    """Instance segmentation (masks for its own detections) or promptable segmentation (masks for given boxes/points)."""

    model_id: str

    def segment(self, image_bgr: np.ndarray, prompts: Sequence[Detection] | None = None) -> InferenceResult: ...


@dataclass(slots=True)
class GlobalMotion:
    """Apparent motion of the whole image between two frames (camera-induced), in pixels."""

    dx: float
    dy: float
    confidence: float  # 0..1; low values mean the estimate should not be used
    method: str


@runtime_checkable
class MotionEstimator(Protocol):
    def estimate(self, previous_gray: np.ndarray, current_gray: np.ndarray) -> GlobalMotion: ...


@dataclass(slots=True)
class AnomalyResult:
    score: float  # image-level anomaly score (higher = more anomalous); not a probability
    heatmap: np.ndarray | None  # HxW float32 anomaly map in source-image coordinates, if the method localises
    threshold: float | None = None  # decision threshold if calibrated on validation data, else None
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AnomalyDetector(Protocol):
    model_id: str

    def fit(self, normal_images: Sequence[np.ndarray]) -> None: ...

    def score(self, image_bgr: np.ndarray) -> AnomalyResult: ...


@runtime_checkable
class Tracker(Protocol):
    """Online multi-object tracker. The production tracker is TypeScript
    (frontend/src/vision/tracker.ts); the evaluation engine drives it through
    a Node runner that implements this protocol per sequence."""

    def run_sequence(self, detections_per_frame: Sequence[Sequence[Detection]], timestamps_ms: Sequence[float], camera_motion: Sequence[GlobalMotion] | None = None) -> list[list[Detection]]: ...


@dataclass(slots=True)
class Sample:
    """One evaluation item from a dataset adapter."""

    id: str
    image_path: str | None
    width: int
    height: int
    annotations: list[dict[str, Any]]  # adapter-specific but normalised: category, box (xyxy px), iscrowd, track_id, mask...
    group: str | None = None  # sequence / clip id, used for leakage-free splits
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DatasetAdapter(Protocol):
    dataset_id: str

    @property
    def categories(self) -> Sequence[str]: ...

    def samples(self, split: str | None = None) -> Iterator[Sample]: ...

    def stats(self) -> dict[str, Any]: ...


@runtime_checkable
class EvaluationRunner(Protocol):
    task: str

    def run(self) -> dict[str, Any]:
        """Run the evaluation and return a benchmark record (app.evaluation.records)."""
        ...


@runtime_checkable
class DomainPipeline(Protocol):
    profile_id: str

    def availability(self) -> dict[str, Any]:
        """Which required models/datasets are present; what is unsupported and why."""
        ...
