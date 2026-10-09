"""Anomaly detection against known-good references.

Two methods implement the ``AnomalyDetector`` protocol (fit on normal images,
then score a query image):

* ``PatchCore`` - Roth et al., "Towards Total Recall in Industrial Anomaly
  Detection", CVPR 2022 (arXiv:2106.08265). Locally aggregated mid-level CNN
  features of every image patch of the normal images form a memory bank
  (optionally reduced by greedy coreset selection); a query patch's anomaly
  score is the distance to its nearest neighbour in the bank, the image score
  is the maximum patch score. Deviations from the paper, all forced by what is
  available here: ResNet-50 v1 instead of WideResNet-50-2, plain maximum
  without the paper's neighbourhood re-weighting (the official code does the
  same by default), and full-image resize instead of a centre crop so that
  defects near the border stay in view.

* ``ReferenceDifference`` - the classic golden-template baseline: pick the
  most similar reference, align it to the query (ECC, affine), and measure the
  colour difference (CIE76 Delta E on blurred images). Works only when parts
  are presented in a repeatable pose and lighting.

Scores are distances, not probabilities. A decision threshold only exists
after ``calibrate`` on held-out normal images (never on anomalous test data).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.vision.interfaces import AnomalyResult

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _box3(f: np.ndarray) -> np.ndarray:
    """3x3 neighbourhood mean per channel with zero padding (PatchCore's unfold + average)."""
    h, w = f.shape[2:]
    p = np.pad(f, ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.zeros_like(f)
    for dy in range(3):
        for dx in range(3):
            out += p[:, :, dy : dy + h, dx : dx + w]
    return out / 9.0


def _interp_matrix(n_out: int, n_in: int) -> np.ndarray:
    """Linear interpolation weights with half-pixel centres and edge clamping
    (the convention of OpenCV INTER_LINEAR and PyTorch align_corners=False)."""
    src = np.clip((np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5, 0, n_in - 1)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_in - 1)
    m = np.zeros((n_out, n_in), np.float32)
    np.add.at(m, (np.arange(n_out), i0), 1 - (src - i0))
    np.add.at(m, (np.arange(n_out), i1), src - i0)
    return m


def _upsample(f: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear resize of (N, C, h0, w0) feature maps to (N, C, h, w)."""
    return _interp_matrix(h, f.shape[2]) @ (f @ _interp_matrix(w, f.shape[3]).T)


class PatchFeatureExtractor:
    """ResNet-50 stage-2 and stage-3 features, aggregated per patch (PatchCore section 3.1).

    Output: (N, h, w, 1024) for an input of size x size, h = w = size / 8:
    512 stage-2 channels plus the 1024 stage-3 channels averaged in adjacent
    pairs (so both stages weigh equally, as PatchCore's per-layer pooling to a
    common dimension does), stage 3 upsampled to the stage-2 grid.
    """

    model_id = "resnet50_v1_patch_features"

    def __init__(self, weights: Path, size: int = 224, threads: int | None = None) -> None:
        from app.vision.runtime import ort_session

        self.session = ort_session(weights, threads=threads)
        self.size = size

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(image_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        rgb = img[..., ::-1].astype(np.float32) / 255.0
        return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)

    def __call__(self, images: Sequence[np.ndarray], batch: int = 8) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch):
            x = np.ascontiguousarray(np.stack([self.preprocess(im) for im in images[i : i + batch]]))
            f2, f3 = self.session.run(None, {"data": x})
            out.append(self.aggregate(f2, f3))
        return np.concatenate(out) if out else np.zeros((0, self.size // 8, self.size // 8, 1024), np.float32)

    @staticmethod
    def aggregate(f2: np.ndarray, f3: np.ndarray) -> np.ndarray:
        f2, f3 = _box3(f2), _box3(f3)
        n, c3, h3, w3 = f3.shape
        f3 = f3.reshape(n, c3 // 2, 2, h3, w3).mean(axis=2)
        h, w = f2.shape[2:]
        merged = np.concatenate([f2, _upsample(f3, h, w)], axis=1)
        return np.ascontiguousarray(merged.transpose(0, 2, 3, 1), dtype=np.float32)


def greedy_coreset(features: np.ndarray, n_select: int, projection_dim: int = 128, seed: int = 0) -> np.ndarray:
    """Indices of a greedy k-centre coreset (PatchCore Algorithm 1) on a random projection."""
    n = len(features)
    if n_select >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((features.shape[1], projection_dim)).astype(np.float32) / np.sqrt(projection_dim)
    z = features @ proj
    sq = np.einsum("ij,ij->i", z, z)
    selected = np.empty(n_select, dtype=np.int64)
    selected[0] = int(rng.integers(n))
    min_d = sq + sq[selected[0]] - 2.0 * (z @ z[selected[0]])
    for k in range(1, n_select):
        i = int(np.argmax(min_d))
        selected[k] = i
        np.minimum(min_d, sq + sq[i] - 2.0 * (z @ z[i]), out=min_d)
    return selected


def nearest_distance(queries: np.ndarray, bank: np.ndarray, bank_sq: np.ndarray | None = None, chunk: int = 4096) -> np.ndarray:
    """Euclidean distance from each query row to its nearest bank row."""
    bank_sq = np.einsum("ij,ij->i", bank, bank) if bank_sq is None else bank_sq
    out = np.empty(len(queries), dtype=np.float32)
    for i in range(0, len(queries), chunk):
        q = queries[i : i + chunk]
        d2 = np.einsum("ij,ij->i", q, q)[:, None] + bank_sq[None, :] - 2.0 * (q @ bank.T)
        out[i : i + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def leave_one_out_scores(features: np.ndarray) -> tuple[list[float], list[float]]:
    """Score each reference against a bank of all the others: (max patch distance, median patch distance) per reference.

    With k references the highest of the k scores is a split-conformal threshold for false-alarm rate about
    1/(k+1) - conservative, because each reference was compared with k-1 others rather than k.
    """
    n, h, w, d = features.shape
    flat = features.reshape(n, h * w, d)
    maxima, medians = [], []
    for i in range(n):
        bank = np.concatenate([flat[j] for j in range(n) if j != i])
        dist = nearest_distance(flat[i], bank)
        maxima.append(float(dist.max()))
        medians.append(float(np.median(dist)))
    return maxima, medians


@dataclass(slots=True)
class PatchCoreConfig:
    coreset_ratio: float = 0.01  # 1.0 keeps every patch
    projection_dim: int = 128  # random projection used only for coreset selection
    blur_sigma: float = 4.0  # Gaussian smoothing of the upsampled anomaly map (paper: sigma = 4)
    seed: int = 0


class PatchCore:
    model_id = "patchcore-resnet50"

    def __init__(self, extractor: PatchFeatureExtractor, config: PatchCoreConfig | None = None) -> None:
        self.extractor = extractor
        self.config = config or PatchCoreConfig()
        self.bank: np.ndarray | None = None
        self.bank_sq: np.ndarray | None = None
        self.threshold: float | None = None
        self.calibration: dict[str, Any] = {}
        self.n_references = 0

    # -- fitting --------------------------------------------------------------------------------
    def fit(self, normal_images: Sequence[np.ndarray]) -> None:
        self.fit_features(self.extractor(normal_images))

    def fit_features(self, features: np.ndarray) -> None:
        """features: (N, h, w, D) from the extractor."""
        flat = features.reshape(-1, features.shape[-1])
        n_select = max(1, int(round(len(flat) * self.config.coreset_ratio)))
        idx = greedy_coreset(flat, n_select, self.config.projection_dim, self.config.seed)
        self.bank = np.ascontiguousarray(flat[idx])
        self.bank_sq = np.einsum("ij,ij->i", self.bank, self.bank)
        self.n_references = len(features)
        self.threshold = None

    def calibrate(self, normal_images: Sequence[np.ndarray]) -> float:
        return self.calibrate_features(self.extractor(normal_images))

    def calibrate_features(self, features: np.ndarray) -> float:
        """Threshold = highest image score among held-out normal images (no anomalous data used)."""
        scores = [float(self.patch_scores(f).max()) for f in features]
        self.threshold = max(scores)
        self.calibration = {"method": "max image score of held-out normal images", "n": len(scores),
                            "normal_scores_p50": float(np.median(scores))}
        return self.threshold

    # -- scoring ---------------------------------------------------------------------------------
    def patch_scores(self, features: np.ndarray) -> np.ndarray:
        """(h, w, D) features of one image -> (h, w) nearest-neighbour distances."""
        if self.bank is None:
            raise RuntimeError("PatchCore.fit() has not been called")
        h, w, d = features.shape
        return nearest_distance(features.reshape(-1, d), self.bank, self.bank_sq).reshape(h, w)

    def anomaly_map(self, patch_scores: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
        """Upsample to the network input (or ``size`` = (w, h)) and smooth, as PatchCore's segmentor."""
        s = self.extractor.size
        m = cv2.resize(patch_scores.astype(np.float32), (s, s), interpolation=cv2.INTER_LINEAR)
        m = cv2.GaussianBlur(m, (0, 0), self.config.blur_sigma)
        return cv2.resize(m, size, interpolation=cv2.INTER_LINEAR) if size else m

    def score(self, image_bgr: np.ndarray) -> AnomalyResult:
        feats = self.extractor([image_bgr])[0]
        ps = self.patch_scores(feats)
        h, w = image_bgr.shape[:2]
        return AnomalyResult(
            score=float(ps.max()),
            heatmap=self.anomaly_map(ps, (w, h)),
            threshold=self.threshold,
            details={"method": "patchcore", "references": self.n_references, "memory_bank": int(len(self.bank)),
                     "input_size": self.extractor.size, "calibration": self.calibration or None},
        )


@dataclass(slots=True)
class ReferenceDifferenceConfig:
    size: int = 256  # working resolution (square, like the PatchCore input)
    blur_sigma: float = 2.0  # pre-blur against noise and sub-pixel misalignment
    map_sigma: float = 4.0
    align: bool = True  # ECC affine alignment of the chosen reference onto the query


class ReferenceDifference:
    model_id = "reference-difference"

    def __init__(self, config: ReferenceDifferenceConfig | None = None) -> None:
        self.config = config or ReferenceDifferenceConfig()
        self.refs_lab: list[np.ndarray] = []
        self.refs_gray: list[np.ndarray] = []
        self.refs_key: np.ndarray | None = None
        self.threshold: float | None = None
        self.calibration: dict[str, Any] = {}

    def _prep(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = self.config.size
        img = cv2.resize(image_bgr, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        img = cv2.GaussianBlur(img, (0, 0), self.config.blur_sigma)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)  # L 0..100, a/b about -127..127: Delta E units
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        key = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).ravel()
        key = (key - key.mean()) / (key.std() + 1e-6)
        return lab, gray, key

    def fit(self, normal_images: Sequence[np.ndarray]) -> None:
        prepped = [self._prep(im) for im in normal_images]
        self.refs_lab = [p[0] for p in prepped]
        self.refs_gray = [p[1] for p in prepped]
        self.refs_key = np.stack([p[2] for p in prepped])
        self.threshold = None

    def _difference(self, image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if self.refs_key is None:
            raise RuntimeError("ReferenceDifference.fit() has not been called")
        lab, gray, key = self._prep(image_bgr)
        j = int(np.argmin(((self.refs_key - key) ** 2).sum(axis=1)))
        ref_lab, ref_gray = self.refs_lab[j], self.refs_gray[j]
        s = self.config.size
        aligned, ecc = False, None
        valid = np.ones((s, s), np.float32)
        if self.config.align:
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                ecc, warp = cv2.findTransformECC(gray, ref_gray, warp, cv2.MOTION_AFFINE,
                                                 (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
                flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                ref_lab = cv2.warpAffine(ref_lab, warp, (s, s), flags=flags, borderMode=cv2.BORDER_REPLICATE)
                valid = cv2.warpAffine(valid, warp, (s, s), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                aligned = True
            except cv2.error:  # ECC did not converge: compare unaligned and say so
                pass
        delta_e = np.sqrt(((lab - ref_lab) ** 2).sum(axis=2)) * (valid > 0.5)
        return cv2.GaussianBlur(delta_e, (0, 0), self.config.map_sigma), {
            "reference_index": j, "aligned": aligned, "ecc": None if ecc is None else round(float(ecc), 4)}

    def calibrate(self, normal_images: Sequence[np.ndarray]) -> float:
        scores = [float(self._difference(im)[0].max()) for im in normal_images]
        self.threshold = max(scores)
        self.calibration = {"method": "max image score of held-out normal images", "n": len(scores),
                            "normal_scores_p50": float(np.median(scores))}
        return self.threshold

    def score(self, image_bgr: np.ndarray) -> AnomalyResult:
        m, info = self._difference(image_bgr)
        h, w = image_bgr.shape[:2]
        return AnomalyResult(
            score=float(m.max()),
            heatmap=cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR),
            threshold=self.threshold,
            details={"method": "reference-difference", "units": "CIE76 Delta E (blurred)", "references": len(self.refs_lab),
                     "calibration": self.calibration or None, **info},
        )

    def map_at_working_size(self, image_bgr: np.ndarray) -> tuple[float, np.ndarray]:
        m, _ = self._difference(image_bgr)
        return float(m.max()), m


def anomaly_regions(heatmap: np.ndarray, threshold: float, min_area_fraction: float = 0.0005) -> list[dict[str, Any]]:
    """Connected regions of the map above the threshold, as boxes in the map's pixel coordinates."""
    mask = (heatmap > threshold).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = min_area_fraction * heatmap.size
    regions = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area >= min_area:
            peak = float(heatmap[y : y + h, x : x + w].max())
            regions.append({"box": (x, y, x + w, y + h), "area_px": area, "peak_score": round(peak, 4)})
    return sorted(regions, key=lambda r: -r["peak_score"])
