"""Controlled image corruptions for robustness measurements.

Same idea as ImageNet-C (Hendrycks & Dietterich, ICLR 2019) and COCO-C
(Michaelis et al., 2019): apply a known degradation to held-out images and
measure the accuracy drop. The parameters below are this project's own (not
the COCO-C severities) and are recorded with every result. Synthetic
corruptions approximate, but are not, real low-light or motion-blurred
footage.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np


def gaussian_blur(sigma: float) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    def apply(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    return apply


def motion_blur(length: int) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    """Linear motion blur along a random direction (hand shake / fast pan)."""
    def apply(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        k = np.zeros((length, length), np.float32)
        k[length // 2, :] = 1.0
        angle = float(rng.uniform(0, 180))
        rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
        k = cv2.warpAffine(k, rot, (length, length))
        k /= max(k.sum(), 1e-6)
        return cv2.filter2D(img, -1, k)
    return apply


def low_light(scale: float, read_noise: float = 2.0, photons_at_white: float = 1000.0) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    """Fewer photons: linear-light scaling with shot (Poisson) and read (Gaussian) noise, then gamma back."""
    def apply(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        linear = (img.astype(np.float32) / 255.0) ** 2.2
        photons = linear * scale * photons_at_white
        noisy = rng.poisson(photons).astype(np.float32) + rng.normal(0, read_noise, photons.shape).astype(np.float32)
        out = np.clip(noisy / photons_at_white, 0, 1) ** (1 / 2.2)
        return (out * 255).astype(np.uint8)
    return apply


def jpeg(quality: int) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    def apply(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img
    return apply


def low_resolution(factor: float) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    """Capture at lower resolution, displayed at the original size (information is lost, not recovered)."""
    def apply(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return apply


#: name -> (function, parameters as recorded)
CORRUPTIONS: dict[str, tuple[Callable[[np.ndarray, np.random.Generator], np.ndarray], dict[str, float]]] = {
    "gaussian_blur_s2": (gaussian_blur(2.0), {"sigma_px": 2.0}),
    "gaussian_blur_s4": (gaussian_blur(4.0), {"sigma_px": 4.0}),
    "motion_blur_15": (motion_blur(15), {"length_px": 15}),
    "motion_blur_31": (motion_blur(31), {"length_px": 31}),
    "low_light_x0.1": (low_light(0.1), {"light_scale": 0.1, "read_noise_e": 2.0, "photons_at_white": 1000}),
    "low_light_x0.02": (low_light(0.02), {"light_scale": 0.02, "read_noise_e": 2.0, "photons_at_white": 1000}),
    "jpeg_q10": (jpeg(10), {"jpeg_quality": 10}),
    "low_resolution_x0.25": (low_resolution(0.25), {"scale": 0.25}),
}
