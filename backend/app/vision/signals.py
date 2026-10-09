"""Pixel signals shared by the browser and the server (config/vision.json "signals").

Kept free of web-framework imports so that offline tools can use exactly the
same definitions as the app.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def gray_signals(gray: np.ndarray, *, log_offset: float, log_span: float) -> tuple[float, float]:
    """Brightness and sharpness of a luminance thumbnail (0..1 floats), computed
    exactly like the browser's signal analyser (config/vision.json "signals"):
    sharpness = clip((log10(var(Laplacian)) + offset) / span, 0, 1)."""
    brightness = float(gray.mean())
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return brightness, 0.0
    lap = gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1] + gray[2:, 1:-1] - 4 * gray[1:-1, 1:-1]
    variance = float(lap.var())
    sharpness = min(1.0, max(0.0, (float(np.log10(variance + 1e-6)) + log_offset) / log_span))
    return round(brightness, 4), round(sharpness, 4)


def thumbnail_gray(img: Image.Image, width: int, height: int) -> np.ndarray:
    small = img.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
