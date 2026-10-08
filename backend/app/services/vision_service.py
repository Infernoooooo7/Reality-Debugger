"""Server-side image handling: upload validation, decoding, normalisation.

Images are processed in memory, EXIF/GPS metadata is stripped by re-encoding,
and nothing is written to permanent storage.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from app.errors import InvalidUploadError, PayloadTooLargeError, UnsupportedMediaError

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Pillow's own bomb guard is replaced by an explicit pixel check below.
Image.MAX_IMAGE_PIXELS = None


@dataclass(slots=True)
class ImageStats:
    brightness: float  # mean luminance 0..1
    contrast: float  # luminance std-dev 0..1
    sharpness: float  # normalised Laplacian energy 0..1
    colorfulness: float  # Hasler–Süsstrunk metric scaled to 0..1


@dataclass(slots=True)
class PreparedImage:
    jpeg: bytes
    width: int
    height: int
    original_width: int
    original_height: int
    source_type: str
    stats: ImageStats
    label: str | None = None
    extra: dict = field(default_factory=dict)


def sniff_image_type(data: bytes) -> str | None:
    """Identify the format from magic bytes (the client MIME type is not trusted)."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


async def read_upload(upload: UploadFile, limit: int, what: str = "file") -> bytes:
    """Read an upload fully but never more than ``limit`` bytes."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1 << 20)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise PayloadTooLargeError(
                f"The {what} is larger than {limit // (1024 * 1024)} MB.",
                hint="Use a smaller file or lower the camera resolution.",
            )
        chunks.append(chunk)
    if total == 0:
        raise InvalidUploadError(f"The {what} is empty.", hint="Pick the file again.")
    return b"".join(chunks)


def _stats(img: Image.Image) -> ImageStats:
    small = img.copy()
    small.thumbnail((256, 256))
    rgb = np.asarray(small.convert("RGB"), dtype=np.float32) / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    if gray.shape[0] > 2 and gray.shape[1] > 2:
        lap = (
            -4 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        sharpness = float(min(1.0, lap.var() * 40.0))
    else:
        sharpness = 0.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return ImageStats(
        brightness=round(brightness, 3),
        contrast=round(min(1.0, contrast * 2), 3),
        sharpness=round(sharpness, 3),
        colorfulness=round(min(1.0, colorfulness * 2), 3),
    )


def prepare_image(data: bytes, *, max_edge: int, max_pixels: int, quality: int = 85) -> PreparedImage:
    """Validate and normalise an uploaded image to an EXIF-free RGB JPEG.

    Runs synchronously (CPU bound); call it from a worker thread.
    """
    kind = sniff_image_type(data)
    if kind not in ALLOWED_IMAGE_TYPES:
        raise UnsupportedMediaError(
            "Unsupported image format.",
            hint="Use a JPG, PNG or WEBP image.",
        )
    try:
        img = Image.open(io.BytesIO(data))
        width, height = img.size
        if width * height > max_pixels:
            raise PayloadTooLargeError(
                f"Image is {width}×{height} pixels, which exceeds the {max_pixels // 1_000_000} MP limit.",
                hint="Resize the image before uploading.",
            )
        img.load()
    except (PayloadTooLargeError, UnsupportedMediaError):
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise InvalidUploadError(
            "The image could not be decoded (it may be corrupted or truncated).",
            hint="Try exporting the image again as JPG or PNG.",
        ) from exc

    img = ImageOps.exif_transpose(img) or img
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    original_width, original_height = img.size
    if max(img.size) > max_edge:
        img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

    stats = _stats(img)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=quality, optimize=True)
    return PreparedImage(
        jpeg=out.getvalue(),
        width=img.size[0],
        height=img.size[1],
        original_width=original_width,
        original_height=original_height,
        source_type=kind,
        stats=stats,
    )
