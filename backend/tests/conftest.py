from __future__ import annotations

import io
import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.config import Settings
from app.main import create_app


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "ai_provider": "demo",
        "demo_latency_ms": 0,
        "anthropic_api_key": None,
        "anthropic_base_url": None,
        "openai_api_key": None,
        "openai_base_url": None,
        "openai_model": None,
        "ai_max_retries": 0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def jpeg_bytes(width: int = 320, height: int = 240, color: tuple[int, int, int] = (110, 90, 70)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([width // 4, height // 4, width // 2, height // 2], fill=(220, 220, 220))
    draw.line([0, height - 10, width, 10], fill=(20, 20, 20), width=4)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()


def png_bytes(width: int = 64, height: int = 64) -> bytes:
    img = Image.new("RGBA", (width, height), (10, 200, 10, 128))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


LAPTOP = {"label": "laptop", "confidence": 0.9, "box": {"x": 0.3, "y": 0.4, "w": 0.35, "h": 0.3}}
CUP = {"label": "cup", "confidence": 0.85, "box": {"x": 0.66, "y": 0.45, "w": 0.08, "h": 0.12}}


def context(*objects: dict, **extra: Any) -> str:
    return json.dumps({"detector": "test-detector", "objects": list(objects), **extra})


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(make_settings())
    with TestClient(app) as test_client:
        yield test_client
