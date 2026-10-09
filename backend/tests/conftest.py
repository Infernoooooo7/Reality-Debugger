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
from app.services.ai_providers import AIError, Completion, ProviderImage


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never pick up real keys or endpoints from the developer's shell."""
    for var in (
        "AI_PROVIDER",
        "AI_FALLBACK_PROVIDER",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "ai_provider": "auto",
        "ai_fallback_provider": "none",
        "gemini_api_key": None,
        "gemini_model": None,
        "anthropic_api_key": None,
        "anthropic_base_url": None,
        "openai_api_key": None,
        "openai_base_url": None,
        "openai_model": None,
        "ai_max_retries": 0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def jpeg_bytes(width: int = 320, height: int = 240, color: tuple[int, int, int] = (110, 90, 70), seed: int = 0) -> bytes:
    img = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([width // 4 + seed * 20, height // 4, width // 2 + seed * 20, height // 2], fill=(220, 220, 220))
    draw.line([0, height - 10, width, 10 + seed * 30], fill=(20, 20, 20), width=4)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()


def png_bytes(width: int = 64, height: int = 64) -> bytes:
    img = Image.new("RGBA", (width, height), (10, 200, 10, 128))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def obj(oid: str, label: str, x: float, y: float, w: float, h: float, conf: float = 0.85, **extra: Any) -> dict[str, Any]:
    return {"id": oid, "label": label, "confidence": conf, "box": {"x": x, "y": y, "w": w, "h": h}, **extra}


# A laptop and a cup touching its right edge (spill risk, HIGH).
LAPTOP = obj("t1", "laptop", 0.30, 0.40, 0.35, 0.30, 0.9, age_ms=6000)
CUP = obj("t2", "cup", 0.64, 0.45, 0.08, 0.12, 0.85, age_ms=6000)


def scene(*objects: dict[str, Any], at_ms: float = 0.0, view_id: int = 0, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "at_ms": at_ms,
        "view_id": view_id,
        "width": 1280,
        "height": 720,
        "objects": list(objects),
        "stats": {"detectors": ["efficientdet_lite0"]},
    }
    body.update(extra)
    return body


def scene_json(*objects: dict[str, Any], **extra: Any) -> str:
    return json.dumps(scene(*objects, **extra))


VALID_AI = {
    "scene": {"name": "GAMING_SETUP", "version": "3.1", "summary": "A desk with a laptop and a mug.", "confidence": 0.9},
    "objects": [{"id": "ai_01", "label": "power strip", "confidence": 0.7, "box": {"x": 0.1, "y": 0.8, "w": 0.2, "h": 0.1}}],
    "relationships": [{"subject": "t2", "relation": "on", "object": "t1", "observation": "The mug stands on the palm rest."}],
    "local_notes": [{"finding_id": "BUG-001", "note": "The mug is full and sits right by the keyboard.", "agrees": True}],
    "findings": [
        {
            "id": "new_1",
            "severity": "LOW",
            "category": "ERGONOMICS",
            "title": "Laptop screen sits far below eye level",
            "evidence": "The laptop is flat on the desk.",
            "inference": "The user looks down while working.",
            "impact": "Neck strain over long sessions.",
            "recommendation": "Raise the laptop on a stand and use an external keyboard.",
            "confidence": 0.8,
            "quip": "",
            "box": {"x": 0.3, "y": 0.4, "w": 0.35, "h": 0.3},
            "related_objects": ["laptop"],
            "frames": [],
        }
    ],
    "status_updates": [],
    "optimizations": [{"id": "opt_1", "title": "Add a coaster", "description": "Fixed drink spot.", "effort": "LOW", "impact": "Fewer spills."}],
    "timeline": [],
    "system_score": 73,
    "final_diagnosis": "Functional, with a spill waiting to happen.",
}


class FakeProvider:
    """A scripted AI provider: returns the next answer, or raises the next error."""

    def __init__(self, name: str = "gemini", answers: list[Any] | None = None, model: str = "fake-model") -> None:
        self.name = name
        self.model = model
        self.answers = list(answers if answers is not None else [VALID_AI])
        self.calls: list[dict[str, Any]] = []
        self.checks = 0
        self.closed = False

    async def complete(
        self,
        *,
        system: str,
        user_text: str,
        images: list[ProviderImage],
        schema: dict,
        effort: str | None,
        timeout: float,
        max_tokens: int,
    ) -> Completion:
        self.calls.append({"system": system, "user_text": user_text, "images": len(images), "effort": effort})
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, AIError):
            raise answer
        text = answer if isinstance(answer, str) else json.dumps(answer)
        return Completion(text=text, model=self.model, stop_reason="stop", input_tokens=100, output_tokens=50)

    async def check(self) -> None:
        self.checks += 1
        answer = self.answers[0]
        if isinstance(answer, AIError):
            raise answer

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def client() -> Iterator[TestClient]:
    """App in local-only mode (no API key anywhere)."""
    app = create_app(make_settings())
    with TestClient(app) as test_client:
        yield test_client
