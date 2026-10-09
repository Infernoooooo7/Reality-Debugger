"""Environment-based configuration.

Values are read from (later wins): process environment, ``<repo>/.env``,
``<repo>/backend/.env``. See ``.env.example`` at the repository root.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent

Effort = Literal["", "low", "medium", "high", "xhigh", "max"]

# Origins of a typical home/office LAN plus localhost. Used so a phone on the
# same Wi-Fi can call the API directly (the Vite proxy makes this optional).
LAN_ORIGIN_REGEX = (
    r"^https?://("
    r"localhost|127\.0\.0\.1|\[::1\]|"
    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"192\.168\.\d{1,3}\.\d{1,3}|"
    r"172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|"
    r"[a-zA-Z0-9-]+\.local"
    r")(:\d{1,5})?$"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(ROOT_DIR / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = "INFO"

    # --- AI provider -------------------------------------------------------
    # auto: anthropic if ANTHROPIC_API_KEY is set, else openai if OPENAI_API_KEY
    # or OPENAI_BASE_URL is set, else demo (simulated, clearly labelled).
    ai_provider: Literal["auto", "anthropic", "openai", "demo"] = "auto"

    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-5-5"
    anthropic_base_url: str | None = None
    # Server-side refusal fallback (beta). Disable if your endpoint rejects it.
    anthropic_refusal_fallback: bool = True

    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    openai_model: str | None = None

    ai_timeout_seconds: float = 120.0
    ai_max_retries: int = 2
    ai_max_tokens: int = 16000
    ai_max_concurrency: int = 4
    # Effort for live frames (latency matters) vs. deep scans / images / video.
    # Empty string = do not send an effort value (for models that reject it).
    ai_effort_live: Effort = "low"
    ai_effort_deep: Effort = "medium"
    # Long edge (px) of images sent to the vision model.
    ai_image_max_edge_live: int = 1024
    ai_image_max_edge_deep: int = 1600
    ai_image_max_edge_video: int = 1024
    ai_malformed_retries: int = 1
    # Short artificial pause for demo answers so the UI states stay readable.
    demo_latency_ms: int = 700

    # --- Upload limits -----------------------------------------------------
    max_image_bytes: int = 15 * 1024 * 1024
    max_frame_bytes: int = 5 * 1024 * 1024
    max_video_bytes: int = 300 * 1024 * 1024
    max_video_keyframes: int = 12
    max_image_pixels: int = 50_000_000
    max_context_chars: int = 64_000

    # --- Live scan sessions (in memory only) -------------------------------
    scan_ttl_seconds: int = 2 * 60 * 60
    max_scans: int = 200
    scan_ai_calls_per_minute: int = 12

    # --- Network -----------------------------------------------------------
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "https://localhost:5173",
            "http://127.0.0.1:5173",
            "https://127.0.0.1:5173",
            "http://localhost:4173",
            "https://localhost:4173",
        ]
    )
    # Allow any localhost / private-LAN origin (phones on the same Wi-Fi).
    cors_allow_lan: bool = True

    # --- Single-service deployment -----------------------------------------
    # Built frontend (frontend/dist). When set, this server also serves the
    # app itself, so one process and one URL host everything (see Dockerfile).
    frontend_dist: Path | None = None

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("anthropic_api_key", "openai_api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("openai_base_url", "openai_model", "anthropic_base_url", "frontend_dist", mode="before")
    @classmethod
    def _blank_str_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def max_request_bytes(self) -> int:
        """Hard cap for any request body (largest legitimate upload + slack)."""
        keyframes = self.max_video_keyframes * self.max_frame_bytes
        return max(self.max_video_bytes, self.max_image_bytes, keyframes) + 2 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
