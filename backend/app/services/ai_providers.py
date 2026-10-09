"""Optional AI reasoning providers.

* :class:`GeminiProvider` - Gemini via the official ``google-genai`` SDK
  (structured JSON output with ``response_json_schema``).
* :class:`AnthropicProvider` - Claude via the official ``anthropic`` SDK
  (structured JSON output, optional server-side refusal fallback).
* :class:`OpenAICompatibleProvider` - any ``/chat/completions`` endpoint that
  accepts images (OpenAI, Ollama, LM Studio, vLLM, OpenRouter, ...).

All raise :class:`AIError` subclasses with user-safe messages; raw provider
errors and API keys never reach the client. None of them is ever required:
the local computer-vision pipeline and diagnostics work without any provider.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import anthropic
import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.config import Settings
from app.errors import AppError

log = logging.getLogger("reality.ai")

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AIError(AppError):
    status_code = 502
    code = "AI_ERROR"


class AINotConfiguredError(AIError):
    status_code = 503
    code = "AI_NOT_CONFIGURED"


class AIAuthError(AIError):
    status_code = 502
    code = "AI_AUTH_FAILED"


class AIModelNotFoundError(AIError):
    status_code = 502
    code = "AI_MODEL_NOT_FOUND"


class AIRateLimitError(AIError):
    status_code = 429
    code = "AI_RATE_LIMITED"
    retryable = True


class AITimeoutError(AIError):
    status_code = 504
    code = "AI_TIMEOUT"
    retryable = True


class AIUnavailableError(AIError):
    status_code = 503
    code = "AI_UNAVAILABLE"
    retryable = True


class AIRefusedError(AIError):
    status_code = 422
    code = "AI_REFUSED"


class AIBadRequestError(AIError):
    status_code = 502
    code = "AI_REQUEST_REJECTED"


class AIMalformedError(AIError):
    status_code = 502
    code = "AI_MALFORMED_RESPONSE"
    retryable = True


@dataclass(slots=True)
class ProviderImage:
    jpeg: bytes
    label: str | None = None


@dataclass(slots=True)
class Completion:
    text: str
    model: str
    stop_reason: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None


class VisionProvider(Protocol):
    name: str
    model: str

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
    ) -> Completion: ...

    async def check(self) -> None: ...

    async def aclose(self) -> None: ...


def _missing_credentials() -> AINotConfiguredError:
    return AINotConfiguredError(
        "No Anthropic credentials were found.",
        hint="Set ANTHROPIC_API_KEY in .env (see .env.example) and restart the backend, or use DEMO MODE.",
    )


def _b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


# --------------------------------------------------------------------------
# Anthropic (Claude)
# --------------------------------------------------------------------------


class AnthropicProvider:
    name = "claude"

    def __init__(self, settings: Settings, *, http_client: Any = None) -> None:
        kwargs: dict[str, Any] = {
            "timeout": settings.ai_timeout_seconds,
            "max_retries": settings.ai_max_retries,
        }
        if http_client is not None:  # injected by tests
            kwargs["http_client"] = http_client
        if settings.anthropic_api_key is not None:
            kwargs["api_key"] = settings.anthropic_api_key.get_secret_value()
        if settings.anthropic_base_url:
            kwargs["base_url"] = settings.anthropic_base_url
        # Raises if no credentials can be resolved at all.
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self.model = settings.anthropic_model
        self._use_fallback = settings.anthropic_refusal_fallback

    @staticmethod
    def _content(user_text: str, images: list[ProviderImage]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for image in images:
            if image.label:
                blocks.append({"type": "text", "text": image.label})
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(image.jpeg)},
                }
            )
        blocks.append({"type": "text", "text": user_text})
        return blocks

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
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if effort:
            output_config["effort"] = effort
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            # The system prompt is identical for every request -> cache it.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": self._content(user_text, images)}],
            "output_config": output_config,
        }
        client = self._client.with_options(timeout=timeout)
        try:
            if self._use_fallback:
                try:
                    response = await client.beta.messages.create(
                        **params, betas=[FALLBACK_BETA], fallbacks="default"
                    )
                except anthropic.BadRequestError as exc:
                    if "fallback" not in str(exc).lower():
                        raise
                    log.warning("Refusal fallback rejected by the API; continuing without it.")
                    self._use_fallback = False
                    response = await client.messages.create(**params)
            else:
                response = await client.messages.create(**params)
        except anthropic.AuthenticationError as exc:
            raise AIAuthError(
                "The Anthropic API rejected the API key.",
                hint="Check ANTHROPIC_API_KEY in your .env file and restart the backend.",
            ) from exc
        except anthropic.PermissionDeniedError as exc:
            raise AIAuthError(
                "The Anthropic API key is not allowed to use this model.",
                hint="Check the key's workspace permissions or set ANTHROPIC_MODEL to a model you can access.",
            ) from exc
        except anthropic.NotFoundError as exc:
            raise AIModelNotFoundError(
                f"Model '{self.model}' was not found.",
                hint="Set ANTHROPIC_MODEL to a valid vision-capable model id.",
            ) from exc
        except anthropic.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after") if exc.response is not None else None
            raise AIRateLimitError(
                "The vision model is rate limited.",
                hint="Wait a few seconds and try again, or lower the live analysis rate.",
                headers={"Retry-After": retry_after} if retry_after else None,
            ) from exc
        except anthropic.BadRequestError as exc:
            log.warning("Anthropic rejected the request: %s", getattr(exc, "message", "bad request"))
            raise AIBadRequestError(
                "The vision model rejected the request.",
                hint="If you changed ANTHROPIC_MODEL, make sure it supports images and structured outputs.",
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise AITimeoutError(
                "The vision model took too long to answer.",
                hint="Try again; local tracking keeps running meanwhile.",
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise AIUnavailableError(
                "Could not reach the Anthropic API.",
                hint="Check the computer's internet connection.",
            ) from exc
        except anthropic.APIStatusError as exc:
            raise AIUnavailableError(
                f"The vision model is temporarily unavailable (HTTP {exc.status_code}).",
                hint="Try again in a moment.",
            ) from exc
        except (anthropic.CredentialsError, TypeError) as exc:
            if isinstance(exc, TypeError) and "authentication" not in str(exc).lower():
                raise
            raise _missing_credentials() from exc

        if response.stop_reason == "refusal":
            raise AIRefusedError(
                "The vision model declined to analyse this frame.",
                hint="Try a different frame or angle.",
            )
        text = "".join(getattr(block, "text", "") for block in response.content if block.type == "text")
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )

    async def check(self) -> None:
        try:
            await self._client.with_options(timeout=15.0, max_retries=0).models.retrieve(self.model)
        except anthropic.AuthenticationError as exc:
            raise AIAuthError("The Anthropic API rejected the API key.", hint="Check ANTHROPIC_API_KEY.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise AIAuthError("The API key cannot access this model.", hint="Check ANTHROPIC_MODEL.") from exc
        except anthropic.NotFoundError as exc:
            raise AIModelNotFoundError(f"Model '{self.model}' was not found.", hint="Check ANTHROPIC_MODEL.") from exc
        except anthropic.APITimeoutError as exc:
            raise AITimeoutError("The Anthropic API did not answer in time.") from exc
        except anthropic.APIConnectionError as exc:
            raise AIUnavailableError("Could not reach the Anthropic API.", hint="Check the internet connection.") from exc
        except anthropic.APIStatusError as exc:
            raise AIUnavailableError(f"Anthropic API error (HTTP {exc.status_code}).") from exc
        except (anthropic.CredentialsError, TypeError) as exc:
            if isinstance(exc, TypeError) and "authentication" not in str(exc).lower():
                raise
            raise _missing_credentials() from exc

    async def aclose(self) -> None:
        await self._client.close()


# --------------------------------------------------------------------------
# OpenAI-compatible (/chat/completions)
# --------------------------------------------------------------------------


class OpenAICompatibleProvider:
    name = "openai"
    _FORMAT_MODES = ("json_schema", "json_object", "none")

    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.base_url = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
        self.model = settings.openai_model or ""
        self._api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        self._client = httpx.AsyncClient(timeout=settings.ai_timeout_seconds, transport=transport)
        self._max_retries = settings.ai_max_retries
        self._format_index = 0

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _body(self, system: str, user_text: str, images: list[ProviderImage], schema: dict) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for image in images:
            if image.label:
                content.append({"type": "text", "text": image.label})
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(image.jpeg)}"}}
            )
        content.append({"type": "text", "text": user_text})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        }
        mode = self._FORMAT_MODES[self._format_index]
        if mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reality_diagnosis", "strict": True, "schema": schema},
            }
        elif mode == "json_object":
            body["response_format"] = {"type": "json_object"}
        return body

    async def _post(self, path: str, body: dict[str, Any] | None, timeout: float) -> httpx.Response:
        attempt = 0
        while True:
            try:
                if body is None:
                    response = await self._client.get(f"{self.base_url}{path}", headers=self._headers(), timeout=timeout)
                else:
                    response = await self._client.post(
                        f"{self.base_url}{path}", json=body, headers=self._headers(), timeout=timeout
                    )
            except httpx.TimeoutException as exc:
                raise AITimeoutError("The vision model took too long to answer.") from exc
            except httpx.HTTPError as exc:
                raise AIUnavailableError(
                    f"Could not reach the model endpoint at {self.base_url}.",
                    hint="Check OPENAI_BASE_URL and that the server is running.",
                ) from exc
            if response.status_code in (429, 500, 502, 503, 504) and attempt < self._max_retries:
                attempt += 1
                await asyncio.sleep(min(8.0, 0.8 * 2**attempt))
                continue
            return response

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return
        if status in (401, 403):
            raise AIAuthError("The model endpoint rejected the API key.", hint="Check OPENAI_API_KEY.")
        if status == 404:
            raise AIModelNotFoundError(
                f"Model '{self.model}' or endpoint was not found.", hint="Check OPENAI_MODEL and OPENAI_BASE_URL."
            )
        if status == 429:
            raise AIRateLimitError("The vision model is rate limited.", hint="Wait a few seconds and retry.")
        if status >= 500:
            raise AIUnavailableError(f"The model endpoint failed (HTTP {status}).", hint="Try again in a moment.")
        raise AIBadRequestError(
            f"The model endpoint rejected the request (HTTP {status}).",
            hint="Make sure OPENAI_MODEL is a vision-capable model.",
        )

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
        if not self.model:
            raise AINotConfiguredError(
                "OPENAI_MODEL is not set.", hint="Set OPENAI_MODEL to a vision-capable model your endpoint serves."
            )
        while True:
            response = await self._post("/chat/completions", self._body(system, user_text, images, schema), timeout)
            # Older/local servers may not support structured response formats.
            if (
                response.status_code in (400, 422)
                and self._format_index < len(self._FORMAT_MODES) - 1
                and "response_format" in response.text
            ):
                self._format_index += 1
                log.info("Endpoint rejected response format; falling back to %s", self._FORMAT_MODES[self._format_index])
                continue
            break
        self._raise_for_status(response)
        try:
            payload = response.json()
            choice = payload["choices"][0]
            message = choice.get("message") or {}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise AIMalformedError("The model endpoint returned an unexpected payload.") from exc
        if message.get("refusal"):
            raise AIRefusedError("The vision model declined to analyse this frame.", hint="Try a different frame.")
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        usage = payload.get("usage") or {}
        return Completion(
            text=content or "",
            model=str(payload.get("model") or self.model),
            stop_reason=choice.get("finish_reason"),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    async def check(self) -> None:
        if not self.model:
            raise AINotConfiguredError("OPENAI_MODEL is not set.", hint="Set OPENAI_MODEL in .env.")
        response = await self._post("/models", None, 15.0)
        self._raise_for_status(response)

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------
# Google Gemini
# --------------------------------------------------------------------------

_BLOCKED_FINISH = {
    "SAFETY",
    "PROHIBITED_CONTENT",
    "BLOCKLIST",
    "SPII",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "RECITATION",
}


class GeminiProvider:
    """Gemini through the official google-genai SDK (Gemini Developer API).

    The API key stays server-side; requests carry the scene model as text and,
    when available, one compressed frame. Structured output uses
    ``response_mime_type="application/json"`` + ``response_json_schema`` as in
    the SDK documentation; if the API rejects the schema the provider falls back
    to plain JSON mode and relies on the backend's own validation.
    """

    name = "gemini"

    def __init__(
        self,
        settings: Settings,
        *,
        default_model: str,
        temperature: float,
        max_output_tokens: int,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if settings.gemini_api_key is None:
            raise AINotConfiguredError(
                "No Gemini API key was found.", hint="Set GEMINI_API_KEY in .env, or use AI_PROVIDER=none (local only)."
            )
        options: dict[str, Any] = {
            "timeout": int(settings.ai_timeout_seconds * 1000),
            # Retries are decided by the app (AI calls are optional and throttled).
            "retry_options": genai_types.HttpRetryOptions(attempts=1),
        }
        if settings.gemini_base_url:
            options["base_url"] = settings.gemini_base_url
        if http_client is not None:  # injected by tests
            options["httpx_async_client"] = http_client
        self._client = genai.Client(
            api_key=settings.gemini_api_key.get_secret_value(), http_options=genai_types.HttpOptions(**options)
        )
        self.model = settings.gemini_model or default_model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._json_schema = True

    def _config(self, system: str, schema: dict, timeout: float, max_tokens: int) -> genai_types.GenerateContentConfig:
        return genai_types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema if self._json_schema else None,
            temperature=self._temperature,
            max_output_tokens=min(max_tokens, self._max_output_tokens),
            http_options=genai_types.HttpOptions(timeout=int(timeout * 1000)),
        )

    @staticmethod
    def _contents(user_text: str, images: list[ProviderImage]) -> list[genai_types.Content]:
        parts: list[genai_types.Part] = []
        for image in images:
            if image.label:
                parts.append(genai_types.Part.from_text(text=image.label))
            parts.append(genai_types.Part.from_bytes(data=image.jpeg, mime_type="image/jpeg"))
        parts.append(genai_types.Part.from_text(text=user_text))
        return [genai_types.Content(role="user", parts=parts)]

    def _map_api_error(self, exc: genai_errors.APIError) -> AIError:
        code = int(getattr(exc, "code", 0) or 0)
        status = str(getattr(exc, "status", "") or "")
        message = str(getattr(exc, "message", "") or "")
        details = str(getattr(exc, "details", "") or "")
        lowered = f"{message} {details}".lower()
        if code in (401, 403) or "api_key_invalid" in lowered or "api key not valid" in lowered or status == "UNAUTHENTICATED":
            return AIAuthError(
                "The Gemini API rejected the API key.",
                hint="Check GEMINI_API_KEY in .env (and that the Generative Language API is enabled for it), then restart the backend.",
            )
        if code == 404:
            return AIModelNotFoundError(f"Gemini model '{self.model}' was not found.", hint="Set GEMINI_MODEL to a current model id.")
        if code == 429 or status == "RESOURCE_EXHAUSTED":
            quota = "quota" in lowered
            return AIRateLimitError(
                "The Gemini quota is exhausted." if quota else "The Gemini API is rate limited.",
                hint="Local diagnostics keep working. Wait, or check the API key's quota and billing." if quota else "Wait a moment; local diagnostics keep working.",
            )
        if code == 504 or status == "DEADLINE_EXCEEDED":
            return AITimeoutError("Gemini did not answer in time.", hint="Local diagnostics keep working.")
        if code >= 500:
            return AIUnavailableError(f"Gemini is temporarily unavailable (HTTP {code}).", hint="Try again later.")
        log.warning("Gemini rejected the request: HTTP %s %s", code, status)
        return AIBadRequestError(
            "Gemini rejected the request.",
            hint="If you changed GEMINI_MODEL, make sure it accepts images and JSON output.",
        )

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
        for _ in range(2):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self.model,
                    contents=self._contents(user_text, images),
                    config=self._config(system, schema, timeout, max_tokens),
                )
                break
            except genai_errors.ClientError as exc:
                text = f"{getattr(exc, 'message', '')} {getattr(exc, 'details', '')}".lower()
                if self._json_schema and getattr(exc, "code", 0) == 400 and "schema" in text:
                    log.info("Gemini rejected the JSON schema; retrying in plain JSON mode.")
                    self._json_schema = False
                    continue
                raise self._map_api_error(exc) from exc
            except genai_errors.APIError as exc:
                raise self._map_api_error(exc) from exc
            except httpx.TimeoutException as exc:
                raise AITimeoutError("Gemini did not answer in time.", hint="Local diagnostics keep working.") from exc
            except httpx.HTTPError as exc:
                raise AIUnavailableError("Could not reach the Gemini API.", hint="Check the internet connection.") from exc
        else:  # pragma: no cover - loop always breaks or raises
            raise AIBadRequestError("Gemini rejected the request.")

        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise AIRefusedError("Gemini declined to analyse this frame.", hint="Try a different frame or angle.")
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise AIMalformedError("Gemini returned no answer.", hint="Try again.")
        finish = getattr(candidates[0], "finish_reason", None)
        finish_name = getattr(finish, "name", None) or (str(finish) if finish else None)
        if finish_name in _BLOCKED_FINISH:
            raise AIRefusedError("Gemini declined to analyse this frame.", hint="Try a different frame or angle.")
        try:
            text = response.text or ""
        except ValueError:  # pragma: no cover - SDK raises for non-text parts
            text = ""
        usage = getattr(response, "usage_metadata", None)
        return Completion(
            text=text,
            model=str(getattr(response, "model_version", None) or self.model),
            stop_reason="max_tokens" if finish_name == "MAX_TOKENS" else (finish_name or "").lower() or None,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
        )

    async def check(self) -> None:
        try:
            await self._client.aio.models.get(model=self.model)
        except genai_errors.APIError as exc:
            raise self._map_api_error(exc) from exc
        except httpx.TimeoutException as exc:
            raise AITimeoutError("Gemini did not answer in time.") from exc
        except httpx.HTTPError as exc:
            raise AIUnavailableError("Could not reach the Gemini API.", hint="Check the internet connection.") from exc

    async def aclose(self) -> None:
        await self._client.aio.aclose()


def monotonic_ms() -> float:
    return time.perf_counter() * 1000.0
