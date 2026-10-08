"""Error envelope shared by every endpoint.

Every failure is returned as::

    {"error": {"code": "...", "message": "...", "hint": "...", "retryable": false}}

Stack traces are logged server-side only and never sent to clients.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("reality.errors")


class AppError(Exception):
    status_code = 400
    code = "BAD_REQUEST"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        code: str | None = None,
        status_code: int | None = None,
        retryable: bool | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code
        if retryable is not None:
            self.retryable = retryable
        self.headers = headers or {}

    def to_body(self) -> dict:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "hint": self.hint,
                "retryable": self.retryable,
            }
        }


class UnsupportedMediaError(AppError):
    status_code = 415
    code = "UNSUPPORTED_MEDIA_TYPE"


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "PAYLOAD_TOO_LARGE"


class InvalidUploadError(AppError):
    status_code = 422
    code = "INVALID_UPLOAD"


class NotFoundError(AppError):
    status_code = 404
    code = "NOT_FOUND"


class RateLimitedError(AppError):
    status_code = 429
    code = "RATE_LIMITED"
    retryable = True


def _json(status: int, body: dict, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content=body, headers=headers)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        level = logging.WARNING if exc.status_code >= 500 else logging.INFO
        log.log(level, "request failed: %s (%s)", exc.code, exc.message)
        return _json(exc.status_code, exc.to_body(), exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        fields = []
        for err in exc.errors()[:8]:
            loc = ".".join(str(part) for part in err.get("loc", ()) if part != "body")
            fields.append(f"{loc}: {err.get('msg', 'invalid')}")
        body = AppError(
            "The request was malformed.",
            code="INVALID_REQUEST",
            hint="; ".join(fields) or None,
        ).to_body()
        return _json(422, body)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return _json(exc.status_code, AppError(message, code=code).to_body())

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error: %s", type(exc).__name__)
        body = AppError(
            "Something went wrong inside the diagnostic core.",
            code="INTERNAL_ERROR",
            hint="Try again. If it keeps happening, check the backend terminal for details.",
            retryable=True,
        ).to_body()
        return _json(500, body)
