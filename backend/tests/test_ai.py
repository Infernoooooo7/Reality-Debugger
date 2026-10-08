"""AI integration tests: real SDK request building against mock transports,
error mapping, malformed-output handling and schema coercion."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.analysis import MalformedDiagnosisError, parse_diagnosis
from app.services.ai_providers import AnthropicProvider, OpenAICompatibleProvider
from app.services.ai_service import extract_json
from app.services.prompts import DIAGNOSIS_SCHEMA
from tests.conftest import CUP, LAPTOP, context, jpeg_bytes, make_settings

VALID = {
    "scene": {"name": "WORKSPACE", "version": "3.1", "summary": "A desk with a laptop and a mug.", "confidence": 0.9},
    "objects": [
        {"id": "obj_01", "label": "laptop", "confidence": 0.97, "box": {"x": 0.3, "y": 0.4, "w": 0.35, "h": 0.3}},
        {"id": "obj_02", "label": "mug", "confidence": 0.9, "box": {"x": 0.66, "y": 0.45, "w": 0.08, "h": 0.12}},
    ],
    "relationships": [{"subject": "obj_02", "relation": "next to", "object": "obj_01", "observation": "Mug 2 cm from keyboard."}],
    "findings": [
        {
            "id": "new_1",
            "severity": "HIGH",
            "category": "SAFETY",
            "title": "Mug next to laptop keyboard",
            "evidence": "A full mug sits beside the keyboard.",
            "inference": "It could be knocked over.",
            "impact": "Liquid damage.",
            "recommendation": "Move the mug to the far side of the desk.",
            "confidence": 0.89,
            "quip": "",
            "box": {"x": 0.6, "y": 0.4, "w": 0.2, "h": 0.2},
            "related_objects": ["obj_01", "obj_02"],
            "frames": [],
        }
    ],
    "status_updates": [],
    "optimizations": [{"id": "opt_1", "title": "Add a coaster", "description": "Fixed drink spot.", "effort": "LOW", "impact": "Fewer spills."}],
    "timeline": [],
    "system_score": 73,
    "final_diagnosis": "Functional, but suffering from technical debt.",
}


def _message(text: str, stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1200, "output_tokens": 600},
    }


def _anthropic_app(handler: Callable[[httpx2.Request], httpx2.Response], **settings: Any) -> TestClient:
    s = make_settings(ai_provider="anthropic", anthropic_api_key="sk-ant-test-key-123456", **settings)
    app = create_app(s)
    http_client = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))
    app.state.ai._provider = AnthropicProvider(s, http_client=http_client)
    return TestClient(app)


def _post_frame(client: TestClient) -> httpx.Response:
    return client.post(
        "/api/analyze/frame",
        files={"image": ("f.jpg", jpeg_bytes(), "image/jpeg")},
        data={"context": context(LAPTOP, CUP), "trigger": "new_object", "personality": "brutal"},
    )


def test_anthropic_request_shape_and_success() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=_message(json.dumps(VALID)))

    with _anthropic_app(handler) as client:
        assert client.get("/api/health").json()["ai"]["provider"] == "anthropic"
        res = _post_frame(client)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["report"]["simulated"] is False
    assert body["report"]["provider"] == "anthropic"
    assert body["report"]["model"] == "claude-opus-5-5"
    assert body["scan"]["findings"][0]["id"] == "BUG-001"
    assert body["scan"]["findings"][0]["status"] == "DISCOVERED"
    assert body["report"]["system_name"] == "WORKSPACE_v3.1"

    assert len(seen) == 1
    request = seen[0]
    assert request.url.path == "/v1/messages"
    assert "server-side-fallback-2026-07-01" in request.headers.get("anthropic-beta", "")
    assert request.headers.get("x-api-key") == "sk-ant-test-key-123456"
    payload = json.loads(request.content)
    assert payload["model"] == "claude-opus-5-5"
    assert payload["fallbacks"] == "default"
    assert payload["output_config"]["format"]["type"] == "json_schema"
    assert payload["output_config"]["format"]["schema"] == DIAGNOSIS_SCHEMA
    assert payload["output_config"]["effort"] == "low"
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/jpeg"
    assert len(content[0]["source"]["data"]) > 100
    user_text = content[-1]["text"]
    assert "MODE: LIVE SCAN" in user_text and "BRUTAL DEBUG" in user_text
    assert "laptop" in user_text and "new object" in user_text
    assert "thinking" not in payload


def test_anthropic_deep_scan_uses_deep_effort_and_lists_active_findings() -> None:
    payloads: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        payloads.append(json.loads(request.content))
        return httpx2.Response(200, json=_message(json.dumps(VALID)))

    with _anthropic_app(handler) as client:
        first = _post_frame(client).json()
        res = client.post(
            "/api/scan/deep",
            files={"image": ("f.jpg", jpeg_bytes(), "image/jpeg")},
            data={"scan_id": first["scan"]["scan_id"]},
        )
    assert res.status_code == 200, res.text
    deep = payloads[1]
    assert deep["output_config"]["effort"] == "medium"
    text = deep["messages"][0]["content"][-1]["text"]
    assert "MODE: DEEP SCAN" in text
    assert "BUG-001" in text  # active findings are passed back for lifecycle tracking


@pytest.mark.parametrize(
    ("status", "body", "expected_status", "expected_code"),
    [
        (401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}, 502, "AI_AUTH_FAILED"),
        (404, {"type": "error", "error": {"type": "not_found_error", "message": "model: nope"}}, 502, "AI_MODEL_NOT_FOUND"),
        (429, {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}, 429, "AI_RATE_LIMITED"),
        (529, {"type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}}, 503, "AI_UNAVAILABLE"),
    ],
)
def test_anthropic_errors_are_mapped(status: int, body: dict, expected_status: int, expected_code: str) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=body, headers={"retry-after": "3"})

    with _anthropic_app(handler) as client:
        res = _post_frame(client)
    assert res.status_code == expected_status, res.text
    error = res.json()["error"]
    assert error["code"] == expected_code
    assert "sk-ant-test-key" not in res.text


def test_anthropic_refusal_is_graceful() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_message("", stop_reason="refusal"))

    with _anthropic_app(handler) as client:
        res = _post_frame(client)
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "AI_REFUSED"


def test_anthropic_fallback_rejection_retries_without_beta() -> None:
    calls: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        if "fallbacks" in json.loads(request.content):
            return httpx2.Response(
                400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "fallbacks: unsupported"}}
            )
        return httpx2.Response(200, json=_message(json.dumps(VALID)))

    with _anthropic_app(handler) as client:
        assert _post_frame(client).status_code == 200
        assert _post_frame(client).status_code == 200
    # first request: rejected + retried; second request goes straight without the beta
    assert [("fallbacks" in json.loads(c.content)) for c in calls] == [True, False, False]


def test_malformed_output_is_retried_then_reported() -> None:
    answers = iter(["this is not json", json.dumps(VALID)])

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_message(next(answers)))

    with _anthropic_app(handler) as client:
        res = _post_frame(client)
    assert res.status_code == 200, res.text

    def always_bad(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_message('{"objects": "nope"'))

    with _anthropic_app(always_bad) as client:
        res = _post_frame(client)
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "AI_MALFORMED_RESPONSE"
    assert res.json()["error"]["retryable"] is True


def test_partially_invalid_output_keeps_valid_parts() -> None:
    messy = dict(VALID)
    messy["findings"] = [
        VALID["findings"][0],
        {"severity": "HIGH"},  # no title -> dropped
        {**VALID["findings"][0], "title": "Second", "confidence": 87, "severity": "warning", "category": "hazard"},
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_message(json.dumps(messy)))

    with _anthropic_app(handler) as client:
        res = client.post("/api/analyze/image", files={"image": ("d.jpg", jpeg_bytes(), "image/jpeg")})
    assert res.status_code == 200, res.text
    report = res.json()
    assert len(report["findings"]) == 2
    second = next(f for f in report["findings"] if f["title"] == "Second")
    assert second["confidence"] == pytest.approx(0.87)
    assert second["severity"] == "MEDIUM"
    assert second["category"] == "SAFETY"
    assert any("Dropped 1 malformed" in w for w in report["warnings"])


def test_missing_api_key_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    s = make_settings(ai_provider="anthropic", anthropic_api_key=None)
    app = create_app(s)

    def handler(request: httpx2.Request) -> httpx2.Response:  # pragma: no cover - must not be reached
        raise AssertionError("no request expected without credentials")

    app.state.ai._provider = AnthropicProvider(
        s, http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))
    )
    with TestClient(app) as client:
        res = client.post("/api/analyze/image", files={"image": ("d.jpg", jpeg_bytes(), "image/jpeg")})
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "AI_NOT_CONFIGURED"


def test_demo_flag_overrides_configured_provider() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:  # pragma: no cover
        raise AssertionError("demo requests must not reach the provider")

    with _anthropic_app(handler) as client:
        res = client.post(
            "/api/analyze/image",
            files={"image": ("d.jpg", jpeg_bytes(), "image/jpeg")},
            data={"demo": "true", "context": context(LAPTOP, CUP)},
        )
    assert res.status_code == 200
    assert res.json()["simulated"] is True


# --------------------------------------------------------------------------
# OpenAI-compatible provider
# --------------------------------------------------------------------------


def _openai_reply(text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "model": "local-vision",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }


def test_openai_compatible_falls_back_to_json_object() -> None:
    formats: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        fmt = (payload.get("response_format") or {}).get("type")
        formats.append(fmt)
        if fmt == "json_schema":
            return httpx.Response(400, json={"error": {"message": "response_format json_schema not supported"}})
        return httpx.Response(200, json=_openai_reply("```json\n" + json.dumps(VALID) + "\n```"))

    s = make_settings(ai_provider="openai", openai_base_url="http://localhost:11434/v1", openai_model="local-vision")
    app = create_app(s)
    app.state.ai._provider = OpenAICompatibleProvider(s, transport=httpx.MockTransport(handler))
    with TestClient(app) as client:
        res = client.post("/api/analyze/image", files={"image": ("d.jpg", jpeg_bytes(), "image/jpeg")})
    assert res.status_code == 200, res.text
    assert res.json()["provider"] == "openai"
    assert formats == ["json_schema", "json_object"]


def test_openai_compatible_requires_model() -> None:
    s = make_settings(ai_provider="openai", openai_base_url="http://localhost:11434/v1", openai_model=None)
    app = create_app(s)
    with TestClient(app) as client:
        assert client.get("/api/health").json()["ai"]["configured"] is False
        res = client.post("/api/analyze/image", files={"image": ("d.jpg", jpeg_bytes(), "image/jpeg")})
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "AI_NOT_CONFIGURED"


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def test_parse_diagnosis_coerces_values() -> None:
    raw = {
        "scene": {"name": "kitchen", "confidence": 140},
        "objects": [{"id": "o1", "label": "cup", "confidence": "0.5", "box": [10, 20, 30, 40]}, {"id": "o2", "label": ""}],
        "findings": [
            {"title": "Pixel box", "box": {"x": 300, "y": 20, "w": 40, "h": 40}, "severity": "blocker", "category": "nonsense"}
        ],
        "system_score": 0.62,
    }
    diagnosis, warnings = parse_diagnosis(raw)
    assert diagnosis.scene.confidence == 1.0
    assert diagnosis.objects[0].box is not None and diagnosis.objects[0].box.x == pytest.approx(0.1)
    assert len(diagnosis.objects) == 1
    finding = diagnosis.findings[0]
    assert finding.box is None
    assert finding.severity.value == "CRITICAL"
    assert finding.category.value == "WORKFLOW"
    assert diagnosis.system_score == 62
    assert warnings


@pytest.mark.parametrize("bad", [[], "text", 3, {}, {"findings": "x"}])
def test_parse_diagnosis_rejects_unusable(bad: Any) -> None:
    with pytest.raises(MalformedDiagnosisError):
        parse_diagnosis(bad)


def test_extract_json_variants() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Here you go:\n```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('prefix {"a": 3} suffix') == {"a": 3}
    with pytest.raises(ValueError):
        extract_json("no json here")
    with pytest.raises(ValueError):
        extract_json("")


def test_schema_is_structured_output_compatible() -> None:
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node["required"]) == set(node["properties"])
            for forbidden in ("minimum", "maximum", "minLength", "maxLength", "maxItems"):
                assert forbidden not in node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(DIAGNOSIS_SCHEMA)
