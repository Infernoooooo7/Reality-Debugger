"""The optional AI layer: provider selection, no accidental paid fallback,
Gemini/Claude request building against mock transports, failure isolation
(local results always survive), call budgeting and output validation."""

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
from app.services.ai_providers import (
    AIAuthError,
    AIRateLimitError,
    AnthropicProvider,
    GeminiProvider,
    OpenAICompatibleProvider,
)
from app.services.ai_service import extract_json
from app.services.prompts import DIAGNOSIS_SCHEMA
from tests.conftest import CUP, LAPTOP, VALID_AI, FakeProvider, jpeg_bytes, make_settings, scene, scene_json

GEMINI_KEY = "AIzaSyTEST_key_000000000000000000000"
CLAUDE_KEY = "sk-ant-test-key-123456"


def frame(client: TestClient, *, trigger: str = "deep_scan", scan_id: str | None = None, seed: int = 0, **data: Any) -> httpx.Response:
    form = {"scene": scene_json(LAPTOP, CUP), "trigger": trigger, "personality": "brutal", **data}
    if scan_id:
        form["scan_id"] = scan_id
    return client.post("/api/analyze/frame", files={"image": ("f.jpg", jpeg_bytes(seed=seed), "image/jpeg")}, data=form)


def fake_app(*providers: FakeProvider, **settings: Any) -> TestClient:
    s = make_settings(**settings)
    return TestClient(create_app(s, providers={p.name: p for p in providers}))


# --------------------------------------------------------------------------
# Provider selection
# --------------------------------------------------------------------------


def test_no_key_means_local_only(client: TestClient) -> None:
    ai = client.get("/api/health").json()["ai"]
    assert ai == {
        "provider": "none",
        "model": None,
        "configured": False,
        "state": "off",
        "detail": ai["detail"],
        "fallback": None,
        "last_error": None,
    }
    assert "GEMINI_API_KEY" in ai["detail"]
    check = client.get("/api/health/ai").json()
    assert check["ok"] is False and check["error"]["code"] == "AI_OFF"


def test_auto_selects_gemini_only_when_its_key_is_set() -> None:
    with TestClient(create_app(make_settings(gemini_api_key=GEMINI_KEY))) as client:
        ai = client.get("/api/health").json()["ai"]
    assert ai["provider"] == "gemini" and ai["configured"] is True and ai["state"] == "unverified"
    assert ai["model"] == "gemini-flash-latest"  # config/ai.json gemini.defaultModel
    assert GEMINI_KEY not in json.dumps(ai)


def test_auto_never_selects_a_paid_provider_implicitly() -> None:
    with TestClient(create_app(make_settings(anthropic_api_key=CLAUDE_KEY, openai_api_key="sk-openai-0000000000"))) as client:
        ai = client.get("/api/health").json()["ai"]
    assert ai["provider"] == "none" and ai["state"] == "off"


def test_explicit_provider_without_key_is_off_not_broken() -> None:
    with TestClient(create_app(make_settings(ai_provider="gemini"))) as client:
        ai = client.get("/api/health").json()["ai"]
        res = frame(client)
    assert ai["provider"] == "gemini" and ai["configured"] is False and ai["state"] == "off"
    assert "GEMINI_API_KEY" in ai["detail"]
    assert res.status_code == 200
    assert res.json()["report"]["ai"]["status"] == "off"
    assert res.json()["report"]["findings"]


@pytest.mark.parametrize(("value", "expected"), [("anthropic", "claude"), ("claude", "claude"), ("demo", "none"), ("none", "none")])
def test_provider_aliases(value: str, expected: str) -> None:
    with TestClient(create_app(make_settings(ai_provider=value, anthropic_api_key=CLAUDE_KEY))) as client:
        assert client.get("/api/health").json()["ai"]["provider"] == expected


def test_fallback_is_off_by_default_and_ignored_when_equal_to_primary() -> None:
    from app.services.ai_service import resolve_fallback

    assert make_settings().ai_fallback_provider == "none"
    assert resolve_fallback(make_settings(ai_fallback_provider="gemini"), "gemini") is None


# --------------------------------------------------------------------------
# Failure isolation and no paid fallback
# --------------------------------------------------------------------------


def test_gemini_failure_never_calls_claude_unless_configured() -> None:
    gemini = FakeProvider("gemini", [AIRateLimitError("quota exhausted")])
    claude = FakeProvider("claude")
    with fake_app(gemini, claude, ai_provider="gemini") as client:
        res = frame(client)
        health = client.get("/api/health").json()["ai"]
    assert res.status_code == 200, res.text
    body = res.json()
    assert len(gemini.calls) == 1 and claude.calls == []
    assert body["report"]["ai"]["status"] == "unavailable"
    assert body["report"]["ai"]["error"]["code"] == "AI_RATE_LIMITED"
    # Local CV results are intact.
    assert any(f["rule"] == "spill_risk" and f["source"] == "local" for f in body["report"]["findings"])
    assert health["state"] == "unavailable" and health["last_error"]["code"] == "AI_RATE_LIMITED"


def test_explicit_fallback_is_used_once_after_a_failure() -> None:
    gemini = FakeProvider("gemini", [AIAuthError("bad key")])
    claude = FakeProvider("claude", model="claude-test")
    with fake_app(gemini, claude, ai_provider="gemini", ai_fallback_provider="claude") as client:
        body = frame(client).json()
        metrics = client.get("/api/metrics").json()
    assert len(gemini.calls) == 1 and len(claude.calls) == 1
    run = body["report"]["ai"]
    assert run["status"] == "ok" and run["provider"] == "claude" and "fallback" in run["reason"]
    assert metrics["ai"]["fallback_used"] == 1 and metrics["ai"]["errors"] == {"AI_AUTH_FAILED": 1}


def test_successful_reasoning_is_merged_separately_from_measurements() -> None:
    gemini = FakeProvider("gemini")
    with fake_app(gemini, ai_provider="gemini") as client:
        body = frame(client).json()
    report = body["report"]
    assert report["ai"]["status"] == "ok" and report["provider"] == "gemini"
    local = next(f for f in report["findings"] if f["rule"] == "spill_risk")
    assert local["source"] == "local" and local["ai_note"].startswith("The mug is full")
    assert local["ai_agrees"] is True
    assert local["measurements"]["edge_gap"] >= 0  # measurement untouched
    ai_finding = next(f for f in report["findings"] if f["source"] == "ai")
    assert ai_finding["category"] == "ERGONOMICS"
    assert any(o["source"] == "ai" and o["label"] == "power strip" for o in report["objects"])
    assert any(r["source"] == "local" for r in report["relationships"])
    assert report["system_name"] == "GAMING_SETUP_v3.1"
    prompt = gemini.calls[0]["user_text"]
    assert "BUG-001" in prompt and "spill_risk" in prompt and "t2 cup" in prompt
    assert "MODE: LIVE SCAN" in prompt


def test_ai_duplicate_of_a_local_finding_becomes_a_note() -> None:
    duplicate = dict(VALID_AI)
    duplicate["local_notes"] = []
    duplicate["findings"] = [
        {**VALID_AI["findings"][0], "category": "SAFETY", "title": "Cup next to laptop", "related_objects": ["cup", "laptop"]}
    ]
    with fake_app(FakeProvider("gemini", [duplicate]), ai_provider="gemini") as client:
        report = frame(client).json()["report"]
    assert [f["source"] for f in report["findings"]] == ["local"]
    assert report["findings"][0]["ai_note"]


def test_malformed_output_is_retried_then_reported_without_failing() -> None:
    gemini = FakeProvider("gemini", ["not json", "still {not json"])
    with fake_app(gemini, ai_provider="gemini") as client:
        res = frame(client)
    assert res.status_code == 200
    assert len(gemini.calls) == 2  # one repair attempt
    assert res.json()["report"]["ai"]["error"]["code"] == "AI_MALFORMED_RESPONSE"
    assert res.json()["report"]["findings"]


# --------------------------------------------------------------------------
# Intelligent calling: triggers, cooldown, budget, de-duplication, back-off
# --------------------------------------------------------------------------


def test_cooldown_applies_to_automatic_triggers_only() -> None:
    gemini = FakeProvider("gemini")
    with fake_app(gemini, ai_provider="gemini") as client:
        first = frame(client, trigger="confirmed_finding").json()
        scan_id = first["scan"]["scan_id"]
        second = frame(client, trigger="confirmed_finding", scan_id=scan_id, seed=3).json()
        deep = client.post(
            "/api/scan/deep",
            files={"image": ("f.jpg", jpeg_bytes(seed=5), "image/jpeg")},
            data={"scene": scene_json(LAPTOP, CUP), "scan_id": scan_id},
        ).json()
    assert first["report"]["ai"]["status"] == "ok"
    assert second["report"]["ai"]["status"] == "skipped" and "cooldown" in second["report"]["ai"]["reason"]
    assert deep["report"]["ai"]["status"] == "ok"  # user request bypasses the cooldown
    assert len(gemini.calls) == 2


def test_duplicate_frames_reuse_the_cached_answer() -> None:
    gemini = FakeProvider("gemini")
    with fake_app(gemini, ai_provider="gemini") as client:
        scan_id = frame(client).json()["scan"]["scan_id"]
        again = frame(client, scan_id=scan_id).json()
    assert again["report"]["ai"]["status"] == "cached"
    assert len(gemini.calls) == 1


def test_per_minute_budget_caps_user_requests_too() -> None:
    gemini = FakeProvider("gemini")
    with fake_app(gemini, ai_provider="gemini", scan_ai_calls_per_minute=2) as client:
        scan_id = None
        runs = []
        for seed in range(3):
            body = frame(client, scan_id=scan_id, seed=seed * 2).json()
            scan_id = body["scan"]["scan_id"]
            runs.append(body["report"]["ai"]["status"])
    assert runs == ["ok", "ok", "skipped"]
    assert len(gemini.calls) == 2


def test_hard_failure_pauses_automatic_calls() -> None:
    gemini = FakeProvider("gemini", [AIAuthError("bad key")])
    with fake_app(gemini, ai_provider="gemini") as client:
        scan_id = frame(client).json()["scan"]["scan_id"]
        auto = frame(client, trigger="scene_change", scan_id=scan_id, seed=2).json()
        user = frame(client, trigger="user_explain", scan_id=scan_id, seed=4, focus="BUG-001").json()
    assert auto["report"]["ai"]["status"] == "skipped" and "paused" in auto["report"]["ai"]["reason"]
    assert user["report"]["ai"]["status"] == "unavailable"
    assert len(gemini.calls) == 2


def test_new_objects_alone_never_call_the_ai() -> None:
    gemini = FakeProvider("gemini")
    with fake_app(gemini, ai_provider="gemini") as client:
        body = frame(client, trigger="new_object").json()
    assert body["report"]["ai"]["status"] == "skipped"
    assert gemini.calls == []


def test_observe_never_uploads_or_calls_but_suggests() -> None:
    gemini = FakeProvider("gemini")
    with fake_app(gemini, ai_provider="gemini") as client:
        first = client.post("/api/scan/observe", json={"scene": scene(LAPTOP, CUP, at_ms=0)}).json()
        assert first["ai_suggestion"]["trigger"] == "first_look"
        scan_id = first["scan"]["scan_id"]
        followed = frame(client, trigger="first_look", scan_id=scan_id).json()
        assert followed["report"]["ai"]["status"] == "ok"
        later = client.post("/api/scan/observe", json={"scan_id": scan_id, "scene": scene(LAPTOP, CUP, at_ms=1600)}).json()
    assert later["ai_suggestion"] is None  # same view, finding already explained, cooldown running
    assert len(gemini.calls) == 1
    assert later["scan"]["ai"]["status"] == "ok"


def test_confirmed_relation_is_suggested_after_cooldown() -> None:
    from app.services.ai_policy import monotonic_ms

    gemini = FakeProvider("gemini")
    with fake_app(gemini, ai_provider="gemini") as client:
        app = client.app
        scan_id = client.post("/api/scan/observe", json={"scene": scene(LAPTOP, at_ms=0)}).json()["scan"]["scan_id"]
        frame(client, trigger="first_look", scan_id=scan_id, scene=scene_json(LAPTOP))
        client.post("/api/scan/observe", json={"scan_id": scan_id, "scene": scene(LAPTOP, CUP, at_ms=1000)})
        blocked = client.post("/api/scan/observe", json={"scan_id": scan_id, "scene": scene(LAPTOP, CUP, at_ms=2700)}).json()
        assert blocked["ai_suggestion"] is None  # confirmed, but the cooldown is running
        session = app.state.scans.get(scan_id)
        session.ai.last_call_ms = monotonic_ms() - 60_000
        session.ai.calls.clear()
        ready = client.post("/api/scan/observe", json={"scan_id": scan_id, "scene": scene(LAPTOP, CUP, at_ms=4300)}).json()
    spill = next(f for f in ready["scan"]["findings"] if f["rule"] == "spill_risk")
    assert spill["status"] in ("CONFIRMED", "TRACKING")
    assert ready["ai_suggestion"]["trigger"] == "relationship"
    assert ready["ai_suggestion"]["finding_ids"] == [spill["id"]]


# --------------------------------------------------------------------------
# Gemini through the real SDK (mock transport)
# --------------------------------------------------------------------------


def _gemini_reply(text: str, finish: str = "STOP") -> dict[str, Any]:
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish}],
        "usageMetadata": {"promptTokenCount": 1200, "candidatesTokenCount": 300},
        "modelVersion": "gemini-flash-latest",
    }


def _gemini_app(handler: Callable[[httpx.Request], httpx.Response], **settings: Any) -> TestClient:
    s = make_settings(ai_provider="gemini", gemini_api_key=GEMINI_KEY, **settings)
    provider = GeminiProvider(
        s,
        default_model="gemini-flash-latest",
        temperature=0.4,
        max_output_tokens=8192,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return TestClient(create_app(s, providers={"gemini": provider}))


def test_gemini_request_shape_and_success() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_gemini_reply(json.dumps(VALID_AI)))

    with _gemini_app(handler) as client:
        res = frame(client)
        health = client.get("/api/health").json()["ai"]
    assert res.status_code == 200, res.text
    assert res.json()["report"]["ai"] == {
        "status": "ok",
        "provider": "gemini",
        "model": "gemini-flash-latest",
        "latency_ms": res.json()["report"]["ai"]["latency_ms"],
        "trigger": "deep_scan",
        "reason": None,
        "error": None,
    }
    assert health["state"] == "ready"
    request = seen[0]
    assert request.url.path.endswith("/models/gemini-flash-latest:generateContent")
    assert request.headers["x-goog-api-key"] == GEMINI_KEY
    body = json.loads(request.content)
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == DIAGNOSIS_SCHEMA
    assert "Reality Debugger" in body["systemInstruction"]["parts"][0]["text"]
    parts = body["contents"][0]["parts"]
    inline = parts[0]["inlineData"]
    assert (inline.get("mimeType") or inline.get("mime_type")) == "image/jpeg" and inline["data"]
    assert "MODE: LIVE SCAN" in parts[-1]["text"] and "spill_risk" in parts[-1]["text"]
    assert GEMINI_KEY not in res.text


def _error(code: int, status: str, message: str, reason: str | None = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message, "status": status}
    if reason:
        error["details"] = [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason, "domain": "googleapis.com"}]
    return {"error": error}


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (400, _error(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.", "API_KEY_INVALID"), "AI_AUTH_FAILED"),
        (403, _error(403, "PERMISSION_DENIED", "Generative Language API has not been used in project."), "AI_AUTH_FAILED"),
        (404, _error(404, "NOT_FOUND", "models/gemini-nope is not found for API version v1beta"), "AI_MODEL_NOT_FOUND"),
        (429, _error(429, "RESOURCE_EXHAUSTED", "You exceeded your current quota, please check your plan."), "AI_RATE_LIMITED"),
        (503, _error(503, "UNAVAILABLE", "The model is overloaded."), "AI_UNAVAILABLE"),
        (504, _error(504, "DEADLINE_EXCEEDED", "Deadline expired."), "AI_TIMEOUT"),
    ],
)
def test_gemini_errors_keep_local_results(status: int, body: dict, expected: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    with _gemini_app(handler) as client:
        res = frame(client)
    assert res.status_code == 200, res.text
    report = res.json()["report"]
    assert report["ai"]["status"] == "unavailable" and report["ai"]["error"]["code"] == expected
    assert report["provider"] == "local"
    assert any(f["rule"] == "spill_risk" for f in report["findings"])
    assert GEMINI_KEY not in res.text


@pytest.mark.parametrize(
    ("exc", "expected"),
    [(httpx.ConnectError("no route"), "AI_UNAVAILABLE"), (httpx.ReadTimeout("slow"), "AI_TIMEOUT")],
)
def test_gemini_network_failures(exc: Exception, expected: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    with _gemini_app(handler) as client:
        res = frame(client)
    assert res.status_code == 200
    assert res.json()["report"]["ai"]["error"]["code"] == expected


def test_gemini_quota_message_and_blocked_prompt() -> None:
    def quota(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json=_error(429, "RESOURCE_EXHAUSTED", "You exceeded your current quota."))

    with _gemini_app(quota) as client:
        assert "quota" in frame(client).json()["report"]["ai"]["error"]["message"].lower()

    def blocked(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []})

    with _gemini_app(blocked) as client:
        assert frame(client).json()["report"]["ai"]["error"]["code"] == "AI_REFUSED"


def test_gemini_schema_rejection_falls_back_to_json_mode() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "responseJsonSchema" in body["generationConfig"]:
            return httpx.Response(400, json=_error(400, "INVALID_ARGUMENT", "Invalid JSON payload: response_json_schema is not supported"))
        return httpx.Response(200, json=_gemini_reply("```json\n" + json.dumps(VALID_AI) + "\n```"))

    with _gemini_app(handler) as client:
        res = frame(client)
    assert res.json()["report"]["ai"]["status"] == "ok"
    assert ["responseJsonSchema" in b["generationConfig"] for b in bodies] == [True, False]


def test_gemini_check_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET" and request.url.path.endswith("/models/gemini-flash-latest")
        return httpx.Response(200, json={"name": "models/gemini-flash-latest", "displayName": "Gemini Flash Latest"})

    with _gemini_app(handler) as client:
        res = client.get("/api/health/ai").json()
    assert res["ok"] is True and res["provider"] == "gemini"


# --------------------------------------------------------------------------
# Claude through the real SDK (mock transport)
# --------------------------------------------------------------------------


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


def _claude_app(handler: Callable[[httpx2.Request], httpx2.Response]) -> TestClient:
    s = make_settings(ai_provider="claude", anthropic_api_key=CLAUDE_KEY)
    provider = AnthropicProvider(s, http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)))
    return TestClient(create_app(s, providers={"claude": provider}))


def test_claude_request_shape_and_success() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=_message(json.dumps(VALID_AI)))

    with _claude_app(handler) as client:
        res = frame(client, trigger="confirmed_finding")
    assert res.status_code == 200, res.text
    assert res.json()["report"]["ai"]["provider"] == "claude"
    payload = json.loads(seen[0].content)
    assert seen[0].headers.get("x-api-key") == CLAUDE_KEY
    assert payload["model"] == "claude-opus-5-5"
    assert payload["output_config"]["format"] == {"type": "json_schema", "schema": DIAGNOSIS_SCHEMA}
    assert payload["output_config"]["effort"] == "low"
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "image" and "MODE: LIVE SCAN" in content[-1]["text"]


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}, "AI_AUTH_FAILED"),
        (429, {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}, "AI_RATE_LIMITED"),
        (529, {"type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}}, "AI_UNAVAILABLE"),
    ],
)
def test_claude_errors_keep_local_results(status: int, body: dict, expected: str) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=body)

    with _claude_app(handler) as client:
        res = frame(client)
    assert res.status_code == 200
    assert res.json()["report"]["ai"]["error"]["code"] == expected
    assert CLAUDE_KEY not in res.text


def test_claude_refusal_is_graceful() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_message("", stop_reason="refusal"))

    with _claude_app(handler) as client:
        res = frame(client)
    assert res.status_code == 200
    assert res.json()["report"]["ai"]["error"]["code"] == "AI_REFUSED"


# --------------------------------------------------------------------------
# OpenAI-compatible provider (explicit opt-in only)
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
        fmt = (json.loads(request.content).get("response_format") or {}).get("type")
        formats.append(fmt)
        if fmt == "json_schema":
            return httpx.Response(400, json={"error": {"message": "response_format json_schema not supported"}})
        return httpx.Response(200, json=_openai_reply("```json\n" + json.dumps(VALID_AI) + "\n```"))

    s = make_settings(ai_provider="openai", openai_base_url="http://localhost:11434/v1", openai_model="local-vision")
    provider = OpenAICompatibleProvider(s, transport=httpx.MockTransport(handler))
    with TestClient(create_app(s, providers={"openai": provider})) as client:
        res = client.post("/api/analyze/image", files={"image": ("d.jpg", jpeg_bytes(), "image/jpeg")})
    assert res.status_code == 200, res.text
    assert res.json()["provider"] == "openai"
    assert formats == ["json_schema", "json_object"]


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
        "local_notes": [{"finding_id": "BUG-001", "note": "ok"}, {"finding_id": "", "note": "x"}],
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
    assert [n.finding_id for n in diagnosis.local_notes] == ["BUG-001"]
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
    assert "local_notes" in DIAGNOSIS_SCHEMA["properties"]
