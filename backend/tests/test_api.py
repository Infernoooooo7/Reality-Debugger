"""HTTP-level tests (DEMO MODE provider, no network)."""

from __future__ import annotations

import io
import json

from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import CUP, LAPTOP, context, jpeg_bytes, make_settings, png_bytes


def test_health_reports_demo_mode(client: TestClient) -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["ai"]["provider"] == "demo"
    assert body["ai"]["simulated"] is True
    assert body["features"]["demo_mode"] is True
    assert body["limits"]["max_image_bytes"] > 0


def test_ai_check_explains_missing_provider(client: TestClient) -> None:
    body = client.get("/api/health/ai").json()
    assert body["ok"] is False
    assert body["error"]["code"] == "AI_NOT_CONFIGURED"


def test_image_demo_report(client: TestClient) -> None:
    res = client.post(
        "/api/analyze/image",
        files={"image": ("desk.jpg", jpeg_bytes(), "image/jpeg")},
        data={"personality": "unhinged", "context": context(LAPTOP, CUP)},
    )
    assert res.status_code == 200, res.text
    report = res.json()
    assert report["simulated"] is True
    assert report["provider"] == "demo"
    assert report["mode"] == "image"
    assert 0 <= report["system_score"] <= 100
    assert report["system_name"].startswith("WORKSPACE_v")
    titles = [f["title"] for f in report["findings"]]
    assert any("0xC0FFEE" in t for t in titles), titles
    first = report["findings"][0]
    for key in ("severity", "category", "title", "evidence", "inference", "impact", "recommendation", "confidence", "status"):
        assert key in first
    assert report["image"]["width"] == 320


def test_png_with_alpha_is_accepted(client: TestClient) -> None:
    res = client.post("/api/analyze/image", files={"image": ("a.png", png_bytes(), "image/png")})
    assert res.status_code == 200, res.text


def test_rejects_non_image_payload(client: TestClient) -> None:
    res = client.post("/api/analyze/image", files={"image": ("notes.txt", b"hello world", "text/plain")})
    assert res.status_code == 415
    assert res.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_rejects_spoofed_mime_type(client: TestClient) -> None:
    res = client.post("/api/analyze/image", files={"image": ("x.jpg", b"GIF89a....", "image/jpeg")})
    assert res.status_code == 415


def test_rejects_corrupted_image(client: TestClient) -> None:
    data = jpeg_bytes()[:200]
    res = client.post("/api/analyze/image", files={"image": ("broken.jpg", data, "image/jpeg")})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "INVALID_UPLOAD"


def test_rejects_empty_and_missing_files(client: TestClient) -> None:
    assert client.post("/api/analyze/image", files={"image": ("e.jpg", b"", "image/jpeg")}).status_code == 422
    res = client.post("/api/analyze/image", data={"personality": "serious"})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "INVALID_REQUEST"


def test_rejects_oversized_image() -> None:
    app = create_app(make_settings(max_image_bytes=2_000))
    with TestClient(app) as client:
        res = client.post("/api/analyze/image", files={"image": ("big.jpg", jpeg_bytes(800, 600), "image/jpeg")})
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_rejects_too_many_pixels() -> None:
    app = create_app(make_settings(max_image_pixels=10_000))
    with TestClient(app) as client:
        res = client.post("/api/analyze/image", files={"image": ("big.jpg", jpeg_bytes(400, 300), "image/jpeg")})
    assert res.status_code == 413


def test_request_body_cap() -> None:
    app = create_app(make_settings(max_video_bytes=1_000, max_image_bytes=1_000, max_frame_bytes=1_000, max_video_keyframes=1))
    with TestClient(app) as client:
        res = client.post("/api/analyze/image", files={"image": ("big.jpg", b"\xff\xd8\xff" + b"0" * 5_000_000, "image/jpeg")})
    assert res.status_code == 413


def test_rejects_bad_personality_and_context(client: TestClient) -> None:
    files = {"image": ("d.jpg", jpeg_bytes(), "image/jpeg")}
    res = client.post("/api/analyze/image", files=files, data={"personality": "sarcastic"})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "INVALID_PERSONALITY"
    res = client.post("/api/analyze/image", files={"image": ("d.jpg", jpeg_bytes(), "image/jpeg")}, data={"context": "{not json"})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "INVALID_CONTEXT"


def test_live_frame_lifecycle(client: TestClient) -> None:
    with_cup = context(LAPTOP, CUP, scene_change=0.05)
    without_cup = context(LAPTOP, scene_change=0.05)
    scan_id = None
    statuses = []
    for ctx in (with_cup, with_cup, with_cup, without_cup):
        data = {"context": ctx, "trigger": "interval", "personality": "brutal"}
        if scan_id:
            data["scan_id"] = scan_id
        res = client.post("/api/analyze/frame", files={"image": ("f.jpg", jpeg_bytes(), "image/jpeg")}, data=data)
        assert res.status_code == 200, res.text
        body = res.json()
        scan_id = body["scan"]["scan_id"]
        bug = next(f for f in body["scan"]["findings"] if f["id"] == "BUG-001")
        statuses.append(bug["status"])
        assert body["report"]["simulated"] is True
    assert statuses == ["DISCOVERED", "CONFIRMED", "TRACKING", "RESOLVED"]

    state = client.get(f"/api/scan/{scan_id}").json()
    assert state["counts"]["resolved"] == 1
    assert [e["type"] for e in state["events"]][:4] == ["DISCOVERED", "CONFIRMED", "TRACKING", "RESOLVED"]
    assert state["status"] == "STABLE"

    assert client.delete(f"/api/scan/{scan_id}").status_code == 204
    missing = client.get(f"/api/scan/{scan_id}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "SCAN_NOT_FOUND"


def test_moving_camera_keeps_finding_open(client: TestClient) -> None:
    first = client.post(
        "/api/analyze/frame",
        files={"image": ("f.jpg", jpeg_bytes(), "image/jpeg")},
        data={"context": context(LAPTOP, CUP)},
    ).json()
    scan_id = first["scan"]["scan_id"]
    moved = client.post(
        "/api/analyze/frame",
        files={"image": ("f.jpg", jpeg_bytes(), "image/jpeg")},
        data={"context": context(scene_change=0.9), "scan_id": scan_id},
    ).json()
    bug = moved["scan"]["findings"][0]
    assert bug["status"] == "DISCOVERED"
    assert bug["out_of_view"] is True


def test_deep_scan_confirms_high_confidence(client: TestClient) -> None:
    res = client.post(
        "/api/scan/deep",
        files={"image": ("f.jpg", jpeg_bytes(), "image/jpeg")},
        data={"context": context(LAPTOP, CUP), "personality": "serious"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["report"]["mode"] == "deep"
    spill = next(f for f in body["report"]["findings"] if f["category"] == "SAFETY")
    assert spill["status"] == "CONFIRMED"
    assert [e["type"] for e in body["events"]] == ["DISCOVERED", "CONFIRMED"]


def test_create_scan_and_unknown_scan(client: TestClient) -> None:
    res = client.post("/api/scan", json={"personality": "unhinged"})
    assert res.status_code == 201
    assert res.json()["personality"] == "unhinged"
    assert client.get("/api/scan/does-not-exist").status_code == 404
    assert client.get("/api/scan/../../etc").status_code == 404


def _manifest(n: int) -> dict:
    frames = []
    for i in range(n):
        objects = [LAPTOP, CUP] if i < n - 1 else [{"label": "cat", "confidence": 0.9, "box": {"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.6}}]
        frames.append({"index": i, "t": 1.0 + i * 4, "scene": 0 if i < n - 1 else 1, "objects": objects})
    return {
        "duration_s": 4.0 * n,
        "width": 320,
        "height": 240,
        "name": "C:\\Users\\someone\\clip.mp4",
        "detector": "test-detector",
        "sampled_frames": 30,
        "redundant_removed": 12,
        "scenes": [{"index": 0, "start_t": 0, "end_t": 8}, {"index": 1, "start_t": 8, "end_t": 12}],
        "frames": frames,
        "events": [{"t": 8.5, "kind": "SCENE_CHANGE", "text": "Hard cut"}],
    }


def test_video_keyframes_timeline(client: TestClient) -> None:
    files = [("frames", (f"k{i}.jpg", jpeg_bytes(), "image/jpeg")) for i in range(3)]
    res = client.post("/api/analyze/video", files=files, data={"manifest": json.dumps(_manifest(3)), "personality": "brutal"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["sampling"]["processed_on"] == "browser"
    assert body["sampling"]["keyframes"] == 3
    assert body["video"]["name"] == "clip.mp4"
    kinds = [(e["kind"], e["source"]) for e in body["timeline"]]
    assert ("DISCOVERED", "demo") in kinds and ("CONFIRMED", "demo") in kinds and ("RESOLVED", "demo") in kinds
    assert ("SCENE_CHANGE", "local") in kinds
    times = [e["t"] for e in body["timeline"]]
    assert times == sorted(times)
    spill = next(f for f in body["report"]["findings"] if f["category"] == "SAFETY")
    assert spill["status"] == "RESOLVED"
    assert spill["first_seen_s"] == 1.0


def test_video_manifest_validation(client: TestClient) -> None:
    files = [("frames", (f"k{i}.jpg", jpeg_bytes(), "image/jpeg")) for i in range(2)]
    res = client.post("/api/analyze/video", files=files, data={"manifest": json.dumps(_manifest(3))})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "INVALID_MANIFEST"
    res = client.post("/api/analyze/video", files=files, data={"manifest": "{}"})
    assert res.status_code == 422
    res = client.post("/api/analyze/video", data={"personality": "serious"})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "MISSING_FILE"


def _avi_bytes(tmp_path) -> bytes:
    import cv2
    import numpy as np

    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for i in range(40):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        if i < 20:
            frame[:, :] = (40, 40, 200)
            frame[30:60, 20 + i : 60 + i] = (255, 255, 255)
        else:
            frame[:, :] = (200, 160, 30)
        writer.write(frame)
    writer.release()
    with open(path, "rb") as fh:
        return fh.read()


def test_video_server_fallback(client: TestClient, tmp_path) -> None:
    data = _avi_bytes(tmp_path)
    res = client.post("/api/analyze/video", files={"video": ("clip.avi", data, "video/x-msvideo")})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["sampling"]["processed_on"] == "server"
    assert body["sampling"]["scenes"] >= 2
    assert any(e["kind"] == "SCENE_CHANGE" for e in body["timeline"])
    assert body["video"]["duration_s"] > 3


def test_corrupted_video_is_rejected(client: TestClient) -> None:
    res = client.post("/api/analyze/video", files={"video": ("bad.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 500, "video/mp4")})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "VIDEO_UNREADABLE"


def test_cors_allows_lan_origins_only(client: TestClient) -> None:
    for origin in ("http://192.168.1.23:5173", "https://10.0.0.5:5173", "http://localhost:5173"):
        res = client.options(
            "/api/analyze/image",
            headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
        )
        assert res.status_code == 200
        assert res.headers.get("access-control-allow-origin") == origin
    res = client.options(
        "/api/analyze/image",
        headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in res.headers


def test_unknown_route_uses_error_envelope(client: TestClient) -> None:
    res = client.get("/api/nope")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "NOT_FOUND"


def test_no_stack_traces_leak(monkeypatch) -> None:
    async def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    app = create_app(make_settings())
    monkeypatch.setattr(app.state.ai, "diagnose", boom)
    # Starlette re-raises after sending the 500 so servers can log it; the
    # client must still only see the generic envelope.
    with TestClient(app, raise_server_exceptions=False) as client:
        res = client.post("/api/analyze/image", files={"image": ("d.jpg", jpeg_bytes(), "image/jpeg")})
    assert res.status_code == 500
    assert "secret internal detail" not in res.text
    assert res.json()["error"]["code"] == "INTERNAL_ERROR"


def test_exif_is_stripped(client: TestClient) -> None:
    from PIL import Image

    img = Image.new("RGB", (200, 100), (50, 60, 70))
    exif = Image.Exif()
    exif[0x010F] = "SecretCam"  # Make
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    from app.services.vision_service import prepare_image

    prepared = prepare_image(buf.getvalue(), max_edge=1024, max_pixels=10_000_000)
    assert b"SecretCam" not in prepared.jpeg
