"""HTTP-level tests in local-only mode (no API key anywhere, no network)."""

from __future__ import annotations

import io
import json

from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import CUP, LAPTOP, jpeg_bytes, make_settings, obj, png_bytes, scene, scene_json


def test_health_reports_local_cv_active(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["features"] == {"local_cv": True, "ai_reasoning": False, "server_video_fallback": True}
    assert body["local"]["labels"] == 80
    assert body["local"]["detectors"] == ["efficientdet_lite0", "yolox_s"]
    assert "spill_risk" in body["local"]["rules"]
    assert body["limits"]["ai_calls_per_minute"] == 6
    assert body["ai"]["state"] == "off"


def test_metrics_endpoint(client: TestClient) -> None:
    client.post("/api/analyze/scene", json={"scene": scene(LAPTOP, CUP)})
    body = client.get("/api/metrics").json()
    assert body["local"]["evaluations"]["count"] == 1
    assert body["local"]["findings_by_rule"]["spill_risk"] == 1
    assert body["ai"]["calls"] == 0 and body["ai"]["state"] == "off"


def test_scene_analysis_needs_no_upload(client: TestClient) -> None:
    res = client.post("/api/analyze/scene", json={"personality": "unhinged", "scene": scene(LAPTOP, CUP)})
    assert res.status_code == 200, res.text
    report = res.json()
    assert report["provider"] == "local" and report["engine"].startswith("local-diagnostics/")
    assert report["ai"]["status"] == "off"
    assert report["mode"] == "image" and report["image"] is None
    spill = next(f for f in report["findings"] if f["rule"] == "spill_risk")
    assert spill["source"] == "local" and spill["status"] == "CONFIRMED"
    assert "0xC0FFEE" in spill["title"]
    for key in ("severity", "category", "title", "evidence", "inference", "impact", "recommendation", "confidence", "measurements"):
        assert key in spill
    assert report["system_name"].startswith("DEVICE_AREA_v")
    assert report["objects"][0]["source"] == "fast"
    assert report["relationships"] and report["relationships"][0]["source"] == "local"
    assert 0 <= report["system_score"] < 100


def test_image_debug_without_ai_uses_the_scene(client: TestClient) -> None:
    res = client.post(
        "/api/analyze/image",
        files={"image": ("desk.jpg", jpeg_bytes(), "image/jpeg")},
        data={"personality": "brutal", "scene": scene_json(LAPTOP, CUP)},
    )
    assert res.status_code == 200, res.text
    report = res.json()
    assert report["ai"]["status"] == "off"
    assert any(f["rule"] == "spill_risk" for f in report["findings"])
    assert report["image"]["width"] == 320


def test_image_without_scene_gets_signal_diagnostics(client: TestClient) -> None:
    dark = jpeg_bytes(color=(8, 8, 8))
    report = client.post("/api/analyze/image", files={"image": ("dark.jpg", dark, "image/jpeg")}).json()
    assert any(f["rule"] == "lighting" for f in report["findings"])
    assert any("only image signals" in w for w in report["warnings"])


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
    res = client.post("/api/analyze/image", files={"image": ("broken.jpg", jpeg_bytes()[:200], "image/jpeg")})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "INVALID_UPLOAD"


def test_rejects_empty_and_missing_files(client: TestClient) -> None:
    assert client.post("/api/analyze/image", files={"image": ("e.jpg", b"", "image/jpeg")}).status_code == 422
    res = client.post("/api/analyze/image", data={"personality": "serious"})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "INVALID_REQUEST"


def test_rejects_oversized_image() -> None:
    with TestClient(create_app(make_settings(max_image_bytes=2_000))) as client:
        res = client.post("/api/analyze/image", files={"image": ("big.jpg", jpeg_bytes(800, 600), "image/jpeg")})
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_rejects_too_many_pixels() -> None:
    with TestClient(create_app(make_settings(max_image_pixels=10_000))) as client:
        res = client.post("/api/analyze/image", files={"image": ("big.jpg", jpeg_bytes(400, 300), "image/jpeg")})
    assert res.status_code == 413


def test_request_body_cap() -> None:
    app = create_app(make_settings(max_video_bytes=1_000, max_image_bytes=1_000, max_frame_bytes=1_000, max_video_keyframes=1))
    with TestClient(app) as client:
        res = client.post("/api/analyze/image", files={"image": ("big.jpg", b"\xff\xd8\xff" + b"0" * 5_000_000, "image/jpeg")})
    assert res.status_code == 413


def test_rejects_bad_personality_and_scene(client: TestClient) -> None:
    res = client.post("/api/analyze/image", files={"image": ("d.jpg", jpeg_bytes(), "image/jpeg")}, data={"personality": "sarcastic"})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "INVALID_PERSONALITY"
    res = client.post("/api/analyze/image", files={"image": ("d.jpg", jpeg_bytes(), "image/jpeg")}, data={"scene": "{not json"})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "INVALID_SCENE"
    huge = json.dumps(scene(*[obj(f"t{i}", "cup", 0.1, 0.1, 0.1, 0.1) for i in range(80)]))
    with TestClient(create_app(make_settings(max_scene_chars=500))) as small:
        res = small.post("/api/analyze/frame", data={"scene": huge})
    assert res.status_code == 413


def test_observe_rejects_invalid_scene(client: TestClient) -> None:
    res = client.post("/api/scan/observe", json={"scene": {"objects": "nope"}})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "INVALID_REQUEST"


def test_live_frame_without_ai_never_needs_the_image(client: TestClient) -> None:
    res = client.post("/api/analyze/frame", data={"scene": scene_json(LAPTOP, CUP), "trigger": "deep_scan"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["report"]["ai"]["status"] == "off"
    assert body["scan"]["observations"] == 1
    res = client.post("/api/analyze/frame", data={"scene": scene_json(LAPTOP), "focus": "not-a-bug"})
    assert res.status_code == 422 and res.json()["error"]["code"] == "INVALID_FOCUS"


def test_create_and_delete_scan(client: TestClient) -> None:
    res = client.post("/api/scan", json={"personality": "unhinged"})
    assert res.status_code == 201
    scan_id = res.json()["scan_id"]
    assert res.json()["personality"] == "unhinged" and res.json()["system_score"] is None
    observed = client.post("/api/scan/observe", json={"scan_id": scan_id, "scene": scene(LAPTOP, CUP)}).json()
    assert observed["scan"]["scan_id"] == scan_id and observed["scan"]["personality"] == "unhinged"
    assert client.delete(f"/api/scan/{scan_id}").status_code == 204
    missing = client.get(f"/api/scan/{scan_id}")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "SCAN_NOT_FOUND"
    assert client.get("/api/scan/../../etc").status_code == 404


# --------------------------------------------------------------------------
# Video
# --------------------------------------------------------------------------


def _samples(n: int = 12) -> list[dict]:
    samples = []
    for i in range(n):
        t = i * 0.5
        objects = [LAPTOP, CUP] if t < 4.0 else [LAPTOP]
        if t >= 5.0:
            objects = [obj("t9", "cat", 0.1, 0.1, 0.4, 0.6, 0.9)]
        samples.append({"t": t, "scene": 0 if t < 5.0 else 1, "objects": objects, "brightness": 0.5, "sharpness": 0.7})
    return samples


def _manifest(**extra) -> dict:
    return {
        "duration_s": 6.0,
        "width": 1280,
        "height": 720,
        "name": "C:\\Users\\someone\\clip.mp4",
        "detector": "efficientdet_lite0",
        "sampled_frames": 12,
        "redundant_removed": 3,
        "scenes": [{"index": 0, "start_t": 0, "end_t": 4.5}, {"index": 1, "start_t": 5.0, "end_t": 5.5}],
        "frames": [
            {"index": 0, "t": 1.0, "scene": 0, "objects": [{"label": "laptop", "confidence": 0.9}]},
            {"index": 1, "t": 5.5, "scene": 1, "objects": [{"label": "cat", "confidence": 0.9}]},
        ],
        "samples": _samples(),
        "events": [{"t": 5.0, "kind": "SCENE_CHANGE", "text": "Hard cut"}],
        **extra,
    }


def test_video_manifest_only_local_timeline(client: TestClient) -> None:
    res = client.post("/api/analyze/video", data={"manifest": json.dumps(_manifest()), "personality": "brutal"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["report"]["ai"]["status"] == "off" and body["report"]["mode"] == "video"
    assert body["sampling"]["processed_on"] == "browser" and body["sampling"]["keyframes"] == 2
    assert body["sampling"]["tracked_objects"] == 3
    assert body["video"]["name"] == "clip.mp4"
    spill = next(f for f in body["report"]["findings"] if f["rule"] == "spill_risk")
    assert spill["first_seen_s"] == 0.0
    # Cup removed at 4.0 s with the laptop still in view -> resolved 3 s later... but the
    # cut to scene 1 at 5.0 s happens first, so the finding is out of view, not resolved.
    assert spill["status"] in ("CONFIRMED", "TRACKING")
    kinds = [(e["kind"], e["source"]) for e in body["timeline"]]
    assert ("DISCOVERED", "local") in kinds and ("CONFIRMED", "local") in kinds and ("SCENE_CHANGE", "local") in kinds
    times = [e["t"] for e in body["timeline"]]
    assert times == sorted(times)


def test_video_resolution_in_the_same_scene(client: TestClient) -> None:
    samples = [{"t": i * 0.5, "scene": 0, "objects": [LAPTOP, CUP] if i < 6 else [LAPTOP]} for i in range(16)]
    manifest = {"duration_s": 8.0, "samples": samples}
    body = client.post("/api/analyze/video", data={"manifest": json.dumps(manifest)}).json()
    spill = next(f for f in body["report"]["findings"] if f["rule"] == "spill_risk")
    assert spill["status"] == "RESOLVED"
    resolved = next(e for e in body["timeline"] if e["kind"] == "RESOLVED")
    assert resolved["t"] >= 2.5 + 3.0


def test_old_clients_with_keyframes_only_still_work(client: TestClient) -> None:
    frames = [
        {"index": i, "t": 1.0 + 2 * i, "scene": 0, "objects": [
            {"track_id": "t1", "label": "laptop", "confidence": 0.9, "box": LAPTOP["box"]},
            {"track_id": "t2", "label": "cup", "confidence": 0.85, "box": CUP["box"]},
        ]}
        for i in range(3)
    ]
    res = client.post("/api/analyze/video", data={"manifest": json.dumps({"duration_s": 7.0, "frames": frames})})
    assert res.status_code == 200, res.text
    assert next(f for f in res.json()["report"]["findings"] if f["rule"] == "spill_risk")["status"] == "TRACKING"


def test_video_manifest_validation(client: TestClient) -> None:
    files = [("frames", (f"k{i}.jpg", jpeg_bytes(), "image/jpeg")) for i in range(3)]
    res = client.post("/api/analyze/video", files=files, data={"manifest": json.dumps(_manifest())})
    assert res.status_code == 422 and res.json()["error"]["code"] == "INVALID_MANIFEST"
    res = client.post("/api/analyze/video", data={"manifest": json.dumps({"duration_s": 3})})
    assert res.status_code == 422 and res.json()["error"]["code"] == "INVALID_MANIFEST"
    res = client.post("/api/analyze/video", data={"personality": "serious"})
    assert res.status_code == 422 and res.json()["error"]["code"] == "MISSING_FILE"


def _avi_bytes(tmp_path, dark: bool = False) -> bytes:
    import cv2
    import numpy as np

    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for i in range(40):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        if i < 20:
            frame[:, :] = (10, 10, 12) if dark else (40, 40, 200)
            frame[30:60, 20 + i : 60 + i] = (40, 40, 40) if dark else (255, 255, 255)
        else:
            frame[:, :] = (200, 160, 30)
        writer.write(frame)
    writer.release()
    with open(path, "rb") as fh:
        return fh.read()


def test_video_server_fallback_local_signals(client: TestClient, tmp_path) -> None:
    res = client.post("/api/analyze/video", files={"video": ("clip.avi", _avi_bytes(tmp_path, dark=True), "video/x-msvideo")})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["sampling"]["processed_on"] == "server" and body["sampling"]["scenes"] >= 2
    assert any(e["kind"] == "SCENE_CHANGE" for e in body["timeline"])
    assert body["video"]["duration_s"] > 3
    assert any(f["rule"] == "lighting" for f in body["report"]["findings"])
    assert any("No object detector runs on the server" in w for w in body["report"]["warnings"])
    assert body["report"]["ai"]["status"] == "off"


def test_corrupted_video_is_rejected(client: TestClient) -> None:
    res = client.post("/api/analyze/video", files={"video": ("bad.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 500, "video/mp4")})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "VIDEO_UNREADABLE"


# --------------------------------------------------------------------------
# Transport, privacy and deployment
# --------------------------------------------------------------------------


def test_cors_allows_lan_origins_only(client: TestClient) -> None:
    for origin in ("http://192.168.1.23:5173", "https://10.0.0.5:5173", "http://localhost:5173"):
        res = client.options("/api/analyze/image", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
        assert res.status_code == 200
        assert res.headers.get("access-control-allow-origin") == origin
    res = client.options(
        "/api/analyze/image", headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "POST"}
    )
    assert "access-control-allow-origin" not in res.headers


def test_unknown_route_uses_error_envelope(client: TestClient) -> None:
    res = client.get("/api/nope")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "NOT_FOUND"


def test_serves_built_frontend_when_configured(tmp_path) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>Reality Debugger</title>")
    (tmp_path / "assets" / "vision.wasm").write_bytes(b"\0asm\x01\0\0\0")
    with TestClient(create_app(make_settings(frontend_dist=tmp_path))) as client:
        page = client.get("/")
        assert page.status_code == 200 and "<title>Reality Debugger</title>" in page.text
        assert client.get("/assets/vision.wasm").headers["content-type"] == "application/wasm"
        assert client.get("/api/health").json()["status"] == "ok"
        missing = client.get("/api/nope")
        assert missing.status_code == 404 and missing.json()["error"]["code"] == "NOT_FOUND"


def test_no_stack_traces_leak(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    app = create_app(make_settings())
    monkeypatch.setattr(app.state.engine, "evaluate", boom)
    with TestClient(app, raise_server_exceptions=False) as client:
        res = client.post("/api/analyze/scene", json={"scene": scene(LAPTOP)})
    assert res.status_code == 500
    assert "secret internal detail" not in res.text
    assert res.json()["error"]["code"] == "INTERNAL_ERROR"


def test_exif_is_stripped() -> None:
    from PIL import Image

    from app.services.vision_service import prepare_image

    img = Image.new("RGB", (200, 100), (50, 60, 70))
    exif = Image.Exif()
    exif[0x010F] = "SecretCam"  # Make
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    prepared = prepare_image(buf.getvalue(), max_edge=1024, max_pixels=10_000_000)
    assert b"SecretCam" not in prepared.jpeg


def test_secrets_are_redacted_from_logs(caplog) -> None:
    import logging

    from app.main import _RedactSecrets

    record = logging.LogRecord("x", logging.INFO, __file__, 1, "key=%s", ("AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ012345",), None)
    _RedactSecrets().filter(record)
    assert "AIza" not in record.getMessage()
