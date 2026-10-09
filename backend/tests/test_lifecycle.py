"""Temporal lifecycle of local findings (config/temporal.json "lifecycle")."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import CUP, LAPTOP, obj, scene


def observe(client: TestClient, scan_id: str | None, *objects: dict, at_ms: float, view_id: int = 0, **extra) -> dict:
    body = {"scan_id": scan_id, "scene": scene(*objects, at_ms=at_ms, view_id=view_id, **extra)}
    res = client.post("/api/scan/observe", json=body)
    assert res.status_code == 200, res.text
    return res.json()


def bug(body: dict, rule: str = "spill_risk") -> dict:
    return next(f for f in body["scan"]["findings"] if f["rule"] == rule)


def test_full_lifecycle_without_ai(client: TestClient) -> None:
    first = observe(client, None, LAPTOP, CUP, at_ms=0)
    scan_id = first["scan"]["scan_id"]
    assert bug(first)["status"] == "DISCOVERED"
    assert [e["type"] for e in first["events"]] == ["DISCOVERED"]
    assert first["report"]["ai"]["status"] == "off"
    assert first["ai_suggestion"] is None

    # Seen again, but too soon: one noisy pair of frames never confirms a bug.
    early = observe(client, scan_id, LAPTOP, CUP, at_ms=500)
    assert bug(early)["status"] == "DISCOVERED"
    confirmed = observe(client, scan_id, LAPTOP, CUP, at_ms=1600)
    assert bug(confirmed)["status"] == "CONFIRMED"
    tracking = observe(client, scan_id, LAPTOP, CUP, at_ms=3200)
    assert bug(tracking)["status"] == "TRACKING"

    # The cup is taken away (laptop still in view): resolved only after
    # resolveAfterMs (3 s) and resolveObservations (2).
    gone1 = observe(client, scan_id, LAPTOP, at_ms=4000)
    assert bug(gone1)["status"] == "TRACKING"
    gone2 = observe(client, scan_id, LAPTOP, at_ms=5500)
    assert bug(gone2)["status"] == "TRACKING"
    resolved = observe(client, scan_id, LAPTOP, at_ms=6500)
    assert bug(resolved)["status"] == "RESOLVED"
    assert "no longer measured" in bug(resolved)["resolved_note"].lower()
    assert resolved["scan"]["counts"]["resolved"] == 1

    # It comes back: REOPENED, and needs confirmation again.
    back = observe(client, scan_id, LAPTOP, CUP, at_ms=8000)
    assert bug(back)["status"] == "DISCOVERED"
    assert "REOPENED" in [e["type"] for e in back["events"]]

    state = client.get(f"/api/scan/{scan_id}").json()
    assert [e["type"] for e in state["events"]] == ["DISCOVERED", "CONFIRMED", "TRACKING", "RESOLVED", "REOPENED"]
    assert state["observations"] == 8 and state["analyses"] == 0
    assert state["engine"].startswith("local-diagnostics/")


def test_view_change_marks_out_of_view_instead_of_resolving(client: TestClient) -> None:
    scan_id = observe(client, None, LAPTOP, CUP, at_ms=0)["scan"]["scan_id"]
    observe(client, scan_id, LAPTOP, CUP, at_ms=1600)
    for at in (3000, 5000, 9000):
        moved = observe(client, scan_id, obj("t9", "chair", 0.2, 0.2, 0.3, 0.5), at_ms=at, view_id=1)
    finding = bug(moved)
    assert finding["status"] == "CONFIRMED"
    assert finding["out_of_view"] is True


def test_object_leaving_through_the_edge_is_not_resolved(client: TestClient) -> None:
    edge_cup = obj("t2", "cup", 0.66, 0.45, 0.33, 0.12)  # touches the right frame edge
    scan_id = observe(client, None, LAPTOP, edge_cup, at_ms=0)["scan"]["scan_id"]
    observe(client, scan_id, LAPTOP, edge_cup, at_ms=1600)
    for at in (3000, 5000, 7000):
        after = observe(client, scan_id, LAPTOP, at_ms=at)
    assert bug(after)["status"] == "CONFIRMED"
    assert bug(after)["out_of_view"] is True


def test_new_track_id_continues_the_same_finding(client: TestClient) -> None:
    scan_id = observe(client, None, LAPTOP, CUP, at_ms=0)["scan"]["scan_id"]
    # The cup's track was lost and re-acquired with a new id at the same place.
    reacquired = {**CUP, "id": "t7"}
    body = observe(client, scan_id, LAPTOP, reacquired, at_ms=1600)
    spills = [f for f in body["scan"]["findings"] if f["rule"] == "spill_risk"]
    assert len(spills) == 1
    assert spills[0]["status"] == "CONFIRMED"


def test_deep_scan_with_both_detectors_confirms_immediately(client: TestClient) -> None:
    verified = [{**LAPTOP, "verified": True, "source": "fused"}, {**CUP, "verified": True, "source": "fused"}]
    res = client.post("/api/scan/deep", data={"scene": __import__("json").dumps(scene(*verified, at_ms=0))})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["report"]["mode"] == "deep"
    assert bug(body)["status"] == "CONFIRMED"
    assert [e["type"] for e in body["events"]] == ["DISCOVERED", "CONFIRMED"]
    assert body["report"]["ai"]["status"] == "off"


def test_scores_follow_open_findings(client: TestClient) -> None:
    # No findings among the recognised objects is not an all-clear: unrated, and the status says why.
    clean = observe(client, None, LAPTOP, at_ms=0)
    assert clean["scan"]["system_score"] is None and clean["scan"]["issue_score"] is None
    assert clean["scan"]["status"] == "LIMITED"
    assert clean["scan"]["inspection"]["coverage"] == "limited"
    risky = observe(client, clean["scan"]["scan_id"], LAPTOP, CUP, at_ms=1000)
    assert risky["scan"]["system_score"] is None  # the scene is still not rated (coverage limited)...
    assert risky["scan"]["issue_score"] < 100  # ...but the measured issues are
    assert risky["scan"]["status"] == "DEGRADED"  # an open HIGH finding is reported whatever the coverage
    assert risky["report"]["system_name"].endswith(risky["report"]["scene"]["version"])


def test_scene_naming_uses_attributes(client: TestClient) -> None:
    body = observe(client, None, LAPTOP, obj("t3", "keyboard", 0.3, 0.75, 0.3, 0.1), at_ms=0)
    assert body["scan"]["scene"]["name"] == "WORKSPACE"
    assert body["scan"]["system_name"].startswith("WORKSPACE_v")
