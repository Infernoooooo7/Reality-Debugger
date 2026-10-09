"""Local engine: ontology, configuration, geometry and the diagnostic rules (no AI)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from app.config import ROOT_DIR
from app.runtime_config import ConfigError, load_config, values
from app.schemas.common import AnalysisMode, Box, Personality, Severity
from app.schemas.scene import SceneModel
from app.services import geometry as geo
from app.services.local_diagnostics import RULE_REQUIREMENTS, LocalDiagnosticEngine
from app.services.ontology import load_ontology
from tests.conftest import CUP, LAPTOP, obj, scene

CONFIG = load_config()
ONTOLOGY = load_ontology()
ENGINE = LocalDiagnosticEngine(CONFIG, ONTOLOGY)


def run(*objects: dict, mode: AnalysisMode = AnalysisMode.IMAGE, timers: dict | None = None, **extra) -> dict[str, list]:
    model = SceneModel.model_validate(scene(*objects, **extra))
    found: dict[str, list] = {}
    for f in ENGINE.evaluate(model, mode=mode, personality=Personality.SERIOUS, timers=timers):
        found.setdefault(f.rule, []).append(f)
    return found


# -- ontology & configuration ------------------------------------------------------------


def test_ontology_comes_from_model_metadata() -> None:
    manifests = {
        p.stem.split(".")[0]: json.loads(p.read_text())
        for p in (ROOT_DIR / "frontend" / "public" / "models").glob("*.manifest.json")
    }
    labels = set()
    for manifest in manifests.values():
        labels |= {label for label in manifest["labels"] if label}
    assert set(ONTOLOGY.labels) == labels
    assert len(labels) == 80  # COCO classes of the bundled detectors
    assert ONTOLOGY.models == sorted(manifests)


@pytest.mark.parametrize(
    ("label", "attribute"),
    [
        ("cup", "liquid_container"),
        ("wine glass", "liquid_container"),
        ("laptop", "electronic"),
        ("cell phone", "electronic"),
        ("pizza", "food"),
        ("knife", "sharp"),
        ("dining table", "surface"),
        ("cat", "animal"),
    ],
)
def test_attributes_are_derived_from_wordnet(label: str, attribute: str) -> None:
    assert ONTOLOGY.has(label, attribute)


def test_people_are_not_objects_with_attributes() -> None:
    assert not (ONTOLOGY.attributes("person") & {"electronic", "food", "liquid_container", "sharp"})


def test_lexicon_maps_free_text_to_attributes() -> None:
    labels, attributes = ONTOLOGY.match_text("coffee mugs")
    assert "liquid_container" in attributes or "cup" in labels
    assert "electronic" in ONTOLOGY.attributes("notebook computer") or ONTOLOGY.match_text("notebook computer")[0]


def test_every_parameter_is_documented() -> None:
    for name in ("vision", "detection", "tracking", "temporal", "diagnostics", "ai"):
        raw = json.loads((ROOT_DIR / "config" / f"{name}.json").read_text())

        def walk(node: object, path: str) -> None:
            if isinstance(node, dict):
                if "value" in node:
                    assert node.get("purpose"), path
                    assert node.get("source") in {"model-doc", "paper", "empirical", "default", "heuristic"}, path
                    return
                for key, child in node.items():
                    walk(child, f"{path}.{key}")

        walk(raw, name)
        assert values(raw)


def test_config_errors_are_explicit(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path)


def test_rules_never_test_label_strings() -> None:
    """The engine reasons over ontology attributes, never over class names."""
    source = (ROOT_DIR / "backend" / "app" / "services" / "local_diagnostics.py").read_text()
    tree = ast.parse(source)
    strings = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert not strings & set(ONTOLOGY.labels)


def test_armed_rules_reflect_the_vocabulary() -> None:
    armed = set(ENGINE.armed_rules())
    assert {"spill_risk", "food_near_electronics", "sharp_exposed", "dense_region", "overlap_cluster", "lighting"} <= armed
    # No bundled detector has cable or outlet classes.
    assert "cable_congestion" not in armed and "liquid_near_outlet" not in armed
    assert set(RULE_REQUIREMENTS) <= set(ENGINE.rule_names)


# -- geometry ---------------------------------------------------------------------------------------


def test_geometry_measurements() -> None:
    a = Box(x=0.1, y=0.1, w=0.2, h=0.2)
    b = Box(x=0.3, y=0.1, w=0.2, h=0.2)
    assert geo.edge_gap(a, b) == pytest.approx(0.0)
    assert geo.iou(a, a) == pytest.approx(1.0)
    assert geo.iou(a, b) == pytest.approx(0.0)
    far = Box(x=0.8, y=0.8, w=0.1, h=0.1)
    prox = geo.proximity(a, far, touch_gap=0.01, near_relative_gap=0.35)
    assert not prox.touching and not prox.near
    assert geo.occupancy([Box(x=0, y=0, w=0.5, h=1)]) == pytest.approx(0.5, abs=0.02)
    assert geo.near_edge(Box(x=0.0, y=0.4, w=0.1, h=0.1), 0.04)
    assert geo.direction(a, b) == "left of"


# -- rules --------------------------------------------------------------------------------------------


def test_spill_risk_severity_depends_on_distance() -> None:
    touching = run(LAPTOP, CUP)["spill_risk"][0]
    assert touching.severity == Severity.HIGH
    assert touching.object_ids == ["t2", "t1"]
    assert touching.measurements["edge_gap"] == pytest.approx(0.0, abs=0.011)
    near_cup = obj("t2", "cup", 0.70, 0.45, 0.08, 0.12)
    near = run(LAPTOP, near_cup)["spill_risk"][0]
    assert near.severity == Severity.MEDIUM
    far_cup = obj("t2", "cup", 0.92, 0.05, 0.06, 0.08)
    assert "spill_risk" not in run(LAPTOP, far_cup)


def test_food_near_electronics_and_animal() -> None:
    found = run(LAPTOP, obj("t3", "sandwich", 0.20, 0.45, 0.10, 0.10), obj("t4", "cat", 0.62, 0.30, 0.2, 0.3))
    assert "food_near_electronics" in found
    assert "animal_near_equipment" in found


def test_sharp_object_needs_persistence_in_live_mode() -> None:
    knife = obj("t5", "knife", 0.4, 0.4, 0.2, 0.05, age_ms=500)
    assert "sharp_exposed" in run(knife)  # single image: no persistence available
    assert "sharp_exposed" not in run(knife, mode=AnalysisMode.LIVE)
    persistent = obj("t5", "knife", 0.4, 0.4, 0.2, 0.05, age_ms=4000)
    assert "sharp_exposed" in run(persistent, mode=AnalysisMode.LIVE)


def test_edge_placement_on_surface() -> None:
    table = obj("t6", "dining table", 0.1, 0.5, 0.8, 0.45)
    glass = obj("t7", "wine glass", 0.11, 0.6, 0.05, 0.12)
    found = run(table, glass)["edge_placement"][0]
    assert found.measurements["side"] == "left"
    assert "edge_placement" not in run(table, obj("t7", "wine glass", 0.45, 0.6, 0.05, 0.12))


def test_object_count_alone_is_not_clutter() -> None:
    # Nine items spread over the whole frame: many objects, but no measured concentration or overlap.
    spread = [obj(f"t{i}", "book", 0.02 + 0.11 * (i % 9), 0.05 + 0.3 * (i % 3), 0.05, 0.06) for i in range(9)]
    found = run(*spread)
    assert "dense_region" not in found and "overlap_cluster" not in found


def test_dense_region_measures_concentration() -> None:
    cluster = [obj(f"t{i}", "book", 0.40 + 0.03 * (i % 3), 0.40 + 0.04 * (i // 3), 0.025, 0.035) for i in range(6)]
    far = obj("t9", "cup", 0.05, 0.05, 0.05, 0.08)
    found = run(*cluster, far)["dense_region"][0]
    assert found.measurements["objects_in_region"] == 6 and found.measurements["recognised_loose_objects"] == 7
    assert "t9" not in found.object_ids
    assert "not counted" in found.inference  # interpretation stays cautious and states the coverage limit
    people = [obj(f"p{i}", "person", 0.40 + 0.03 * (i % 3), 0.40 + 0.04 * (i // 3), 0.025, 0.035) for i in range(6)]
    assert "dense_region" not in run(*people)


def test_overlap_cluster_and_surface_congestion() -> None:
    stack = [obj("a", "book", 0.4, 0.4, 0.2, 0.2), obj("b", "book", 0.42, 0.42, 0.2, 0.2), obj("c", "book", 0.44, 0.44, 0.2, 0.2)]
    group = run(*stack)["overlap_cluster"][0]
    assert sorted(group.object_ids) == ["a", "b", "c"] and group.measurements["max_overlap_of_smaller"] > 0.8
    assert "cannot tell" in group.inference
    assert "overlap_cluster" not in run(*stack[:2])  # one overlapping pair is normal
    table = obj("t", "dining table", 0.2, 0.4, 0.4, 0.4)
    crowded = [table] + [obj(f"o{i}", "bowl", 0.2 + 0.1 * (i % 4), 0.4 + 0.1 * (i // 4), 0.1, 0.1) for i in range(12)]
    assert "surface_congestion" in run(*crowded)


def test_keep_clear_zone_reports_objects_inside_it() -> None:
    zone = {"id": "z1", "name": "keyboard area", "box": {"x": 0.3, "y": 0.6, "w": 0.4, "h": 0.3}}
    cup_inside = obj("c1", "cup", 0.4, 0.65, 0.08, 0.12)
    found = run(LAPTOP, cup_inside, zones=[zone])["keep_clear_zone"][0]
    assert found.object_ids == ["c1"] and "keyboard area" in found.title
    assert "keep_clear_zone" not in run(LAPTOP, obj("c1", "cup", 0.85, 0.1, 0.08, 0.12), zones=[zone])
    assert "keep_clear_zone" not in run(LAPTOP, cup_inside)  # no zone defined, nothing to check


def test_signal_rules() -> None:
    dark = run(signals={"brightness": 0.1})["lighting"][0]
    assert dark.measurements["brightness"] == pytest.approx(0.1)
    assert "lighting" not in run(signals={"brightness": 0.5})
    assert "blur" in run(signals={"sharpness": 0.1})
    assert "blur" not in run(signals={"sharpness": 0.8})


def test_view_obstruction_needs_duration() -> None:
    timers: dict[str, float] = {}
    assert "view_obstruction" not in run(signals={"brightness": 0.02}, mode=AnalysisMode.LIVE, timers=timers, at_ms=0)
    assert "view_obstruction" not in run(signals={"brightness": 0.02}, mode=AnalysisMode.LIVE, timers=timers, at_ms=1000)
    assert "view_obstruction" in run(signals={"brightness": 0.02}, mode=AnalysisMode.LIVE, timers=timers, at_ms=2500)
    assert "view_obstruction" not in run(signals={"brightness": 0.5}, mode=AnalysisMode.LIVE, timers=timers, at_ms=3000)


def test_temporal_object_rules() -> None:
    occluded = obj("t1", "bottle", 0.4, 0.4, 0.1, 0.2, occlusion=0.6, occluded_ms=6000)
    assert "persistent_occlusion" in run(occluded, mode=AnalysisMode.LIVE)
    assert "persistent_occlusion" not in run(occluded, mode=AnalysisMode.IMAGE)
    mover = obj("t2", "mouse", 0.4, 0.4, 0.05, 0.05, reversals=5)
    assert "repeated_movement" in run(mover, mode=AnalysisMode.LIVE)
    stale = obj("t3", "cup", 0.4, 0.4, 0.05, 0.08, static_ms=400_000)
    assert "long_presence" in run(stale, mode=AnalysisMode.LIVE)


def test_findings_carry_evidence_and_measurements() -> None:
    finding = run(LAPTOP, CUP)["spill_risk"][0]
    assert "cup" in finding.evidence and "laptop" in finding.evidence
    assert finding.recommendation and finding.inference and finding.impact
    assert 0 < finding.confidence <= 1
    assert finding.box is not None
    assert {"edge_gap", "relative_gap", "overlap_iou"} <= set(finding.measurements)


def test_personality_changes_tone_not_facts() -> None:
    model = SceneModel.model_validate(scene(LAPTOP, CUP))
    serious = ENGINE.evaluate(model, mode=AnalysisMode.IMAGE, personality=Personality.SERIOUS)[0]
    unhinged = ENGINE.evaluate(model, mode=AnalysisMode.IMAGE, personality=Personality.UNHINGED)[0]
    assert serious.title != unhinged.title
    assert serious.evidence == unhinged.evidence and serious.measurements == unhinged.measurements
    assert serious.quip == "" and unhinged.quip


def test_relations_are_computed_from_boxes() -> None:
    table = obj("t9", "dining table", 0.1, 0.3, 0.8, 0.6)
    model = SceneModel.model_validate(scene(LAPTOP, CUP, table))
    relations = ENGINE.relations(model)
    kinds = {(r.subject_label, r.relation, r.object_label) for r in relations}
    assert ("laptop", "on", "dining table") in kinds
    assert any(r.relation in {"touching", "overlaps"} and {r.subject_label, r.object_label} == {"cup", "laptop"} for r in relations)


def test_invalid_objects_are_dropped_individually() -> None:
    model = SceneModel.model_validate(scene(LAPTOP, {"id": "x", "label": "cup"}, {"label": "box", "box": [0, 0, 2, 2]}))
    assert [o.id for o in model.objects] == ["t1"]
