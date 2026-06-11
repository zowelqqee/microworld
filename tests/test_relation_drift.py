import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.relations import Relation
from core.pattern_prediction import PatternBasedPredictor
from core.relation_drift import (
    ATOMIC_COMPONENT,
    ABSTRACT_COMPONENT,
    DIRECT_MATERIAL,
    RAW_MATERIAL,
    DEFAULT_DRIFT_PENALTY_TABLE,
    RelationDriftEngine,
    material_category,
)
from examples.pattern_audit_export import build_audit_rows


def _rels(*triples) -> list[Relation]:
    return [Relation(s, r, t) for s, r, t in triples]


def _write_csv(tmp_path, triples) -> str:
    path = tmp_path / "relations.csv"
    lines = ["source,relation_type,target"]
    lines.extend(f"{s},{r},{t}" for s, r, t in triples)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_relation_depth_tracking_depths_1_2_3():
    engine = RelationDriftEngine(_rels(
        ("table", "made_of", "tree"),
        ("tree", "made_of", "wood"),
        ("wood", "made_of", "carbon"),
    ))

    paths = engine.discover_relation_depths(max_depth=3)

    assert len(paths[1]) == 3
    assert any(path.nodes == ("table", "tree", "wood") for path in paths[2])
    assert any(path.nodes == ("table", "tree", "wood", "carbon") for path in paths[3])


def test_audit_export_outputs_path_length(tmp_path):
    csv_path = _write_csv(tmp_path, [
        ("a", "part_of", "b"),
        ("b", "part_of", "c"),
    ])

    rows = build_audit_rows(csv_path, min_count=1, limit=10)

    assert rows
    assert rows[0]["path_length"] == "2"


def test_material_categories():
    assert material_category("paper") == DIRECT_MATERIAL
    assert material_category("tree") == RAW_MATERIAL
    assert material_category("iron") == ATOMIC_COMPONENT


def test_detects_made_of_semantic_level_change():
    engine = RelationDriftEngine(_rels(
        ("blood", "made_of", "haemoglobin"),
        ("haemoglobin", "made_of", "iron"),
    ))

    examples = engine.detect_relation_drift(max_depth=2)

    assert len(examples) == 1
    assert examples[0].drift == f"{DIRECT_MATERIAL}->{ATOMIC_COMPONENT}"
    assert examples[0].path_length == 2


def test_same_material_level_not_flagged():
    engine = RelationDriftEngine(_rels(
        ("tool", "made_of", "steel"),
        ("steel", "made_of", "metal"),
    ))

    examples = engine.detect_relation_drift(max_depth=2)

    assert examples == []


def test_report_includes_audit_accuracy_and_examples():
    audit_rows = [
        {
            "source": "blood",
            "relation_type": "made_of",
            "target": "iron",
            "manual_label": "wrong",
        },
        {
            "source": "bowl",
            "relation_type": "made_of",
            "target": "iron",
            "manual_label": "plausible",
        },
    ]
    engine = RelationDriftEngine(_rels(
        ("blood", "made_of", "haemoglobin"),
        ("haemoglobin", "made_of", "iron"),
        ("bowl", "made_of", "steel"),
        ("steel", "made_of", "iron"),
    ))

    reports = engine.build_report(max_depth=2, audit_rows=audit_rows)
    made_of = next(report for report in reports if report.relation_type == "made_of")

    assert made_of.support == 2
    assert made_of.drift_support == 2
    assert made_of.reviewed == 2
    assert made_of.wrong == 1
    assert made_of.audit_accuracy == pytest.approx(0.5)
    assert made_of.examples


class TestDriftAwarePredictionScoring:
    def _pred(self, triples, source, target, **kwargs):
        preds = PatternBasedPredictor(_rels(*triples)).predict_from_bigrams(
            min_count=1,
            min_confidence=0.0,
            hub_penalty=False,
            use_relation_drift=True,
            **kwargs,
        )
        return next(
            p for p in preds
            if p.source == source and p.relation_type == "made_of" and p.target == target
        )

    def test_direct_material_has_no_penalty(self):
        pred = self._pred(
            [
                ("tool", "made_of", "steel"),
                ("steel", "made_of", "metal"),
            ],
            "tool",
            "metal",
        )

        assert pred.drift_type is None
        assert pred.drift_penalty == pytest.approx(1.0)
        assert "drift=" not in pred.reason

    def test_raw_material_gets_penalty(self):
        raw = PatternBasedPredictor(_rels(
            ("book", "made_of", "paper"),
            ("paper", "made_of", "wood"),
        )).predict_from_bigrams(
            min_count=1,
            min_confidence=0.0,
            hub_penalty=False,
            use_relation_drift=False,
        )[0]
        drift = self._pred(
            [
                ("book", "made_of", "paper"),
                ("paper", "made_of", "wood"),
            ],
            "book",
            "wood",
        )

        assert drift.drift_type == RAW_MATERIAL
        assert drift.drift_penalty == pytest.approx(DEFAULT_DRIFT_PENALTY_TABLE[RAW_MATERIAL])
        assert drift.confidence == pytest.approx(raw.confidence * 0.85)
        assert "drift=raw_material" in drift.reason
        assert "drift_penalty=0.85" in drift.reason

    def test_atomic_component_gets_stronger_penalty(self):
        drift = self._pred(
            [
                ("blood", "made_of", "haemoglobin"),
                ("haemoglobin", "made_of", "iron"),
            ],
            "blood",
            "iron",
        )

        assert drift.drift_type == ATOMIC_COMPONENT
        assert drift.drift_penalty == pytest.approx(0.65)
        assert "drift=atomic_component" in drift.reason

    def test_abstract_component_gets_penalty(self):
        drift = self._pred(
            [
                ("community", "made_of", "culture"),
                ("culture", "made_of", "ideals"),
            ],
            "community",
            "ideals",
        )

        assert drift.drift_type == ABSTRACT_COMPONENT
        assert drift.drift_penalty == pytest.approx(0.70)
        assert "drift=abstract_component" in drift.reason

    def test_non_made_of_relations_unaffected(self):
        raw = PatternBasedPredictor(_rels(
            ("a", "part_of", "b"),
            ("b", "part_of", "c"),
        )).predict_from_bigrams(
            min_count=1,
            min_confidence=0.0,
            hub_penalty=False,
        )[0]
        drift = PatternBasedPredictor(_rels(
            ("a", "part_of", "b"),
            ("b", "part_of", "c"),
        )).predict_from_bigrams(
            min_count=1,
            min_confidence=0.0,
            hub_penalty=False,
            use_relation_drift=True,
        )[0]

        assert drift.confidence == pytest.approx(raw.confidence)
        assert drift.drift_type is None
        assert drift.drift_penalty == pytest.approx(1.0)

    def test_audit_export_flag_passes_through(self, tmp_path):
        csv_path = _write_csv(tmp_path, [
            ("blood", "made_of", "haemoglobin"),
            ("haemoglobin", "made_of", "iron"),
        ])

        rows = build_audit_rows(
            csv_path,
            min_count=1,
            min_confidence=0.0,
            use_relation_drift=True,
            limit=10,
        )

        assert rows
        assert "drift=atomic_component" in rows[0]["reason"]
