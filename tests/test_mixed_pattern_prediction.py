import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.relations import Relation
from core.pattern_prediction import PatternBasedPredictor
from examples.pattern_audit_export import build_audit_rows


def _rels(*triples) -> list[Relation]:
    return [Relation(s, r, t) for s, r, t in triples]


def _keys(preds):
    return {(p.source, p.relation_type, p.target) for p in preds}


def _conf(count: int) -> float:
    return min(0.95, 0.5 + 0.05 * math.log(count + 1))


def _write_csv(tmp_path, triples) -> str:
    path = tmp_path / "relations.csv"
    lines = ["source,relation_type,target"]
    lines.extend(f"{s},{r},{t}" for s, r, t in triples)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


class TestMixedRules:
    def test_is_a_capable_of_predicts_capable_of(self):
        preds = PatternBasedPredictor(_rels(
            ("sparrow", "is_a", "bird"),
            ("bird", "capable_of", "fly"),
        )).predict_from_mixed_bigrams(min_count=1)

        assert ("sparrow", "capable_of", "fly") in _keys(preds)

    def test_is_a_has_property_predicts_has_property(self):
        preds = PatternBasedPredictor(_rels(
            ("rose", "is_a", "flower"),
            ("flower", "has_property", "colorful"),
        )).predict_from_mixed_bigrams(min_count=1)

        assert ("rose", "has_property", "colorful") in _keys(preds)

    def test_is_a_used_for_predicts_used_for(self):
        preds = PatternBasedPredictor(_rels(
            ("hammer", "is_a", "tool"),
            ("tool", "used_for", "fixing"),
        )).predict_from_mixed_bigrams(min_count=1)

        assert ("hammer", "used_for", "fixing") in _keys(preds)

    def test_is_a_has_a_predicts_has_a(self):
        preds = PatternBasedPredictor(_rels(
            ("car", "is_a", "vehicle"),
            ("vehicle", "has_a", "wheel"),
        )).predict_from_mixed_bigrams(min_count=1)

        assert ("car", "has_a", "wheel") in _keys(preds)

    def test_part_of_made_of_predicts_made_of(self):
        preds = PatternBasedPredictor(_rels(
            ("door", "part_of", "car"),
            ("car", "made_of", "metal"),
        )).predict_from_mixed_bigrams(min_count=1)

        assert ("door", "made_of", "metal") in _keys(preds)

    def test_unsafe_mixed_patterns_are_ignored(self):
        preds = PatternBasedPredictor(_rels(
            ("wheel", "part_of", "car"),
            ("car", "capable_of", "move"),
        )).predict_from_mixed_bigrams(min_count=1)

        assert preds == []

    def test_existing_output_edge_is_not_duplicated(self):
        preds = PatternBasedPredictor(_rels(
            ("sparrow", "is_a", "bird"),
            ("bird", "capable_of", "fly"),
            ("sparrow", "capable_of", "fly"),
        )).predict_from_mixed_bigrams(min_count=1)

        assert ("sparrow", "capable_of", "fly") not in _keys(preds)

    def test_disabled_mixed_relation_skipped_by_default(self):
        preds = PatternBasedPredictor(_rels(
            ("statue", "is_a", "landmark"),
            ("landmark", "at_location", "city"),
        )).predict_from_mixed_bigrams(
            min_count=1,
            min_confidence=0.0,
            allowed_rules={("is_a", "at_location"): "at_location"},
        )

        assert preds == []

    def test_disabled_mixed_relation_can_be_included(self):
        preds = PatternBasedPredictor(_rels(
            ("statue", "is_a", "landmark"),
            ("landmark", "at_location", "city"),
        )).predict_from_mixed_bigrams(
            min_count=1,
            min_confidence=0.0,
            allowed_rules={("is_a", "at_location"): "at_location"},
            include_disabled_relations=True,
        )

        assert ("statue", "at_location", "city") in _keys(preds)


class TestMixedConfidenceFactors:
    def test_hub_penalty_applies(self):
        triples = [
            *[(f"leaf{i}", "is_a", "hub") for i in range(15)],
            ("hub", "capable_of", "thing"),
        ]
        predictor = PatternBasedPredictor(_rels(*triples))

        with_penalty = predictor.predict_from_mixed_bigrams(
            min_count=1,
            hub_penalty=True,
            min_confidence=0.0,
        )
        without_penalty = predictor.predict_from_mixed_bigrams(
            min_count=1,
            hub_penalty=False,
            min_confidence=0.0,
        )

        pen = {(p.source, p.target): p.confidence for p in with_penalty}
        no_pen = {(p.source, p.target): p.confidence for p in without_penalty}
        assert pen[("leaf0", "thing")] < no_pen[("leaf0", "thing")]
        assert "hub_penalty=" in next(p.reason for p in with_penalty if p.source == "leaf0")

    def test_relation_trust_applies_to_output_relation(self):
        preds = PatternBasedPredictor(_rels(
            ("sparrow", "is_a", "bird"),
            ("bird", "capable_of", "fly"),
        )).predict_from_mixed_bigrams(
            min_count=1,
            min_confidence=0.0,
            hub_penalty=False,
            relation_trust={"capable_of": 0.25},
        )

        pred = next(p for p in preds if p.source == "sparrow")
        assert pred.confidence == pytest.approx(_conf(1) * 0.25)
        assert "trust=0.250" in pred.reason

    def test_node_quality_penalizes_surviving_chains(self):
        noisy = "caf\u00e9"
        preds = PatternBasedPredictor(_rels(
            (noisy, "is_a", "drink"),
            ("drink", "has_property", "wet"),
        )).predict_from_mixed_bigrams(
            min_count=1,
            min_confidence=0.0,
            hub_penalty=False,
            use_node_quality=True,
            min_node_quality=0.0,
        )

        pred = next(p for p in preds if p.source == noisy)
        assert pred.confidence == pytest.approx(_conf(1) * 0.2)
        assert "nq=0.200" in pred.reason

    def test_node_quality_threshold_filters_low_quality_nodes(self):
        preds = PatternBasedPredictor(_rels(
            ("epic_fail", "is_a", "bird"),
            ("bird", "capable_of", "fly"),
        )).predict_from_mixed_bigrams(
            min_count=1,
            use_node_quality=True,
            min_node_quality=0.3,
        )

        assert preds == []


class TestMixedAuditExport:
    def test_audit_export_mode_mixed_works(self, tmp_path):
        csv_path = _write_csv(tmp_path, [
            ("sparrow", "is_a", "bird"),
            ("bird", "capable_of", "fly"),
        ])

        rows = build_audit_rows(csv_path, min_count=1, mode="mixed", limit=20)

        assert rows
        assert rows[0]["source"] == "sparrow"
        assert rows[0]["relation_type"] == "capable_of"
        assert rows[0]["target"] == "fly"
        assert rows[0]["reason"].startswith("mixed pattern:")

    def test_audit_export_mode_all_dedupes_and_keeps_higher_confidence(self, tmp_path):
        triples = [
            ("a", "capable_of", "b"),
            ("b", "capable_of", "c"),
            ("a", "is_a", "x"),
            ("x", "capable_of", "c"),
            ("p0", "is_a", "t0"),
            ("t0", "capable_of", "act0"),
            ("p1", "is_a", "t1"),
            ("t1", "capable_of", "act1"),
            ("p2", "is_a", "t2"),
            ("t2", "capable_of", "act2"),
            ("p3", "is_a", "t3"),
            ("t3", "capable_of", "act3"),
            ("p4", "is_a", "t4"),
            ("t4", "capable_of", "act4"),
        ]
        csv_path = _write_csv(tmp_path, triples)

        rows = build_audit_rows(
            csv_path,
            min_count=1,
            mode="all",
            limit=100,
            hub_penalty=False,
        )

        matches = [
            r for r in rows
            if r["source"] == "a"
            and r["relation_type"] == "capable_of"
            and r["target"] == "c"
        ]
        assert len(matches) == 1
        assert matches[0]["reason"].startswith("mixed pattern:")
        assert float(matches[0]["confidence"]) == pytest.approx(_conf(6), abs=1e-6)

    def test_old_transitive_mode_unchanged(self, tmp_path):
        csv_path = _write_csv(tmp_path, [
            ("a", "part_of", "b"),
            ("b", "part_of", "c"),
            ("x", "part_of", "y"),
            ("y", "part_of", "z"),
        ])

        default_rows = build_audit_rows(csv_path, min_count=1, limit=20)
        explicit_rows = build_audit_rows(
            csv_path,
            min_count=1,
            mode="transitive",
            limit=20,
        )

        assert default_rows == explicit_rows
        assert all(row["reason"].startswith("transitive pattern:") for row in default_rows)
