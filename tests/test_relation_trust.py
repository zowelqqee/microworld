"""Tests for core/relation_trust.py and its integration in PatternBasedPredictor."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.relations import Relation
from core.relation_trust import (
    DEFAULT_RELATION_TRUST,
    UNKNOWN_RELATION_TRUST,
    get_trust,
)
from core.pattern_prediction import PatternBasedPredictor, PatternPrediction


# ── helpers ────────────────────────────────────────────────────────────────────

def _rels(*triples) -> list[Relation]:
    return [Relation(s, r, t) for s, r, t in triples]


def _chains(rel: str, n: int) -> list[Relation]:
    """n independent 2-hop chains all using *rel*."""
    return _rels(*[(f"s{i}", rel, f"m{i}") for i in range(n)],
                 *[(f"m{i}", rel, f"e{i}") for i in range(n)])


def _base_conf(count: int) -> float:
    return min(0.95, 0.5 + 0.05 * math.log(count + 1))


# ── DEFAULT_RELATION_TRUST ─────────────────────────────────────────────────────

class TestDefaultRelationTrust:
    def test_made_of_value(self):
        assert DEFAULT_RELATION_TRUST["made_of"] == pytest.approx(0.862)

    def test_part_of_value(self):
        assert DEFAULT_RELATION_TRUST["part_of"] == pytest.approx(0.767)

    def test_is_a_value(self):
        assert DEFAULT_RELATION_TRUST["is_a"] == pytest.approx(0.756)

    def test_at_location_value(self):
        assert DEFAULT_RELATION_TRUST["at_location"] == pytest.approx(0.1)

    def test_all_values_between_zero_and_one(self):
        for rel, v in DEFAULT_RELATION_TRUST.items():
            assert 0.0 < v <= 1.0, f"{rel}={v} out of range"

    def test_high_trust_relations_above_low(self):
        assert DEFAULT_RELATION_TRUST["made_of"] > DEFAULT_RELATION_TRUST["at_location"]


# ── get_trust ──────────────────────────────────────────────────────────────────

class TestGetTrust:
    def test_known_relation_returns_table_value(self):
        assert get_trust("made_of") == pytest.approx(0.862)

    def test_unknown_relation_returns_unknown_default(self):
        assert get_trust("no_such_relation") == pytest.approx(UNKNOWN_RELATION_TRUST)

    def test_custom_table_override(self):
        assert get_trust("r", {"r": 0.9}) == pytest.approx(0.9)

    def test_custom_table_unknown_still_uses_default(self):
        assert get_trust("missing", {"other": 0.5}) == pytest.approx(UNKNOWN_RELATION_TRUST)

    def test_clamps_above_one(self):
        assert get_trust("r", {"r": 1.5}) == pytest.approx(1.0)

    def test_clamps_zero_to_epsilon(self):
        v = get_trust("r", {"r": 0.0})
        assert v > 0.0

    def test_clamps_negative(self):
        v = get_trust("r", {"r": -0.5})
        assert v > 0.0

    def test_unknown_relation_trust_constant_in_range(self):
        assert 0.0 < UNKNOWN_RELATION_TRUST <= 1.0


# ── integration: relation_trust in predict_from_bigrams ───────────────────────

class TestRelationTrustInPredictor:

    # -- trust=None leaves confidence unchanged --------------------------------

    def test_no_trust_confidence_unchanged(self):
        rels = _chains("part_of", 4)
        p_notrust = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False, relation_trust=None
        )
        p_trust = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False,
            relation_trust=DEFAULT_RELATION_TRUST,
        )
        notrust_confs = {(p.source, p.target): p.confidence for p in p_notrust}
        trust_confs   = {(p.source, p.target): p.confidence for p in p_trust}
        # trust < 1 means penalised predictions should have lower confidence
        for key in notrust_confs:
            if key in trust_confs:
                assert trust_confs[key] <= notrust_confs[key] + 1e-9

    def test_trust_none_is_default_behavior(self):
        rels = _chains("is_a", 4)
        pred = PatternBasedPredictor(rels)
        p1 = pred.predict_from_bigrams(min_count=1, hub_penalty=False)
        p2 = pred.predict_from_bigrams(min_count=1, hub_penalty=False, relation_trust=None)
        assert len(p1) == len(p2)
        for a, b in zip(p1, p2):
            assert a.confidence == pytest.approx(b.confidence)

    # -- high-trust relation preserves high confidence -------------------------

    def test_high_trust_relation_has_high_confidence(self):
        # made_of trust = 0.862; confidence should be ~base * 0.862
        rels = _chains("made_of", 10)
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False,
            relation_trust=DEFAULT_RELATION_TRUST,
        )
        assert len(preds) > 0
        base = _base_conf(10)
        expected = min(0.95, base * 0.862)
        for p in preds:
            assert p.confidence == pytest.approx(expected, abs=1e-6)

    def test_high_trust_confidence_above_low_trust(self):
        # made_of (0.862) should yield higher confidence than at_location (0.1)
        # for the same base pattern count.
        count = 8
        rels_hi = _chains("made_of",     count)
        rels_lo = _chains("at_location", count)
        p_hi = PatternBasedPredictor(rels_hi).predict_from_bigrams(
            min_count=1, hub_penalty=False,
            relation_trust=DEFAULT_RELATION_TRUST,
        )
        p_lo = PatternBasedPredictor(rels_lo).predict_from_bigrams(
            min_count=1, hub_penalty=False, min_confidence=0.0,
            relation_trust=DEFAULT_RELATION_TRUST,
            include_disabled_relations=True,
        )
        assert len(p_hi) > 0
        assert len(p_lo) > 0
        assert p_hi[0].confidence > p_lo[0].confidence

    # -- low-trust relation is penalised ---------------------------------------

    def test_low_trust_lowers_confidence(self):
        # at_location trust = 0.1; base conf * 0.1 should be < base conf
        count = 8
        rels = _chains("at_location", count)
        p_notrust = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False, min_confidence=0.0,
            relation_trust=None,
            include_disabled_relations=True,
        )
        p_trust = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False, min_confidence=0.0,
            relation_trust=DEFAULT_RELATION_TRUST,
            include_disabled_relations=True,
        )
        # both runs should have predictions
        assert len(p_notrust) > 0
        assert len(p_trust) > 0
        assert p_trust[0].confidence < p_notrust[0].confidence

    def test_low_trust_confidence_formula(self):
        count = 8
        rels = _chains("at_location", count)
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False, min_confidence=0.0,
            relation_trust=DEFAULT_RELATION_TRUST,
            include_disabled_relations=True,
        )
        assert len(preds) > 0
        base = _base_conf(count)
        expected = min(0.95, base * 0.1)
        for p in preds:
            assert p.confidence == pytest.approx(expected, abs=1e-6)

    def test_low_trust_filtered_by_min_confidence(self):
        # at_location trust=0.1; base≈0.62 → conf≈0.062 < default min_confidence=0.5
        count = 8
        rels = _chains("at_location", count)
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False,
            relation_trust=DEFAULT_RELATION_TRUST,
            include_disabled_relations=True,
        )
        assert preds == []

    # -- unknown relation defaults to UNKNOWN_RELATION_TRUST ------------------

    def test_unknown_relation_uses_unknown_default(self):
        count = 4
        rels = _chains("has_property", count)
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False, min_confidence=0.0,
            relation_trust=DEFAULT_RELATION_TRUST,
        )
        assert len(preds) > 0
        base = _base_conf(count)
        expected = min(0.95, base * UNKNOWN_RELATION_TRUST)
        for p in preds:
            assert p.confidence == pytest.approx(expected, abs=1e-6)

    def test_unknown_trust_is_not_one(self):
        # UNKNOWN_RELATION_TRUST < 1.0 encodes uncertainty
        assert UNKNOWN_RELATION_TRUST < 1.0

    def test_custom_trust_table_applied(self):
        count = 4
        custom_trust = 0.3
        rels = _chains("custom_rel", count)
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False, min_confidence=0.0,
            relation_trust={"custom_rel": custom_trust},
        )
        assert len(preds) > 0
        base = _base_conf(count)
        expected = min(0.95, base * custom_trust)
        for p in preds:
            assert p.confidence == pytest.approx(expected, abs=1e-6)

    def test_empty_trust_table_uses_unknown_default(self):
        count = 4
        rels = _chains("some_rel", count)
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False, min_confidence=0.0,
            relation_trust={},
        )
        assert len(preds) > 0
        base = _base_conf(count)
        expected = min(0.95, base * UNKNOWN_RELATION_TRUST)
        for p in preds:
            assert p.confidence == pytest.approx(expected, abs=1e-6)

    # -- confidence cap still applies -----------------------------------------

    def test_trust_one_caps_at_0_95(self):
        # Even with trust=1.0 and huge count, confidence stays ≤ 0.95
        rels = _chains("made_of", 10_000)
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False,
            relation_trust={"made_of": 1.0},
        )
        for p in preds:
            assert p.confidence <= 0.95

    # -- hub_penalty and relation_trust compose --------------------------------

    def test_trust_and_hub_penalty_both_applied(self):
        # Use a hub graph + a low-trust relation; both factors should reduce conf.
        hub_rels = _rels(
            *[(f"n{i}", "at_location", "hub") for i in range(10)],
            ("hub", "at_location", "end"),
            ("s",   "at_location", "mid"),
            ("mid", "at_location", "d"),
        )
        preds_pen_only = PatternBasedPredictor(hub_rels).predict_from_bigrams(
            min_count=1, hub_penalty=True, min_confidence=0.0, relation_trust=None,
            include_disabled_relations=True,
        )
        preds_both = PatternBasedPredictor(hub_rels).predict_from_bigrams(
            min_count=1, hub_penalty=True, min_confidence=0.0,
            relation_trust=DEFAULT_RELATION_TRUST,
            include_disabled_relations=True,
        )
        pen_map  = {(p.source, p.target): p.confidence for p in preds_pen_only}
        both_map = {(p.source, p.target): p.confidence for p in preds_both}
        for key in both_map:
            if key in pen_map:
                assert both_map[key] <= pen_map[key] + 1e-9

    # -- reason string includes trust -----------------------------------------

    def test_reason_contains_trust_when_enabled(self):
        rels = _chains("part_of", 4)
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False, min_confidence=0.0,
            relation_trust=DEFAULT_RELATION_TRUST,
        )
        assert len(preds) > 0
        for p in preds:
            assert "trust=" in p.reason

    def test_reason_no_trust_when_disabled(self):
        rels = _chains("part_of", 4)
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False, relation_trust=None
        )
        for p in preds:
            assert "trust=" not in p.reason

    def test_reason_trust_value_matches_table(self):
        rels = _chains("is_a", 4)
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False, min_confidence=0.0,
            relation_trust=DEFAULT_RELATION_TRUST,
        )
        assert len(preds) > 0
        # trust=0.756 should appear as "trust=0.756"
        for p in preds:
            assert "trust=0.756" in p.reason


# ── audit export: use_relation_trust passthrough ──────────────────────────────

class TestAuditExportRelationTrust:
    _TINY_CSV = (
        "source,relation_type,target\n"
        + "".join(f"s{i},is_a,m{i}\n" for i in range(6))
        + "".join(f"m{i},is_a,end\n" for i in range(6))
        + "".join(f"a{i},at_location,hub\n" for i in range(6))
        + "".join(f"hub,at_location,loc\n")
    )

    def _write(self, tmp_path) -> str:
        p = tmp_path / "tiny.csv"
        p.write_text(self._TINY_CSV)
        return str(p)

    def test_use_relation_trust_false_ignores_priors(self, tmp_path):
        from examples.pattern_audit_export import build_audit_rows
        rows_notrust = build_audit_rows(self._write(tmp_path), limit=200,
                                        hub_penalty=False, use_relation_trust=False)
        rows_trust   = build_audit_rows(self._write(tmp_path), limit=200,
                                        hub_penalty=False, use_relation_trust=True)
        notrust_map = {(r["source"], r["relation_type"], r["target"]): float(r["confidence"])
                       for r in rows_notrust}
        trust_map   = {(r["source"], r["relation_type"], r["target"]): float(r["confidence"])
                       for r in rows_trust}
        # trust should not increase confidence
        for key in trust_map:
            if key in notrust_map:
                assert trust_map[key] <= notrust_map[key] + 1e-6

    def test_use_relation_trust_reduces_at_location_confidence(self, tmp_path):
        from examples.pattern_audit_export import build_audit_rows
        rows_no = build_audit_rows(self._write(tmp_path), limit=200,
                                   hub_penalty=False, use_relation_trust=False,
                                   relation_filter={"at_location"},
                                   include_disabled_relations=True)
        rows_yes = build_audit_rows(self._write(tmp_path), limit=200,
                                    hub_penalty=False, use_relation_trust=True,
                                    min_count=1 if False else 5,
                                    relation_filter={"at_location"},
                                    include_disabled_relations=True)
        # at_location trust=0.1, so if any survive they must be lower
        for r_no, r_yes in zip(rows_no, rows_yes):
            assert float(r_yes["confidence"]) <= float(r_no["confidence"]) + 1e-6
