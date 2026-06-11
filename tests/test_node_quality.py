"""Tests for core/node_quality.py and its integration in PatternBasedPredictor."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.relations import Relation
from core.node_quality import DEFAULT_BAD_TOKENS, node_quality, is_low_quality_node
from core.pattern_prediction import PatternBasedPredictor


# ── helpers ────────────────────────────────────────────────────────────────────

def _rels(*triples) -> list[Relation]:
    return [Relation(s, r, t) for s, r, t in triples]


def _chains(src_prefix: str, via: str, tgt: str, n: int, rel: str = "made_of") -> list[Relation]:
    """n sources each connected to *via*, which connects to *tgt*."""
    return _rels(
        *[(f"{src_prefix}{i}", rel, via) for i in range(n)],
        (via, rel, tgt),
    )


def _base_conf(count: int) -> float:
    return min(0.95, 0.5 + 0.05 * math.log(count + 1))


# ── DEFAULT_BAD_TOKENS ─────────────────────────────────────────────────────────

class TestDefaultBadTokens:
    def test_sister_naked_present(self):
        assert "sister_naked" in DEFAULT_BAD_TOKENS

    def test_epic_fail_present(self):
        assert "epic_fail" in DEFAULT_BAD_TOKENS

    def test_fail_present(self):
        assert "fail" in DEFAULT_BAD_TOKENS

    def test_naked_present(self):
        assert "naked" in DEFAULT_BAD_TOKENS

    def test_bolas_present(self):
        assert "bolas" in DEFAULT_BAD_TOKENS


# ── node_quality ───────────────────────────────────────────────────────────────

class TestNodeQuality:
    # clean nodes → 1.0
    def test_simple_word_is_clean(self):
        assert node_quality("steel") == pytest.approx(1.0)

    def test_two_token_word_is_clean(self):
        assert node_quality("red_car") == pytest.approx(1.0)

    def test_three_token_word_is_clean(self):
        assert node_quality("big_red_car") == pytest.approx(1.0)

    def test_normal_name_quality_is_one(self):
        for name in ("iron", "knife", "metal", "internet", "fruit"):
            assert node_quality(name) == pytest.approx(1.0), f"unexpected penalty for {name!r}"

    # explicit bad tokens → 0.0
    def test_sister_naked_is_zero(self):
        assert node_quality("sister_naked") == pytest.approx(0.0)

    def test_epic_fail_is_zero(self):
        assert node_quality("epic_fail") == pytest.approx(0.0)

    def test_bare_naked_token_is_zero(self):
        assert node_quality("naked") == pytest.approx(0.0)

    def test_bare_fail_token_is_zero(self):
        assert node_quality("fail") == pytest.approx(0.0)

    def test_bare_bolas_token_is_zero(self):
        assert node_quality("bolas") == pytest.approx(0.0)

    def test_compound_containing_bolas_is_zero(self):
        # tu_hermana_en_bolas — caught via "bolas" token
        assert node_quality("tu_hermana_en_bolas") == pytest.approx(0.0)

    def test_compound_containing_naked_is_zero(self):
        assert node_quality("something_naked_else") == pytest.approx(0.0)

    def test_compound_containing_fail_is_zero(self):
        assert node_quality("epic_fail_test") == pytest.approx(0.0)

    # length penalty
    def test_short_name_no_length_penalty(self):
        assert node_quality("a" * 25) == pytest.approx(1.0)

    def test_long_name_gets_penalty(self):
        q = node_quality("a" * 50)
        assert q < 1.0
        assert q > 0.0

    def test_length_penalty_decreases_with_length(self):
        q30 = node_quality("a" * 30)
        q60 = node_quality("a" * 60)
        assert q30 > q60

    # token count penalty
    def test_four_tokens_gets_penalty(self):
        q = node_quality("a_b_c_d")
        assert q < 1.0

    def test_many_tokens_lower_than_few_tokens(self):
        q_few  = node_quality("a_b_c")
        q_many = node_quality("a_b_c_d_e_f")
        assert q_few > q_many

    # non-ASCII penalty
    def test_non_ascii_gets_penalty(self):
        q = node_quality("café")
        assert q < 1.0

    def test_ascii_no_ascii_penalty(self):
        q = node_quality("cafe")
        assert q == pytest.approx(1.0)

    # digit ratio penalty
    def test_digit_heavy_name_penalised(self):
        q = node_quality("123456")
        assert q < 0.5

    def test_mostly_letters_not_penalised(self):
        assert node_quality("abc1") == pytest.approx(1.0)

    # score range
    def test_quality_always_between_zero_and_one(self):
        for name in ["", "x", "steel", "sister_naked", "a" * 100,
                     "tu_hermana_en_bolas", "café_123456"]:
            q = node_quality(name)
            assert 0.0 <= q <= 1.0, f"out of range for {name!r}: {q}"


# ── is_low_quality_node ────────────────────────────────────────────────────────

class TestIsLowQualityNode:
    def test_bad_token_is_low_quality(self):
        assert is_low_quality_node("sister_naked") is True

    def test_clean_node_is_not_low_quality(self):
        assert is_low_quality_node("steel") is False

    def test_custom_threshold_zero_passes_bad(self):
        # threshold=0 → nothing is low quality
        assert is_low_quality_node("naked", threshold=0.0) is False

    def test_custom_threshold_one_fails_everything(self):
        # threshold=1.0 → even clean nodes fail (quality < 1.0 includes most)
        # "steel" has quality 1.0, so threshold=1.0 still passes it
        assert is_low_quality_node("steel", threshold=1.01) is True


# ── integration: use_node_quality in predict_from_bigrams ─────────────────────

class TestNodeQualityInPredictor:

    # -- bad intermediate suppressed -------------------------------------------

    def test_bad_intermediate_suppresses_prediction(self):
        # sister_naked as intermediate → prediction suppressed
        rels = _chains("src", "sister_naked", "material", n=8)
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False,
            use_node_quality=True, min_node_quality=0.3,
        )
        keys = {(p.source, p.relation_type, p.target) for p in preds}
        for i in range(8):
            assert (f"src{i}", "made_of", "material") not in keys

    def test_epic_fail_intermediate_suppressed(self):
        rels = _chains("thing", "epic_fail", "stuff", n=8)
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False,
            use_node_quality=True, min_node_quality=0.3,
        )
        assert preds == []

    def test_non_english_intermediate_suppressed(self):
        rels = _chains("obj", "tu_hermana_en_bolas", "whatever", n=8)
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False,
            use_node_quality=True, min_node_quality=0.3,
        )
        assert preds == []

    # -- bad source or target suppressed ---------------------------------------

    def test_bad_source_node_suppressed(self):
        # source = "naked_thing" → contains "naked" token
        rels = _rels(
            *[(f"naked_thing", "made_of", f"mid{i}") for i in range(8)],
            *[(f"mid{i}", "made_of", "iron") for i in range(8)],
        )
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False,
            use_node_quality=True, min_node_quality=0.3,
        )
        keys = {(p.source, p.relation_type, p.target) for p in preds}
        assert ("naked_thing", "made_of", "iron") not in keys

    def test_bad_target_node_suppressed(self):
        rels = _rels(
            *[(f"src{i}", "made_of", "mid") for i in range(8)],
            ("mid", "made_of", "epic_fail"),
        )
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False,
            use_node_quality=True, min_node_quality=0.3,
        )
        keys = {(p.source, p.relation_type, p.target) for p in preds}
        for i in range(8):
            assert (f"src{i}", "made_of", "epic_fail") not in keys

    # -- good nodes survive ----------------------------------------------------

    def test_clean_chain_survives(self):
        rels = _chains("tool", "steel", "iron", n=8)
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False,
            use_node_quality=True, min_node_quality=0.3,
        )
        keys = {(p.source, p.relation_type, p.target) for p in preds}
        for i in range(8):
            assert (f"tool{i}", "made_of", "iron") in keys

    def test_mixed_graph_only_clean_survive(self):
        # good chains through "steel", bad through "sister_naked"
        good = _chains("tool", "steel",         "iron",     n=8)
        bad  = _chains("src",  "sister_naked",  "material", n=8)
        rels = good + bad
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False,
            use_node_quality=True, min_node_quality=0.3,
        )
        keys = {(p.source, p.relation_type, p.target) for p in preds}
        for i in range(8):
            assert (f"tool{i}", "made_of", "iron") in keys
            assert (f"src{i}",  "made_of", "material") not in keys

    # -- confidence scaled by node quality ------------------------------------

    def test_node_quality_reduces_confidence(self):
        # Use a 4-token intermediate (quality < 1.0) alongside a clean chain.
        # long_mid: 4 tokens → quality < 1.0
        rels_clean = _chains("clean", "steel",                "iron",    n=8)
        rels_noisy = _chains("noisy", "a_b_c_d",              "iron",    n=8)
        pred_clean = PatternBasedPredictor(rels_clean).predict_from_bigrams(
            min_count=1, hub_penalty=False, use_node_quality=True,
        )
        pred_noisy = PatternBasedPredictor(rels_noisy).predict_from_bigrams(
            min_count=1, hub_penalty=False, use_node_quality=True,
        )
        # clean should have higher or equal confidence per prediction
        if pred_clean and pred_noisy:
            avg_clean = sum(p.confidence for p in pred_clean) / len(pred_clean)
            avg_noisy = sum(p.confidence for p in pred_noisy) / len(pred_noisy)
            assert avg_clean >= avg_noisy

    def test_nq_factor_in_confidence_formula(self):
        # "a_b_c_d" has 4 tokens → nq = 3/4 = 0.75
        rels = _chains("s", "a_b_c_d", "end", n=8)
        pred_nq  = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False, use_node_quality=True, min_node_quality=0.0,
        )
        pred_raw = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False, use_node_quality=False,
        )
        if pred_nq and pred_raw:
            assert pred_nq[0].confidence <= pred_raw[0].confidence + 1e-9

    # -- use_node_quality=False restores old behavior -------------------------

    def test_disabled_includes_bad_intermediate(self):
        rels = _chains("src", "sister_naked", "material", n=8)
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False, use_node_quality=False,
        )
        keys = {(p.source, p.relation_type, p.target) for p in preds}
        # at least one prediction should exist (bad node is not filtered)
        assert len(keys) > 0

    def test_disabled_confidence_unchanged(self):
        rels = _chains("tool", "steel", "iron", n=8)
        p_on  = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False, use_node_quality=True,
        )
        p_off = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False, use_node_quality=False,
        )
        # steel is quality 1.0, so confidences should be identical
        on_map  = {(p.source, p.target): p.confidence for p in p_on}
        off_map = {(p.source, p.target): p.confidence for p in p_off}
        for key in on_map:
            if key in off_map:
                assert on_map[key] == pytest.approx(off_map[key])

    # -- reason string contains nq when enabled --------------------------------

    def test_reason_contains_nq_when_enabled(self):
        rels = _chains("s", "a_b_c_d", "end", n=8)
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False, use_node_quality=True,
            min_node_quality=0.0, min_confidence=0.0,
        )
        assert len(preds) > 0
        for p in preds:
            assert "nq=" in p.reason

    def test_reason_no_nq_when_disabled(self):
        rels = _chains("tool", "steel", "iron", n=8)
        preds = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False, use_node_quality=False,
        )
        for p in preds:
            assert "nq=" not in p.reason

    # -- composing with hub_penalty and relation_trust -------------------------

    def test_all_three_factors_compose(self):
        from core.relation_trust import DEFAULT_RELATION_TRUST
        rels = _chains("tool", "steel", "iron", n=8, rel="made_of")
        p_all = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=True,
            relation_trust=DEFAULT_RELATION_TRUST,
            use_node_quality=True, min_node_quality=0.0,
        )
        p_none = PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, hub_penalty=False,
            relation_trust=None,
            use_node_quality=False,
        )
        if p_all and p_none:
            # all factors ≤ 1 means applying them can only reduce confidence
            assert p_all[0].confidence <= p_none[0].confidence + 1e-9


# ── audit export: node quality passthrough ─────────────────────────────────────

class TestAuditExportNodeQuality:
    _CSV = (
        "source,relation_type,target\n"
        + "".join(f"tool{i},made_of,steel\n" for i in range(8))
        + "steel,made_of,iron\n"
        + "".join(f"src{i},made_of,sister_naked\n" for i in range(8))
        + "sister_naked,made_of,material\n"
    )

    def _write(self, tmp_path) -> str:
        p = tmp_path / "nq.csv"
        p.write_text(self._CSV)
        return str(p)

    def test_use_node_quality_false_includes_bad_nodes(self, tmp_path):
        from examples.pattern_audit_export import build_audit_rows
        rows = build_audit_rows(self._write(tmp_path), limit=200,
                                hub_penalty=False, use_node_quality=False)
        targets = {r["target"] for r in rows}
        assert "material" in targets   # sister_naked → material survives without filter

    def test_use_node_quality_true_removes_bad_intermediate(self, tmp_path):
        from examples.pattern_audit_export import build_audit_rows
        rows = build_audit_rows(self._write(tmp_path), limit=200,
                                hub_penalty=False, use_node_quality=True,
                                min_node_quality=0.3)
        targets = {r["target"] for r in rows}
        assert "material" not in targets   # sister_naked chain suppressed

    def test_use_node_quality_preserves_clean_chain(self, tmp_path):
        from examples.pattern_audit_export import build_audit_rows
        rows = build_audit_rows(self._write(tmp_path), limit=200,
                                hub_penalty=False, use_node_quality=True,
                                min_node_quality=0.3)
        targets = {r["target"] for r in rows}
        assert "iron" in targets   # tool → steel → iron survives

    def test_min_node_quality_zero_disables_filter(self, tmp_path):
        from examples.pattern_audit_export import build_audit_rows
        rows_nq0  = build_audit_rows(self._write(tmp_path), limit=200,
                                     hub_penalty=False, use_node_quality=True,
                                     min_node_quality=0.0)
        rows_nq03 = build_audit_rows(self._write(tmp_path), limit=200,
                                     hub_penalty=False, use_node_quality=True,
                                     min_node_quality=0.3)
        # with threshold 0, more (or equal) predictions survive
        assert len(rows_nq0) >= len(rows_nq03)

    def test_confidence_column_is_float_string(self, tmp_path):
        from examples.pattern_audit_export import build_audit_rows
        rows = build_audit_rows(self._write(tmp_path), limit=10,
                                hub_penalty=False, use_node_quality=True)
        for r in rows:
            float(r["confidence"])  # must not raise
