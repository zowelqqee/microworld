"""Tests for core/suppression_policy.py"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.suppression_policy import (
    KNOWN_NOISY_TOKENS,
    TOKEN_ALLOWLIST,
    apply_suppression_policy,
    is_noisy_token,
    should_suppress_quality_aware,
    token_quality_score,
)


def _row(source: str, target: str, relation: str = "made_of") -> dict:
    return {
        "source": source,
        "relation_type": relation,
        "target": target,
        "baseline_confidence": "0.50",
        "learned_confidence": "0.20",
        "delta": "-0.30",
        "evidence": "",
        "reason": "",
        "manual_label": "",
        "notes": "",
    }


# ── is_noisy_token ────────────────────────────────────────────────────────────


class TestIsNoisyToken:
    def test_oxegen_is_noisy(self):
        assert is_noisy_token("oxegen") is True

    def test_talbe_is_noisy(self):
        assert is_noisy_token("talbe") is True

    def test_h2o_is_not_noisy(self):
        assert is_noisy_token("h2o") is False

    def test_molten_sand_is_not_noisy(self):
        assert is_noisy_token("molten_sand") is False

    def test_silica_is_not_noisy(self):
        assert is_noisy_token("silica") is False

    def test_normal_material_tokens_not_noisy(self):
        for token in ("wood", "trees", "sand", "paper", "water", "glass"):
            assert is_noisy_token(token) is False, f"Expected {token!r} to be non-noisy"

    def test_empty_token_is_noisy(self):
        assert is_noisy_token("") is True

    def test_weird_garbage_token_is_noisy(self):
        assert is_noisy_token("x!@#$") is True

    def test_all_known_noisy_tokens_flagged(self):
        for t in KNOWN_NOISY_TOKENS:
            assert is_noisy_token(t) is True, f"Expected KNOWN_NOISY token {t!r} to be noisy"

    def test_all_allowlist_tokens_not_noisy(self):
        for t in TOKEN_ALLOWLIST:
            assert is_noisy_token(t) is False, f"Expected allowlist token {t!r} to be non-noisy"


# ── token_quality_score ───────────────────────────────────────────────────────


class TestTokenQualityScore:
    def test_empty_returns_zero(self):
        assert token_quality_score("") == pytest.approx(0.0)

    def test_whitespace_only_returns_zero(self):
        assert token_quality_score("   ") == pytest.approx(0.0)

    def test_known_noisy_returns_zero(self):
        assert token_quality_score("oxegen") == pytest.approx(0.0)
        assert token_quality_score("talbe") == pytest.approx(0.0)

    def test_allowlist_returns_one(self):
        for t in TOKEN_ALLOWLIST:
            assert token_quality_score(t) == pytest.approx(1.0), (
                f"Allowlist token {t!r} should score 1.0"
            )

    def test_garbage_chars_penalized(self):
        assert token_quality_score("x!@#$") < 0.5

    def test_normal_word_scores_high(self):
        for token in ("apple", "fruit", "plant", "organism", "banana"):
            assert token_quality_score(token) >= 0.55, (
                f"{token!r} should score >= 0.55"
            )

    def test_score_in_range(self):
        for token in ("", "oxegen", "h2o", "wood", "x!@#", "abc", "molten_sand"):
            s = token_quality_score(token)
            assert 0.0 <= s <= 1.0, f"Score out of range for {token!r}: {s}"


# ── should_suppress_quality_aware ─────────────────────────────────────────────


class TestShouldSuppressQualityAware:
    def test_suppresses_row_with_noisy_target_oxegen(self):
        assert should_suppress_quality_aware(_row("ice", "oxegen")) is True

    def test_does_not_suppress_noisy_source_clean_target(self):
        # talbe is a typo for "table" — noisy source, but target "wood" is fine.
        # Source noise is a canonicalization issue, not a suppression reason.
        assert should_suppress_quality_aware(_row("talbe", "wood")) is False

    def test_does_not_suppress_row_with_target_h2o(self):
        assert should_suppress_quality_aware(_row("ice", "h2o")) is False

    def test_does_not_suppress_row_with_target_silica(self):
        assert should_suppress_quality_aware(_row("glasses", "silica")) is False

    def test_does_not_suppress_clean_material_row(self):
        assert should_suppress_quality_aware(_row("envelope", "trees")) is False

    def test_suppresses_garbage_target(self):
        assert should_suppress_quality_aware(_row("apple", "!@#$%")) is True

    def test_does_not_suppress_garbage_source_clean_target(self):
        assert should_suppress_quality_aware(_row("!!!", "wood")) is False

    def test_custom_min_token_quality_allowlist_still_passes(self):
        # h2o is in allowlist and scores 1.0, so even strict threshold does not suppress
        assert should_suppress_quality_aware(_row("ice", "h2o"), min_token_quality=0.99) is False

    def test_does_not_suppress_both_clean(self):
        assert should_suppress_quality_aware(_row("iron", "steel")) is False

    # ── spec-required explicit cases ──────────────────────────────────────────

    def test_spec_noisy_source_clean_target_not_suppressed(self):
        # talbe --made_of--> wood: typo source, good prediction → do NOT suppress
        assert should_suppress_quality_aware(_row("talbe", "wood")) is False

    def test_spec_clean_source_noisy_target_suppressed(self):
        # ice --made_of--> oxegen: misspelled target → suppress
        assert should_suppress_quality_aware(_row("ice", "oxegen")) is True

    def test_spec_noisy_source_noisy_target_suppressed(self):
        # talbe --made_of--> oxegen: both noisy, but target is the decisive signal
        assert should_suppress_quality_aware(_row("talbe", "oxegen")) is True


# ── apply_suppression_policy ──────────────────────────────────────────────────


class TestApplySuppressionPolicy:
    def _sample_rows(self) -> list[dict]:
        return [
            _row("ice", "oxegen"),   # noisy target
            _row("ice", "h2o"),      # both clean (h2o in allowlist)
            _row("talbe", "wood"),   # noisy source
        ]

    def test_naive_returns_all_rows(self):
        rows = self._sample_rows()
        assert len(apply_suppression_policy(rows, "naive")) == len(rows)

    def test_naive_output_equals_input(self):
        rows = self._sample_rows()
        assert apply_suppression_policy(rows, "naive") == rows

    def test_quality_aware_filters_to_noisy_target_rows_only(self):
        # Only ice/oxegen has a noisy target; ice/h2o and talbe/wood do not
        rows = self._sample_rows()
        result = apply_suppression_policy(rows, "quality_aware")
        assert len(result) == 1

    def test_quality_aware_keeps_noisy_target_row(self):
        rows = self._sample_rows()
        result = apply_suppression_policy(rows, "quality_aware")
        assert any(r["target"] == "oxegen" for r in result)

    def test_quality_aware_does_not_export_noisy_source_only_row(self):
        # talbe/wood: noisy source, clean target — should NOT appear in suppression output
        rows = self._sample_rows()
        result = apply_suppression_policy(rows, "quality_aware")
        assert not any(r["source"] == "talbe" for r in result)

    def test_quality_aware_excludes_clean_row(self):
        rows = self._sample_rows()
        result = apply_suppression_policy(rows, "quality_aware")
        assert not any(r["target"] == "h2o" for r in result)

    def test_unknown_policy_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown suppression policy"):
            apply_suppression_policy([], "bogus")

    def test_unknown_policy_message_contains_policy_name(self):
        with pytest.raises(ValueError, match="bogus"):
            apply_suppression_policy([], "bogus")

    def test_empty_rows_naive_returns_empty(self):
        assert apply_suppression_policy([], "naive") == []

    def test_empty_rows_quality_aware_returns_empty(self):
        assert apply_suppression_policy([], "quality_aware") == []

    def test_custom_min_token_quality_respected(self):
        # With min_token_quality=0.0 nobody scores below 0.0, so nothing suppressed
        rows = self._sample_rows()
        result = apply_suppression_policy(rows, "quality_aware", min_token_quality=0.0)
        assert len(result) == 0  # even noisy tokens score 0.0, which is not < 0.0
