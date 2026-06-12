"""Tests for core/surname_policy.py

The policy applies to any personal-name dataset: given names, family names,
or a mix.  Tests cover both surname-like and given-name-like examples.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.surname_policy import (
    explain_surname_quality,
    is_valid_generated_surname,
    surname_quality_score,
)


# ── rejection ─────────────────────────────────────────────────────────────────

class TestRejects:
    def test_rejects_empty(self):
        assert is_valid_generated_surname("") is False

    def test_rejects_single_char(self):
        assert is_valid_generated_surname("a") is False

    def test_rejects_too_long(self):
        assert is_valid_generated_surname("a" * 30) is False

    def test_rejects_no_vowel_garbage(self):
        assert is_valid_generated_surname("qzxqz") is False

    def test_rejects_leading_punctuation(self):
        assert is_valid_generated_surname("-ivanov") is False

    def test_duplicate_rejected_when_avoiding(self):
        src = {"ivanov"}
        assert is_valid_generated_surname(
            "ivanov", source_names=src, avoid_duplicates=True
        ) is False
        # still allowed when avoid_duplicates is off
        assert is_valid_generated_surname(
            "ivanov", source_names=src, avoid_duplicates=False
        ) is True


# ── two-char names: medium/low, not fatal ────────────────────────────────────

class TestTwoCharNames:
    def test_two_char_passes_threshold(self):
        # 2-char real short names should still be valid (threshold=0.5)
        assert is_valid_generated_surname("ea") is True
        assert is_valid_generated_surname("li") is True

    def test_two_char_scores_below_full(self):
        # Should NOT score 1.0 — marked "very short" with a soft penalty
        for name in ("ea", "li", "ty"):
            score = surname_quality_score(name)
            assert score < 1.0, f"{name!r} should score < 1.0, got {score}"
            assert score >= 0.5, f"{name!r} should still pass 0.5 threshold"

    def test_two_char_reason_is_very_short(self):
        reasons = explain_surname_quality("ea")
        assert any("very short" in r for r in reasons)


# ── acceptance: classic surnames ─────────────────────────────────────────────

class TestAllowsSurnames:
    def test_allows_slavic_surnames(self):
        for name in ("ivanov", "petrov", "abramov", "dmitriev"):
            assert is_valid_generated_surname(name) is True, name

    def test_allows_special_endings(self):
        for name in ("abramidze", "gelashvili", "saroyan", "petrosyan"):
            assert is_valid_generated_surname(name) is True, name

    def test_allows_unicode_surname(self):
        assert is_valid_generated_surname("иванов") is True

    def test_allows_short_surname_like(self):
        # dyer, east, dutton are real surname-style short names
        for name in ("dutton", "dyer", "east"):
            assert is_valid_generated_surname(name) is True, name

    def test_dutton_dyer_east_score_high(self):
        for name in ("dutton", "dyer", "east"):
            score = surname_quality_score(name)
            assert score >= 0.8, f"{name!r} expected >= 0.8, got {score}"


# ── acceptance: given-name-like inputs ───────────────────────────────────────

class TestAllowsGivenNames:
    def test_allows_english_given_names(self):
        for name in ("eleanor", "eldrick", "ebraheem"):
            assert is_valid_generated_surname(name) is True, name

    def test_given_names_score_high(self):
        for name in ("eleanor", "eldrick", "ebraheem"):
            score = surname_quality_score(name)
            assert score >= 0.8, f"{name!r} expected >= 0.8, got {score}"

    def test_allows_short_given_names(self):
        # 3–4 char given names
        for name in ("ava", "mia", "leo", "emma", "liam"):
            assert is_valid_generated_surname(name) is True, name

    def test_allows_longer_given_names(self):
        for name in ("alexandra", "sebastian", "christopher"):
            assert is_valid_generated_surname(name) is True, name


# ── scoring / explanation ─────────────────────────────────────────────────────

class TestScoring:
    def test_score_in_range(self):
        for name in ("ivanov", "qzxqz", "", "a", "abramidze", "eleanor", "ea"):
            s = surname_quality_score(name)
            assert 0.0 <= s <= 1.0, f"{name!r} score out of range: {s}"

    def test_good_name_scores_higher_than_garbage(self):
        assert surname_quality_score("ivanov") > surname_quality_score("qzxqz")

    def test_long_garbage_scores_low(self):
        # Extremely long random consonant string should score low
        assert surname_quality_score("zzzzzzzzzzzzzzzzzzzzzzz") < 0.5

    def test_explain_returns_reasons_for_bad(self):
        reasons = explain_surname_quality("qzxqz")
        assert isinstance(reasons, list)
        assert reasons
        assert any("vowel" in r for r in reasons)

    def test_explain_returns_something_for_good(self):
        for name in ("ivanov", "eleanor", "dutton"):
            reasons = explain_surname_quality(name)
            assert isinstance(reasons, list)
            assert reasons, f"expected non-empty reasons for {name!r}"

    def test_explain_good_name_says_plausible(self):
        reasons = explain_surname_quality("ivanov")
        assert any("plausible" in r for r in reasons)

    def test_explain_flags_too_short(self):
        assert any("short" in r for r in explain_surname_quality("a"))

    def test_explain_flags_very_short_for_two_chars(self):
        reasons = explain_surname_quality("ea")
        assert any("very short" in r for r in reasons)


# ── generated-name plausibility thresholds ────────────────────────────────────

class TestGeneratedNameScoreThresholds:
    """Verify that plausible generated names score high and glued/overlong names
    score proportionally lower.  These are the canonical quality benchmarks."""

    # High-quality generated examples — must score above threshold
    @pytest.mark.parametrize("name,threshold", [
        ("majanis",   0.75),
        ("chelis",    0.75),
        ("romaura",   0.70),
        ("levaughn",  0.75),
        ("eleanor",   0.80),
        ("dutton",    0.80),
    ])
    def test_plausible_names_score_above_threshold(self, name, threshold):
        score = surname_quality_score(name)
        assert score >= threshold, (
            f"{name!r} expected >= {threshold}, got {score:.4f}; "
            f"reasons: {explain_surname_quality(name)}"
        )

    # Overlong / glued names — must score below ceiling
    @pytest.mark.parametrize("name,ceiling", [
        ("nalicelahuvaanvita", 0.45),
        ("lochukroyannah",     0.60),
        ("skyylanahogan",      0.65),
    ])
    def test_overlong_names_score_below_ceiling(self, name, ceiling):
        score = surname_quality_score(name)
        assert score < ceiling, (
            f"{name!r} expected < {ceiling}, got {score:.4f}; "
            f"reasons: {explain_surname_quality(name)}"
        )

    def test_two_letter_names_below_one_and_has_very_short(self):
        for name in ("ea", "li", "ty"):
            score = surname_quality_score(name)
            assert score < 1.0, f"{name!r} should score < 1.0, got {score}"
            reasons = explain_surname_quality(name)
            assert any("very short" in r for r in reasons), (
                f"{name!r} missing 'very short' in reasons: {reasons}"
            )


# ── audit-derived bad-name diagnostics ───────────────────────────────────────

class TestAuditBadNameDiagnostics:
    @pytest.mark.parametrize("name,reason,ceiling", [
        ("lovelo", "nickname_like", 0.80),
        ("march", "common_word_like", 0.80),
        ("avito", "brand_like", 0.80),
        ("all", "common_word_like", 0.70),
        ("loch", "common_word_like", 0.80),
    ])
    def test_common_brand_and_nickname_like_forms_penalized(
        self, name, reason, ceiling
    ):
        reasons = explain_surname_quality(name)
        score = surname_quality_score(name)
        assert reason in reasons, reasons
        assert score < ceiling, (name, score, reasons)

    @pytest.mark.parametrize("name", ["kyn", "yia", "gen", "kha", "ter", "nne", "jid"])
    def test_short_fragments_are_not_perfect_names(self, name):
        reasons = explain_surname_quality(name)
        score = surname_quality_score(name)
        assert (
            "too_fragmentary" in reasons or "awkward_short_form" in reasons
        ), reasons
        assert score < 0.75, (name, score, reasons)

    def test_qweslienna_has_q_or_glued_diagnostic(self):
        reasons = explain_surname_quality("qweslienna")
        score = surname_quality_score("qweslienna")
        assert (
            "weird_q_usage" in reasons or "medium_glued_name" in reasons
        ), reasons
        assert score < 0.80, (score, reasons)

    def test_gateuillis_has_poor_readability(self):
        reasons = explain_surname_quality("gateuillis")
        score = surname_quality_score("gateuillis")
        assert "poor_readability" in reasons, reasons
        assert score < 0.80, (score, reasons)

    @pytest.mark.parametrize("name", [
        "latalille",
        "brighteme",
        "nafranimi",
        "roaryonnia",
        "sauldenyx",
        "evreekahl",
        "journesten",
    ])
    def test_medium_artificial_glued_forms_are_penalized(self, name):
        reasons = explain_surname_quality(name)
        score = surname_quality_score(name)
        assert (
            "medium_glued_name" in reasons or "poor_readability" in reasons
        ), (name, reasons)
        assert score < 0.85, (name, score, reasons)

    @pytest.mark.parametrize("name,threshold", [
        ("majanis", 0.75),
        ("chelis", 0.75),
        ("romaura", 0.70),
        ("levaughn", 0.75),
        ("khadan", 0.75),
        ("zephana", 0.75),
        ("jazeli", 0.75),
        ("jakaria", 0.75),
        ("selyn", 0.75),
        ("kataliza", 0.75),
        ("ahrianna", 0.75),
        ("delaine", 0.75),
    ])
    def test_plausible_audit_examples_remain_high_enough(self, name, threshold):
        score = surname_quality_score(name)
        reasons = explain_surname_quality(name)
        assert score >= threshold, (name, score, reasons)


# ── quality reason content for overlong names ─────────────────────────────────

class TestOverlongNameReasons:
    """Check that the right penalty tags appear for overlong/glued names."""

    def test_extremely_long_reason(self):
        reasons = explain_surname_quality("nalicelahuvaanvita")
        assert "extremely_long" in reasons, reasons

    def test_too_long_reason_for_15plus(self):
        # shlizevoneenaton is 16 chars → too_long tier (15-17)
        reasons = explain_surname_quality("shlizevoneenaton")
        assert "too_long" in reasons, reasons

    def test_long_name_reason_for_lochukroyannah(self):
        # lochukroyannah is 14 chars → long_name tier (12-14)
        reasons = explain_surname_quality("lochukroyannah")
        assert "long_name" in reasons, reasons

    def test_long_name_reason_for_skyylanahogan(self):
        reasons = explain_surname_quality("skyylanahogan")
        assert "long_name" in reasons, reasons

    def test_too_many_syllable_chunks_in_nalicelahuvaanvita(self):
        reasons = explain_surname_quality("nalicelahuvaanvita")
        assert "too_many_syllable_chunks" in reasons, reasons

    def test_too_many_syllable_chunks_in_lochukroyannah(self):
        # 4 nuclei + length 14 → too_many_syllable_chunks
        reasons = explain_surname_quality("lochukroyannah")
        assert "too_many_syllable_chunks" in reasons, reasons


# ── positive informational reasons ────────────────────────────────────────────

class TestPositiveReasons:
    """Check that positive signals appear in the reasons for clean names."""

    def test_reasonable_length_in_short_names(self):
        for name in ("majanis", "chelis", "dutton"):
            reasons = explain_surname_quality(name)
            assert "reasonable_length" in reasons, (
                f"{name!r} missing reasonable_length: {reasons}"
            )

    def test_balanced_vowels_in_clean_names(self):
        for name in ("majanis", "levaughn", "dutton"):
            reasons = explain_surname_quality(name)
            assert "balanced_vowels" in reasons, (
                f"{name!r} missing balanced_vowels: {reasons}"
            )

    def test_common_name_ending_for_known_suffixes(self):
        # "ivanov" ends in "ov" which is in ALLOWED_ENDINGS
        reasons = explain_surname_quality("ivanov")
        assert "common_name_ending" in reasons, reasons
