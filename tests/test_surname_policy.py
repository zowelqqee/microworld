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
