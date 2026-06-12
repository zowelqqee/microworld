"""Tests for audit-driven generated-name pattern mining."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.name_pattern_mining import extract_name_patterns, mine_pattern_trust


def _row(name: str, label: str) -> dict[str, str]:
    return {"name": name, "manual_label": label}


def test_extract_ngrams_includes_boundary_patterns():
    patterns = extract_name_patterns("manyn")
    assert "^ma" in patterns
    assert "ma" in patterns
    assert "yn$" in patterns
    assert "^man" in patterns
    assert "nyn$" in patterns


def test_bad_heavy_pattern_is_learned():
    trust, stats = mine_pattern_trust(
        [
            _row("qweslienna", "bad"),
            _row("qwemara", "bad"),
            _row("majanis", "good"),
        ],
    )
    assert trust["qwe"] == pytest.approx(0.70)
    assert stats["qwe"] == {"good": 0, "bad": 2, "unclear": 0}


def test_good_heavy_pattern_is_not_penalized():
    trust, stats = mine_pattern_trust(
        [
            _row("majanis", "good"),
            _row("marina", "good"),
            _row("mavito", "bad"),
        ],
    )
    assert "ma" not in trust
    assert "ma" not in stats


def test_pattern_with_too_little_support_is_ignored():
    trust, stats = mine_pattern_trust([_row("qweslienna", "bad")])
    assert "qwe" not in trust
    assert "qwe" not in stats


def test_unclear_does_not_create_bad_penalty_alone():
    trust, stats = mine_pattern_trust(
        [
            _row("qweslienna", "unclear"),
            _row("qwemara", "unclear"),
            _row("qwelin", "unclear"),
        ],
    )
    assert trust == {}
    assert stats == {}


def test_moderate_bad_heavy_pattern_uses_soft_penalty():
    trust, stats = mine_pattern_trust(
        [
            _row("qweslienna", "bad"),
            _row("qwemara", "bad"),
            _row("qwelin", "bad"),
            _row("qweanna", "good"),
        ],
    )
    assert trust["qwe"] == pytest.approx(0.85)
    assert stats["qwe"] == {"good": 1, "bad": 3, "unclear": 0}

