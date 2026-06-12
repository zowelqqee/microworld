"""Tests for core/surname_generator.py (loader + transition graph).

Works for given names, family names, or mixed personal-name datasets.
"""
from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.surname_generator import (
    END,
    START,
    SurnameTransitionGraph,
    iter_transitions,
    load_surnames,
    normalize_surname,
    transition_key,
)


# ── loader ────────────────────────────────────────────────────────────────────

class TestLoader:
    def test_loads_txt_one_per_line(self, tmp_path):
        p = tmp_path / "names.txt"
        p.write_text("Ivanov\nPetrov\nSmith\n", encoding="utf-8")
        assert load_surnames(str(p)) == ["ivanov", "petrov", "smith"]

    def test_loads_csv_first_column(self, tmp_path):
        p = tmp_path / "names.csv"
        p.write_text("Ivanov,42\nPetrov,7\n", encoding="utf-8")
        assert load_surnames(str(p)) == ["ivanov", "petrov"]

    def test_explicit_column(self, tmp_path):
        p = tmp_path / "names.csv"
        p.write_text("id,surname\n1,Ivanov\n2,Petrov\n", encoding="utf-8")
        # column 1 = surname; header row 'surname' is also read (still valid)
        result = load_surnames(str(p), column=1)
        assert result == ["surname", "ivanov", "petrov"]

    def test_strips_whitespace(self, tmp_path):
        p = tmp_path / "names.txt"
        p.write_text("  Ivanov  \n\tPetrov\t\n", encoding="utf-8")
        assert load_surnames(str(p)) == ["ivanov", "petrov"]

    def test_lowercases_by_default(self, tmp_path):
        p = tmp_path / "names.txt"
        p.write_text("ABRAMIDZE\n", encoding="utf-8")
        assert load_surnames(str(p)) == ["abramidze"]

    def test_keep_case_when_disabled(self, tmp_path):
        p = tmp_path / "names.txt"
        p.write_text("Ivanov\n", encoding="utf-8")
        assert load_surnames(str(p), lowercase=False) == ["Ivanov"]

    def test_skips_empty_rows(self, tmp_path):
        p = tmp_path / "names.txt"
        p.write_text("Ivanov\n\n   \nPetrov\n", encoding="utf-8")
        assert load_surnames(str(p)) == ["ivanov", "petrov"]

    def test_skips_too_short(self, tmp_path):
        p = tmp_path / "names.txt"
        p.write_text("a\nIvanov\nx\n", encoding="utf-8")
        assert load_surnames(str(p)) == ["ivanov"]

    def test_keeps_apostrophe_and_hyphen(self, tmp_path):
        p = tmp_path / "names.txt"
        p.write_text("O'Brien\nSaint-Claire\n", encoding="utf-8")
        assert load_surnames(str(p)) == ["o'brien", "saint-claire"]

    def test_keeps_unicode_letters(self, tmp_path):
        p = tmp_path / "names.txt"
        p.write_text("Иванов\n", encoding="utf-8")
        assert load_surnames(str(p)) == ["иванов"]


class TestNormalize:
    def test_drops_digits_and_symbols(self):
        assert normalize_surname("Iv4nov!") == "ivnov"

    def test_strip_and_lower(self):
        assert normalize_surname("  PeTROV \n") == "petrov"


# ── transition graph: order 1 ─────────────────────────────────────────────────

class TestOrderOne:
    def test_order_one_transitions(self):
        g = SurnameTransitionGraph(order=1).build(["ab"])
        assert g.transition_counts[(START,)] == {"a": 1}
        assert g.transition_counts[("a",)] == {"b": 1}
        assert g.transition_counts[("b",)] == {END: 1}

    def test_start_and_end_exist(self):
        g = SurnameTransitionGraph(order=1).build(["ab", "ac"])
        assert (START,) in g.transition_counts
        assert g.transition_counts[("b",)] == {END: 1}
        assert g.transition_counts[("c",)] == {END: 1}

    def test_probabilities_sum_to_one(self):
        g = SurnameTransitionGraph(order=1).build(["ab", "ac", "ad"])
        probs = g.transition_probs(("a",))
        assert pytest.approx(sum(probs.values())) == 1.0
        assert probs == pytest.approx({"b": 1 / 3, "c": 1 / 3, "d": 1 / 3})

    def test_unknown_context_empty(self):
        g = SurnameTransitionGraph(order=1).build(["ab"])
        assert g.transition_probs(("z",)) == {}


# ── transition graph: order 2 ─────────────────────────────────────────────────

class TestOrderTwo:
    def test_order_two_transitions(self):
        g = SurnameTransitionGraph(order=2).build(["abr"])
        assert g.transition_counts[(START, START)] == {"a": 1}
        assert g.transition_counts[(START, "a")] == {"b": 1}
        assert g.transition_counts[("a", "b")] == {"r": 1}
        assert g.transition_counts[("b", "r")] == {END: 1}

    def test_abramidze_chain(self):
        g = SurnameTransitionGraph(order=2).build(["abramidze"])
        assert g.transition_counts[("a", "b")] == {"r": 1}
        assert g.transition_counts[("d", "z")] == {"e": 1}
        assert g.transition_counts[("z", "e")] == {END: 1}


# ── iter_transitions / keys ───────────────────────────────────────────────────

class TestIterTransitions:
    def test_reconstructs_keys_order_two(self):
        keys = [transition_key(ctx, ch) for ctx, ch in iter_transitions("ab", 2)]
        assert keys == [
            "<START><START>->a",
            "<START>a->b",
            "ab-><END>",
        ]

    def test_order_must_be_positive(self):
        with pytest.raises(ValueError):
            list(iter_transitions("ab", 0))


# ── generation ────────────────────────────────────────────────────────────────

class TestGeneration:
    def test_generation_stops_at_end(self):
        g = SurnameTransitionGraph(order=1).build(["ab"])
        # only one possible path: a -> b -> END
        name = g.generate(rng=random.Random(0))
        assert name == "ab"

    def test_generation_respects_max_length(self):
        # "ab"*: a->b->a->b ... never reaches END within max_length
        g = SurnameTransitionGraph(order=1)
        g.transition_counts = {
            (START,): {"a": 1},
            ("a",): {"b": 1},
            ("b",): {"a": 1},
        }
        name = g.generate(max_length=5, rng=random.Random(0))
        assert len(name) == 5

    def test_seeded_generation_is_deterministic(self):
        names = ["ivanov", "petrov", "sidorov", "abramov", "smirnov"]
        g = SurnameTransitionGraph(order=2).build(names)
        a = [g.generate(rng=random.Random(123)) for _ in range(5)]
        b = [g.generate(rng=random.Random(123)) for _ in range(5)]
        assert a == b

    def test_different_seeds_can_differ(self):
        names = ["ivanov", "petrov", "sidorov", "abramov", "smirnov", "kuznetsov"]
        g = SurnameTransitionGraph(order=1).build(names)
        out_a = [g.generate(rng=random.Random(1)) for _ in range(20)]
        out_b = [g.generate(rng=random.Random(99)) for _ in range(20)]
        assert out_a != out_b

    def test_trust_multiplier_affects_choice(self):
        # From START there are two equally likely first chars: a and b.
        g = SurnameTransitionGraph(order=1)
        g.transition_counts = {
            (START,): {"a": 1, "b": 1},
            ("a",): {END: 1},
            ("b",): {END: 1},
        }
        # Strongly suppress 'a' and boost 'b': generation should favour 'b'.
        trust = {"transition_trust": {"<START>->a": 0.1, "<START>->b": 2.0}}
        rng = random.Random(7)
        out = [g.generate(rng=rng, trust_profile=trust) for _ in range(200)]
        n_b = sum(1 for x in out if x == "b")
        assert n_b > 150  # overwhelmingly 'b'

    def test_zero_weights_fall_back_to_base(self):
        g = SurnameTransitionGraph(order=1)
        g.transition_counts = {
            (START,): {"a": 1, "b": 1},
            ("a",): {END: 1},
            ("b",): {END: 1},
        }
        # All START transitions zeroed → fallback to base probabilities, still valid.
        trust = {"transition_trust": {"<START>->a": 0.0, "<START>->b": 0.0}}
        out = g.generate(rng=random.Random(3), trust_profile=trust)
        assert out in {"a", "b"}
