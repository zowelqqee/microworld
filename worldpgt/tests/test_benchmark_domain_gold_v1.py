"""Tests for benchmark_domain_gold_v1.

Verifies the domain-matched gold-set builder and runner's output shape.
Does not assert specific answer_rate/precision values since those depend on
overlay content and the NLU pipeline's own coverage, and may shift as either
evolves -- that's the whole point of this benchmark (see its module
docstring): it's meant to move over time as the non-LLM system improves.
"""

from __future__ import annotations

import pytest


def test_benchmark_domain_gold_imports_cleanly():
    import worldpgt.experiments.benchmark_domain_gold_v1 as mod
    assert callable(mod.run)
    assert callable(mod.main)
    assert callable(mod.build_gold_set)


def test_is_correct_matches_normalized_substring():
    from worldpgt.experiments.benchmark_domain_gold_v1 import _is_correct
    assert _is_correct("Tesla produces electric cars and battery storage.", "electric cars")
    assert _is_correct("", "electric cars") is False
    assert _is_correct("Tesla produces electric cars.", "") is False
    assert _is_correct("Tesla produces solar panels.", "electric cars") is False


@pytest.fixture(scope="module")
def gold_set():
    from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
    from worldpgt.experiments.benchmark_domain_gold_v1 import build_gold_set
    orch = AnswerOrchestrator(overlay_mode="pump-dry-run")
    return build_gold_set(orch)


def test_gold_set_is_nonempty(gold_set):
    assert len(gold_set) > 50


def test_gold_set_items_have_required_fields(gold_set):
    for item in gold_set:
        assert item["question"].strip()
        assert item["gold_answer"].strip()
        assert item["predicate"]
        assert item["kind"] in {"relation", "definition"}


def test_gold_set_has_no_excluded_relation_indices_leaking_through(gold_set):
    """The manually-curated exclusion list should mean none of the known-bad
    fragments show up as a gold answer (spot check a couple of the worst
    ones from the curation pass)."""
    bad_fragments = {"300 U", "few officers", "This precision", "The user can"}
    for item in gold_set:
        assert item["gold_answer"] not in bad_fragments
        assert item["subject"] not in bad_fragments


@pytest.fixture(scope="module")
def domain_result():
    from worldpgt.experiments.benchmark_domain_gold_v1 import run
    return run(overlay_mode="pump-dry-run")


def test_result_has_required_keys(domain_result):
    required = {
        "timestamp", "overlay_mode", "total_questions", "answer_count",
        "answer_rate", "audit_count", "no_count", "audit_rate",
        "correct_count", "wrong_count", "precision", "route_counts",
        "support_kind_counts", "by_predicate", "rows",
    }
    assert required <= set(domain_result.keys())


def test_result_counts_sum_to_total(domain_result):
    r = domain_result
    assert r["answer_count"] + r["audit_count"] + r["no_count"] == r["total_questions"]


def test_result_correct_plus_wrong_equals_answered(domain_result):
    r = domain_result
    assert r["correct_count"] + r["wrong_count"] == r["answer_count"]


def test_result_precision_consistent_with_counts(domain_result):
    r = domain_result
    if r["answer_count"] == 0:
        assert r["precision"] == 0.0
    else:
        assert r["precision"] == pytest.approx(r["correct_count"] / r["answer_count"], abs=1e-4)


def test_by_predicate_totals_sum_to_grand_total(domain_result):
    r = domain_result
    assert sum(c["total"] for c in r["by_predicate"].values()) == r["total_questions"]


def test_unanswered_rows_are_not_marked_correct(domain_result):
    for row in domain_result["rows"]:
        if row["decision"] != "answer":
            assert row["correct"] is False


def test_this_is_meaningfully_more_answerable_than_open_domain_webquestions(domain_result):
    """The whole point of this benchmark: domain-matched facts should score
    far higher than the ~40% seen on open-domain WebQuestions, since these
    facts genuinely exist in the graph. A regression toward WebQuestions-tier
    scores would mean something broke in the core NLU pipeline, not just
    "unlucky open-domain phrasing" -- worth a hard floor here."""
    assert domain_result["answer_rate"] >= 0.85
    assert domain_result["precision"] >= 0.9
