"""Tests for benchmark_external_v1.

Verifies the bundled WebQuestions sample shape and the runner's output shape.
Does not assert specific answer_rate/precision values since those depend on
overlay content and may shift as the overlay evolves.
"""

from __future__ import annotations

import pytest

_FAST_N = 15


def _fast_sample() -> dict:
    from worldpgt.experiments.benchmark_external_v1 import load_sample
    full = load_sample()
    return {**full, "items": full["items"][:_FAST_N]}


def test_benchmark_external_imports_cleanly():
    import worldpgt.experiments.benchmark_external_v1 as mod
    assert callable(mod.run)
    assert callable(mod.main)
    assert callable(mod.load_sample)


def test_sample_file_has_at_least_200_items():
    from worldpgt.experiments.benchmark_external_v1 import load_sample
    data = load_sample()
    assert len(data["items"]) >= 200


def test_sample_items_have_required_fields():
    from worldpgt.experiments.benchmark_external_v1 import load_sample
    data = load_sample()
    for item in data["items"]:
        assert item["id"]
        assert item["question"].strip()
        assert isinstance(item["answers"], list) and item["answers"]


def test_normalize_and_match_helpers():
    from worldpgt.experiments.benchmark_external_v1 import _is_correct, _normalize
    assert _normalize("Tesla, Inc.") == "tesla inc"
    assert _is_correct("Tesla is an electric car company.", ["Tesla"])
    assert _is_correct("The answer is University of Oregon.", ["Oregon Ducks"]) is False
    assert _is_correct("", ["Tesla"]) is False


@pytest.fixture(scope="module")
def external_result():
    from worldpgt.experiments.benchmark_external_v1 import run
    return run(overlay_mode="pump-dry-run", sample=_fast_sample())


def test_result_has_required_keys(external_result):
    required = {
        "timestamp", "overlay_mode", "total_questions", "answer_count",
        "answer_rate", "audit_count", "no_count", "audit_rate",
        "correct_count", "wrong_count", "precision", "diagnostics", "rows",
    }
    assert required <= set(external_result.keys())


def test_result_diagnostics_have_required_fields(external_result):
    required = {
        "route_counts", "support_kind_counts", "question_entity_mention_count",
        "gold_answer_entity_mention_count", "direct_gold_support_count",
        "direct_gold_support_answer_count", "direct_gold_support_audit_count",
    }
    assert required <= set(external_result["diagnostics"].keys())


def test_result_total_questions(external_result):
    assert external_result["total_questions"] == _FAST_N


def test_result_counts_sum_to_total(external_result):
    r = external_result
    assert r["answer_count"] + r["audit_count"] + r["no_count"] == r["total_questions"]


def test_result_correct_plus_wrong_equals_answered(external_result):
    r = external_result
    assert r["correct_count"] + r["wrong_count"] == r["answer_count"]


def test_result_precision_consistent_with_counts(external_result):
    r = external_result
    if r["answer_count"] == 0:
        assert r["precision"] is None
    else:
        assert r["precision"] == pytest.approx(r["correct_count"] / r["answer_count"])


def test_result_rows_have_required_fields(external_result):
    for row in external_result["rows"]:
        for field in (
            "id", "question", "gold_answers", "decision", "route",
            "support_kind", "answer_text", "correct",
            "question_entity_mentioned", "gold_answer_entity_mentioned",
            "direct_gold_support", "latency_ms",
        ):
            assert field in row, f"Missing field '{field}' in row {row.get('id')}"


def test_result_rows_decision_values(external_result):
    valid = {"answer", "no", "audit"}
    for row in external_result["rows"]:
        assert row["decision"] in valid


def test_unanswered_rows_are_not_marked_correct(external_result):
    for row in external_result["rows"]:
        if row["decision"] != "answer":
            assert row["correct"] is False
