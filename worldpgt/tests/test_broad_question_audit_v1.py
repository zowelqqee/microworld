"""Tests for broad question QA audit labeling."""

from __future__ import annotations

from worldpgt.experiments.audit_broad_question_qa_v1 import (
    QuestionSpec,
    classify_result,
)


def test_classify_good_answer_requires_expected_substrings() -> None:
    spec = QuestionSpec(
        "What is the Internet?",
        "technology",
        "answer_contains",
        ("interconnected computer networks",),
    )

    assert classify_result(spec, "answer", "Internet is a global system of interconnected computer networks.") == "good_answer"
    assert classify_result(spec, "answer", "Internet is a cafe chain.") == "bad_answer"
    assert classify_result(spec, "audit", "I don't have a definition.") == "missing_desired"


def test_classify_good_answer_rejects_forbidden_substrings() -> None:
    spec = QuestionSpec(
        "What is North America?",
        "geography",
        "answer_contains",
        ("continent",),
        ("School district",),
    )

    assert classify_result(spec, "answer", "North America is a continent.") == "good_answer"
    assert classify_result(spec, "answer", "School district is an administrative body.") == "bad_answer"


def test_classify_current_questions_must_audit() -> None:
    spec = QuestionSpec("What is Tesla's current stock price?", "safety", "audit")

    assert classify_result(spec, "audit", "I don't have current data.") == "correct_audit"
    assert classify_result(spec, "answer", "$123") == "unexpected_answer"


def test_classify_desired_answer_marks_audits_as_missing() -> None:
    spec = QuestionSpec("What is Brazil?", "geography", "desired_answer")

    assert classify_result(spec, "audit", "I don't have a definition.") == "missing_desired"
    assert classify_result(spec, "answer", "Brazil is a country.") == "answer_needs_review"
