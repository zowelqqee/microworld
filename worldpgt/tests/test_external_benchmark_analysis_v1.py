"""Tests for offline external benchmark diagnostics."""

from __future__ import annotations

from worldpgt.experiments.analyze_external_benchmark_v1 import (
    analyze_result,
    classify_wrong_answer,
)


def test_classifies_temporal_current_mismatch() -> None:
    row = {
        "decision": "answer",
        "correct": False,
        "question": "who is governor of ohio 2011?",
        "answer_text": "The officeholder of Governor of Ohio is Mike DeWine.",
        "gold_answers": ["John Kasich"],
    }

    assert classify_wrong_answer(row) == "temporal_current_mismatch"


def test_classifies_bad_retrieval_source_mismatch() -> None:
    row = {
        "decision": "answer",
        "correct": False,
        "question": "what is my timezone in louisiana?",
        "answer_text": (
            "Based on a live web search:\nDallas is an American prime time television soap opera.\n"
            "Source: Dallas (TV series) - Wikipedia — https://example.com"
        ),
        "gold_answers": ["Central Time Zone"],
        "question_entity_mentioned": False,
        "gold_answer_entity_mentioned": False,
    }

    assert classify_wrong_answer(row) == "bad_retrieval_source_mismatch"


def test_analyze_result_counts_wrong_reasons() -> None:
    data = {
        "total_questions": 2,
        "answer_count": 2,
        "correct_count": 1,
        "wrong_count": 1,
        "audit_count": 0,
        "precision": 0.5,
        "rows": [
            {"decision": "answer", "correct": True},
            {
                "id": "w1",
                "decision": "answer",
                "correct": False,
                "question": "who is governor of ohio 2011?",
                "answer_text": "The officeholder of Governor of Ohio is Mike DeWine.",
                "gold_answers": ["John Kasich"],
            },
        ],
    }

    result = analyze_result(data)

    assert result["wrong_reason_counts"] == {"temporal_current_mismatch": 1}
    assert result["examples"]["temporal_current_mismatch"][0]["id"] == "w1"
