"""Risk and coverage metrics for controlled continuation outputs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional


def _norm(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _empty_counts() -> dict:
    return {
        "total": 0,
        "expected_answerable_count": 0,
        "expected_no_answer_count": 0,
        "continue_count": 0,
        "audit_count": 0,
        "suppress_count": 0,
        "correct_continue_count": 0,
        "wrong_continue_count": 0,
        "audited_answerable_count": 0,
        "correct_no_answer_audit_count": 0,
    }


def _add_row(counts: dict, row: dict) -> None:
    expected = _norm(row.get("expected_sense"))
    selected = _norm(row.get("selected_sense"))
    decision = row.get("decision", "")
    answerable = expected is not None

    counts["total"] += 1
    if answerable:
        counts["expected_answerable_count"] += 1
    else:
        counts["expected_no_answer_count"] += 1

    if decision == "continue":
        counts["continue_count"] += 1
        if answerable and selected == expected:
            counts["correct_continue_count"] += 1
        elif answerable and selected != expected:
            counts["wrong_continue_count"] += 1
    elif decision == "audit":
        counts["audit_count"] += 1
        if answerable:
            counts["audited_answerable_count"] += 1
        else:
            counts["correct_no_answer_audit_count"] += 1
    elif decision == "suppress":
        counts["suppress_count"] += 1


def _finalize(counts: dict) -> dict:
    continued_evaluable = counts["correct_continue_count"] + counts["wrong_continue_count"]
    answerable = counts["expected_answerable_count"]
    total = counts["total"]
    result = dict(counts)
    result.update(
        {
            "coverage_rate": _rate(counts["continue_count"], total),
            "audit_rate": _rate(counts["audit_count"], total),
            "wrong_continue_rate": _rate(counts["wrong_continue_count"], total),
            "precision_on_continued": _rate(counts["correct_continue_count"], continued_evaluable),
            "answerable_recall": _rate(counts["correct_continue_count"], answerable),
            "abstention_rate_on_answerable": _rate(counts["audited_answerable_count"], answerable),
        }
    )
    return result


def summarize_rows(rows: Iterable[dict]) -> dict:
    global_counts = _empty_counts()
    grouped: dict[str, dict] = {}

    for row in rows:
        _add_row(global_counts, row)
        difficulty = row.get("difficulty_type", "") or "unknown"
        grouped.setdefault(difficulty, _empty_counts())
        _add_row(grouped[difficulty], row)

    summary = _finalize(global_counts)
    summary["by_difficulty_type"] = {
        difficulty: _finalize(counts)
        for difficulty, counts in sorted(grouped.items())
    }
    return summary
