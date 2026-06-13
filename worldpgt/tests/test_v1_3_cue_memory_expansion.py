from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from worldpgt.continuation.continuation_engine import ControlledContinuationEngine
from worldpgt.continuation.continuation_policy import ContinuationPolicy
from worldpgt.experiments.audit_reason_miner import mine_rows
from worldpgt.experiments.check_semantic_render_quality import check_rows
from worldpgt.experiments.risk_coverage_metrics import summarize_rows


_PROMPTS = Path(__file__).resolve().parents[1] / "experiments" / "continuation_prompts_v1.csv"

_V1_2_CONTINUED_IDS = {
    "v1-001",
    "v1-002",
    "v1-003",
    "v1-004",
    "v1-005",
    "v1-006",
    "v1-007",
    "v1-008",
    "v1-009",
    "v1-010",
    "v1-011",
    "v1-012",
    "v1-013",
    "v1-014",
    "v1-015",
    "v1-016",
    "v1-017",
    "v1-018",
    "v1-019",
    "v1-020",
    "v1-042",
    "v1-043",
    "v1-046",
    "v1-048",
    "v1-049",
    "v1-050",
    "v1-052",
    "v1-054",
    "v1-058",
    "v1-059",
    "v1-060",
    "v1-067",
    "v1-072",
    "v1-084",
    "v1-087",
    "v1-095",
    "v1-110",
}

_NEW_MISSING_OR_WEAK_CONTINUED_IDS = {
    "v1-021",
    "v1-022",
    "v1-023",
    "v1-024",
    "v1-025",
    "v1-026",
    "v1-027",
    "v1-028",
    "v1-029",
    "v1-031",
    "v1-032",
    "v1-033",
    "v1-034",
    "v1-035",
    "v1-036",
    "v1-037",
    "v1-038",
    "v1-039",
    "v1-040",
}

_TRUE_UNSAFE_IDS = {
    "v1-081",
    "v1-082",
    "v1-083",
    "v1-085",
    "v1-086",
    "v1-088",
    "v1-089",
    "v1-090",
    "v1-091",
    "v1-092",
    "v1-093",
    "v1-094",
}

_NO_CLEAR_ANSWER_IDS = {"v1-111", "v1-112", "v1-113", "v1-114", "v1-115"}
_NO_KNOWN_TERM_IDS = {"v1-116", "v1-117", "v1-118", "v1-119", "v1-120"}


@lru_cache(maxsize=1)
def _run_rows() -> tuple[dict[str, dict], ...]:
    engine = ControlledContinuationEngine()
    rows: list[dict] = []
    with _PROMPTS.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            result = engine.continue_prompt(record["prompt"])
            rows.append(
                {
                    **record,
                    "continuation": result.continuation,
                    "selected_sense": result.selected_sense or "",
                    "confidence": f"{result.confidence:.4f}",
                    "decision": result.decision,
                    "reasons": " | ".join(result.reasons),
                    "memory_hits": " | ".join(result.memory_hits),
                }
            )
    return tuple(rows)


def _rows() -> list[dict]:
    return list(_run_rows())


def _by_id() -> dict[str, dict]:
    return {row["id"]: row for row in _rows()}


def test_existing_v1_2_safe_continued_rows_still_continue_with_same_sense():
    rows = _by_id()
    assert len(_V1_2_CONTINUED_IDS) == 37
    for row_id in _V1_2_CONTINUED_IDS:
        row = rows[row_id]
        assert row["decision"] == "continue", row_id
        assert row["selected_sense"] == row["expected_sense"], row_id
        assert row["continuation"].strip(), row_id


def test_previous_missing_or_weak_cue_rows_now_continue_from_explicit_cues():
    rows = _by_id()
    for row_id in _NEW_MISSING_OR_WEAK_CONTINUED_IDS:
        row = rows[row_id]
        assert row["decision"] == "continue", row_id
        assert row["selected_sense"] == row["expected_sense"], row_id
        assert "positive_cue=" in row["memory_hits"], row_id


def test_sense_tie_rows_continue_only_when_added_cues_break_tie_safely():
    rows = _by_id()

    assert rows["v1-063"]["decision"] == "continue"
    assert rows["v1-063"]["selected_sense"] == "sports_equipment"
    assert "positive_cue=swung -> sports_equipment" in rows["v1-063"]["memory_hits"]

    for row_id in ("v1-061", "v1-065", "v1-068", "v1-071"):
        assert rows[row_id]["decision"] == "audit", row_id


def test_low_margin_rows_continue_only_when_added_evidence_raises_margin():
    rows = _by_id()

    assert rows["v1-077"]["decision"] == "continue"
    assert rows["v1-077"]["selected_sense"] == "animal"
    assert "positive_cue=splash -> animal" in rows["v1-077"]["memory_hits"]

    for row_id in ("v1-066", "v1-074", "v1-076", "v1-078", "v1-079"):
        assert rows[row_id]["decision"] == "audit", row_id


def test_true_unsafe_rows_remain_audited():
    rows = _by_id()
    for row_id in _TRUE_UNSAFE_IDS:
        row = rows[row_id]
        assert row["decision"] == "audit", row_id
        assert row["continuation"] == "", row_id


def test_unsupported_no_clear_answer_rows_remain_audited_without_fallback():
    rows = _by_id()
    for row_id in _NO_CLEAR_ANSWER_IDS:
        row = rows[row_id]
        assert row["decision"] == "audit", row_id
        assert row["continuation"] == "", row_id


def test_missing_sense_memory_rows_remain_audited_without_fallback():
    rows = _by_id()
    for row_id in _NO_KNOWN_TERM_IDS:
        row = rows[row_id]
        assert row["decision"] == "audit", row_id
        assert row["continuation"] == "", row_id
        assert "no_ambiguous_term" in row["reasons"], row_id


def test_v1_051_remains_no_safe_repaired_candidate_audit():
    row = _by_id()["v1-051"]
    assert row["decision"] == "audit"
    assert row["continuation"] == ""
    assert "audit_reason=no_safe_repaired_candidate" in row["reasons"]


def test_v1_3_benchmark_safety_numbers_locked():
    summary = summarize_rows(_rows())
    assert summary["continue_count"] == 58
    assert summary["audit_count"] == 62
    assert summary["wrong_continue_count"] == 0
    assert summary["precision_on_continued"] == 1.0


def test_v1_3_semantic_quality_flagged_count_is_zero():
    quality = check_rows(_rows())
    assert quality["flagged_count"] == 0, quality["flagged_rows"]


def test_v1_3_audit_reason_distribution_locked():
    reason_counts = mine_rows(_rows())["summary"]["reason_counts"]
    assert reason_counts == {
        "low_margin": 12,
        "missing_or_weak_cue_support": 10,
        "missing_sense_memory": 5,
        "no_safe_repaired_candidate": 1,
        "sense_tie": 11,
        "surface_validation_failed": 6,
        "true_unsafe": 12,
        "unsupported_or_underconstrained_context": 5,
    }


def test_v1_3_output_is_deterministic():
    first = summarize_rows(_rows())
    _run_rows.cache_clear()
    second = summarize_rows(_rows())
    assert first == second


def test_policy_thresholds_were_not_lowered():
    policy = ContinuationPolicy()
    assert policy.min_score == 1.0
    assert policy.min_margin == 1.0
