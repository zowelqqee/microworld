from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from worldpgt.continuation.continuation_engine import ControlledContinuationEngine
from worldpgt.continuation.continuation_policy import ContinuationPolicy
from worldpgt.continuation.prompt_tail_validator import validate_prompt_tail_compatibility
from worldpgt.continuation.semantic_frame import SemanticFrame
from worldpgt.experiments.check_semantic_render_quality import check_rows
from worldpgt.experiments.risk_coverage_metrics import summarize_rows


_PROMPTS = Path(__file__).resolve().parents[1] / "experiments" / "continuation_prompts_v1.csv"

_BAD_PHRASES = [
    "could and searched",
    "before and hit",
    "motioned for and completed",
    "while tourists and swam",
    "turned toward and carried",
    "as the hook the operator",
    "made everyone and brought",
    "after and filled",
    "would get louder after and",
    "before the player swung, he steadied",
]


def _frame(sense_id: str) -> SemanticFrame:
    return SemanticFrame(
        term="term",
        sense_id=sense_id,
        actor=None,
        action=None,
        object=None,
        location=None,
        intent=None,
        connector_type="neutral_extension",
    )


@lru_cache(maxsize=1)
def _run_rows() -> tuple[dict, ...]:
    engine = ControlledContinuationEngine()
    rows = []
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


def _by_id() -> dict[str, dict]:
    return {row["id"]: row for row in _run_rows()}


def test_modal_tail_repairs_could_and_searched_to_bare_infinitive():
    result = validate_prompt_tail_compatibility(
        "The bat could",
        "The bat could and searched for insects",
        _frame("animal"),
    )

    assert result.passed
    assert result.repair_applied
    assert result.text == "The bat could search for insects"
    assert result.rule_name == "modal_and_past_to_bare_infinitive"


def test_before_and_hit_repairs_to_gerund():
    result = validate_prompt_tail_compatibility(
        "The batter tapped the bat before",
        "The batter tapped the bat before and hit the ball",
        _frame("sports_equipment"),
    )

    assert result.passed
    assert result.text == "The batter tapped the bat before hitting the ball"


def test_motioned_for_and_completed_repairs_to_object_phrase():
    result = validate_prompt_tail_compatibility(
        "Near closing time the bank manager motioned for",
        "Near closing time the bank manager motioned for and completed the transaction",
        _frame("financial_institution"),
    )

    assert result.passed
    assert result.text == "Near closing time the bank manager motioned for the client to come forward"


def test_while_tourists_and_swam_is_rejected_without_invented_predicate():
    result = validate_prompt_tail_compatibility(
        "On the pier the seal raised its head while tourists",
        "On the pier the seal raised its head while tourists and swam through the cold water",
        _frame("animal"),
    )

    assert not result.passed
    assert result.rejection_reason == "while_subject_requires_predicate_for_subject"


def test_turned_toward_and_carried_repairs_to_object():
    result = validate_prompt_tail_compatibility(
        "The foreman signaled and the crane turned toward",
        "The foreman signaled and the crane turned toward and carried the load",
        _frame("machine"),
    )

    assert result.passed
    assert result.text == "The foreman signaled and the crane turned toward the load"


def test_as_the_hook_operator_clause_repairs_to_hook_predicate():
    result = validate_prompt_tail_compatibility(
        "The crew waited below the crane as the hook",
        "The crew waited below the crane as the hook the operator checked the cables",
        _frame("machine"),
    )

    assert result.passed
    assert result.text == "The crew waited below the crane as the hook rose"


def test_made_everyone_and_brought_repairs_to_complement():
    result = validate_prompt_tail_compatibility(
        "After the thaw the spring mornings made everyone",
        "After the thaw the spring mornings made everyone and brought warmer days",
        _frame("season"),
    )

    assert result.passed
    assert result.text == "After the thaw the spring mornings made everyone feel warmer"


def test_existing_subject_redundant_and_is_removed_for_spring_handle():
    result = validate_prompt_tail_compatibility(
        "The latch clicked when the spring inside the handle",
        "The latch clicked when the spring inside the handle and snapped back into place",
        _frame("coil"),
    )

    assert result.passed
    assert result.text == "The latch clicked when the spring inside the handle snapped back into place"


def test_unmapped_preposition_tail_rejects_instead_of_generic_fallback():
    result = validate_prompt_tail_compatibility(
        "The worker waited for",
        "The worker waited for and completed the task",
        _frame("machine"),
    )

    assert not result.passed
    assert result.rejection_reason == "object_or_purpose_required"


def test_reviewed_rows_have_final_safe_text_or_audit():
    rows = _by_id()
    expected = {
        "v1-022": ("continue", "Near closing time the bank manager motioned for the client to come forward"),
        "v1-025": ("continue", "The ranger dimmed the attic light so the bat could search for insects"),
        "v1-027": ("continue", "The batter tapped the bat on the plate before hitting the ball"),
        "v1-030": ("audit", ""),
        "v1-035": ("continue", "The foreman signaled and the crane turned toward the load"),
        "v1-036": ("continue", "The crew waited below the crane as the hook rose"),
        "v1-037": ("continue", "After the thaw the spring mornings made everyone feel warmer"),
        "v1-038": ("continue", "The latch clicked when the spring inside the handle snapped back into place"),
        "v1-040": ("continue", "The crowd knew the rock would get louder after the band started playing"),
        "v1-063": ("continue", "The bat had wings painted on it before the player swung, and he steadied himself"),
    }
    for row_id, (decision, text) in expected.items():
        row = rows[row_id]
        assert row["decision"] == decision, row_id
        assert row["continuation"] == text, row_id

    assert "audit_reason=prompt_tail_incompatible" in rows["v1-030"]["reasons"]


def test_v1_051_remains_no_safe_repaired_candidate():
    row = _by_id()["v1-051"]
    assert row["decision"] == "audit"
    assert row["continuation"] == ""
    assert "audit_reason=no_safe_repaired_candidate" in row["reasons"]


def test_true_unsafe_rows_remain_audited():
    rows = _by_id()
    for row_id in (
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
    ):
        assert rows[row_id]["decision"] == "audit", row_id


def test_policy_thresholds_were_not_lowered():
    policy = ContinuationPolicy()
    assert policy.min_score == 1.0
    assert policy.min_margin == 1.0


def test_output_is_deterministic():
    first = summarize_rows(list(_run_rows()))
    _run_rows.cache_clear()
    second = summarize_rows(list(_run_rows()))
    assert first == second


def test_manual_bad_phrase_sweep_is_clean():
    rows = list(_run_rows())
    text = "\n".join(row["continuation"] for row in rows)
    for phrase in _BAD_PHRASES:
        assert phrase not in text
    assert check_rows(rows)["flagged_count"] == 0


def test_semantic_quality_checker_flags_prompt_tail_bad_phrases():
    rows = [
        {
            "id": "bad",
            "prompt": "The bat could",
            "ambiguous_term": "bat",
            "selected_sense": "animal",
            "decision": "continue",
            "continuation": "The bat could and searched for insects",
        }
    ]
    quality = check_rows(rows)
    assert quality["flagged_count"] == 1
    assert "prompt_tail:could_and_searched" in quality["flagged_rows"][0]["flags"]
