from __future__ import annotations

import csv
from pathlib import Path

from worldpgt.continuation.continuation_engine import ControlledContinuationEngine
from worldpgt.continuation.sense_memory import ExplicitSenseMemory

_PROMPTS = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "continuation_prompts_v1.csv"
)


def test_continued_rows_use_semantic_renderer_reason():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt("The customer reached the bank teller with cash to")
    assert result.decision == "continue"
    assert "renderer=semantic_renderer_v2" in result.memory_hits
    assert "semantic_frame_intent=transaction" in result.memory_hits
    assert any(h.startswith("candidate_count=") for h in result.memory_hits)
    assert any(h.startswith("selected_candidate=") for h in result.memory_hits)


def test_unsafe_candidate_falls_back_to_audit():
    memory = ExplicitSenseMemory(include_builtin=False)
    memory.add_sense(
        "bank",
        "river_edge",
        ["river"],
        ["current cast his line"],
        {"neutral_extension": ["current cast his line"]},
    )
    engine = ControlledContinuationEngine(memory=memory)
    result = engine.continue_prompt("The river bank")
    assert result.decision == "audit"
    assert result.continuation == ""
    assert result.selected_sense == "river_edge"
    assert "audit_reason=no_safe_semantic_candidate" in result.reasons
    # Surface-risk reporting is preserved alongside the new reason.
    assert "audit_reason=surface_realization_risk" in result.reasons
    assert "surface_risk=current_cast_his_line" in result.memory_hits


def test_sense_selection_unchanged_on_full_v1_set():
    """Every continued row keeps the correct sense (wrong_continue_count == 0)."""
    engine = ControlledContinuationEngine()
    continue_count = 0
    wrong_continue = 0
    with _PROMPTS.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            result = engine.continue_prompt(record.get("prompt", ""))
            if result.decision != "continue":
                continue
            continue_count += 1
            expected = (record.get("expected_sense", "") or "").strip() or None
            if expected is not None and result.selected_sense != expected:
                wrong_continue += 1

    # v1-051 still audits rather than emitting a subject-drift continuation.
    assert continue_count == 58
    assert wrong_continue == 0


def test_improved_rows_drop_repeated_words():
    engine = ControlledContinuationEngine()
    cases = {
        "In April the spring rain warmed the garden and": ("season", "the flowers opened"),
        "The bat was not for a game as it tucked its wings and": ("animal", "it searched for insects"),
        "The seal was not on a document when it dove for fish and": ("animal", "it surfaced for air"),
        "The rock song described a heavy stone on the ground and": ("stone", "it rolled downhill"),
    }
    for prompt, (sense, ending) in cases.items():
        result = engine.continue_prompt(prompt)
        assert result.decision == "continue"
        assert result.selected_sense == sense
        assert result.continuation.endswith(ending)
