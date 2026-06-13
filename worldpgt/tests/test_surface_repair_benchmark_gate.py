"""Benchmark regression gate for the deterministic surface-repair layer.

Locks the measured v1.2 safety numbers after surface repair. The engine is run
directly over the frozen v1 dataset (no external services, no regenerated output
files), so the gate is fully deterministic and self-contained.

If any of these constraints change, this test fails on purpose: it is the
contract that surface repair did not regress safety. See ``surface_repair.py``
and the "Surface Repair Layer" section of ``worldpgt/README.md``.
"""

from __future__ import annotations

import csv
from pathlib import Path

from worldpgt.continuation.continuation_engine import ControlledContinuationEngine
from worldpgt.experiments.check_semantic_render_quality import check_rows
from worldpgt.experiments.risk_coverage_metrics import summarize_rows

_PROMPTS = Path(__file__).resolve().parents[1] / "experiments" / "continuation_prompts_v1.csv"

# Old broken surface fragments that surface repair removed unconditionally.
# None of these may appear in any emitted continuation again.
#
# ("and lay near the river" is deliberately NOT here: it is only a bug when it
# attaches across an intervening actor (v1-051, now audited). It is a valid
# compound predicate over a single subject in v1-019, which still continues.)
_OLD_BROKEN_PHRASES = [
    "before the swing he steadied himself",
    "chasing fish to catch another fish",
    "close the envelope",
    "its wings searched for insects",
]


def _run_benchmark() -> list[dict]:
    """Run the engine over the frozen dataset; return output-shaped rows."""
    engine = ControlledContinuationEngine()
    rows: list[dict] = []
    with _PROMPTS.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            prompt = record.get("prompt", "")
            result = engine.continue_prompt(prompt)
            rows.append(
                {
                    "id": record.get("id", ""),
                    "prompt": prompt,
                    "ambiguous_term": record.get("ambiguous_term", "")
                    or result.ambiguous_term
                    or "",
                    "expected_sense": record.get("expected_sense", "") or "",
                    "difficulty_type": record.get("difficulty_type", ""),
                    "continuation": result.continuation,
                    "selected_sense": result.selected_sense or "",
                    "decision": result.decision,
                    "reasons": " | ".join(result.reasons),
                }
            )
    return rows


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["id"]: row for row in rows}


def test_benchmark_safety_numbers_locked():
    summary = summarize_rows(_run_benchmark())
    assert summary["wrong_continue_count"] == 0
    assert summary["continue_count"] == 37
    assert summary["audit_count"] == 83
    assert summary["precision_on_continued"] == 1.0


def test_semantic_quality_flagged_count_is_zero():
    quality = check_rows(_run_benchmark())
    assert quality["flagged_count"] == 0, quality["flagged_rows"]


def test_v1_051_audited_with_no_safe_repaired_candidate():
    row = _by_id(_run_benchmark())["v1-051"]
    assert row["decision"] == "audit"
    assert row["continuation"] == ""
    assert "audit_reason=no_safe_repaired_candidate" in row["reasons"]


def test_repaired_rows_continued_and_clean():
    rows = _by_id(_run_benchmark())
    for row_id in ("v1-007", "v1-009", "v1-011", "v1-043"):
        row = rows[row_id]
        assert row["decision"] == "continue", row_id
        assert row["continuation"].strip(), row_id


def test_no_emitted_continuation_contains_old_broken_phrase():
    """Row-level regression: the exact old bugs must never reappear."""
    rows = _by_id(_run_benchmark())

    # Each old phrase is pinned to the row that used to emit it.
    pinned = {
        "v1-007": "before the swing he steadied himself",
        "v1-009": "chasing fish to catch another fish",
        "v1-011": "close the envelope",
        "v1-043": "its wings searched for insects",
        "v1-051": "and lay near the river",
    }
    for row_id, broken in pinned.items():
        assert broken not in rows[row_id]["continuation"], (row_id, broken)

    # And globally: no emitted continuation contains any of the old fragments.
    for row in rows.values():
        for broken in _OLD_BROKEN_PHRASES:
            assert broken not in row["continuation"], (row["id"], broken)


def test_gate_is_deterministic():
    first = summarize_rows(_run_benchmark())
    second = summarize_rows(_run_benchmark())
    assert first == second
