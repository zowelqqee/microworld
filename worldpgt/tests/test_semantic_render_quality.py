from __future__ import annotations

from worldpgt.experiments.check_semantic_render_quality import check_rows


def _row(prompt: str, continuation: str, term: str = "bank") -> dict:
    return {
        "id": "1",
        "prompt": prompt,
        "ambiguous_term": term,
        "selected_sense": "x",
        "decision": "continue",
        "continuation": continuation,
    }


def test_flags_story_drift():
    summary = check_rows(
        [_row("She reached the bank to", 'She reached the bank to ask her boyfriend "where"')]
    )
    assert summary["flagged_count"] == 1
    flags = summary["flagged_rows"][0]["flags"]
    assert any(f.startswith("story_drift") for f in flags)


def test_flags_duplicated_action_noun():
    summary = check_rows(
        [_row("The metal spring compressed inside the mechanism and",
              "The metal spring compressed inside the mechanism and the mechanism snapped back",
              term="spring")]
    )
    assert summary["flagged_count"] == 1
    flags = summary["flagged_rows"][0]["flags"]
    assert any(f.startswith("repeated_from_prompt") for f in flags)


def test_flags_empty_continuation_on_continue_row():
    summary = check_rows([_row("The customer reached the bank to", "")])
    assert summary["flagged_count"] == 1
    assert summary["flagged_rows"][0]["flags"] == ["empty_continuation"]


def test_passes_clean_semantic_continuation():
    summary = check_rows(
        [_row("The customer reached the bank teller with cash to",
              "The customer reached the bank teller with cash to open an account")]
    )
    assert summary["flagged_count"] == 0
    assert summary["flagged_rate"] == 0.0


def test_ignores_non_continue_rows():
    rows = [
        {"id": "1", "prompt": "p", "ambiguous_term": "bank", "decision": "audit", "continuation": ""},
    ]
    summary = check_rows(rows)
    assert summary["total_continued"] == 0
    assert summary["flagged_count"] == 0


def test_softened_repeat_is_not_flagged():
    # "catch another fish" after "...chasing fish" is acceptable (softened by "another").
    summary = check_rows(
        [_row("The seal swam through ocean water chasing fish to",
              "The seal swam through ocean water chasing fish to catch another fish",
              term="seal")]
    )
    assert summary["flagged_count"] == 0
