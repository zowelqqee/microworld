from __future__ import annotations

import csv
import sys

import pytest

from worldpgt.baselines.gpt2.compare_microworld_vs_gpt2_audit import compare
from worldpgt.baselines.gpt2.create_gpt2_audit_csv import AUDIT_FIELDS
from worldpgt.baselines.gpt2.summarize_gpt2_audit import summarize_rows
from worldpgt.experiments.run_v1_microworld_continuation import OUTPUT_FIELDS as MW_FIELDS


def test_summarize_counts_good_bad_unclear_and_precision():
    summary = summarize_rows(
        [
            _gpt("1", "animal", "animal", "good", "cue_rich"),
            _gpt("2", "animal", "sports_equipment", "bad", "cue_rich"),
            _gpt("3", "animal", "", "unclear", "weak_cue"),
        ]
    )

    assert summary["total"] == 3
    assert summary["audited_count"] == 3
    assert summary["good"] == 1
    assert summary["bad"] == 1
    assert summary["unclear"] == 1
    assert summary["precision"] == pytest.approx(0.5)


def test_summarize_sense_correctness():
    summary = summarize_rows(
        [
            _gpt("1", "animal", "animal", "good", "cue_rich"),
            _gpt("2", "animal", "sports_equipment", "bad", "cue_rich"),
            _gpt("3", "animal", "", "unclear", "weak_cue"),
            _gpt("4", "", "", "good", "no_known_term"),
        ]
    )

    assert summary["correct_sense_count"] == 2
    assert summary["wrong_sense_count"] == 1
    assert summary["no_sense_count"] == 1
    assert summary["correct_sense_rate"] == pytest.approx(0.5)
    assert summary["wrong_sense_rate"] == pytest.approx(0.25)


def test_summarize_grouped_difficulty_metrics():
    summary = summarize_rows(
        [
            _gpt("1", "animal", "animal", "good", "cue_rich"),
            _gpt("2", "sports_equipment", "", "unclear", "cue_rich"),
            _gpt("3", "machine", "bird", "bad", "conflicting_cue"),
        ]
    )

    cue = summary["by_difficulty_type"]["cue_rich"]
    assert cue["total"] == 2
    assert cue["good"] == 1
    assert cue["unclear"] == 1
    assert cue["precision"] == pytest.approx(1.0)
    assert cue["correct_sense_count"] == 1
    assert cue["no_sense_count"] == 1

    conflict = summary["by_difficulty_type"]["conflicting_cue"]
    assert conflict["bad"] == 1
    assert conflict["wrong_sense_count"] == 1


def test_comparison_handles_microworld_audit_vs_gpt2_good(tmp_path):
    mw_path = tmp_path / "mw.csv"
    gpt_path = tmp_path / "gpt.csv"
    _write_mw(mw_path, [_mw("1", "animal", "", "audit")])
    _write_gpt(gpt_path, [_gpt("1", "animal", "animal", "good", "weak_cue")])

    summary = compare(str(mw_path), str(gpt_path))

    assert summary["head_to_head"]["microworld_audit_gpt2_good"] == 1
    assert summary["head_to_head"]["gpt2_good_microworld_audit"] == 1
    assert len(summary["head_to_head"]["disagreements"]) == 1


def test_comparison_handles_microworld_continue_vs_gpt2_bad(tmp_path):
    mw_path = tmp_path / "mw.csv"
    gpt_path = tmp_path / "gpt.csv"
    _write_mw(mw_path, [_mw("1", "animal", "animal", "continue")])
    _write_gpt(gpt_path, [_gpt("1", "animal", "sports_equipment", "bad", "cue_rich")])

    summary = compare(str(mw_path), str(gpt_path))

    assert summary["head_to_head"]["microworld_continue_gpt2_bad"] == 1
    assert summary["head_to_head"]["microworld_safe_gpt2_bad"] == 1
    assert len(summary["head_to_head"]["disagreements"]) == 1


def test_comparison_includes_disagreements(tmp_path):
    mw_path = tmp_path / "mw.csv"
    gpt_path = tmp_path / "gpt.csv"
    _write_mw(
        mw_path,
        [
            _mw("1", "animal", "", "audit"),
            _mw("2", "machine", "machine", "continue"),
        ],
    )
    _write_gpt(
        gpt_path,
        [
            _gpt("1", "animal", "animal", "good", "weak_cue"),
            _gpt("2", "machine", "bird", "bad", "cue_rich"),
        ],
    )

    summary = compare(str(mw_path), str(gpt_path))

    ids = {row["id"] for row in summary["head_to_head"]["disagreements"]}
    assert ids == {"1", "2"}


def test_no_torch_tiktoken_transformers_needed_for_summary_or_compare():
    for name in ["torch", "tiktoken", "transformers"]:
        sys.modules.pop(name, None)

    summarize_rows([_gpt("1", "animal", "animal", "good", "cue_rich")])

    for name in ["torch", "tiktoken", "transformers"]:
        assert name not in sys.modules


def _write_mw(path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_gpt(path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _mw(row_id: str, expected: str, selected: str, decision: str) -> dict:
    return {
        "id": row_id,
        "prompt": f"prompt {row_id}",
        "ambiguous_term": "bat",
        "expected_sense": expected,
        "difficulty_type": "cue_rich",
        "notes": "test",
        "continuation": "",
        "selected_sense": selected,
        "confidence": "1.0",
        "decision": decision,
        "reasons": "",
        "memory_hits": "",
    }


def _gpt(row_id: str, expected: str, judged: str, label: str, difficulty: str) -> dict:
    return {
        "id": row_id,
        "prompt": f"prompt {row_id}",
        "ambiguous_term": "bat",
        "expected_sense": expected,
        "difficulty_type": difficulty,
        "notes": "test",
        "model": "nanogpt:gpt2",
        "completion": "completion",
        "full_text": "full text",
        "judged_text": "judged text",
        "judged_sense": judged,
        "label": label,
        "audit_notes": "notes",
    }
