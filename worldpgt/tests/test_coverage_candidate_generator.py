from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from worldpgt.continuation.continuation_policy import ContinuationPolicy
from worldpgt.continuation.coverage_candidate_generator import generate_untrusted_candidate
from worldpgt.experiments import run_v1_2_coverage_mode as coverage


_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
_PROMPTS = _EXPERIMENTS / "continuation_prompts_v1.csv"
_TRUSTED = _EXPERIMENTS / "microworld_continuation_v1_2_outputs.csv"
_PLAN = _EXPERIMENTS / "microworld_continuation_v1_2_audit_improvement_plan.json"


def _run_tmp(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    output_csv = tmp_path / "coverage.csv"
    output_json = tmp_path / "coverage.json"
    rows, summary = coverage.run(
        str(_PROMPTS),
        str(_TRUSTED),
        str(_PLAN),
        str(output_csv),
        str(output_json),
    )
    return rows, summary, output_csv, output_json


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["id"]: row for row in rows}


def test_coverage_mode_preserves_trusted_benchmark_counts(tmp_path):
    _, summary, _, _ = _run_tmp(tmp_path)

    assert summary["trusted_continue_count"] == 58
    assert summary["trusted_audit_count"] == 62
    assert summary["trusted_wrong_continue_count"] == 0


def test_existing_trusted_rows_keep_same_trusted_continuation(tmp_path):
    rows, _, _, _ = _run_tmp(tmp_path)
    trusted_rows = {
        row["id"]: row
        for row in csv.DictReader(_TRUSTED.open(newline="", encoding="utf-8"))
        if row["decision"] == "continue"
    }
    coverage_rows = _by_id(rows)

    for row_id, trusted in trusted_rows.items():
        row = coverage_rows[row_id]
        assert row["trusted_decision"] == "continue", row_id
        assert row["trusted_continuation"] == trusted["continuation"], row_id
        assert row["candidate_full_text"] == trusted["continuation"], row_id
        assert row["candidate_status"] == "trusted", row_id


def test_audited_rows_may_get_untrusted_candidate_but_remain_audit(tmp_path):
    rows, _, _, _ = _run_tmp(tmp_path)
    row = _by_id(rows)["v1-064"]

    assert row["trusted_decision"] == "audit"
    assert row["trusted_continuation"] == ""
    assert row["candidate_status"] == "untrusted"
    assert row["candidate_full_text"]
    assert row["candidate_selected_sense"] == "animal"


def test_true_unsafe_rows_remain_trusted_audit_and_unavailable(tmp_path):
    rows, _, _, _ = _run_tmp(tmp_path)
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
        row = _by_id(rows)[row_id]
        assert row["trusted_decision"] == "audit", row_id
        assert row["candidate_status"] == "unavailable", row_id
        assert row["candidate_review_action"] == "keep_audit", row_id


def test_v1_051_remains_trusted_audit_and_unavailable_keep_audit(tmp_path):
    rows, _, _, _ = _run_tmp(tmp_path)
    row = _by_id(rows)["v1-051"]

    assert row["trusted_decision"] == "audit"
    assert row["trusted_continuation"] == ""
    assert row["candidate_status"] == "unavailable"
    assert row["candidate_review_action"] == "keep_audit"


def test_prompt_tail_validation_applies_to_untrusted_candidates():
    proposal = {
        "proposal_type": "guard_rule_addition",
        "recommended_action": "human_review",
        "risk_level": "medium",
        "evidence": {"prompt_cues": ["pier"], "conflicting_cues": []},
        "proposed_change": {"add_positive_cues": ["pier"]},
    }
    candidate = generate_untrusted_candidate(
        "On the pier the seal raised its head while tourists",
        "seal",
        "animal",
        proposal,
    )

    assert candidate.candidate_status == "unavailable"
    assert candidate.candidate_full_text == ""
    assert "prompt_tail_validator=rejected" in candidate.candidate_trace


def test_candidate_with_broken_prompt_tail_becomes_unavailable():
    proposal = {
        "proposal_type": "phrase_candidate_addition",
        "recommended_action": "human_review",
        "risk_level": "medium",
        "evidence": {"prompt_cues": ["boat"], "conflicting_cues": []},
        "proposed_change": {},
    }
    candidate = generate_untrusted_candidate(
        "The credit card floated to the bank where the boat",
        "bank",
        "river_edge",
        proposal,
    )

    assert candidate.candidate_status == "unavailable"
    assert candidate.candidate_full_text == ""
    assert any("candidate_pattern=unfinished_where_subject" in item for item in candidate.candidate_trace)


def test_candidate_coverage_summary_counts_are_correct(tmp_path):
    _, summary, _, _ = _run_tmp(tmp_path)

    assert summary["total_rows"] == 120
    assert summary["candidate_available_count"] == 79
    assert summary["candidate_unavailable_count"] == 41
    assert summary["total_candidate_coverage"] == 0.6583
    assert summary["candidate_by_status"] == {
        "trusted": 58,
        "unavailable": 41,
        "untrusted": 21,
    }
    assert summary["candidate_by_source"] == {
        "audit_fix_proposal": 21,
        "none": 41,
        "trusted_continuation": 58,
    }


def test_candidate_output_is_deterministic(tmp_path):
    first_rows, first_summary, _, _ = _run_tmp(tmp_path / "a")
    second_rows, second_summary, _, _ = _run_tmp(tmp_path / "b")

    assert first_rows == second_rows
    assert first_summary == second_summary


def test_no_disallowed_modeling_references_in_coverage_output(tmp_path):
    rows, summary, _, _ = _run_tmp(tmp_path)
    text = json.dumps({"rows": rows, "summary": summary}, sort_keys=True).lower()
    for phrase in ("torch", "transformers", "openai", "gpt", "neural"):
        assert phrase not in text


def test_policy_thresholds_unchanged():
    policy = ContinuationPolicy()
    assert policy.min_score == 1.0
    assert policy.min_margin == 1.0


def test_cli_writes_csv_and_json(tmp_path, monkeypatch):
    out_csv = tmp_path / "coverage.csv"
    out_json = tmp_path / "coverage.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_v1_2_coverage_mode",
            "--input",
            str(_PROMPTS),
            "--trusted-output",
            str(_TRUSTED),
            "--audit-plan",
            str(_PLAN),
            "--output-csv",
            str(out_csv),
            "--output-json",
            str(out_json),
        ],
    )
    coverage.main()

    assert out_csv.exists()
    assert out_json.exists()
    summary = json.loads(out_json.read_text())
    assert summary["trusted_continue_count"] == 58
    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 120
    assert "candidate_full_text" in rows[0]


def test_coverage_runner_does_not_modify_trusted_output(tmp_path):
    before = hashlib.sha256(_TRUSTED.read_bytes()).hexdigest()
    _run_tmp(tmp_path)
    after = hashlib.sha256(_TRUSTED.read_bytes()).hexdigest()
    assert before == after
