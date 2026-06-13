from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from worldpgt.experiments import audit_fix_proposer as proposer


_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
_AUDIT_REASONS = _EXPERIMENTS / "microworld_continuation_v1_2_audit_reasons.json"
_OUTPUTS = _EXPERIMENTS / "microworld_continuation_v1_2_outputs.csv"


def _load_plan() -> dict:
    return proposer.build_plan(
        proposer.read_audit_report(str(_AUDIT_REASONS)),
        proposer.read_output_rows(str(_OUTPUTS)),
    )


def _by_row(plan: dict) -> dict[str, dict]:
    return {proposal["row_ids"][0]: proposal for proposal in plan["proposals"]}


def _mini_report(row: dict) -> dict:
    return {
        "summary": {"audit_count": 1, "continue_count": 0},
        "rows": [row],
    }


def _diag(row_id: str, reason: str, term: str, expected: str, prompt: str) -> dict:
    return {
        "row_id": row_id,
        "prompt": prompt,
        "term": term,
        "expected_sense": expected,
        "selected_sense": None,
        "confidence": 0.0,
        "decision": "audit",
        "primary_audit_reason": reason,
        "secondary_reasons": [],
        "evidence": {},
        "suggested_next_action": "",
    }


def _output(row_id: str, term: str, expected: str, prompt: str, reasons: str = "") -> dict:
    return {
        "id": row_id,
        "prompt": prompt,
        "ambiguous_term": term,
        "expected_sense": expected,
        "selected_sense": "",
        "confidence": "0.0000",
        "decision": "audit",
        "reasons": reasons,
        "memory_hits": "",
    }


def test_proposer_reads_audit_reasons_and_is_deterministic():
    first = _load_plan()
    second = _load_plan()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["policy"] == {
        "mode": "propose_only",
        "auto_apply": False,
        "generation_behavior_changed": False,
        "thresholds_changed": False,
        "validators_changed": False,
    }


def test_true_unsafe_rows_become_keep_audit():
    proposals = _by_row(_load_plan())
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
        proposal = proposals[row_id]
        assert proposal["proposal_type"] == "keep_audit", row_id
        assert proposal["recommended_action"] == "keep_audit", row_id


def test_unsupported_rows_become_keep_audit():
    proposals = _by_row(_load_plan())
    for row_id in ("v1-111", "v1-112", "v1-113", "v1-114", "v1-115"):
        assert proposals[row_id]["recommended_action"] == "keep_audit", row_id


def test_v1_051_no_safe_repaired_candidate_becomes_keep_audit():
    proposal = _by_row(_load_plan())["v1-051"]
    assert proposal["proposal_type"] == "keep_audit"
    assert proposal["recommended_action"] == "keep_audit"
    assert proposal["source_reason"] == "no_safe_repaired_candidate"


def test_missing_or_weak_concrete_cue_produces_cue_memory_addition():
    row = _diag(
        "x-001",
        "missing_or_weak_cue_support",
        "bat",
        "sports_equipment",
        "In the dugout the bat waited",
    )
    output = _output("x-001", "bat", "sports_equipment", row["prompt"], "all_sense_scores_zero")
    plan = proposer.build_plan(_mini_report(row), [output])
    proposal = plan["proposals"][0]

    assert proposal["proposal_type"] == "cue_memory_addition"
    assert proposal["recommended_action"] == "auto_safe_later"
    assert proposal["risk_level"] == "low"
    assert proposal["proposed_change"]["add_positive_cues"] == ["dugout"]


def test_broad_cue_is_not_auto_safe():
    row = _diag(
        "x-002",
        "missing_or_weak_cue_support",
        "seal",
        "animal",
        "The seal moved through water",
    )
    output = _output("x-002", "seal", "animal", row["prompt"], "all_sense_scores_zero")
    plan = proposer.build_plan(_mini_report(row), [output])
    proposal = plan["proposals"][0]

    assert proposal["proposal_type"] == "cue_memory_addition"
    assert proposal["risk_level"] == "high"
    assert proposal["recommended_action"] == "human_review"


def test_v1_062_mixed_bank_cues_are_guard_rule_human_review():
    proposal = _by_row(_load_plan())["v1-062"]

    assert proposal["proposal_type"] == "guard_rule_addition"
    assert proposal["recommended_action"] == "human_review"
    assert proposal["risk_level"] != "low"
    assert "cash" not in proposal["proposed_change"]["add_positive_cues"]
    assert "cash" in proposal["evidence"]["conflicting_cues"]
    assert "stream" in proposal["evidence"]["supporting_cues"]
    assert "stream" in proposal["proposed_change"]["add_positive_cues"]


def test_v1_064_mixed_bat_cues_are_guard_rule_human_review():
    proposal = _by_row(_load_plan())["v1-064"]

    assert proposal["proposal_type"] == "guard_rule_addition"
    assert proposal["recommended_action"] == "human_review"
    assert proposal["risk_level"] != "low"
    assert "player" not in proposal["proposed_change"]["add_positive_cues"]
    assert "player" in proposal["evidence"]["conflicting_cues"]
    assert "rafters" in proposal["evidence"]["supporting_cues"]
    assert "rafters" in proposal["proposed_change"]["add_positive_cues"]


def test_conflicting_cue_proposals_are_never_low_or_auto_safe():
    for proposal in _load_plan()["proposals"]:
        if proposal["evidence"]["conflicting_cues"]:
            assert proposal["risk_level"] != "low", proposal["row_ids"]
            assert proposal["recommended_action"] != "auto_safe_later", proposal["row_ids"]


def test_broad_cue_proposals_are_never_auto_safe():
    plan = _load_plan()
    synthetic = _mini_report(
        _diag(
            "x-broad",
            "missing_or_weak_cue_support",
            "seal",
            "animal",
            "The seal moved under the table near people",
        )
    )
    synthetic_output = _output(
        "x-broad",
        "seal",
        "animal",
        "The seal moved under the table near people",
        "all_sense_scores_zero",
    )
    plan["proposals"].extend(proposer.build_plan(synthetic, [synthetic_output])["proposals"])

    for proposal in plan["proposals"]:
        if proposal["evidence"]["broad_cues"]:
            assert proposal["recommended_action"] != "auto_safe_later", proposal["row_ids"]


def test_sense_tie_produces_human_review_guard_rule():
    proposal = _by_row(_load_plan())["v1-061"]
    assert proposal["proposal_type"] == "guard_rule_addition"
    assert proposal["recommended_action"] == "human_review"
    assert proposal["risk_level"] == "medium"


def test_low_margin_produces_human_review():
    proposal = _by_row(_load_plan())["v1-066"]
    assert proposal["source_reason"] == "low_margin"
    assert proposal["recommended_action"] == "human_review"


def test_surface_validation_failed_proposes_phrase_or_keep_audit():
    proposals = _by_row(_load_plan())
    assert proposals["v1-041"]["proposal_type"] == "phrase_candidate_addition"
    assert proposals["v1-041"]["recommended_action"] == "human_review"
    assert proposals["v1-030"]["proposal_type"] == "keep_audit"
    assert proposals["v1-030"]["recommended_action"] == "keep_audit"


def test_plan_does_not_suggest_threshold_or_validator_changes_or_generic_fallback():
    text = json.dumps(_load_plan(), sort_keys=True).lower()
    forbidden = [
        "lower threshold",
        "threshold lowering",
        "weaken validator",
        "validator weakening",
        "generic fallback",
        "force continuation",
    ]
    for phrase in forbidden:
        assert phrase not in text


def test_plan_does_not_mention_disallowed_modeling_tools():
    text = json.dumps(_load_plan(), sort_keys=True).lower()
    for phrase in ("torch", "transformers", "openai", "gpt", "neural", "training"):
        assert phrase not in text


def test_summary_counts_match_proposals():
    plan = _load_plan()
    summary = plan["summary"]
    proposals = plan["proposals"]

    assert summary["total_audits"] == 62
    assert summary["proposals_total"] == len(proposals) == 62
    assert summary["auto_safe_proposals"] == sum(p["recommended_action"] == "auto_safe_later" for p in proposals)
    assert summary["review_required"] == sum(p["recommended_action"] == "human_review" for p in proposals)
    assert summary["keep_audit"] == sum(p["recommended_action"] == "keep_audit" for p in proposals)
    assert summary["needs_instrumentation"] == sum(
        p["recommended_action"] == "needs_instrumentation" for p in proposals
    )
    assert sum(summary["by_proposal_type"].values()) == len(proposals)
    assert sum(summary["by_risk_level"].values()) == len(proposals)
    assert sum(summary["by_action"].values()) == len(proposals)


def test_cli_writes_json_and_csv(tmp_path, monkeypatch):
    out_json = tmp_path / "plan.json"
    out_csv = tmp_path / "plan.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_fix_proposer",
            "--audit-reasons",
            str(_AUDIT_REASONS),
            "--outputs",
            str(_OUTPUTS),
            "--output-json",
            str(out_json),
            "--output-csv",
            str(out_csv),
        ],
    )
    proposer.main()

    assert out_json.exists()
    assert out_csv.exists()
    plan = json.loads(out_json.read_text())
    assert plan["summary"]["total_audits"] == 62
    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 62
    assert set(proposer.CSV_FIELDS) == set(rows[0].keys())


def test_running_proposer_does_not_modify_benchmark_output(tmp_path):
    before = hashlib.sha256(_OUTPUTS.read_bytes()).hexdigest()
    plan = proposer.build_plan(
        proposer.read_audit_report(str(_AUDIT_REASONS)),
        proposer.read_output_rows(str(_OUTPUTS)),
    )
    proposer.write_json(plan, str(tmp_path / "plan.json"))
    proposer.write_csv(plan, str(tmp_path / "plan.csv"))
    after = hashlib.sha256(_OUTPUTS.read_bytes()).hexdigest()
    assert before == after


def test_current_benchmark_sanity_and_keep_audit_classification():
    outputs = proposer.read_output_rows(str(_OUTPUTS))
    plan = _load_plan()
    proposals = _by_row(plan)
    assert sum(row["decision"] == "audit" for row in outputs) == 62
    assert sum(row["decision"] == "continue" for row in outputs) == 58
    assert outputs == proposer.read_output_rows(str(_OUTPUTS))
    assert proposals["v1-051"]["recommended_action"] == "keep_audit"
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
        assert proposals[row_id]["recommended_action"] == "keep_audit"
