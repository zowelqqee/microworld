from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from worldpgt.experiments import coverage_gap_curriculum as curriculum


_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
_COVERAGE_OUTPUT = _EXPERIMENTS / "microworld_continuation_v1_2_coverage_mode_outputs.csv"
_COVERAGE_SUMMARY = _EXPERIMENTS / "microworld_continuation_v1_2_coverage_mode_summary.json"
_AUDIT_REASONS = _EXPERIMENTS / "microworld_continuation_v1_2_audit_reasons.json"
_AUDIT_PLAN = _EXPERIMENTS / "microworld_continuation_v1_2_audit_improvement_plan.json"
_DATASET = _EXPERIMENTS / "continuation_prompts_v1.csv"
_TRUSTED_OUTPUT = _EXPERIMENTS / "microworld_continuation_v1_2_outputs.csv"


def _build() -> dict:
    return curriculum.build_curriculum(
        curriculum._read_csv(str(_COVERAGE_OUTPUT)),
        curriculum._read_json(str(_COVERAGE_SUMMARY)),
        curriculum._read_json(str(_AUDIT_REASONS)),
        curriculum._read_json(str(_AUDIT_PLAN)),
        curriculum._read_csv(str(_DATASET)),
    )


def _tasks_by_id(data: dict) -> dict[str, dict]:
    return {task["row_id"]: task for task in data["learning_tasks"]}


def test_curriculum_identifies_exactly_unavailable_rows():
    coverage_rows = curriculum._read_csv(str(_COVERAGE_OUTPUT))
    unavailable = {row["id"] for row in coverage_rows if row["candidate_status"] == "unavailable"}
    tasks = _tasks_by_id(_build())

    assert set(tasks) == unavailable
    assert len(tasks) == 41


def test_summary_counts_operational_coverage():
    summary = _build()["summary"]

    assert summary["trusted_continue_count"] == 58
    assert summary["untrusted_candidate_count"] == 21
    assert summary["learning_task_count"] == 41
    assert summary["operational_coverage_count"] == 120
    assert summary["operational_coverage_rate"] == 1.0


def test_every_unavailable_row_gets_exactly_one_learning_task():
    task_ids = [task["row_id"] for task in _build()["learning_tasks"]]
    assert len(task_ids) == len(set(task_ids)) == 41


def test_trusted_and_untrusted_rows_do_not_get_learning_tasks():
    coverage_rows = curriculum._read_csv(str(_COVERAGE_OUTPUT))
    covered = {
        row["id"]
        for row in coverage_rows
        if row["candidate_status"] in {"trusted", "untrusted"}
    }
    tasks = set(_tasks_by_id(_build()))
    assert covered.isdisjoint(tasks)


def test_true_unsafe_rows_are_keep_audit_learning_tasks():
    tasks = _tasks_by_id(_build())
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
        task = tasks[row_id]
        assert task["gap_type"] == "true_unsafe", row_id
        assert task["required_input"] == ["keep_audit_policy"], row_id
        assert task["proposed_learning_payload"]["keep_audit_rationale"], row_id


def test_v1_051_is_no_safe_rewrite_keep_audit_task():
    task = _tasks_by_id(_build())["v1-051"]
    assert task["gap_type"] == "no_safe_rewrite"
    assert "phrase_candidates" in task["required_input"]
    assert task["proposed_learning_payload"]["keep_audit_rationale"]


def test_missing_sense_memory_rows_request_sense_memory_entry():
    tasks = _tasks_by_id(_build())
    for row_id in ("v1-116", "v1-117", "v1-118", "v1-119", "v1-120"):
        task = tasks[row_id]
        assert task["gap_type"] == "missing_sense_memory", row_id
        assert "sense_memory_entry" in task["required_input"], row_id
        assert "semantic_frame" in task["required_input"], row_id


def test_needs_instrumentation_fallback_requests_trace_instrumentation():
    row = {
        "id": "x-001",
        "prompt": "A prompt",
        "ambiguous_term": "bank",
        "expected_sense": "river_edge",
        "trusted_decision": "audit",
        "candidate_status": "unavailable",
        "candidate_reason": "unclear",
        "candidate_review_action": "needs_memory",
        "reasons": "unrecognized",
    }
    data = curriculum.build_curriculum(
        [row],
        {"total_rows": 1, "trusted_continue_count": 0, "candidate_by_status": {"untrusted": 0}},
        {"rows": []},
        {"proposals": []},
        [],
    )
    task = data["learning_tasks"][0]
    assert task["gap_type"] == "needs_trace_instrumentation"
    assert "trace_instrumentation" in task["required_input"]


def test_task_ids_are_deterministic():
    first = _build()
    second = _build()
    assert [task["task_id"] for task in first["learning_tasks"]] == [
        task["task_id"] for task in second["learning_tasks"]
    ]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_cli_writes_json_and_csv(tmp_path, monkeypatch):
    out_json = tmp_path / "curriculum.json"
    out_csv = tmp_path / "curriculum.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "coverage_gap_curriculum",
            "--coverage-output",
            str(_COVERAGE_OUTPUT),
            "--coverage-summary",
            str(_COVERAGE_SUMMARY),
            "--audit-reasons",
            str(_AUDIT_REASONS),
            "--audit-plan",
            str(_AUDIT_PLAN),
            "--dataset",
            str(_DATASET),
            "--output-json",
            str(out_json),
            "--output-csv",
            str(out_csv),
        ],
    )
    curriculum.main()

    assert out_json.exists()
    assert out_csv.exists()
    data = json.loads(out_json.read_text())
    assert data["summary"]["learning_task_count"] == 41
    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 41
    assert set(curriculum.CSV_FIELDS) == set(rows[0].keys())


def test_curriculum_builder_does_not_modify_benchmark_or_coverage_files(tmp_path):
    before_trusted = hashlib.sha256(_TRUSTED_OUTPUT.read_bytes()).hexdigest()
    before_coverage = hashlib.sha256(_COVERAGE_OUTPUT.read_bytes()).hexdigest()
    data = _build()
    curriculum.write_json(data, str(tmp_path / "curriculum.json"))
    curriculum.write_csv(data, str(tmp_path / "curriculum.csv"))
    after_trusted = hashlib.sha256(_TRUSTED_OUTPUT.read_bytes()).hexdigest()
    after_coverage = hashlib.sha256(_COVERAGE_OUTPUT.read_bytes()).hexdigest()

    assert before_trusted == after_trusted
    assert before_coverage == after_coverage


def test_curriculum_does_not_suggest_forbidden_changes():
    text = json.dumps(_build(), sort_keys=True).lower()
    forbidden = [
        "lower threshold",
        "threshold lowering",
        "weaken validator",
        "validator weakening",
        "generic trusted fallback",
        "generic fallback",
        "force continuation",
        "torch",
        "transformers",
        "openai",
        "gpt",
        "neural",
        "training",
    ]
    for phrase in forbidden:
        assert phrase not in text
