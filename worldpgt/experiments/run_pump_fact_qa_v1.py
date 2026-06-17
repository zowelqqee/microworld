"""Pump Fact QA Benchmark v1 runner.

Generates deterministic questions from the precision-filtered pump answerable
delta, runs them through the assistant surface with ``--overlay pump-dry-run``,
validates the answers, and writes detailed artifacts. It also merges a compact
QA section into ``pump_summary.json`` / ``pump_report.json``.

Local-only and read-only with respect to memory: no network, no neural code, no
mutation of accepted memory, accepted/promoted overlays, or the snapshot
dry-run overlay. The pump overlay stays a proposal/dry-run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.experiments.ask_microworld_v1 import ask
from worldpgt.pump_fact_qa.fact_loader import load_pump_facts
from worldpgt.pump_fact_qa.question_generator import generate_all_prompts
from worldpgt.pump_fact_qa.qa_runner import run_prompts
from worldpgt.pump_fact_qa.report import (
    build_summary,
    collect_failures,
    collect_planner_gaps,
    compact_pump_section,
)
from worldpgt.pump_fact_qa.types import CATEGORY_ADVERSARIAL

_EXPERIMENTS = Path(__file__).resolve().parent
_PUMP_OUT = _EXPERIMENTS / "knowledge_pump_v1"
_OUT = _PUMP_OUT / "pump_fact_qa_v1"

_PRECISION_FACTS = _PUMP_OUT / "pump_precision_answerable_delta.json"
_FALLBACK_FACTS = _PUMP_OUT / "pump_answerable_delta.json"

_PUMP_SUMMARY = _PUMP_OUT / "pump_summary.json"
_PUMP_REPORT = _PUMP_OUT / "pump_report.json"

# Protected files that must never change.
_ACCEPTED = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_PROMOTED = _EXPERIMENTS / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json"
_SNAPSHOT_DRY_RUN = _EXPERIMENTS / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json"
_TRUSTED = _EXPERIMENTS / "accepted_knowledge_memory_v1.json"
_SENSE_MEMORY = _ROOT / "worldpgt" / "continuation" / "sense_memory.py"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING"


def _protected_hashes() -> dict[str, str]:
    return {
        "trusted_memory": _hash(_TRUSTED),
        "accepted_overlay": _hash(_ACCEPTED),
        "promoted_overlay": _hash(_PROMOTED),
        "snapshot_dry_run_overlay": _hash(_SNAPSHOT_DRY_RUN),
        "sense_memory": _hash(_SENSE_MEMORY),
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _merge_pump_section(path: Path, section: dict[str, Any]) -> None:
    """Additively merge the compact QA section into an existing pump artifact."""
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(data, dict):
        return
    if path.name == "pump_report.json" and isinstance(data.get("summary"), dict):
        data["summary"].update(section)
    data.update(section)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(
    *,
    out_dir: Path = _OUT,
    precision_path: Path = _PRECISION_FACTS,
    fallback_path: Path = _FALLBACK_FACTS,
    update_pump_summary: bool = True,
    ask_fn=ask,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    before = _protected_hashes()

    facts = load_pump_facts(precision_path, fallback_path)
    prompt_groups = generate_all_prompts(facts)
    all_prompts = (
        prompt_groups["positive"]
        + prompt_groups["adversarial"]
        + prompt_groups["current"]
    )

    results = run_prompts(all_prompts, ask_fn)
    summary = build_summary(facts, results)
    failures = collect_failures(results)
    planner_gaps = collect_planner_gaps(results)

    # --- questions artifacts ---
    question_rows = [p.to_dict() for p in all_prompts]
    _write_json(out_dir / "pump_fact_qa_questions.json", question_rows)
    _write_csv(
        out_dir / "pump_fact_qa_questions.csv",
        [
            {**r, "expected_tokens": "|".join(r.get("expected_tokens", []))}
            for r in question_rows
        ],
        ["prompt_id", "question", "category", "expected_decision", "predicate",
         "subject", "obj", "expected_tokens", "notes"],
    )

    # --- outputs artifacts ---
    output_rows = [r.to_dict() for r in results]
    _write_json(out_dir / "pump_fact_qa_outputs.json", output_rows)
    _write_csv(
        out_dir / "pump_fact_qa_outputs.csv",
        [
            {
                "prompt_id": r.prompt.prompt_id,
                "question": r.prompt.question,
                "category": r.prompt.category,
                "decision": r.decision,
                "support_kind": r.support_kind,
                "supported_by_context": r.supported_by_context,
                "overlay_mode": r.overlay_mode,
                "valid": r.valid,
                "classification": r.classification,
                "is_critical_failure": r.is_critical_failure,
                "answer_text": r.answer_text,
            }
            for r in results
        ],
        ["prompt_id", "question", "category", "decision", "support_kind",
         "supported_by_context", "overlay_mode", "valid", "classification",
         "is_critical_failure", "answer_text"],
    )

    # --- adversarial-specific artifacts ---
    adversarial_prompts = prompt_groups[CATEGORY_ADVERSARIAL]
    adversarial_results = [r for r in results if r.prompt.category == CATEGORY_ADVERSARIAL]
    _write_json(
        out_dir / "pump_fact_qa_adversarial_questions.json",
        [p.to_dict() for p in adversarial_prompts],
    )
    _write_json(
        out_dir / "pump_fact_qa_adversarial_outputs.json",
        [r.to_dict() for r in adversarial_results],
    )

    # --- failures / planner gaps ---
    _write_json(out_dir / "pump_fact_qa_failures.json", failures)
    _write_json(out_dir / "pump_fact_qa_planner_gaps.json", planner_gaps)

    after = _protected_hashes()
    confirmations = {
        "network_calls": False,
        "trusted_memory_modified": before["trusted_memory"] != after["trusted_memory"],
        "accepted_overlay_modified": before["accepted_overlay"] != after["accepted_overlay"],
        "promoted_overlay_modified": before["promoted_overlay"] != after["promoted_overlay"],
        "snapshot_dry_run_overlay_modified": before["snapshot_dry_run_overlay"] != after["snapshot_dry_run_overlay"],
        "sense_memory_modified": before["sense_memory"] != after["sense_memory"],
        "auto_ingest": False,
        "auto_promote": False,
        "auto_apply": False,
        "weak_context_excluded": True,
        "overlay_mode": "pump-dry-run",
        "safe_for_general_runtime": False,
    }
    summary["confirmations"] = confirmations

    _write_json(out_dir / "pump_fact_qa_summary.json", summary)
    _write_json(
        out_dir / "pump_fact_qa_report.json",
        {
            "summary": summary,
            "failures": failures,
            "planner_gaps": planner_gaps,
            "facts": [f.to_dict() for f in facts],
            "source_artifacts": [str(precision_path), str(fallback_path)],
            "explanation": [
                "Positive prompts test that precision-filtered pump facts answer "
                "safely through the assistant surface.",
                "Adversarial prompts test that reversed relations are audited, not answered.",
                "Current/live prompts test that stale overlay data is never answered as live.",
                "Audited positive prompts are honest planner gaps, not failures.",
            ],
        },
    )

    if update_pump_summary:
        section = compact_pump_section(summary)
        _merge_pump_section(_PUMP_SUMMARY, section)
        _merge_pump_section(_PUMP_REPORT, section)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Pump Fact QA Benchmark v1")
    parser.add_argument("--no-pump-summary-update", action="store_true",
                        help="Do not merge the compact QA section into pump_summary/report.")
    args = parser.parse_args(argv)

    summary = run(update_pump_summary=not args.no_pump_summary_update)

    print("Pump Fact QA Benchmark v1")
    for key in (
        "pump_fact_qa_fact_count",
        "pump_fact_qa_relation_fact_count",
        "pump_fact_qa_definition_fact_count",
        "pump_fact_qa_positive_prompt_count",
        "pump_fact_qa_adversarial_prompt_count",
        "pump_fact_qa_current_prompt_count",
        "pump_fact_qa_total_prompt_count",
        "pump_fact_qa_positive_answer_count",
        "pump_fact_qa_positive_audit_count",
        "pump_fact_qa_positive_planner_gap_count",
        "pump_fact_qa_positive_wrong_count",
        "pump_fact_qa_positive_unsupported_count",
        "pump_fact_qa_adversarial_pass_count",
        "pump_fact_qa_adversarial_fail_count",
        "pump_fact_qa_current_safety_pass_count",
        "pump_fact_qa_current_safety_fail_count",
        "pump_fact_qa_answer_precision",
        "pump_fact_qa_supported_answer_rate",
        "pump_fact_qa_planner_gap_rate",
        "pump_fact_qa_all_critical_passed",
    ):
        print(f"  {key}: {summary.get(key)}")
    return 0 if summary.get("pump_fact_qa_all_critical_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
