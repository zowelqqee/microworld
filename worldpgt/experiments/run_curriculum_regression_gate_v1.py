"""Curriculum Regression Gate v1 runner.

Replays the Self-Learning Loop v1 proposed curriculum + adversarial cases
through the existing entity-QA / cross-page-QA pipelines (over the accepted wiki
overlay) and confirms that dangerous cases still audit/reject and that
missing-knowledge rows audit safely.

SAFETY CONTRACT (enforced by construction — this runner only reads inputs and
the accepted overlay, and writes under ``curriculum_regression_v1/``):

- No trusted memory write. accepted_knowledge_memory_v1.json untouched.
- No overlay write. accepted / promoted overlays untouched (read-only base).
- No planner / validator / threshold / ingestion / overlay-builder change.
- Proposed analyzer rules and extraction patterns are NOT activated.
- No neural / GPT / training / embedding / network code. nanogpt/ untouched.
- safe_for_general_runtime is always False.

Usage::

    python3 worldpgt/experiments/run_curriculum_regression_gate_v1.py
    # or
    python3 -m worldpgt.experiments.run_curriculum_regression_gate_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow running directly as a script.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.curriculum_regression.curriculum_case_loader import CurriculumCaseLoader
from worldpgt.curriculum_regression.curriculum_case_runner import CaseRunner
from worldpgt.curriculum_regression.curriculum_report import build_report, build_summary
from worldpgt.curriculum_regression.curriculum_result_validator import (
    validate_adversarial,
    validate_curriculum,
)

_EXPERIMENTS = Path(__file__).resolve().parent
_DEFAULT_SELF_LEARNING_DIR = _EXPERIMENTS / "self_learning_v1"
_DEFAULT_OUTDIR = _EXPERIMENTS / "curriculum_regression_v1"
_DEFAULT_OVERLAY = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"

_CURRICULUM_FIELDS = [
    "id",
    "category",
    "question",
    "expected_decision",
    "observed_decision",
    "observed_intent",
    "observed_answer",
    "expected_answer_contains",
    "expected_audit_reason",
    "passed",
    "failure_reason",
    "route",
    "source_opportunity_id",
]

_ADVERSARIAL_FIELDS = [
    "id",
    "attack_category",
    "question",
    "expected_decision",
    "observed_decision",
    "observed_answer",
    "expected_safety_reason",
    "passed",
    "failure_reason",
    "route",
    "source_opportunity_id",
]


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
        fh.write("\n")


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_curriculum_regression_gate(
    self_learning_dir: Path = _DEFAULT_SELF_LEARNING_DIR,
    out_dir: Path = _DEFAULT_OUTDIR,
    overlay_json_path: Path = _DEFAULT_OVERLAY,
    write: bool = True,
) -> Dict[str, Any]:
    self_learning_dir = Path(self_learning_dir)
    out_dir = Path(out_dir)

    loaded = CurriculumCaseLoader(self_learning_dir).load()
    runner = CaseRunner(str(overlay_json_path))

    curriculum_results = []
    curriculum_rows = []
    for case in loaded.curriculum_cases:
        observed = runner.run(case.question)
        result = validate_curriculum(case, observed)
        curriculum_results.append(result)
        curriculum_rows.append(
            {
                "id": case.id,
                "category": case.category,
                "question": case.question,
                "expected_decision": case.expected_decision,
                "observed_decision": observed.decision,
                "observed_intent": observed.intent,
                "observed_answer": observed.answer,
                "expected_answer_contains": case.expected_answer_contains,
                "expected_audit_reason": case.expected_audit_reason,
                "passed": result.passed,
                "failure_reason": result.failure_reason,
                "route": observed.route,
                "source_opportunity_id": case.source_opportunity_id,
            }
        )

    adversarial_results = []
    adversarial_rows = []
    for case in loaded.adversarial_cases:
        observed = runner.run(case.question)
        result = validate_adversarial(case, observed)
        adversarial_results.append(result)
        adversarial_rows.append(
            {
                "id": case.id,
                "attack_category": case.attack_category,
                "question": case.question,
                "expected_decision": case.expected_decision,
                "observed_decision": observed.decision,
                "observed_answer": observed.answer,
                "expected_safety_reason": case.expected_safety_reason,
                "passed": result.passed,
                "failure_reason": result.failure_reason,
                "route": observed.route,
                "source_opportunity_id": case.source_opportunity_id,
            }
        )

    summary = build_summary(loaded, curriculum_results, adversarial_results)
    report = build_report(
        summary,
        loaded,
        curriculum_results,
        adversarial_results,
        loaded.source_artifacts_read,
    )

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(
            out_dir / "curriculum_regression_outputs.csv",
            _CURRICULUM_FIELDS,
            curriculum_rows,
        )
        _write_csv(
            out_dir / "curriculum_adversarial_outputs.csv",
            _ADVERSARIAL_FIELDS,
            adversarial_rows,
        )
        _write_json(out_dir / "curriculum_regression_summary.json", summary)
        _write_json(out_dir / "curriculum_regression_report.json", report)

    return {
        "loaded": loaded,
        "curriculum_results": curriculum_results,
        "adversarial_results": adversarial_results,
        "summary": summary,
        "report": report,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-learning-dir", default=str(_DEFAULT_SELF_LEARNING_DIR))
    parser.add_argument("--out-dir", default=str(_DEFAULT_OUTDIR))
    parser.add_argument("--overlay-json", default=str(_DEFAULT_OVERLAY))
    args = parser.parse_args(argv)

    result = run_curriculum_regression_gate(
        self_learning_dir=Path(args.self_learning_dir),
        out_dir=Path(args.out_dir),
        overlay_json_path=Path(args.overlay_json),
        write=True,
    )
    s = result["summary"]
    print("Curriculum Regression Gate v1")
    print(f"  curriculum   : {s['curriculum_passed']}/{s['curriculum_total']} passed")
    print(f"  adversarial  : {s['adversarial_passed']}/{s['adversarial_total']} passed")
    print(f"  decisions    : answer={s['answer_count']} audit={s['audit_count']} "
          f"reject={s['reject_count']}")
    print(f"  unsafe_answer_count          : {s['unsafe_answer_count']}")
    print(f"  missing_knowledge_audit_count: {s['missing_knowledge_audit_count']}")
    print(f"  guard passes : weak_link={s['weak_link_safety_pass_count']} "
          f"inversion={s['inversion_guard_pass_count']} "
          f"current={s['current_claim_guard_pass_count']} "
          f"volatile={s['volatile_guard_pass_count']} "
          f"private={s['private_guard_pass_count']} "
          f"universal={s['unsupported_universal_guard_pass_count']}")
    print(f"  all_passed   : {s['all_passed']}")
    print(f"  rules_activated={s['rules_activated']} "
          f"patterns_activated={s['patterns_activated']}")
    print(f"  trusted_memory_modified={s['trusted_memory_modified']} "
          f"accepted_overlay_modified={s['accepted_overlay_modified']} "
          f"promoted_overlay_modified={s['promoted_overlay_modified']}")
    print(f"  safe_for_general_runtime={s['safe_for_general_runtime']}")
    print(f"  artifacts written to: {args.out_dir}")
    return 0 if s["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
