"""Context-Pack QA Consistency Gate v1 runner.

Dry-run gate that compares EXISTING QA decisions against the read-only Working
Context Pack for the same question, writing consistency artifacts under
``worldpgt/experiments/context_consistency_v1/``.

SAFETY CONTRACT (enforced by construction — read-only over QA outputs and the
overlay; writes only under ``context_consistency_v1/``):

- QA behavior is NOT changed; the context pack is NOT wired into planners.
- No trusted memory / accepted overlay / promoted overlay writes.
- No planner / validator / threshold / ingestion / overlay-builder change.
- No analyzer-rule / extraction-pattern activation. No auto-ingest/promote/apply.
- No neural / GPT / training / embedding / network code. nanogpt/ untouched.
- safe_for_general_runtime is always False.

Usage::

    python3 worldpgt/experiments/run_context_pack_qa_consistency_v1.py
    python3 worldpgt/experiments/run_context_pack_qa_consistency_v1.py --overlay-mode promoted
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

# Allow running directly as a script.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.context_consistency.context_consistency_checker import check_row
from worldpgt.context_consistency.context_consistency_report import (
    build_report,
    build_summary,
)
from worldpgt.context_consistency.qa_result_loader import load_qa_results
from worldpgt.context_pack.context_pack_builder import (
    ContextPackBuilder,
    load_knowledge_requests,
)
from worldpgt.context_pack.context_pack_validator import validate_pack
from worldpgt.context_pack.types import OVERLAY_ACCEPTED, OVERLAY_PROMOTED

_EXPERIMENTS = Path(__file__).resolve().parent
_DEFAULT_OUTDIR = _EXPERIMENTS / "context_consistency_v1"
_ACCEPTED_OVERLAY = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_PROMOTED_OVERLAY = (
    _EXPERIMENTS / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json"
)
_KNOWLEDGE_REQUESTS = (
    _EXPERIMENTS / "knowledge_requests_v1" / "candidate_knowledge_requests.json"
)

_CSV_FIELDS = [
    "id",
    "source_suite",
    "question",
    "qa_decision",
    "qa_intent",
    "qa_answer",
    "context_safe_for_answering",
    "candidate_paths_count",
    "blocked_paths_count",
    "weak_links_count",
    "source_facts_count",
    "missing_hints_count",
    "validation_passed",
    "consistency_status",
    "failure_reason",
    "risk_flags",
]


def _overlay_path(mode: str) -> Path:
    return _PROMOTED_OVERLAY if mode == OVERLAY_PROMOTED else _ACCEPTED_OVERLAY


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
        fh.write("\n")


def run_consistency_gate(
    experiments_dir: Path = _EXPERIMENTS,
    overlay_mode: str = OVERLAY_ACCEPTED,
    out_dir: Path = _DEFAULT_OUTDIR,
    write: bool = True,
) -> Dict[str, Any]:
    experiments_dir = Path(experiments_dir)
    out_dir = Path(out_dir)

    loaded = load_qa_results(experiments_dir)
    knowledge_requests = load_knowledge_requests(_KNOWLEDGE_REQUESTS)
    builder = ContextPackBuilder(
        str(_overlay_path(overlay_mode)),
        overlay_mode=overlay_mode,
        knowledge_requests=knowledge_requests,
    )

    results = []
    for qa in loaded.rows:
        pack = builder.build(qa.question)
        validation = validate_pack(pack)
        results.append(check_row(qa, pack, validation))

    summary = build_summary(
        results, qa_rows_total=len(loaded.rows), skipped_rows_total=0
    )
    report = build_report(
        summary,
        results,
        loaded.source_artifacts_read,
        loaded.warnings,
        loaded.errors,
    )

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            out_dir / "context_consistency_outputs.json",
            [asdict(r) for r in results],
        )
        with (out_dir / "context_consistency_outputs.csv").open(
            "w", encoding="utf-8", newline=""
        ) as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "id": r.id,
                        "source_suite": r.source_suite,
                        "question": r.question,
                        "qa_decision": r.qa_decision,
                        "qa_intent": r.qa_intent,
                        "qa_answer": r.qa_answer,
                        "context_safe_for_answering": r.context_safe_for_answering,
                        "candidate_paths_count": r.context_candidate_paths_count,
                        "blocked_paths_count": r.context_blocked_paths_count,
                        "weak_links_count": r.context_weak_links_count,
                        "source_facts_count": r.context_source_facts_count,
                        "missing_hints_count": r.context_missing_hints_count,
                        "validation_passed": r.validation_passed,
                        "consistency_status": r.consistency_status,
                        "failure_reason": r.failure_reason,
                        "risk_flags": "|".join(r.risk_flags),
                    }
                )
        _write_json(out_dir / "context_consistency_summary.json", summary)
        _write_json(out_dir / "context_consistency_report.json", report)

    return {"loaded": loaded, "results": results, "summary": summary, "report": report}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlay-mode",
        choices=[OVERLAY_ACCEPTED, OVERLAY_PROMOTED],
        default=OVERLAY_ACCEPTED,
    )
    parser.add_argument("--experiments-dir", default=str(_EXPERIMENTS))
    parser.add_argument("--out-dir", default=str(_DEFAULT_OUTDIR))
    args = parser.parse_args(argv)

    result = run_consistency_gate(
        experiments_dir=Path(args.experiments_dir),
        overlay_mode=args.overlay_mode,
        out_dir=Path(args.out_dir),
    )
    s = result["summary"]
    print("Context-Pack QA Consistency Gate v1 (dry-run)")
    print(f"  qa_rows_total                : {s['qa_rows_total']}")
    print(f"  checked_rows_total           : {s['checked_rows_total']}")
    print(f"  skipped_rows_total           : {s['skipped_rows_total']}")
    print(f"  answer_rows / audit_rows     : {s['answer_rows']} / {s['audit_rows']}")
    print(f"  answer_supported_by_context  : {s['answer_supported_by_context']}")
    print(f"  audit_supported_by_context   : {s['audit_supported_by_context']}")
    print(f"  safe_policy_answer_supported : {s['safe_policy_answer_supported']}")
    print(f"  answer_without_context_support: {s['answer_without_context_support']}")
    print(f"  audit_despite_safe_context   : {s['audit_despite_safe_context']}")
    print(f"  false supports               : weak={s['weak_link_false_support_count']} "
          f"volatile={s['volatile_false_stable_count']} "
          f"current={s['current_live_false_support_count']} "
          f"inversion={s['inversion_false_support_count']} "
          f"private={s['private_false_support_count']} "
          f"universal={s['unsupported_universal_false_support_count']}")
    print(f"  context_validation_failed    : {s['context_validation_failed_count']}")
    print(f"  passed/warning/failed        : {s['consistency_passed_count']}/"
          f"{s['consistency_warning_count']}/{s['consistency_failed_count']}")
    print(f"  all_critical_passed          : {s['all_critical_passed']}")
    print(f"  safe_for_general_runtime={s['safe_for_general_runtime']} "
          f"runtime_behavior_modified={s['runtime_behavior_modified']} "
          f"network_calls={s['network_calls']}")
    print(f"  artifacts written to         : {args.out_dir}")
    return 0 if s["all_critical_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
