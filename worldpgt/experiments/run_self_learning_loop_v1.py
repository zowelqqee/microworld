"""Self-Learning Loop v1 runner.

Mines learning opportunities from EXISTING Microworld QA / regression /
quarantine / ingestion / promotion artifacts and writes a proposal layer under
``worldpgt/experiments/self_learning_v1/``.

SAFETY CONTRACT (enforced by construction — this runner only reads inputs and
writes under ``self_learning_v1/``):

- No trusted memory write. accepted_knowledge_memory_v1.json untouched.
- No overlay write. accepted / promoted overlays untouched.
- No planner / validator / threshold / ingestion / overlay-builder change.
- No analyzer rule or extraction pattern is activated — proposals only.
- No neural / GPT / training / embedding / network code. nanogpt/ untouched.
- safe_for_general_runtime is always False.

Usage::

    python3 worldpgt/experiments/run_self_learning_loop_v1.py
    # or
    python3 -m worldpgt.experiments.run_self_learning_loop_v1
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

from worldpgt.self_learning.artifact_loader import ArtifactLoader
from worldpgt.self_learning.curriculum_builder import (
    build_adversarial_cases,
    build_curriculum,
)
from worldpgt.self_learning.learning_signal_miner import mine_signals
from worldpgt.self_learning.opportunity_classifier import (
    classify_signals,
    propose_analyzer_rules,
    propose_extraction_patterns,
)
from worldpgt.self_learning.proposal_report import build_report

_EXPERIMENTS = Path(__file__).resolve().parent
_DEFAULT_OUTDIR = _EXPERIMENTS / "self_learning_v1"


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


def run_self_learning_loop(
    experiments_dir: Path = _EXPERIMENTS,
    out_dir: Path = _DEFAULT_OUTDIR,
    write: bool = True,
) -> Dict[str, Any]:
    """Run the full pipeline. Returns the in-memory result objects."""
    experiments_dir = Path(experiments_dir)
    out_dir = Path(out_dir)

    loaded = ArtifactLoader(experiments_dir).load()
    signals = mine_signals(loaded)
    opportunities = classify_signals(signals)
    pattern_proposals = propose_extraction_patterns(signals)
    analyzer_rule_proposals = propose_analyzer_rules(opportunities)
    curriculum_rows = build_curriculum(opportunities)
    adversarial_cases = build_adversarial_cases(opportunities)
    report = build_report(
        loaded,
        opportunities,
        curriculum_rows,
        adversarial_cases,
        pattern_proposals,
        analyzer_rule_proposals,
    )

    summary = _build_summary(loaded, signals, opportunities, report)

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            out_dir / "learning_opportunities.json",
            [asdict(o) for o in opportunities],
        )
        _write_json(out_dir / "learning_opportunity_summary.json", summary)
        _write_csv(
            out_dir / "proposed_curriculum_v1.csv",
            [
                "id",
                "category",
                "question",
                "expected_decision",
                "expected_answer_contains",
                "expected_audit_reason",
                "source_opportunity_id",
                "activation_status",
            ],
            [asdict(r) for r in curriculum_rows],
        )
        _write_csv(
            out_dir / "proposed_adversarial_cases_v1.csv",
            [
                "id",
                "attack_category",
                "question",
                "expected_decision",
                "expected_safety_reason",
                "source_opportunity_id",
            ],
            [asdict(c) for c in adversarial_cases],
        )
        _write_json(
            out_dir / "proposed_extraction_patterns_v1.json",
            [asdict(p) for p in pattern_proposals],
        )
        _write_json(
            out_dir / "proposed_analyzer_rules_v1.json",
            [asdict(a) for a in analyzer_rule_proposals],
        )
        _write_json(out_dir / "self_learning_report_v1.json", asdict(report))

    return {
        "loaded": loaded,
        "signals": signals,
        "opportunities": opportunities,
        "pattern_proposals": pattern_proposals,
        "analyzer_rule_proposals": analyzer_rule_proposals,
        "curriculum_rows": curriculum_rows,
        "adversarial_cases": adversarial_cases,
        "report": report,
        "summary": summary,
    }


def _build_summary(loaded, signals, opportunities, report) -> Dict[str, Any]:
    from collections import Counter

    quarantine_reason_counts = {
        s.text: s.count for s in signals if s.signal_type == "quarantine_reason_count"
    }
    by_activation = Counter(o.activation_status for o in opportunities)
    return {
        "proposal_only": True,
        "safe_for_general_runtime": False,
        "opportunities_total": report.opportunities_total,
        "opportunities_by_category": report.opportunities_by_category,
        "opportunities_by_priority": report.opportunities_by_priority,
        "opportunities_by_activation_status": dict(sorted(by_activation.items())),
        "quarantine_reason_counts": dict(sorted(quarantine_reason_counts.items())),
        "signals_total": len(signals),
        "warnings": list(loaded.warnings),
        "errors": list(loaded.errors),
        "source_artifacts_read": list(loaded.source_artifacts_read),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", default=str(_EXPERIMENTS))
    parser.add_argument("--out-dir", default=str(_DEFAULT_OUTDIR))
    args = parser.parse_args(argv)

    result = run_self_learning_loop(
        experiments_dir=Path(args.experiments_dir),
        out_dir=Path(args.out_dir),
        write=True,
    )
    report = result["report"]
    print("Self-Learning Loop v1 (proposal layer only)")
    print(f"  source artifacts read : {len(report.source_artifacts_read)}")
    print(f"  warnings              : {len(report.warnings)}")
    print(f"  opportunities total   : {report.opportunities_total}")
    print(f"  by category           : {report.opportunities_by_category}")
    print(f"  by priority           : {report.opportunities_by_priority}")
    print(f"  curriculum rows       : {report.curriculum_rows_total}")
    print(f"  adversarial rows      : {report.adversarial_rows_total}")
    print(f"  pattern proposals     : {report.pattern_proposals_total}")
    print(f"  analyzer rule props   : {report.analyzer_rule_proposals_total}")
    print(f"  regression observed   : {report.regression_status_observed}")
    print(f"  trusted_memory_modified={report.trusted_memory_modified} "
          f"accepted_overlay_modified={report.accepted_overlay_modified} "
          f"promoted_overlay_modified={report.promoted_overlay_modified}")
    print(f"  safe_for_general_runtime={report.safe_for_general_runtime}")
    print(f"  artifacts written to  : {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
