"""Preserve Pump Fact QA compact metrics across plan-only pump runs.

When run_knowledge_pump_v1.py writes a new pump_summary.json it rebuilds
from scratch, so any compact QA fields merged by run_pump_fact_qa_v1.py
are silently dropped. This module re-merges them from the canonical QA
artifact before the summary is written.

Priority order:
  1. pump_fact_qa_v1/pump_fact_qa_summary.json  (canonical QA artifact)
  2. Existing pump_summary.json QA keys          (preserved_from_previous_summary)
  3. Neither exists                              (missing)

Freshness is determined by comparing the artifact's fact_count against
the current pump's pump_answerable_fact_delta_count.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# The stable compact fields that must survive a plan-only pump run.
_QA_COMPACT_TARGET_FIELDS = [
    "pump_fact_qa_enabled",
    "pump_fact_qa_fact_count",
    "pump_fact_qa_relation_fact_count",
    "pump_fact_qa_definition_fact_count",
    "pump_fact_qa_total_prompt_count",
    "pump_fact_qa_positive_prompt_count",
    "pump_fact_qa_positive_answer_count",
    "pump_fact_qa_positive_audit_count",
    "pump_fact_qa_positive_planner_gap_count",
    "pump_fact_qa_wrong_count",
    "pump_fact_qa_wrong_answer_count",
    "pump_fact_qa_unsupported_answer_count",
    "pump_fact_qa_adversarial_fail_count",
    "pump_fact_qa_current_safety_fail_count",
    "pump_fact_qa_answer_precision",
    "pump_fact_qa_supported_answer_rate",
    "pump_fact_qa_planner_gap_rate",
    "pump_fact_qa_all_critical_passed",
]


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _pick(qa: dict, *keys: str, default: Any = None) -> Any:
    """Return the first non-None value found among keys in qa."""
    for k in keys:
        v = qa.get(k)
        if v is not None:
            return v
    return default


def _compact_from_qa_artifact(qa: dict[str, Any]) -> dict[str, Any]:
    """Extract compact QA fields from pump_fact_qa_summary.json.

    Handles both pump_fact_qa_* canonical names and the v1.8 short-name
    aliases (fact_count, wrong_answer_count, etc.) added to report.py.
    """
    return {
        "pump_fact_qa_enabled": _pick(qa, "pump_fact_qa_enabled", default=True),
        "pump_fact_qa_fact_count": _pick(qa, "pump_fact_qa_fact_count", "fact_count", default=0),
        "pump_fact_qa_relation_fact_count": _pick(
            qa, "pump_fact_qa_relation_fact_count", "relation_fact_count", default=0
        ),
        "pump_fact_qa_definition_fact_count": _pick(
            qa, "pump_fact_qa_definition_fact_count", "definition_fact_count", default=0
        ),
        "pump_fact_qa_total_prompt_count": _pick(qa, "pump_fact_qa_total_prompt_count", default=0),
        "pump_fact_qa_positive_prompt_count": _pick(
            qa, "pump_fact_qa_positive_prompt_count", "positive_prompt_count", default=0
        ),
        "pump_fact_qa_positive_answer_count": _pick(
            qa, "pump_fact_qa_positive_answer_count", "positive_answer_count", default=0
        ),
        "pump_fact_qa_positive_audit_count": _pick(
            qa, "pump_fact_qa_positive_audit_count", "positive_audit_count", default=0
        ),
        "pump_fact_qa_positive_planner_gap_count": _pick(
            qa, "pump_fact_qa_positive_planner_gap_count", "positive_planner_gap_count", default=0
        ),
        # "wrong" appears under several names across versions.
        "pump_fact_qa_wrong_count": _pick(
            qa, "pump_fact_qa_positive_wrong_count", "wrong_answer_count",
            "pump_fact_qa_wrong_count", default=0
        ),
        "pump_fact_qa_wrong_answer_count": _pick(
            qa, "pump_fact_qa_positive_wrong_count", "wrong_answer_count", default=0
        ),
        "pump_fact_qa_unsupported_answer_count": _pick(
            qa, "pump_fact_qa_positive_unsupported_count", "unsupported_answer_count",
            "pump_fact_qa_unsupported_answer_count", default=0
        ),
        "pump_fact_qa_adversarial_fail_count": _pick(
            qa, "pump_fact_qa_adversarial_fail_count", "adversarial_fail_count", default=0
        ),
        "pump_fact_qa_current_safety_fail_count": _pick(
            qa, "pump_fact_qa_current_safety_fail_count", "current_safety_fail_count", default=0
        ),
        "pump_fact_qa_answer_precision": _pick(
            qa, "pump_fact_qa_answer_precision", "answer_precision", default=0.0
        ),
        "pump_fact_qa_supported_answer_rate": _pick(
            qa, "pump_fact_qa_supported_answer_rate", "supported_answer_rate", default=0.0
        ),
        "pump_fact_qa_planner_gap_rate": _pick(
            qa, "pump_fact_qa_planner_gap_rate", "planner_gap_rate", default=0.0
        ),
        "pump_fact_qa_all_critical_passed": _pick(
            qa, "pump_fact_qa_all_critical_passed", "all_critical_passed", default=False
        ),
    }


def merge_qa_preservation(
    summary: dict[str, Any],
    out_dir: Path,
    *,
    current_fact_count: int,
) -> dict[str, Any]:
    """Merge preserved Pump Fact QA compact fields into a freshly built pump summary.

    Mutates and returns ``summary``.

    Args:
        summary: The newly built pump summary dict (will be updated in-place).
        out_dir: The pump output directory (e.g. knowledge_pump_v1/).
        current_fact_count: The current pump_answerable_fact_delta_count, used
            for the freshness check against the QA artifact.
    """
    qa_artifact_path = out_dir / "pump_fact_qa_v1" / "pump_fact_qa_summary.json"

    # --- Priority 1: canonical QA artifact ---
    if qa_artifact_path.exists():
        qa = _load_json(qa_artifact_path)
        if qa:
            compact = _compact_from_qa_artifact(qa)
            artifact_fact_count = compact.get("pump_fact_qa_fact_count", 0)

            if artifact_fact_count == current_fact_count:
                status = {
                    "pump_fact_qa_status": "current_from_qa_artifact",
                    "pump_fact_qa_stale": False,
                    "pump_fact_qa_preserved": True,
                }
            else:
                status = {
                    "pump_fact_qa_status": "stale_fact_count_mismatch",
                    "pump_fact_qa_stale": True,
                    "pump_fact_qa_preserved": True,
                    "pump_fact_qa_expected_fact_count": current_fact_count,
                    "pump_fact_qa_artifact_fact_count": artifact_fact_count,
                }

            summary.update(compact)
            summary.update(status)
            return summary

    # --- Priority 2: old pump_summary.json has QA keys ---
    existing_path = out_dir / "pump_summary.json"
    if existing_path.exists():
        old = _load_json(existing_path)
        old_qa_fields = {k: old[k] for k in _QA_COMPACT_TARGET_FIELDS if k in old}
        if old_qa_fields:
            summary.update(old_qa_fields)
            summary.update({
                "pump_fact_qa_status": "preserved_from_previous_summary",
                "pump_fact_qa_stale": True,
                "pump_fact_qa_preserved": True,
            })
            return summary

    # --- Priority 3: nothing available ---
    summary.update({
        "pump_fact_qa_enabled": False,
        "pump_fact_qa_status": "missing",
        "pump_fact_qa_preserved": False,
    })
    return summary
