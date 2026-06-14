"""Report assembly for Wikipedia Self-Ingestion v1.

Builds the deterministic self-ingestion report dict. ``safe_for_general_runtime``
is always False. No ML, no network.
"""

from __future__ import annotations


def build_report(
    *,
    sources_total: int,
    documents_read: int,
    read_errors: int,
    candidates_total: int,
    new_candidates: int,
    duplicate_existing: int,
    conflicts: int,
    quarantined_total: int,
    rejected_total: int,
    overlay_delta_items: int,
    dry_run_overlay_items: int,
    qa_regression_status: dict,
    quarantine_reasons: dict,
    high_risk_in_delta: bool,
    notes: list[str],
) -> dict:
    all_qa_green = bool(qa_regression_status) and all(
        v.get("status") == "PASS" for v in qa_regression_status.values()
    )
    safe_to_apply = all_qa_green and not high_risk_in_delta and conflicts >= 0

    return {
        "sources_total": sources_total,
        "documents_read": documents_read,
        "read_errors": read_errors,
        "candidates_total": candidates_total,
        "new_candidates": new_candidates,
        "duplicate_existing": duplicate_existing,
        "conflicts": conflicts,
        "quarantined_total": quarantined_total,
        "rejected_total": rejected_total,
        "quarantine_reasons": quarantine_reasons,
        "overlay_delta_items": overlay_delta_items,
        "dry_run_overlay_items": dry_run_overlay_items,
        "qa_regression_status": qa_regression_status,
        "all_qa_green": all_qa_green,
        "high_risk_in_delta": high_risk_in_delta,
        "safe_to_apply_overlay_delta": bool(safe_to_apply),
        "safe_for_general_runtime": False,
        "notes": notes,
    }
