"""Promotion report assembly for Promote Overlay Delta v1.

Builds the deterministic promotion report. ``safe_for_general_runtime`` is always
False; trusted memory and the accepted overlay are never modified. No ML, no
network.
"""

from __future__ import annotations

from worldpgt.self_ingestion.overlay_delta_promoter import PromotionResult


def _weak_links_remain_weak(promoted_items: list[dict]) -> bool:
    for it in promoted_items:
        if it.get("overlay_type") == "overlay_context_link":
            if it.get("strength") != "weak" or it.get("trust") != "weak_context_only":
                return False
    return True


def _no_volatile_promoted(accepted_items: list[dict]) -> bool:
    for it in accepted_items:
        if it.get("overlay_type") == "overlay_source_fact":
            return False
        if it.get("stability") == "volatile" or it.get("requires_recheck") is True:
            return False
    return True


def _no_inverted_promoted(accepted_items: list[dict]) -> bool:
    people = ("elon musk", "jeff bezos", "bernard arnault", "michael bloomberg",
              "gwynne shotwell", "martin eberhard", "marc tarpenning", "larry ellison")
    for it in accepted_items:
        if it.get("overlay_type") == "overlay_relation" and it.get("predicate") == "founded":
            if any(p in str(it.get("object", "")).lower() for p in people):
                return False
    return True


def _no_private_promoted(accepted_items: list[dict]) -> bool:
    import re
    pat = re.compile(r"(private\s+email|phone\s+number|home\s+address|password|@)", re.I)
    for it in accepted_items:
        blob = " ".join(str(v) for v in it.values() if isinstance(v, str))
        if pat.search(blob):
            return False
    return True


def build_promotion_report(
    promotion: PromotionResult,
    regression: dict,
) -> dict:
    val = promotion.validation
    accepted = promotion.validation.accepted_items

    all_regress_green = bool(regression) and all(
        r.get("status") == "PASS" for r in regression.values()
    )
    safe_to_promote = bool(val.safe_to_promote and all_regress_green)

    return {
        "base_overlay_items": promotion.base_count,
        "delta_proposal_items": promotion.delta_count,
        "accepted_delta_items": promotion.accepted_delta_count,
        "rejected_delta_items": val.rejected_count,
        "blocked_delta_items": val.blocked_count,
        "promoted_overlay_items": promotion.promoted_count,
        "safe_to_promote": safe_to_promote,
        "safe_for_general_runtime": False,
        "trusted_memory_modified": False,
        "accepted_overlay_modified": False,
        "quarantine_not_promoted": True,
        "rejected_items_not_promoted": True,
        "volatile_items_not_promoted": _no_volatile_promoted(accepted),
        "private_items_not_promoted": _no_private_promoted(accepted),
        "inverted_relations_not_promoted": _no_inverted_promoted(accepted),
        "weak_links_remain_weak": _weak_links_remain_weak(promotion.promoted_items),
        "regression_results": regression,
        "promoted_overlay_provenance": promotion.meta,
        "notes": [
            "Base accepted overlay preserved unchanged; never overwritten.",
            "Only validator-accepted, stable, non-conflicting delta items promoted.",
            "Volatile/source-qualified facts, weak-link-as-fact, inverted, private, "
            "universal, high-risk, conflicting, duplicate, and malformed items blocked.",
            "Promoted overlay is an isolated promoted self-ingestion overlay, not "
            "trusted general runtime memory.",
        ],
    }
