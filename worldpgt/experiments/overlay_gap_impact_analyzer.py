"""Overlay gap impact analyzer for knowledge overlay dry-run v1.

Analyzes why the knowledge overlay did not increase net trusted coverage
and produces a targeted knowledge acquisition plan for the remaining
audited rows.

READ-ONLY: Does not modify any source file, memory, policy, or benchmark output.
"""

from __future__ import annotations

from typing import Any

from worldpgt.continuation.sense_memory import ExplicitSenseMemory


# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

_GAP_TYPE_TO_BLOCKER: dict[str, str] = {
    "true_unsafe": "true_unsafe_keep_audit",
    "unsupported_context": "unsupported_context",
    "missing_sense_memory": "missing_sense_memory",
    "no_safe_rewrite": "no_safe_rewrite",
    "prompt_tail_blocked": "prompt_tail_blocked",
    "missing_phrase_candidate": "missing_phrase_candidate",
}

_AUDIT_REASON_TO_BLOCKER: dict[str, str] = {
    "low_margin": "missing_positive_cue",
    "sense_tie": "missing_positive_cue",
    "missing_or_weak_cue_support": "missing_positive_cue",
    "surface_validation_failed": "missing_phrase_candidate",
    "true_unsafe": "true_unsafe_keep_audit",
    "unsupported_or_underconstrained_context": "unsupported_context",
    "missing_sense_memory": "missing_sense_memory",
    "no_safe_repaired_candidate": "no_safe_rewrite",
}

TARGETABLE_BLOCKERS: frozenset[str] = frozenset({
    "missing_positive_cue",
    "missing_phrase_candidate",
    "prompt_tail_blocked",
})

BLOCKER_TO_REQUIRED_INPUT: dict[str, str] = {
    "missing_positive_cue": "positive_cue",
    "missing_phrase_candidate": "phrase_candidate_hint",
    "prompt_tail_blocked": "prompt_tail_rule",
    "true_unsafe_keep_audit": "keep_audit_policy",
    "no_safe_rewrite": "no_action_safe_audit",
    "unsupported_context": "no_action_safe_audit",
    "missing_sense_memory": "sense_memory_entry",
    "low_margin_conflict": "positive_cue",
}

BLOCKER_TO_EFFECT: dict[str, str] = {
    "missing_positive_cue": "may_unlock_continue",
    "missing_phrase_candidate": "may_unlock_continue",
    "prompt_tail_blocked": "may_unlock_continue",
    "true_unsafe_keep_audit": "should_remain_audit",
    "no_safe_rewrite": "should_remain_audit",
    "unsupported_context": "should_remain_audit",
    "missing_sense_memory": "should_remain_audit",
    "low_margin_conflict": "should_remain_audit",
}

_BLOCKER_TO_SOURCE: dict[str, str] = {
    "missing_positive_cue": "curated_snippet",
    "missing_phrase_candidate": "manual_rule",
    "prompt_tail_blocked": "manual_rule",
    "true_unsafe_keep_audit": "keep_audit_policy",
    "no_safe_rewrite": "keep_audit_policy",
    "unsupported_context": "keep_audit_policy",
    "missing_sense_memory": "manual_rule",
    "low_margin_conflict": "keep_audit_policy",
}

_BLOCKER_TO_RECOMMENDED_STEP: dict[str, str] = {
    "missing_phrase_candidate": "expand_phrase_candidates",
    "prompt_tail_blocked": "expand_phrase_candidates",
    "missing_positive_cue": "add_positive_cues",
    "true_unsafe_keep_audit": "keep_audit",
    "no_safe_rewrite": "keep_audit",
    "unsupported_context": "keep_audit",
    "missing_sense_memory": "add_sense_memory",
    "low_margin_conflict": "keep_audit",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_builtin_cue_set() -> dict[tuple[str, str], set[str]]:
    """Return {(term, sense_id): set_of_cues} for the baseline ExplicitSenseMemory."""
    mem = ExplicitSenseMemory(include_builtin=True)
    result: dict[tuple[str, str], set[str]] = {}
    for term in mem.known_terms():
        for entry in mem.get_senses(term):
            result[(term, entry.sense_id)] = set(entry.cues)
    return result


def _parse_reasons(reasons_str: str) -> dict[str, str]:
    """Parse 'key=value | ...' reasons string into a dict."""
    parts: dict[str, str] = {}
    for segment in reasons_str.split(" | "):
        if "=" in segment:
            k, v = segment.split("=", 1)
            parts[k.strip()] = v.strip()
    return parts


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def extract_overlay_cue_matches(
    overlay_rows: list[dict],
) -> dict[str, list[tuple[str, str, str]]]:
    """Parse overlay_cue trace markers from memory_hits.

    Returns {row_id: [(cue_value, term, sense_id), ...]} for rows where
    overlay-only cues contributed scoring evidence.
    """
    result: dict[str, list[tuple[str, str, str]]] = {}
    for row in overlay_rows:
        matches: list[tuple[str, str, str]] = []
        for part in row.get("memory_hits", "").split(" | "):
            if not part.startswith("overlay_cue="):
                continue
            rest = part[len("overlay_cue="):]
            if " -> " not in rest:
                continue
            cue_val, ts = rest.split(" -> ", 1)
            if ":" in ts:
                term, sense_id = ts.split(":", 1)
                matches.append((cue_val.strip(), term.strip(), sense_id.strip()))
        if matches:
            result[row["id"]] = matches
    return result


def classify_row_changes(
    baseline_rows: list[dict],
    overlay_rows: list[dict],
) -> list[dict]:
    """Classify each row by whether and how the overlay changed its decision.

    Returns one dict per row in overlay_rows order.
    """
    base_idx = {r["id"]: r for r in baseline_rows}
    overlay_cue_matches = extract_overlay_cue_matches(overlay_rows)
    results: list[dict] = []

    for ov in overlay_rows:
        rid = ov["id"]
        b = base_idx.get(rid, {})
        b_dec = b.get("decision", "")
        o_dec = ov.get("decision", "")
        changed = (
            b_dec != o_dec
            or b.get("selected_sense", "") != ov.get("selected_sense", "")
            or b.get("continuation", "") != ov.get("continuation", "")
        )
        overlay_influenced = rid in overlay_cue_matches

        if b_dec == "audit" and o_dec == "continue":
            change_type = "audit_to_continue"
        elif b_dec == "continue" and o_dec == "audit":
            change_type = "continue_to_audit"
        elif o_dec == "audit":
            change_type = "unchanged_audit"
        elif changed:
            change_type = "changed_continue"
        else:
            change_type = "unchanged_continue"

        results.append({
            "row_id": rid,
            "term": ov.get("ambiguous_term", ""),
            "baseline_decision": b_dec,
            "overlay_decision": o_dec,
            "baseline_selected_sense": b.get("selected_sense", ""),
            "overlay_selected_sense": ov.get("selected_sense", ""),
            "changed": changed,
            "overlay_influenced": overlay_influenced,
            "change_type": change_type,
        })
    return results


def explain_changed_rows(
    baseline_rows: list[dict],
    overlay_rows: list[dict],
    audit_reasons_by_id: dict[str, dict],
) -> list[dict]:
    """Produce causal explanations for rows whose decision changed under the overlay.

    Returns one dict per changed row (both audit→continue and continue→audit).
    """
    base_idx = {r["id"]: r for r in baseline_rows}
    overlay_cue_matches = extract_overlay_cue_matches(overlay_rows)
    explanations: list[dict] = []

    for ov in overlay_rows:
        rid = ov["id"]
        b = base_idx.get(rid, {})
        b_dec = b.get("decision", "")
        o_dec = ov.get("decision", "")
        if b_dec == o_dec:
            continue

        cues = overlay_cue_matches.get(rid, [])
        if not cues:
            continue

        parts = _parse_reasons(ov.get("reasons", ""))
        top_score = parts.get("top_score", "?")
        second_score = parts.get("second_score", "?")
        margin = parts.get("margin", "?")
        exp_sense = ov.get("expected_sense", "")
        o_sense = ov.get("selected_sense", "")
        b_sense = b.get("selected_sense", "")
        cue_val, cue_term, cue_sense = cues[0]

        if b_dec == "audit" and o_dec == "continue":
            mechanism = "new_cue_broke_cue_deficit"
            explanation = (
                f"Overlay cue '{cue_val}' ({cue_term}:{cue_sense}) raised top_score to "
                f"{top_score} with margin={margin} and no conflict, satisfying continue thresholds."
            )
            correct_direction = (not exp_sense) or (o_sense == exp_sense)
        else:  # continue → audit
            mechanism = "new_cue_introduced_conflict"
            explanation = (
                f"Overlay cue '{cue_val}' ({cue_term}:{cue_sense}) introduced a competing score, "
                f"triggering conflict_detected with margin={margin} == min_margin=1.0, so policy audited."
            )
            # If baseline was selecting the wrong sense, auditing improves safety
            correct_direction = (not exp_sense) or (b_sense != exp_sense)

        explanations.append({
            "row_id": rid,
            "term": ov.get("ambiguous_term", ""),
            "change_type": f"{b_dec}_to_{o_dec}",
            "overlay_cue": cue_val,
            "overlay_cue_sense": cue_sense,
            "mechanism": mechanism,
            "explanation": explanation,
            "correct_direction": correct_direction,
            "top_score": top_score,
            "second_score": second_score,
            "margin": margin,
        })
    return explanations


def _derive_blocker(
    primary_audit_reason: str,
    curriculum_task: dict | None,
    is_newly_audited: bool,
) -> str:
    """Derive the primary blocker type for a row that is audited in the overlay run."""
    if is_newly_audited:
        return "low_margin_conflict"
    gap_type = (curriculum_task or {}).get("gap_type", "")
    if gap_type and gap_type in _GAP_TYPE_TO_BLOCKER:
        return _GAP_TYPE_TO_BLOCKER[gap_type]
    return _AUDIT_REASON_TO_BLOCKER.get(primary_audit_reason, "missing_positive_cue")


def analyze_unchanged_audits(
    overlay_rows: list[dict],
    baseline_rows: list[dict],
    audit_reasons_by_id: dict[str, dict],
    curriculum_by_id: dict[str, dict],
) -> list[dict]:
    """Analyze every row that is audited in the overlay run.

    Includes both rows that were already auditing in baseline (unchanged_audit)
    and any row newly pushed into audit by the overlay (is_newly_audited=True).

    Returns one dict per overlay-audit row.
    """
    base_idx = {r["id"]: r for r in baseline_rows}
    overlay_cue_matches = extract_overlay_cue_matches(overlay_rows)
    results: list[dict] = []

    for ov in overlay_rows:
        if ov.get("decision") != "audit":
            continue

        rid = ov["id"]
        b_dec = base_idx.get(rid, {}).get("decision", "")
        is_newly_audited = b_dec == "continue"
        overlay_influenced = rid in overlay_cue_matches

        ar = audit_reasons_by_id.get(rid, {})
        primary_reason = ar.get("primary_audit_reason", "")
        curriculum_task = curriculum_by_id.get(rid)
        blocker = _derive_blocker(primary_reason, curriculum_task, is_newly_audited)

        why_insufficient = ""
        if is_newly_audited:
            why_insufficient = "newly_audited_by_overlay"
        elif overlay_influenced:
            parts = _parse_reasons(ov.get("reasons", ""))
            margin = parts.get("margin", "0")
            conflict = "conflict_detected" in ov.get("reasons", "")
            if conflict and margin == "1":
                why_insufficient = "conflict_persists_at_min_margin"
            elif conflict:
                why_insufficient = "conflict_persists_insufficient_margin"
            else:
                why_insufficient = "cue_matched_but_score_insufficient"

        results.append({
            "row_id": rid,
            "term": ov.get("ambiguous_term", ""),
            "is_newly_audited": is_newly_audited,
            "overlay_influenced": overlay_influenced,
            "primary_audit_reason": primary_reason,
            "blocker_type": blocker,
            "curriculum_task_id": (curriculum_task or {}).get("task_id", ""),
            "why_overlay_insufficient": why_insufficient,
            "targetable": blocker in TARGETABLE_BLOCKERS,
        })
    return results


def analyze_accepted_auto_usage(
    auto_review_data: dict[str, Any],
    overlay_rows: list[dict],
    baseline_rows: list[dict],
) -> list[dict]:
    """Classify each accepted_auto item by how (or whether) it was used.

    Categories:
    - duplicate_builtin_memory: positive_cue already present in baseline memory
    - used_changed_decision:    positive_cue matched a prompt and changed the decision
    - used_no_decision_change:  positive_cue matched a prompt but decision was unchanged
    - not_observed_in_prompts:  positive_cue not found in any of the 120 prompts
    - unknown:                  non-scoring item type (typical_action, typical_location, etc.)
    """
    builtin_cues = _build_builtin_cue_set()
    overlay_cue_matches = extract_overlay_cue_matches(overlay_rows)

    # (term, sense_id, cue_lower) -> set of row_ids where it matched
    cue_to_rows: dict[tuple[str, str, str], set[str]] = {}
    for row_id, matches in overlay_cue_matches.items():
        for cue_val, term, sense_id in matches:
            key = (term, sense_id, cue_val.lower())
            cue_to_rows.setdefault(key, set()).add(row_id)

    base_idx = {r["id"]: r for r in baseline_rows}
    changed_ids: set[str] = {
        ov["id"] for ov in overlay_rows
        if base_idx.get(ov["id"], {}).get("decision") != ov.get("decision")
    }

    results: list[dict] = []
    for pr in auto_review_data.get("proposal_reviews", []):
        term = pr["term"]
        sense_id = pr["sense"]
        for item in pr.get("items", []):
            if item.get("decision") != "accepted_auto":
                continue
            itype = item.get("item_type", "")
            value = item.get("value", "")

            if itype != "positive_cue":
                usage_category = "unknown"
                matched_rows: list[str] = []
            else:
                cue_lower = value.lower()
                if cue_lower in builtin_cues.get((term, sense_id), set()):
                    usage_category = "duplicate_builtin_memory"
                    matched_rows = []
                else:
                    key = (term, sense_id, cue_lower)
                    rows_hit = cue_to_rows.get(key, set())
                    if not rows_hit:
                        usage_category = "not_observed_in_prompts"
                        matched_rows = []
                    elif rows_hit & changed_ids:
                        usage_category = "used_changed_decision"
                        matched_rows = sorted(rows_hit & changed_ids)
                    else:
                        usage_category = "used_no_decision_change"
                        matched_rows = sorted(rows_hit)

            results.append({
                "term": term,
                "sense_id": sense_id,
                "item_type": itype,
                "value": value,
                "usage_category": usage_category,
                "matched_row_ids": ",".join(matched_rows),
            })
    return results


def build_acquisition_plan(
    unchanged_audit_items: list[dict],
    curriculum_by_id: dict[str, dict],
) -> list[dict]:
    """Produce a targeted acquisition plan for all targetable overlay-audit rows.

    Each plan item specifies the knowledge needed to potentially unlock a safe
    continuation for that row.  Only targetable rows are included.
    """
    plan: list[dict] = []
    seen: set[str] = set()

    for item in unchanged_audit_items:
        if not item["targetable"]:
            continue
        rid = item["row_id"]
        curriculum_task = curriculum_by_id.get(rid)
        task_id = (curriculum_task or {}).get("task_id", "")
        # deduplicate by curriculum task when one task covers multiple rows
        dedup_key = task_id if task_id else rid
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        blocker = item["blocker_type"]
        plan.append({
            "row_id": rid,
            "term": item["term"],
            "blocker_type": blocker,
            "required_input": BLOCKER_TO_REQUIRED_INPUT.get(blocker, "positive_cue"),
            "expected_effect": BLOCKER_TO_EFFECT.get(blocker, "may_unlock_continue"),
            "source_strategy": _BLOCKER_TO_SOURCE.get(blocker, "curated_snippet"),
            "priority": "high" if task_id else "medium",
            "curriculum_task_id": task_id,
        })
    return plan


def choose_recommended_step(acquisition_plan: list[dict]) -> str:
    """Return the recommended next acquisition step based on dominant blocker type."""
    step_counts: dict[str, int] = {}
    for item in acquisition_plan:
        step = _BLOCKER_TO_RECOMMENDED_STEP.get(item["blocker_type"], "add_positive_cues")
        step_counts[step] = step_counts.get(step, 0) + 1
    if not step_counts:
        return "no_targetable_gaps"
    return max(step_counts, key=step_counts.__getitem__)


def build_analysis_csv_rows(
    unchanged_audit_items: list[dict],
    changed_explanations: list[dict],
    audit_reasons_by_id: dict[str, dict],
) -> list[dict]:
    """Build the flat CSV rows for the gap impact analysis CSV.

    Includes all overlay-audit rows plus rows resolved audit→continue by the overlay.
    """
    rows: list[dict] = []
    newly_audited_ids: set[str] = {
        item["row_id"] for item in unchanged_audit_items if item["is_newly_audited"]
    }

    for item in unchanged_audit_items:
        rid = item["row_id"]
        blocker = item["blocker_type"]
        rows.append({
            "row_id": rid,
            "term": item["term"],
            "baseline_decision": "continue" if item["is_newly_audited"] else "audit",
            "overlay_decision": "audit",
            "change_type": "continue_to_audit" if item["is_newly_audited"] else "unchanged_audit",
            "is_newly_audited": "yes" if item["is_newly_audited"] else "no",
            "overlay_influenced": "yes" if item["overlay_influenced"] else "no",
            "primary_audit_reason": item["primary_audit_reason"],
            "blocker_type": blocker,
            "required_input": BLOCKER_TO_REQUIRED_INPUT.get(blocker, ""),
            "expected_effect": BLOCKER_TO_EFFECT.get(blocker, ""),
            "curriculum_task_id": item["curriculum_task_id"],
            "why_overlay_insufficient": item["why_overlay_insufficient"],
            "targetable": "yes" if item["targetable"] else "no",
        })

    # Append rows resolved from audit → continue (not already in unchanged_audit_items)
    for exp in changed_explanations:
        if exp["change_type"] != "audit_to_continue":
            continue
        rid = exp["row_id"]
        ar = audit_reasons_by_id.get(rid, {})
        rows.append({
            "row_id": rid,
            "term": exp["term"],
            "baseline_decision": "audit",
            "overlay_decision": "continue",
            "change_type": "audit_to_continue",
            "is_newly_audited": "no",
            "overlay_influenced": "yes",
            "primary_audit_reason": ar.get("primary_audit_reason", ""),
            "blocker_type": "resolved_by_overlay",
            "required_input": "",
            "expected_effect": "unlocked_continue",
            "curriculum_task_id": "",
            "why_overlay_insufficient": "",
            "targetable": "no",
        })

    rows.sort(key=lambda r: r["row_id"])
    return rows


def run_analysis(
    baseline_rows: list[dict],
    overlay_rows: list[dict],
    overlay_summary: dict,
    audit_reasons_data: dict,
    curriculum_data: dict,
    auto_review_data: dict[str, Any],
) -> dict:
    """Run the full gap impact analysis.

    Returns a structured dict with summary, changed_rows, unchanged_audit_analysis,
    accepted_auto_usage, targeted_acquisition_plan, and analysis_csv_rows.
    """
    audit_reasons_by_id: dict[str, dict] = {
        r["row_id"]: r for r in audit_reasons_data.get("rows", [])
    }
    curriculum_by_id: dict[str, dict] = {
        t["row_id"]: t for t in curriculum_data.get("learning_tasks", [])
    }

    changed_explanations = explain_changed_rows(baseline_rows, overlay_rows, audit_reasons_by_id)
    unchanged_audit_items = analyze_unchanged_audits(
        overlay_rows, baseline_rows, audit_reasons_by_id, curriculum_by_id
    )
    auto_usage = analyze_accepted_auto_usage(auto_review_data, overlay_rows, baseline_rows)
    acquisition_plan = build_acquisition_plan(unchanged_audit_items, curriculum_by_id)
    recommended_step = choose_recommended_step(acquisition_plan)
    analysis_csv_rows = build_analysis_csv_rows(
        unchanged_audit_items, changed_explanations, audit_reasons_by_id
    )

    newly_continued = overlay_summary.get("delta", {}).get("newly_continued_rows", [])
    newly_audited = overlay_summary.get("delta", {}).get("newly_audited_rows", [])
    overlay_audit_count = sum(1 for r in overlay_rows if r.get("decision") == "audit")
    overlay_continue_count = len(overlay_rows) - overlay_audit_count

    blocker_counts: dict[str, int] = {}
    for item in unchanged_audit_items:
        b = item["blocker_type"]
        blocker_counts[b] = blocker_counts.get(b, 0) + 1

    usage_summary: dict[str, int] = {}
    for u in auto_usage:
        cat = u["usage_category"]
        usage_summary[cat] = usage_summary.get(cat, 0) + 1
    usage_summary["total"] = len(auto_usage)

    targetable_count = sum(1 for i in unchanged_audit_items if i["targetable"])
    keep_audit_count = sum(1 for i in unchanged_audit_items if not i["targetable"])

    summary: dict = {
        "total_prompts": len(overlay_rows),
        "overlay_continue_count": overlay_continue_count,
        "overlay_audit_count": overlay_audit_count,
        "newly_continued_count": len(newly_continued),
        "newly_audited_count": len(newly_audited),
        "unchanged_audit_count": overlay_audit_count - len(newly_audited),
        "net_trusted_coverage_change": len(newly_continued) - len(newly_audited),
        "risk_regression_count": len(
            overlay_summary.get("delta", {}).get("risk_regressions", [])
        ),
        "targetable_audit_count": targetable_count,
        "keep_audit_count": keep_audit_count,
        "recommended_next_step": recommended_step,
        "knowledge_gap_summary": blocker_counts,
        "accepted_auto_usage_summary": usage_summary,
    }

    return {
        "summary": summary,
        "changed_rows": changed_explanations,
        "unchanged_audit_analysis": unchanged_audit_items,
        "accepted_auto_usage": auto_usage,
        "targeted_acquisition_plan": acquisition_plan,
        "analysis_csv_rows": analysis_csv_rows,
    }
