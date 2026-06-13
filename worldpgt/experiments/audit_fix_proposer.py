"""Propose-only audit improvement planning for Microworld continuations.

Reads mined audit diagnostics plus the benchmark output trace and produces a
deterministic, machine-readable plan of explicit future improvements. This
module is diagnostic/planning infrastructure only: it never runs the generator,
changes memory, changes thresholds, or applies fixes.

Usage:
    python3 -m worldpgt.experiments.audit_fix_proposer \
        --audit-reasons worldpgt/experiments/microworld_continuation_v1_2_audit_reasons.json \
        --outputs worldpgt/experiments/microworld_continuation_v1_2_outputs.csv \
        --output-json worldpgt/experiments/microworld_continuation_v1_2_audit_improvement_plan.json \
        --output-csv worldpgt/experiments/microworld_continuation_v1_2_audit_improvement_plan.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from typing import Iterable

from worldpgt.continuation.sense_memory import ExplicitSenseMemory
from worldpgt.experiments.audit_reason_types import (
    LOW_MARGIN,
    MISSING_OR_WEAK_CUE_SUPPORT,
    MISSING_SENSE_MEMORY,
    NO_SAFE_REPAIRED_CANDIDATE,
    SENSE_TIE,
    SURFACE_VALIDATION_FAILED,
    TRUE_UNSAFE,
    UNSUPPORTED_OR_UNDERCONSTRAINED_CONTEXT,
)


PROPOSAL_TYPES = {
    "cue_memory_addition",
    "anti_cue_addition",
    "guard_rule_addition",
    "semantic_frame_addition",
    "phrase_candidate_addition",
    "prompt_tail_rule_addition",
    "keep_audit",
    "needs_trace_instrumentation",
}

CSV_FIELDS = [
    "proposal_id",
    "row_ids",
    "proposal_type",
    "term",
    "target_sense",
    "source_reason",
    "risk_level",
    "recommended_action",
    "supporting_cues",
    "conflicting_cues",
    "broad_cues",
    "rationale",
]

SAFETY_CHECKS = [
    "wrong_continue_count_remains_0",
    "semantic_quality_flagged_remains_0",
    "true_unsafe_rows_remain_audited",
]

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "the", "a", "an", "to", "and", "as", "it", "its", "his", "her", "he", "she",
    "they", "them", "their", "was", "were", "is", "are", "be", "been", "of", "in",
    "on", "at", "with", "for", "by", "from", "into", "below", "above",
    "down", "up", "out", "off", "this", "that", "then", "only",
    "so", "but", "or", "if", "after", "before", "until", "when", "while", "where",
    "there", "here", "him", "had", "has", "have", "did", "do", "does", "would",
    "could", "should", "will", "can", "might", "about", "back", "more", "same",
    "no", "not", "without", "because", "everyone",
}
_BROAD_CUES = {
    "water", "light", "paper", "sound", "night", "table", "people", "place",
    "time", "thing", "object", "near", "under", "over", "looked", "small", "large", "plain", "silent", "still",
    "moved", "returned", "nearby", "comment", "silence",
}
_CONCRETE_CUES = {
    "envelope", "wax", "document", "flap", "parcel", "clerk", "flippers", "fish",
    "hook", "dugout", "batter", "pitcher", "reeds", "latch", "thaw", "rafters",
    "stream", "canoe", "plate", "player", "swing", "swung", "crew", "operator",
    "nest", "wings", "lake", "stage", "band", "concert", "mechanism", "device",
    "cash", "counter", "manager", "bridge", "trail", "cliff",
}
_VERBISH_SUFFIXES = ("ed", "ing")
_MEMORY = ExplicitSenseMemory()
_EXTRA_CUE_SENSES = {
    ("bat", "animal"): {"rafters", "attic", "eaves"},
    ("bat", "sports_equipment"): {"pitcher", "dugout", "batter", "plate", "player"},
    ("bank", "river_edge"): {"stream", "reeds", "bridge"},
    ("bank", "financial_institution"): {"cash", "counter", "manager"},
    ("crane", "machine"): {"hook", "crew", "operator"},
    ("crane", "bird"): {"reeds", "wings", "lake"},
    ("seal", "animal"): {"flippers", "fish", "pier"},
    ("seal", "closure_stamp"): {"wax", "document", "envelope", "parcel", "flap"},
    ("spring", "coil"): {"latch", "mechanism", "device"},
    ("spring", "season"): {"thaw"},
}


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _split_markers(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split("|") if item.strip()]


def _marker_value(markers: Iterable[str], prefix: str) -> str | None:
    for marker in markers:
        if marker.startswith(prefix):
            return marker.split("=", 1)[1]
    return None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _trace_markers(row: dict) -> list[str]:
    markers = _split_markers(row.get("reasons", "")) + _split_markers(row.get("memory_hits", ""))
    return sorted(dict.fromkeys(markers))


def _cue_candidates(prompt: str, term: str) -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []
    for token in _tokens(prompt):
        if token == (term or "").lower() or token in _STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        candidates.append(token)
    return candidates


def _best_cues(prompt: str, term: str, limit: int = 3) -> list[str]:
    candidates = _cue_candidates(prompt, term)
    concrete = [cue for cue in candidates if cue in _CONCRETE_CUES]
    if concrete:
        return sorted(concrete)[:limit]

    def key(cue: str) -> tuple[int, int, str]:
        if cue in _CONCRETE_CUES:
            tier = 0
        elif cue in _BROAD_CUES:
            tier = 2
        elif cue.endswith(_VERBISH_SUFFIXES):
            tier = 1
        else:
            tier = 1
        return (tier, len(cue), cue)

    return sorted(candidates, key=key)[:limit]


def _cue_senses(term: str, cue: str) -> set[str]:
    senses: set[str] = set()
    for entry in _MEMORY.get_senses(term):
        if cue in entry.cues:
            senses.add(entry.sense_id)
    for (extra_term, sense_id), cues in _EXTRA_CUE_SENSES.items():
        if extra_term == term and cue in cues:
            senses.add(sense_id)
    return senses


def _cue_buckets(prompt: str, term: str, target_sense: str | None) -> dict[str, list[str]]:
    candidates = _best_cues(prompt, term)
    supporting: list[str] = []
    conflicting: list[str] = []
    broad: list[str] = []
    unknown: list[str] = []
    for cue in candidates:
        senses = _cue_senses(term, cue)
        supports_target = bool(target_sense and target_sense in senses)
        supports_other = bool(target_sense and any(sense != target_sense for sense in senses))
        if cue in _BROAD_CUES:
            broad.append(cue)
        elif supports_other:
            conflicting.append(cue)
        elif supports_target:
            supporting.append(cue)
        else:
            unknown.append(cue)
    return {
        "prompt_cues": candidates,
        "supporting_cues": supporting,
        "conflicting_cues": conflicting,
        "broad_cues": broad,
        "unknown_cues": unknown,
    }


def _risk_for_buckets(buckets: dict[str, list[str]], source_reason: str) -> str:
    supporting = buckets["supporting_cues"]
    conflicting = buckets["conflicting_cues"]
    broad = buckets["broad_cues"]
    unknown = buckets["unknown_cues"]
    if conflicting:
        return "medium"
    if broad:
        return "high"
    if not supporting and unknown:
        return "high"
    if source_reason in {SENSE_TIE, LOW_MARGIN}:
        return "medium"
    if supporting and not unknown and all(cue in _CONCRETE_CUES for cue in supporting):
        return "low"
    return "medium"


def _proposal_id(row_ids: list[str], proposal_type: str, source_reason: str) -> str:
    seed = "|".join(row_ids + [proposal_type, source_reason])
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
    return f"p-{row_ids[0]}-{proposal_type}-{digest}"


def _empty_change() -> dict:
    return {
        "add_positive_cues": [],
        "add_anti_cues": [],
        "add_guard_rule": None,
        "add_semantic_frame": None,
        "add_phrase_candidate": None,
        "add_prompt_tail_rule": None,
    }


def _evidence(diag: dict, output_row: dict, buckets: dict[str, list[str]]) -> dict:
    markers = _trace_markers(output_row)
    return {
        "prompt_cues": buckets["prompt_cues"],
        "supporting_cues": buckets["supporting_cues"],
        "conflicting_cues": buckets["conflicting_cues"],
        "broad_cues": buckets["broad_cues"],
        "trace_markers": markers,
        "top_score": _parse_float(_marker_value(markers, "top_score=")),
        "second_score": _parse_float(_marker_value(markers, "second_score=")),
        "margin": _parse_float(_marker_value(markers, "margin=")),
    }


def _base_proposal(
    diag: dict,
    output_row: dict,
    proposal_type: str,
    risk_level: str,
    recommended_action: str,
    rationale: str,
    buckets: dict[str, list[str]],
    proposed_change: dict | None = None,
) -> dict:
    row_id = diag.get("row_id", "")
    source_reason = diag.get("primary_audit_reason", "")
    return {
        "proposal_id": _proposal_id([row_id], proposal_type, source_reason),
        "row_ids": [row_id],
        "proposal_type": proposal_type,
        "term": diag.get("term") or output_row.get("ambiguous_term", ""),
        "target_sense": diag.get("expected_sense") or output_row.get("expected_sense", "") or None,
        "source_reason": source_reason,
        "current_decision": output_row.get("decision", "audit"),
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "rationale": rationale,
        "evidence": _evidence(diag, output_row, buckets),
        "proposed_change": proposed_change or _empty_change(),
        "safety_checks_required": list(SAFETY_CHECKS),
        "expected_effect": {
            "may_continue_rows": [] if recommended_action == "keep_audit" else [row_id],
            "risk_of_false_positive": risk_level,
        },
    }


def _keep_audit(diag: dict, output_row: dict, reason: str) -> dict:
    target_sense = diag.get("expected_sense") or output_row.get("expected_sense", "") or None
    buckets = _cue_buckets(
        output_row.get("prompt", ""),
        output_row.get("ambiguous_term", ""),
        target_sense,
    )
    risk = "medium" if buckets["conflicting_cues"] else "high" if buckets["broad_cues"] else "low"
    return _base_proposal(
        diag,
        output_row,
        "keep_audit",
        risk,
        "keep_audit",
        reason,
        buckets,
    )


def _needs_instrumentation(diag: dict, output_row: dict, reason: str) -> dict:
    target_sense = diag.get("expected_sense") or output_row.get("expected_sense", "") or None
    buckets = _cue_buckets(
        output_row.get("prompt", ""),
        output_row.get("ambiguous_term", ""),
        target_sense,
    )
    return _base_proposal(
        diag,
        output_row,
        "needs_trace_instrumentation",
        "medium",
        "needs_instrumentation",
        reason,
        buckets,
    )


def _cue_memory_proposal(diag: dict, output_row: dict, source_reason: str) -> dict:
    term = diag.get("term") or output_row.get("ambiguous_term", "")
    target_sense = diag.get("expected_sense") or output_row.get("expected_sense", "") or None
    buckets = _cue_buckets(output_row.get("prompt", ""), term, target_sense)
    if buckets["conflicting_cues"]:
        return _guard_rule_proposal(diag, output_row, source_reason)

    risk = _risk_for_buckets(buckets, source_reason)
    action = (
        "auto_safe_later"
        if risk == "low" and source_reason == MISSING_OR_WEAK_CUE_SUPPORT
        else "human_review"
    )
    change = _empty_change()
    change["add_positive_cues"] = buckets["supporting_cues"]
    rationale = (
        "Concrete prompt cues may support the expected sense through explicit cue memory."
        if risk == "low"
        else "Cue evidence is plausible but mixed or broad enough to require human review."
    )
    return _base_proposal(
        diag,
        output_row,
        "cue_memory_addition",
        risk,
        action,
        rationale,
        buckets,
        change,
    )


def _guard_rule_proposal(diag: dict, output_row: dict, source_reason: str) -> dict:
    term = diag.get("term") or output_row.get("ambiguous_term", "")
    target_sense = diag.get("expected_sense") or output_row.get("expected_sense", "") or None
    buckets = _cue_buckets(output_row.get("prompt", ""), term, target_sense)
    change = _empty_change()
    change["add_positive_cues"] = buckets["supporting_cues"]
    conflicting = buckets["conflicting_cues"] or ["conflicting"]
    supporting = buckets["supporting_cues"] or ["supporting"]
    change["add_guard_rule"] = (
        f"term={term}: prefer {target_sense or 'target_sense'} only when "
        f"{'/'.join(supporting)} cues dominate {'/'.join(conflicting)} cues"
    )
    return _base_proposal(
        diag,
        output_row,
        "guard_rule_addition",
        "medium",
        "human_review",
        "Competing evidence needs a narrow term-specific guard; do not auto-apply conflict rules.",
        buckets,
        change,
    )


def _surface_proposal(diag: dict, output_row: dict) -> dict:
    markers = _trace_markers(output_row)
    prompt = output_row.get("prompt", "")
    target_sense = diag.get("expected_sense") or output_row.get("expected_sense", "") or None
    buckets = _cue_buckets(prompt, output_row.get("ambiguous_term", ""), target_sense)
    if "audit_reason=prompt_tail_incompatible" in markers:
        return _keep_audit(
            diag,
            output_row,
            "Prompt-tail validation rejected the candidate; repairing this row would require unsupported action/entity invention.",
        )

    change = _empty_change()
    proposal_type = "phrase_candidate_addition"
    if any("malformed_until_subject" in marker for marker in markers):
        change["add_phrase_candidate"] = "explicit safe predicate for unfinished until-subject tail"
    elif "before" in _tokens(prompt) or "until" in _tokens(prompt):
        change["add_phrase_candidate"] = "explicit connector-compatible phrase candidate for the selected sense"
    else:
        change["add_prompt_tail_rule"] = "narrow grammar rule for the observed prompt-tail pattern"
        proposal_type = "prompt_tail_rule_addition"

    return _base_proposal(
        diag,
        output_row,
        proposal_type,
        "medium",
        "human_review",
        "A safe explicit phrase or narrow prompt-tail rule may help, but broad grammar repair is unsafe.",
        buckets,
        change,
    )


def propose_for_row(diag: dict, output_row: dict) -> dict:
    reason = diag.get("primary_audit_reason", "")
    if reason == TRUE_UNSAFE:
        return _keep_audit(diag, output_row, "Contradictory or negated evidence makes continuation unsafe.")
    if reason == UNSUPPORTED_OR_UNDERCONSTRAINED_CONTEXT:
        return _keep_audit(diag, output_row, "Context is intentionally underconstrained; continuation would guess.")
    if reason == NO_SAFE_REPAIRED_CANDIDATE:
        return _keep_audit(diag, output_row, "No safe repaired candidate exists; keep audited unless an explicit tested rewrite is added.")
    if reason == MISSING_SENSE_MEMORY:
        return _needs_instrumentation(
            diag,
            output_row,
            "No known ambiguous term/sense was available; future sense-memory work needs human review and clearer trace.",
        )
    if reason == MISSING_OR_WEAK_CUE_SUPPORT:
        return _cue_memory_proposal(diag, output_row, reason)
    if reason == SENSE_TIE:
        return _guard_rule_proposal(diag, output_row, reason)
    if reason == LOW_MARGIN:
        markers = _trace_markers(output_row)
        if any("conflict_detected" == marker for marker in markers):
            return _guard_rule_proposal(diag, output_row, reason)
        return _cue_memory_proposal(diag, output_row, reason)
    if reason == SURFACE_VALIDATION_FAILED:
        return _surface_proposal(diag, output_row)
    return _needs_instrumentation(diag, output_row, "Trace markers are insufficient for a grounded proposal.")


def _counts(items: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(items).items()))


def build_plan(audit_report: dict, output_rows: Iterable[dict]) -> dict:
    outputs_by_id = {row.get("id", ""): row for row in output_rows}
    diagnostics = list(audit_report.get("rows", []))
    proposals = [
        propose_for_row(diag, outputs_by_id.get(diag.get("row_id", ""), {}))
        for diag in diagnostics
    ]

    actions = [proposal["recommended_action"] for proposal in proposals]
    summary = {
        "total_audits": audit_report.get("summary", {}).get("audit_count", len(diagnostics)),
        "proposals_total": len(proposals),
        "auto_safe_proposals": actions.count("auto_safe_later"),
        "review_required": actions.count("human_review"),
        "keep_audit": actions.count("keep_audit"),
        "needs_instrumentation": actions.count("needs_instrumentation"),
        "by_proposal_type": _counts(p["proposal_type"] for p in proposals),
        "by_risk_level": _counts(p["risk_level"] for p in proposals),
        "by_action": _counts(actions),
    }
    return {
        "summary": summary,
        "policy": {
            "mode": "propose_only",
            "auto_apply": False,
            "generation_behavior_changed": False,
            "thresholds_changed": False,
            "validators_changed": False,
        },
        "proposals": proposals,
    }


def read_audit_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_output_rows(path: str) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(plan: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(plan: dict, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for proposal in plan["proposals"]:
            writer.writerow(
                {
                    "proposal_id": proposal["proposal_id"],
                    "row_ids": ";".join(proposal["row_ids"]),
                    "proposal_type": proposal["proposal_type"],
                    "term": proposal["term"] or "",
                    "target_sense": proposal["target_sense"] or "",
                    "source_reason": proposal["source_reason"],
                    "risk_level": proposal["risk_level"],
                    "recommended_action": proposal["recommended_action"],
                    "supporting_cues": ";".join(proposal["evidence"]["supporting_cues"]),
                    "conflicting_cues": ";".join(proposal["evidence"]["conflicting_cues"]),
                    "broad_cues": ";".join(proposal["evidence"]["broad_cues"]),
                    "rationale": proposal["rationale"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a propose-only audit improvement plan.")
    parser.add_argument("--audit-reasons", required=True, help="Audit reason JSON from audit_reason_miner")
    parser.add_argument("--outputs", required=True, help="Microworld continuation output CSV")
    parser.add_argument("--output-json", required=True, help="Improvement plan JSON path")
    parser.add_argument("--output-csv", required=False, help="Optional improvement plan CSV path")
    args = parser.parse_args()

    plan = build_plan(read_audit_report(args.audit_reasons), read_output_rows(args.outputs))
    write_json(plan, args.output_json)
    if args.output_csv:
        write_csv(plan, args.output_csv)
    print(json.dumps(plan["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
