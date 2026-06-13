"""Coverage-mode candidate generation for audited Microworld rows.

This module is explicitly *not* part of trusted continuation. It proposes
separate, untrusted candidates for supervised review while preserving the
normal policy decision and trusted continuation exactly as-is.

No neural models, no training, no threshold changes, and no generic trusted
fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re

from worldpgt.continuation import phrase_library
from worldpgt.continuation.prompt_tail_validator import validate_prompt_tail_compatibility
from worldpgt.continuation.realization import ENDING_NEUTRAL
from worldpgt.continuation.semantic_frame import build_semantic_frame
from worldpgt.continuation.semantic_renderer import make_frame_candidates, rank_frame_candidates
from worldpgt.continuation.subject_action_validator import validate_subject_action
from worldpgt.continuation.surface_repair import repair_surface_candidate
from worldpgt.continuation.surface_validator import validate_surface_text
from worldpgt.experiments.check_semantic_render_quality import check_rows


_TOKEN_RE = re.compile(r"[a-z0-9']+")
_UNAVAILABLE_ACTIONS = {"keep_audit", "needs_instrumentation"}
_ACTION_TO_REVIEW = {
    "auto_safe_later": "accept_as_training_example",
    "human_review": "revise",
}
_ACTION_TO_RISK = {
    "auto_safe_later": "low",
    "human_review": "medium",
}
_CANDIDATE_BAD_PATTERNS = [
    ("unfinished_to_action", re.compile(r"\bto\s+(deposit|open)\s+and\b", re.IGNORECASE)),
    ("double_and_open", re.compile(r"\band\s+open\s+and\b", re.IGNORECASE)),
    ("unfinished_before_subject", re.compile(r"\bbefore\s+the\s+player,\s+he\b", re.IGNORECASE)),
    ("unfinished_where_subject", re.compile(r"\bwhere\s+the\s+[a-z0-9']+\s+and\b", re.IGNORECASE)),
    ("unfinished_where_bare_subject", re.compile(r"\bwhere\s+reeds\s+and\b", re.IGNORECASE)),
    ("wrong_subject_metal_device", re.compile(r"\bafter\s+the\s+metal\s+device\s+brought\b", re.IGNORECASE)),
    ("repeated_pronoun_subject", re.compile(r"\bas\s+it\s+it\b", re.IGNORECASE)),
    ("as_subject_repeated_subject", re.compile(r"\bas\s+(the\s+)?[a-z0-9']+\s+(it|the)\b", re.IGNORECASE)),
    ("when_subject_repeated_subject", re.compile(r"\bwhen\s+the\s+[a-z0-9']+\s+[a-z0-9']+\s+the\b", re.IGNORECASE)),
    ("device_and_snapped", re.compile(r"\binside\s+the\s+device\s+and\s+snapped\b", re.IGNORECASE)),
    ("ground_and_lay", re.compile(r"\bon\s+the\s+ground\s+and\s+lay\b", re.IGNORECASE)),
    ("stream_and_watched", re.compile(r"\bbeside\s+the\s+stream\s+and\s+watched\b", re.IGNORECASE)),
]


@dataclass
class CoverageCandidate:
    """Candidate metadata written by coverage mode."""

    candidate_continuation: str
    candidate_full_text: str
    candidate_status: str
    candidate_risk: str
    candidate_source: str
    candidate_review_action: str
    candidate_reason: str
    candidate_selected_sense: str
    candidate_validation_status: str
    candidate_trace: list[str] = field(default_factory=list)
    candidate_learning_payload: dict = field(default_factory=dict)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _phrase_part(prompt: str, text: str) -> str:
    base = prompt.rstrip()
    if text.lower().startswith(base.lower()):
        return text[len(base):].strip()
    return text.strip()


def _unavailable(
    reason: str,
    review_action: str,
    trace: list[str] | None = None,
    target_sense: str = "",
    payload: dict | None = None,
) -> CoverageCandidate:
    return CoverageCandidate(
        candidate_continuation="",
        candidate_full_text="",
        candidate_status="unavailable",
        candidate_risk="none",
        candidate_source="none",
        candidate_review_action=review_action,
        candidate_reason=reason,
        candidate_selected_sense=target_sense,
        candidate_validation_status="not_run",
        candidate_trace=list(trace or []),
        candidate_learning_payload=payload or {},
    )


def trusted_candidate(prompt: str, trusted_full_text: str, selected_sense: str) -> CoverageCandidate:
    phrase = _phrase_part(prompt, trusted_full_text)
    return CoverageCandidate(
        candidate_continuation=phrase,
        candidate_full_text=trusted_full_text,
        candidate_status="trusted",
        candidate_risk="low",
        candidate_source="trusted_continuation",
        candidate_review_action="accept_as_training_example",
        candidate_reason="trusted policy continuation reused as candidate",
        candidate_selected_sense=selected_sense,
        candidate_validation_status="trusted",
        candidate_trace=["candidate_status=trusted"],
        candidate_learning_payload={},
    )


def _proposal_payload(prompt: str, term: str, target_sense: str, proposal: dict, frame_intent: str | None) -> dict:
    evidence = proposal.get("evidence", {})
    proposed_change = proposal.get("proposed_change", {})
    needed_memory = []
    guard = proposed_change.get("add_guard_rule")
    if guard:
        needed_memory.append(f"guard_rule:{guard}")
    phrase = proposed_change.get("add_phrase_candidate")
    if phrase:
        needed_memory.append(f"phrase_candidate:{phrase}")
    return {
        "term": term,
        "target_sense": target_sense,
        "prompt_cues": list(evidence.get("prompt_cues", [])),
        "proposed_positive_cues": list(proposed_change.get("add_positive_cues", [])),
        "conflicting_cues": list(evidence.get("conflicting_cues", [])),
        "semantic_frame": frame_intent,
        "candidate_phrase": "",
        "needed_memory": needed_memory,
    }


def _review_action(proposal: dict) -> str:
    action = proposal.get("recommended_action", "")
    if action in _UNAVAILABLE_ACTIONS:
        return "needs_memory" if action == "needs_instrumentation" else "keep_audit"
    return _ACTION_TO_REVIEW.get(action, "revise")


def _risk(proposal: dict) -> str:
    action = proposal.get("recommended_action", "")
    return _ACTION_TO_RISK.get(action, proposal.get("risk_level", "medium") or "medium")


def _passes_final_validation(prompt: str, full_text: str, term: str, selected_sense: str) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for name, pattern in _CANDIDATE_BAD_PATTERNS:
        if pattern.search(full_text):
            failures.append(f"candidate_pattern={name}")
    surface = validate_surface_text(prompt, full_text)
    if not surface.ok:
        failures.extend(f"surface_pattern={pattern}" for pattern in surface.matched_patterns)
    subject = validate_subject_action(full_text)
    if not subject.ok:
        failures.extend(subject.reasons)
    quality = check_rows(
        [
            {
                "id": "candidate",
                "prompt": prompt,
                "ambiguous_term": term,
                "selected_sense": selected_sense,
                "decision": "continue",
                "continuation": full_text,
            }
        ]
    )
    if quality["flagged_count"]:
        failures.extend(quality["flagged_rows"][0]["flags"])
    return not failures, failures


def _candidate_source(proposal: dict) -> str:
    if proposal.get("proposal_type") in {
        "cue_memory_addition",
        "guard_rule_addition",
        "phrase_candidate_addition",
        "prompt_tail_rule_addition",
    }:
        return "audit_fix_proposal"
    return "expected_sense_supervised_label"


def generate_untrusted_candidate(prompt: str, term: str, target_sense: str, proposal: dict) -> CoverageCandidate:
    """Generate one untrusted candidate for a reviewed audited row, if safe enough."""
    trace: list[str] = []
    if not term or not target_sense:
        return _unavailable("missing term or target sense", "needs_memory", target_sense=target_sense)

    action = proposal.get("recommended_action", "")
    if action in _UNAVAILABLE_ACTIONS:
        return _unavailable(
            f"proposal action {action} keeps row out of candidate generation",
            _review_action(proposal),
            [f"proposal_action={action}"],
            target_sense=target_sense,
        )

    frame = build_semantic_frame(
        prompt,
        term,
        target_sense,
        [f"coverage_mode_source={_candidate_source(proposal)}"],
    )
    payload = _proposal_payload(prompt, term, target_sense, proposal, frame.intent)
    phrases = phrase_library.get_phrases(term, target_sense, frame.connector_type)
    if not phrases:
        phrases = phrase_library.get_phrases(term, target_sense, ENDING_NEUTRAL)
    if not phrases:
        return _unavailable(
            "no explicit phrase candidates for target sense",
            "needs_memory",
            ["candidate_generation=no_phrase_candidates"],
            target_sense=target_sense,
            payload=payload,
        )

    candidates = make_frame_candidates(prompt, frame, phrases)
    trace.append(f"candidate_count={len(candidates)}")
    tried = 0
    remaining = list(candidates)
    while remaining:
        best = rank_frame_candidates(prompt, remaining)
        if best is None:
            break
        tried += 1
        trace.extend(best.reasons)
        repair = repair_surface_candidate(prompt, best.text, frame)
        trace.extend(repair.reasons)
        if not repair.safe:
            trace.append(f"surface_repair_rejected={repair.audit_reason}")
            remaining = [candidate for candidate in remaining if candidate is not best]
            continue

        tail = validate_prompt_tail_compatibility(prompt, repair.text, frame)
        if not tail.passed:
            trace.append("prompt_tail_validator=rejected")
            if tail.rule_name:
                trace.append(f"prompt_tail_rule={tail.rule_name}")
            if tail.rejection_reason:
                trace.append(f"prompt_tail_rejection={tail.rejection_reason}")
            remaining = [candidate for candidate in remaining if candidate is not best]
            continue

        final_text = tail.text
        ok, failures = _passes_final_validation(prompt, final_text, term, target_sense)
        if not ok:
            trace.extend(f"candidate_validation_failure={failure}" for failure in failures)
            remaining = [candidate for candidate in remaining if candidate is not best]
            continue

        phrase = _phrase_part(prompt, final_text)
        payload["candidate_phrase"] = phrase
        trace.append("candidate_validation=passed")
        if tail.repair_applied:
            trace.append("prompt_tail_repair=applied")
        return CoverageCandidate(
            candidate_continuation=phrase,
            candidate_full_text=final_text,
            candidate_status="untrusted",
            candidate_risk=_risk(proposal),
            candidate_source=_candidate_source(proposal),
            candidate_review_action=_review_action(proposal),
            candidate_reason="explicit untrusted candidate generated for supervised review",
            candidate_selected_sense=target_sense,
            candidate_validation_status="passed",
            candidate_trace=trace,
            candidate_learning_payload=payload,
        )

    trace.append(f"candidate_attempts={tried}")
    return _unavailable(
        "no candidate survived renderer, repair, prompt-tail, and surface validation",
        "revise",
        trace,
        target_sense=target_sense,
        payload=payload,
    )


def candidate_to_json(candidate: CoverageCandidate) -> str:
    return json.dumps(candidate.candidate_learning_payload, sort_keys=True)
