"""Deterministic taxonomy + classifier for *why* an audited row did not continue.

This is diagnostic only. It reads the trace fields the engine already emits
(``reasons``, ``memory_hits``, ``confidence``, ``selected_sense``,
``expected_sense``) and maps each audited row to one *primary* missing-capability
reason plus any *secondary* contributing markers. It does not change generation,
policy, validators, or the renderer.

Classification is rule-based and grounded in explicit trace markers. It never
infers a confident reason from prompt text alone; when traces are insufficient it
returns :data:`UNKNOWN_REASON` rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# --- Reason taxonomy --------------------------------------------------------

LOW_MARGIN = "low_margin"
SENSE_TIE = "sense_tie"
MISSING_OR_WEAK_CUE_SUPPORT = "missing_or_weak_cue_support"
MISSING_SENSE_MEMORY = "missing_sense_memory"
MISSING_SEMANTIC_FRAME = "missing_semantic_frame"
MISSING_PHRASE_CANDIDATE = "missing_phrase_candidate"
SURFACE_VALIDATION_FAILED = "surface_validation_failed"
SUBJECT_ACTION_INVALID = "subject_action_invalid"
NO_SAFE_REPAIRED_CANDIDATE = "no_safe_repaired_candidate"
UNSUPPORTED_OR_UNDERCONSTRAINED_CONTEXT = "unsupported_or_underconstrained_context"
TRUE_UNSAFE = "true_unsafe"
UNKNOWN_REASON = "unknown_reason"

ALL_REASONS = [
    LOW_MARGIN,
    SENSE_TIE,
    MISSING_OR_WEAK_CUE_SUPPORT,
    MISSING_SENSE_MEMORY,
    MISSING_SEMANTIC_FRAME,
    MISSING_PHRASE_CANDIDATE,
    SURFACE_VALIDATION_FAILED,
    SUBJECT_ACTION_INVALID,
    NO_SAFE_REPAIRED_CANDIDATE,
    UNSUPPORTED_OR_UNDERCONSTRAINED_CONTEXT,
    TRUE_UNSAFE,
    UNKNOWN_REASON,
]

# Deterministic next-action guidance per reason (engineering direction, not code).
SUGGESTED_ACTION = {
    LOW_MARGIN: "add cue / anti-cue memory or tune sense evidence, not thresholds",
    SENSE_TIE: "add disambiguating cues and guard rules",
    MISSING_OR_WEAK_CUE_SUPPORT: "add cue memory",
    MISSING_SENSE_MEMORY: "add sense entry / explicit memory",
    MISSING_SEMANTIC_FRAME: "add semantic frame",
    MISSING_PHRASE_CANDIDATE: "add safe phrase candidates",
    SURFACE_VALIDATION_FAILED: "inspect renderer/validator mismatch",
    SUBJECT_ACTION_INVALID: "add role-aware actor rewrite or audit rule",
    NO_SAFE_REPAIRED_CANDIDATE: "keep audited unless a safe explicit rewrite is added",
    UNSUPPORTED_OR_UNDERCONSTRAINED_CONTEXT: "keep audit by default",
    TRUE_UNSAFE: "keep audit",
    UNKNOWN_REASON: "improve trace instrumentation",
}


@dataclass
class AuditEvidence:
    """Trace fields parsed from one row, used to classify it."""

    audit_reasons: list[str] = field(default_factory=list)
    all_sense_scores_zero: bool = False
    conflict_detected: bool = False
    no_ambiguous_term: bool = False
    no_candidate_senses: bool = False
    has_surface_pattern: bool = False
    surface_patterns: list[str] = field(default_factory=list)
    surface_repair_reason: Optional[str] = None
    subject_action_invalid: bool = False
    has_semantic_frame: bool = False
    candidate_count: Optional[int] = None
    top_score: Optional[float] = None
    second_score: Optional[float] = None
    margin: Optional[float] = None
    confidence: Optional[float] = None
    expected_sense: Optional[str] = None
    selected_sense: Optional[str] = None

    def as_dict(self) -> dict:
        """Compact, JSON-friendly view of only the populated evidence fields."""
        out: dict = {}
        if self.audit_reasons:
            out["audit_reasons"] = list(self.audit_reasons)
        if self.all_sense_scores_zero:
            out["all_sense_scores_zero"] = True
        if self.conflict_detected:
            out["conflict_detected"] = True
        if self.no_ambiguous_term:
            out["no_ambiguous_term"] = True
        if self.no_candidate_senses:
            out["no_candidate_senses"] = True
        if self.has_surface_pattern:
            out["surface_patterns"] = list(self.surface_patterns)
        if self.surface_repair_reason is not None:
            out["surface_repair_reason"] = self.surface_repair_reason
        if self.subject_action_invalid:
            out["subject_action_invalid"] = True
        if self.has_semantic_frame:
            out["has_semantic_frame"] = True
        if self.candidate_count is not None:
            out["candidate_count"] = self.candidate_count
        if self.top_score is not None:
            out["top_score"] = self.top_score
        if self.second_score is not None:
            out["second_score"] = self.second_score
        if self.margin is not None:
            out["margin"] = self.margin
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.expected_sense:
            out["expected_sense"] = self.expected_sense
        if self.selected_sense:
            out["selected_sense"] = self.selected_sense
        return out


@dataclass
class AuditClassification:
    """Result of classifying one audited row."""

    primary_reason: str
    secondary_reasons: list[str]
    evidence: AuditEvidence
    suggested_next_action: str


def _split_markers(value: str) -> list[str]:
    return [tok.strip() for tok in (value or "").split("|") if tok.strip()]


def _parse_float(token: str) -> Optional[float]:
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def parse_evidence(row: dict) -> AuditEvidence:
    """Extract trace evidence from an output row. Pure, deterministic."""
    reasons = _split_markers(row.get("reasons", ""))
    memory_hits = _split_markers(row.get("memory_hits", ""))
    all_markers = reasons + memory_hits

    ev = AuditEvidence()
    for marker in all_markers:
        if marker.startswith("audit_reason="):
            val = marker.split("=", 1)[1]
            if val not in ev.audit_reasons:
                ev.audit_reasons.append(val)
        elif marker.startswith("surface_pattern="):
            ev.has_surface_pattern = True
            ev.surface_patterns.append(marker.split("=", 1)[1])
        elif marker.startswith("surface_repair_reason="):
            ev.surface_repair_reason = marker.split("=", 1)[1]
        elif marker.startswith("semantic_frame_intent="):
            ev.has_semantic_frame = True
        elif marker.startswith("candidate_count="):
            try:
                ev.candidate_count = int(marker.split("=", 1)[1])
            except ValueError:
                pass
        elif marker.startswith("top_score="):
            ev.top_score = _parse_float(marker.split("=", 1)[1])
        elif marker.startswith("second_score="):
            ev.second_score = _parse_float(marker.split("=", 1)[1])
        elif marker.startswith("margin="):
            ev.margin = _parse_float(marker.split("=", 1)[1])
        elif marker == "all_sense_scores_zero":
            ev.all_sense_scores_zero = True
        elif marker == "conflict_detected":
            ev.conflict_detected = True
        elif marker == "no_ambiguous_term":
            ev.no_ambiguous_term = True
        elif marker == "no_candidate_senses":
            ev.no_candidate_senses = True
        elif marker in {"subject_action_invalid", "subject_action_validator=rejected"}:
            ev.subject_action_invalid = True

    ev.confidence = _parse_float(row.get("confidence", ""))
    ev.expected_sense = (row.get("expected_sense", "") or "").strip() or None
    ev.selected_sense = (row.get("selected_sense", "") or "").strip() or None
    return ev


def _margin_tier(margin: Optional[float]) -> str:
    """Tied (margin == 0) vs merely low (0 < margin < threshold)."""
    if margin is not None and margin == 0:
        return SENSE_TIE
    return LOW_MARGIN


def classify(row: dict) -> AuditClassification:
    """Classify one audited row into a primary reason + secondary markers.

    Priority is deterministic and grounded in explicit trace markers, in order:
    surface-repair audit, role failure, surface-validation failure, missing
    candidate/frame, margin/tie, misleading/negated/anti-cue (true unsafe),
    weak-cue guard, raw zero-score, missing term, else unknown.
    """
    ev = parse_evidence(row)
    ar = ev.audit_reasons
    secondary: list[str] = []

    def result(primary: str) -> AuditClassification:
        ordered = [r for r in ALL_REASONS if r in secondary and r != primary]
        return AuditClassification(
            primary_reason=primary,
            secondary_reasons=ordered,
            evidence=ev,
            suggested_next_action=SUGGESTED_ACTION[primary],
        )

    # 1. Surface repair could not preserve sense/roles.
    if "no_safe_repaired_candidate" in ar or ev.surface_repair_reason is not None:
        return result(NO_SAFE_REPAIRED_CANDIDATE)

    # 1b. Prompt tail accepted the sense but rejected the final grammar fit.
    if "prompt_tail_incompatible" in ar:
        return result(SURFACE_VALIDATION_FAILED)

    # 2. Role / action pairing invalid.
    if ev.subject_action_invalid or "subject_action_invalid" in ar:
        return result(SUBJECT_ACTION_INVALID)

    # 3. Candidate existed but failed final surface validation.
    if ev.has_surface_pattern or "surface_realization_risk" in ar:
        if "no_safe_semantic_candidate" in ar:
            secondary.append(MISSING_PHRASE_CANDIDATE)
        return result(SURFACE_VALIDATION_FAILED)

    # 4. Sense selected but no safe frame/phrase candidate.
    if "no_safe_semantic_candidate" in ar:
        return result(MISSING_PHRASE_CANDIDATE if ev.has_semantic_frame else MISSING_SEMANTIC_FRAME)

    # 5. Competing senses below margin threshold.
    if "low_margin_conflict" in ar or "low_confidence_conflict" in ar:
        if ev.conflict_detected:
            secondary.append(SENSE_TIE if _margin_tier(ev.margin) == LOW_MARGIN else LOW_MARGIN)
        return result(_margin_tier(ev.margin))

    # 6. Misleading / contradictory / negated evidence: continuing would guess.
    if "anti_cue_conflict" in ar or "anti_cue" in ar:
        return result(TRUE_UNSAFE)
    if "only_negated_evidence" in ar or "negated_top_sense" in ar:
        return result(TRUE_UNSAFE)

    # 7. Weak cue blocked by a guard rule -> needs stronger cue memory.
    if "guard_failure" in ar:
        return result(MISSING_OR_WEAK_CUE_SUPPORT)

    # 8. Explicit threshold markers without a conflict marker.
    explicit = " ".join(ar)
    if "margin_below_min_margin" in explicit:
        return result(_margin_tier(ev.margin))
    if "top_score_below_min_score" in explicit:
        return result(MISSING_OR_WEAK_CUE_SUPPORT)

    # 9. No positive cue fired for any sense.
    if ev.all_sense_scores_zero:
        # Answerable (expected sense known) -> cue memory would help.
        # No expected sense -> context genuinely does not support a continuation.
        if ev.expected_sense:
            return result(MISSING_OR_WEAK_CUE_SUPPORT)
        return result(UNSUPPORTED_OR_UNDERCONSTRAINED_CONTEXT)

    # 10. Term (or candidate senses) not represented in explicit memory.
    if ev.no_ambiguous_term or ev.no_candidate_senses:
        return result(MISSING_SENSE_MEMORY)

    # 11. Traces insufficient: do not fake a confident reason.
    return result(UNKNOWN_REASON)
