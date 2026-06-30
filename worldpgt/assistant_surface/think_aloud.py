"""Think-aloud composition for the Assistant Surface.

Turns a finished :class:`AssistantAnswer` (plus, optionally, a multi-hop chain
and any relevant inferred facts) into the two-block "think aloud" surface::

    [THINKING]
    ...a plain-language account of how the answer was reached...

    [ANSWER]
    ...the answer itself...

The THINKING block is reconstructed from the deterministic decision trace the
orchestrator already records (route, matched entities, relations, support kind),
so it never adds information the system didn't actually have. The ANSWER block
reuses the existing answer text, except for audits — where it switches to the
honest gap explanation — and for inference-backed answers, where it explains the
derived conclusion and flags it as inferred rather than verified.

This module is presentation-only: importing or calling it never changes the
runtime answer, the overlay, or memory. Inference is surfaced here purely as a
*labelled* explanation, consistent with the system's safety model (an inferred
fact is never promoted to a verified one).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from worldpgt.assistant_surface.types import AssistantAnswer
from worldpgt.cognition.inference_engine import InferredFact
from worldpgt.cognition.verbalization_engine import (
    hop_sentence,
    verbalize_audit,
    verbalize_inferred_fact,
    verbalize_multihop,
    verbalize_synthesis,
)

# Capability-style predicates worth surfacing as an INFERRED tier in a profile.
_PROFILE_INFERENCE_PREDICATES = frozenset(
    {"develops", "produces", "operates", "provides", "owns", "manufactures"}
)


@dataclass
class ThinkAloud:
    """A think-aloud surface: a thinking narrative plus the answer."""

    thinking: str
    answer: str

    def render(self) -> str:
        return f"[THINKING]\n{self.thinking}\n\n[ANSWER]\n{self.answer}"


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def build_think_aloud(
    answer: AssistantAnswer,
    *,
    question: str,
    subject: str | None = None,
    multihop_result=None,
    inferred_facts: Sequence[InferredFact] = (),
    synthesis=None,
) -> ThinkAloud:
    """Compose a :class:`ThinkAloud` from a finished answer and its context.

    When *synthesis* (a ``SynthesisAnswer``) is supplied for a profile question,
    the ANSWER block is rendered as a tiered list (verified / snapshot / inferred)
    so the web UI can badge each tier.
    """
    cs = (answer.trace.context_summary if answer.trace else None) or {}
    if subject is None:
        matched = cs.get("matched_entities") or []
        subject = matched[0] if matched else (answer.question or "the topic")

    relations = [_readable_rel(r) for r in cs.get("direct_relations", [])]

    if multihop_result is not None and getattr(multihop_result, "decision", "") == "answer":
        return _multihop_think_aloud(answer, question, subject, multihop_result)

    if synthesis is not None and getattr(synthesis, "matched", False) and answer.decision != "audit":
        return _synthesis_think_aloud(subject, synthesis, inferred_facts)

    if answer.decision == "audit" and inferred_facts:
        return _inferred_think_aloud(question, subject, relations, inferred_facts)

    if answer.decision == "audit":
        return _audit_think_aloud(answer, question, subject, relations)

    return _direct_think_aloud(answer, question, subject, relations)


# --------------------------------------------------------------------------- #
# Per-mode builders
# --------------------------------------------------------------------------- #

def _direct_think_aloud(
    answer: AssistantAnswer, question: str, subject: str, relations: list[str]
) -> ThinkAloud:
    # Show only the fact relevant to *this* question type (the answer is already
    # scoped to it), not every fact known about the entity.
    lines = [
        f"I identified the entity: {subject}.",
        _search_line(answer.route, question),
    ]
    finding = _relevant_finding(answer)
    if finding:
        lines.append(finding)
    lines.append(f"Support: {_support_phrase(answer.support_kind)}.")
    if answer.route == "entity_relation":
        lines.append("No additional hops needed.")
    lines.append("Conclusion: direct answer available.")
    return ThinkAloud(thinking="\n".join(lines), answer=answer.answer_text)


def _relevant_finding(answer: AssistantAnswer) -> str:
    """The single fact that answers the question, drawn from the answer itself."""
    sentence = _first_sentence(answer.answer_text)
    if not sentence:
        return ""
    label = "I found a definition" if answer.route == "entity_definition" else "I found"
    return f"{label}: {sentence}"


def _first_sentence(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    m = re.search(r"(.+?[.!?])(?:\s|$)", t)
    return m.group(1).strip() if m else t.split("\n", 1)[0].strip()


def _multihop_think_aloud(
    answer: AssistantAnswer, question: str, subject: str, multihop
) -> ThinkAloud:
    lines = [
        f"I identified the entities involved, starting from {subject}.",
        "No direct relation found — searching for a connecting path.",
    ]
    hop1 = getattr(multihop, "hop1_detail", None) or getattr(multihop, "hop1", None)
    hop2 = getattr(multihop, "hop2_detail", None) or getattr(multihop, "hop2", None)
    if hop_sentence(hop1):
        lines.append(f"Hop 1: {hop_sentence(hop1)}.")
    if hop_sentence(hop2):
        lines.append(f"Hop 2: {hop_sentence(hop2)}.")
    lines.append("Conclusion: a supported two-step chain connects them.")
    return ThinkAloud(
        thinking="\n".join(lines),
        answer=verbalize_multihop(multihop) or answer.answer_text,
    )


def _synthesis_think_aloud(
    subject: str, synthesis, inferred_facts: Sequence[InferredFact]
) -> ThinkAloud:
    verified = synthesis.verified_count
    snapshot = synthesis.snapshot_count
    lines = [
        f"I identified the entity: {subject}.",
        "I gathered every fact I hold about it from the knowledge graph.",
        f"I sorted them by trust: {verified} verified, {snapshot} dated snapshot(s)"
        + (f", {len(inferred_facts)} inferred" if inferred_facts else "")
        + ".",
        "Conclusion: I can give a tiered summary, tagging each claim's trust level.",
    ]
    return ThinkAloud(
        thinking="\n".join(lines),
        answer=verbalize_synthesis(synthesis, inferred_facts),
    )


def _inferred_think_aloud(
    question: str,
    subject: str,
    relations: list[str],
    inferred_facts: Sequence[InferredFact],
) -> ThinkAloud:
    fact = inferred_facts[0]
    lines = [
        f"Entity found: {subject}.",
        f"Direct search: no {fact.predicate} relation found for {subject}.",
        "Running inference rules...",
    ]
    chain_clause = " ∧ ".join(f"{s} {p} {o}" for (s, p, o) in fact.chain)
    lines.append(
        f"Rule {fact.rule}: {chain_clause} "
        f"→ Inferred: {fact.subject} {fact.predicate} {fact.object} "
        f"(confidence: {_conf_word(fact.confidence)})"
    )
    answer_text = (
        "Based on my reasoning: "
        + verbalize_inferred_fact(fact)
        + "\nNote: this is an inferred conclusion, not a directly verified fact."
    )
    return ThinkAloud(thinking="\n".join(lines), answer=answer_text)


def _audit_think_aloud(
    answer: AssistantAnswer, question: str, subject: str, relations: list[str]
) -> ThinkAloud:
    known = relations[:3]
    lines = [f"Entity found: {subject}." if known else f"Searching for: {subject}."]
    lines.append(_search_line(answer.route, question))
    if known:
        lines.append("Found: " + "; ".join(known) + ".")
    missing_hints = (answer.trace.context_summary or {}).get("missing_hints", []) if answer.trace else []
    if missing_hints:
        lines.append("Missing: " + "; ".join(missing_hints[:3]) + ".")
    else:
        lines.append("Missing: no facts in the knowledge base cover this.")
    lines.append("Gap identified: this part of the question is not answerable.")

    known_sentence = (
        f"I know that {known[0]}." if known else None
    )
    answer_text = verbalize_audit(
        question,
        reason=_audit_reason(answer),
        known=known_sentence,
        needs=_needs_from_hints(missing_hints),
    )
    return ThinkAloud(thinking="\n".join(lines), answer=answer_text)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _readable_rel(display: str) -> str:
    """Strip the ``[stability=…, trust=…]`` suffix from a relation display."""
    return str(display or "").split(" [")[0].strip()


def _search_line(route: str, question: str) -> str:
    return {
        "entity_relation": "I searched for matching relation facts.",
        "entity_definition": "I searched for a definition and classifying facts.",
        "connection_path": "I searched for a connecting path between the entities.",
        "source_qualified_fact": "I searched for a source-qualified snapshot fact.",
        "weak_link_policy": "I checked how these entities are linked in memory.",
    }.get(route, "I searched my knowledge base for relevant facts.")


def _support_phrase(support_kind: str) -> str:
    return {
        "stable_relation": "a stable, verified relation",
        "semi_stable_relation": "a semi-stable historical relation",
        "stable_definition": "a stable definition",
        "explicit_connection_path": "an explicit connection path",
        "explicit_is_a_chain": "an explicit is-a chain",
        "source_qualified_fact": "a source-qualified snapshot",
        "safe_policy_answer": "a safe policy explanation",
    }.get(support_kind, support_kind.replace("_", " "))


def _audit_reason(answer: AssistantAnswer) -> str | None:
    text = (answer.answer_text or "").strip()
    if not text or text.lower().startswith("no relevant information"):
        return "I don't have a verified answer for it in my knowledge base"
    return text.rstrip(".")


def _needs_from_hints(hints: list[str]) -> list[str]:
    return [h for h in hints[:3] if h]


def _conf_word(value: float) -> str:
    if value >= 0.85:
        return "high"
    if value >= 0.6:
        return "medium"
    return "low"


def select_inferred_facts(
    workspace, subject: str, question: str, limit: int = 3
) -> list[InferredFact]:
    """Pick inferred facts about *subject* relevant to *question*.

    Facts whose predicate word appears in the question are preferred (so "what
    does Starlink *develop*?" surfaces the ``develops`` inference first). Returns
    an empty list when nothing about the subject was inferred.
    """
    if workspace is None or not subject:
        return []
    facts = list(workspace.for_subject(subject))
    if not facts:
        return []
    q = (question or "").lower()
    facts.sort(key=lambda f: (not _predicate_in_question(f.predicate, q), -f.confidence))
    return facts[:limit]


def select_profile_inferred_facts(
    workspace, subject: str, limit: int = 3
) -> list[InferredFact]:
    """Capability-style inferred facts about *subject* for a profile's INFERRED tier."""
    if workspace is None or not subject:
        return []
    facts = [
        f
        for f in workspace.for_subject(subject)
        if f.predicate in _PROFILE_INFERENCE_PREDICATES
    ]
    facts.sort(key=lambda f: -f.confidence)
    return facts[:limit]


def _predicate_in_question(predicate: str, question_lc: str) -> bool:
    """True when the predicate (or its singular stem) is mentioned in the question."""
    word = (predicate or "").replace("_", " ").strip().lower()
    if not word:
        return False
    return word in question_lc or word.rstrip("s") in question_lc
