"""Assistant/CLI adapter for the reasoning layer.

Thin, additive integration mirroring ``multihop_qa.assistant_adapter``: parse
a controlled question form, delegate to the explanation builder or the
counterfactual analyzer, and render an honest answer. Unrecognized forms
return an explicit unsupported result so callers can fall through to the
existing pipeline unchanged.

Deterministic. No ML. No network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from worldpgt.reasoning.counterfactual import analyze_counterfactual
from worldpgt.reasoning.counterfactual import render as render_counterfactual
from worldpgt.reasoning.explanation_builder import explain_fact
from worldpgt.reasoning.explanation_renderer import render as render_explanation
from worldpgt.reasoning.pattern_discovery import (
    relevant_patterns,
    render_pattern_note,
)
from worldpgt.reasoning.types import GraphPattern

# Verb → overlay predicate. Only unambiguous, domain-agnostic mappings.
_VERB_PREDICATES: dict[str, str] = {
    "develop": "develops",
    "develops": "develops",
    "produce": "produces",
    "produces": "produces",
    "manufacture": "manufactures",
    "manufactures": "manufactures",
    "operate": "operates",
    "operates": "operates",
    "publish": "publishes",
    "publishes": "publishes",
    "own": "owns",
    "owns": "owns",
    "use": "uses",
    "uses": "uses",
    "provide": "provides",
    "provides": "provides",
    "enable": "enables",
    "enables": "enables",
    "support": "supports",
    "supports": "supports",
    "require": "requires",
    "requires": "requires",
    "lead": "leads",
    "leads": "leads",
}

_VERB_ALTERNATION = "|".join(sorted(_VERB_PREDICATES, key=len, reverse=True))

_WHY_VERB_RE = re.compile(
    rf"^why\s+(?:does|do|did)\s+(.+?)\s+({_VERB_ALTERNATION})\s+(.+?)[?.]?$",
    re.IGNORECASE,
)
_WHY_IS_A_RE = re.compile(
    r"^why\s+is\s+(.+?)\s+(?:a|an)\s+(.+?)[?.]?$",
    re.IGNORECASE,
)
_WHY_FOUNDED_RE = re.compile(
    r"^why\s+did\s+(.+?)\s+found\s+(.+?)[?.]?$",
    re.IGNORECASE,
)

# Past-tense verbs accepted in counterfactual hypotheses.
_CF_VERB_PREDICATES: dict[str, str] = {
    "founded": "founded",
    "developed": "develops",
    "produced": "produces",
    "manufactured": "manufactures",
    "operated": "operates",
    "published": "publishes",
    "owned": "owns",
    "used": "uses",
    "provided": "provides",
    "led": "leads",
}
_CF_VERBS = "|".join(sorted(_CF_VERB_PREDICATES, key=len, reverse=True))

_WHAT_IF_NOT_RE = re.compile(
    rf"^what\s+(?:if|would\s+(?:change|happen)\s+if)\s+(.+?)\s+"
    rf"(?:had\s+not|hadn't|had\s+never|did\s+not|didn't|never)\s+"
    rf"({_CF_VERBS})\s+(.+?)[?.]?$",
    re.IGNORECASE,
)
_WHAT_IF_NOT_PASSIVE_RE = re.compile(
    rf"^what\s+(?:if|would\s+(?:change|happen)\s+if)\s+(.+?)\s+"
    rf"(?:had|was|were)\s+(?:not|never)\s+been\s+({_CF_VERBS})\s+by\s+(.+?)[?.]?$"
    rf"|^what\s+(?:if|would\s+(?:change|happen)\s+if)\s+(.+?)\s+"
    rf"(?:was|were)\s+(?:not|never)\s+({_CF_VERBS})\s+by\s+(.+?)[?.]?$",
    re.IGNORECASE,
)
_WHAT_IF_NO_ENTITY_RE = re.compile(
    r"^what\s+(?:if|would\s+(?:change|happen)\s+if)\s+(.+?)\s+"
    r"(?:did\s+not|didn't|had\s+never|never)\s+exist(?:ed)?[?.]?$",
    re.IGNORECASE,
)
_WHAT_DEPENDS_RE = re.compile(
    rf"^what\s+depends\s+on\s+the\s+fact\s+that\s+(.+?)\s+"
    rf"({_VERB_ALTERNATION}|{_CF_VERBS})\s+(.+?)[?.]?$",
    re.IGNORECASE,
)


@dataclass
class AssistantReasoningResult:
    """Result object for the CLI / API integration."""

    question: str
    kind: str            # "explanation" | "counterfactual" | "unsupported"
    decision: str        # answer | partial | analysis | audit
    answer_text: str
    audit_reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    pattern_notes: list[str] = field(default_factory=list)
    support_kind: str = "explicit_reasoning_chain"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "kind": self.kind,
            "decision": self.decision,
            "answer_text": self.answer_text,
            "audit_reason": self.audit_reason,
            "detail": dict(self.detail),
            "pattern_notes": list(self.pattern_notes),
            "support_kind": self.support_kind,
        }


def _clean(text: str) -> str:
    return text.strip().strip("\"'")


def parse_reasoning_question(question: str) -> dict[str, Any] | None:
    """Parse a controlled reasoning question; None when unrecognized."""
    q = (question or "").strip()

    m = _WHY_VERB_RE.match(q)
    if m:
        predicate = _VERB_PREDICATES[m.group(2).lower()]
        return {
            "kind": "explanation",
            "subject": _clean(m.group(1)),
            "predicate": predicate,
            "object": _clean(m.group(3)),
        }
    m = _WHY_FOUNDED_RE.match(q)
    if m:
        return {
            "kind": "explanation",
            "subject": _clean(m.group(1)),
            "predicate": "founded",
            "object": _clean(m.group(2)),
        }
    m = _WHY_IS_A_RE.match(q)
    if m:
        return {
            "kind": "explanation",
            "subject": _clean(m.group(1)),
            "predicate": "is_a",
            "object": _clean(m.group(2)),
        }

    m = _WHAT_IF_NOT_RE.match(q)
    if m:
        return {
            "kind": "counterfactual",
            "subject": _clean(m.group(1)),
            "predicate": _CF_VERB_PREDICATES[m.group(2).lower()],
            "object": _clean(m.group(3)),
        }
    m = _WHAT_IF_NOT_PASSIVE_RE.match(q)
    if m:
        groups = [g for g in m.groups() if g is not None]
        obj, verb, subj = groups[0], groups[1], groups[2]
        return {
            "kind": "counterfactual",
            "subject": _clean(subj),
            "predicate": _CF_VERB_PREDICATES[verb.lower()],
            "object": _clean(obj),
        }
    m = _WHAT_IF_NO_ENTITY_RE.match(q)
    if m:
        return {
            "kind": "counterfactual",
            "subject": _clean(m.group(1)),
            "predicate": None,
            "object": None,
        }
    m = _WHAT_DEPENDS_RE.match(q)
    if m:
        verb = m.group(2).lower()
        predicate = _VERB_PREDICATES.get(verb) or _CF_VERB_PREDICATES.get(verb)
        return {
            "kind": "counterfactual",
            "subject": _clean(m.group(1)),
            "predicate": predicate,
            "object": _clean(m.group(3)),
        }
    return None


def is_reasoning_compatible(question: str) -> bool:
    return parse_reasoning_question(question) is not None


def try_answer_reasoning(
    question: str,
    overlay_items: list[dict],
    patterns: list[GraphPattern] | None = None,
) -> AssistantReasoningResult:
    """Try the reasoning layer on a controlled question form.

    *patterns* is optional pre-discovered context (e.g. the nightly artifact);
    when provided, explanations may cite a relevant pattern and the answer
    carries pattern notes. Absence of patterns never blocks an answer.
    """
    parsed = parse_reasoning_question(question)
    if parsed is None:
        return AssistantReasoningResult(
            question=question,
            kind="unsupported",
            decision="audit",
            answer_text="[audit] no recognized reasoning question pattern matched",
            audit_reason="no recognized reasoning question pattern matched",
        )

    if parsed["kind"] == "explanation":
        chain = explain_fact(
            overlay_items,
            parsed["subject"],
            parsed["predicate"],
            parsed["object"],
            patterns=patterns,
        )
        notes = [
            render_pattern_note(p)
            for p in relevant_patterns(
                patterns or [], parsed["subject"], parsed["predicate"]
            )
        ]
        return AssistantReasoningResult(
            question=question,
            kind="explanation",
            decision=chain.decision,
            answer_text=render_explanation(chain),
            audit_reason=chain.audit_reason or "",
            detail=chain.to_dict(),
            pattern_notes=notes,
            support_kind="explanatory_fact_chain",
        )

    trace = analyze_counterfactual(
        overlay_items,
        parsed["subject"],
        parsed["predicate"],
        parsed["object"],
        patterns=patterns,
    )
    return AssistantReasoningResult(
        question=question,
        kind="counterfactual",
        decision=trace.decision,
        answer_text=render_counterfactual(trace),
        audit_reason=trace.audit_reason or "",
        detail=trace.to_dict(),
        support_kind="structural_dependency_analysis",
    )
