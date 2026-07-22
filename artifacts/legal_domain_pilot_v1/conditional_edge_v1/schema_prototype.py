"""Conditional-edge schema prototype — design artifact, NOT production code.

This file deliberately lives under ``artifacts/`` and not under
``worldpgt/relation_extraction_v2/``.  It exists to make the proposed schema
concrete and executable for the retrospective test in this directory.  It
imports nothing from the runtime and changes no runtime behaviour.

Proposed minimal extension of ``ExtractedRelationCandidate``
------------------------------------------------------------
The existing candidate already carries one optional structured field::

    evidence: Optional[RelationExtractionEvidence] = None

The proposal reuses exactly that pattern and adds three more optional fields::

    conditions: list[ConditionClause] = []
    exceptions: list[ConditionClause] = []
    polarity:   "affirm" | "negate"   = "affirm"

Every existing simple edge keeps working untouched: empty lists and
``polarity="affirm"`` reproduce today's semantics exactly, and ``to_dict``
omits the three keys entirely when they are at their defaults, so existing
overlay JSON stays byte-identical.  No parallel edge class is required.

Semantics, stated explicitly so a planner can rely on them
----------------------------------------------------------
- ``conditions`` is a **conjunction**: the edge asserts its triple only when
  *every* condition holds.  An empty list means unconditional.
- ``exceptions`` is a **disjunction of defeaters**: the edge does not assert
  its triple for any case matching *any* exception.
- Disjunctive alternatives ("if A, or B, or C") are represented as *separate
  edges* sharing subject/predicate/object, one condition each.
- ``polarity="negate"`` asserts the triple does **not** hold.  It is not a
  different predicate; it is the sign of the same predicate, which is what
  lets "X shall be entitled unless C" become
  ``entitled_to(person, patent) [negate] if C``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Polarity = Literal["affirm", "negate"]

# A clause is either a fact that must obtain, or a restriction on the context
# in which the rule is being applied ("for purposes of ... under subsection
# (a)(2)").  The distinction was *discovered by the retrospective test*, not
# assumed by the original design; see report.md section 5.
ClauseKind = Literal["factual", "scope"]


@dataclass(frozen=True)
class ConditionClause:
    """One condition or exception, individually anchored to the source text."""

    text: str
    evidence_span: str
    kind: ClauseKind = "factual"

    def to_dict(self) -> dict:
        payload = {"text": self.text, "evidence_span": self.evidence_span}
        if self.kind != "factual":
            payload["kind"] = self.kind
        return payload


@dataclass(frozen=True)
class ConditionalRelationEdge:
    """The three proposed fields attached to an ordinary triple.

    Field names and defaults mirror what would be added to
    ``ExtractedRelationCandidate``; the provenance fields it already has
    (``source_url``, ``evidence_sentence``) are reused unchanged.
    """

    id: str
    subject: str
    predicate: str
    object: str
    evidence_sentence: str
    stated_in: str = ""
    conditions: tuple[ConditionClause, ...] = ()
    exceptions: tuple[ConditionClause, ...] = ()
    polarity: Polarity = "affirm"

    def is_simple(self) -> bool:
        """True when this edge is indistinguishable from a pre-extension edge."""
        return not self.conditions and not self.exceptions and self.polarity == "affirm"

    def to_dict(self) -> dict:
        payload = {
            "id": self.id,
            "subject": self.subject,
            "relation": self.predicate,
            "object": self.object,
            "evidence_sentence": self.evidence_sentence,
            "stated_in": self.stated_in,
        }
        # Defaults are omitted so existing simple edges serialize unchanged.
        if self.conditions:
            payload["conditions"] = [c.to_dict() for c in self.conditions]
        if self.exceptions:
            payload["exceptions"] = [e.to_dict() for e in self.exceptions]
        if self.polarity != "affirm":
            payload["polarity"] = self.polarity
        return payload


# --------------------------------------------------------------------------
# Verification: the same "literal span" discipline the rest of the graph uses.
# --------------------------------------------------------------------------

_CONDITIONAL_CONNECTIVE = re.compile(
    r"\b(?:if|unless|except|when|whenever|provided that|subject to|"
    r"notwithstanding|shall|states that|establishes)\b",
    re.IGNORECASE,
)
_MAX_PREDICATE_WORDS = 6


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _is_literal(span: str, source: str) -> bool:
    return bool(_normalise(span)) and _normalise(span).casefold() in _normalise(source).casefold()


def verify_edge(edge: ConditionalRelationEdge) -> dict:
    """Return deterministic pass/fail checks for one conditional edge.

    ``no_welding`` is the direct test of the failure mode this schema exists to
    remove: a predicate that has swallowed the rule's conditional structure.
    """

    predicate_words = len(_normalise(edge.predicate).replace("_", " ").split())
    failures: list[str] = []

    if predicate_words > _MAX_PREDICATE_WORDS:
        failures.append(f"predicate_too_long({predicate_words}w)")
    if _CONDITIONAL_CONNECTIVE.search(edge.predicate.replace("_", " ")):
        failures.append("conditional_connective_in_predicate")

    for label, clauses in (("condition", edge.conditions), ("exception", edge.exceptions)):
        for number, clause in enumerate(clauses):
            if not _is_literal(clause.evidence_span, edge.evidence_sentence):
                failures.append(f"{label}[{number}]_span_not_literal")

    return {
        "id": edge.id,
        "predicate_words": predicate_words,
        "condition_count": len(edge.conditions),
        "exception_count": len(edge.exceptions),
        "polarity": edge.polarity,
        "no_welding": not failures or all(not f.startswith(("predicate", "conditional")) for f in failures),
        "all_spans_literal": not any("span_not_literal" in f for f in failures),
        "failures": failures,
        "passes": not failures,
    }


_STOPWORDS = frozenset(
    """a an and or the of to in on for by with as is are was were be been being that
    which who whom whose this these those such it its any no not shall may under""".split()
)


def content_tokens(text: str) -> set[str]:
    """Content words used to measure how much of a provision an edge retains."""
    return {
        token
        for token in re.findall(r"[a-z0-9()]+", _normalise(text).lower())
        if token not in _STOPWORDS and len(token) > 1
    }


def coverage(edge: ConditionalRelationEdge, provision_text: str) -> float:
    """Fraction of the provision's content words retained by the whole edge."""
    captured = content_tokens(
        " ".join(
            [edge.subject, edge.predicate.replace("_", " "), edge.object]
            + [c.text for c in edge.conditions]
            + [e.text for e in edge.exceptions]
        )
    )
    target = content_tokens(provision_text)
    return len(target & captured) / len(target) if target else 1.0
