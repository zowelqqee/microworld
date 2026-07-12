"""Maps SemanticQuery + raw question text → QueryPlan.

Handles both the standard lookup types already covered by EntityAnswerPlanner
and the new aggregate/filter patterns (count, size_compare, filtered_lookup)
that the existing planner cannot answer.
"""

from __future__ import annotations

import re

from worldpgt.entity_qa.types import SemanticQuery
from worldpgt.query_engine.primitives import (
    Classify,
    Compare,
    Count,
    Filter,
    Find,
    Primitive,
    Traverse,
)
from worldpgt.query_engine.types import QueryPlan

_HOW_MANY_RE = re.compile(r"\bhow\s+many\b", re.IGNORECASE)
_MORE_COMPARE_RE = re.compile(
    r"\bmore\b.{0,40}\bthan\b|\bmore\b.{0,60}\bor\b|\bwho\s+\S+\s+more\b",
    re.IGNORECASE,
)
_FILTER_THAT_RE = re.compile(
    r"\bwhich\s+\w+\s+(?:that|who)\b|\bwhat\s+\w+\s+(?:that|which)\b",
    re.IGNORECASE,
)

# "did/has/have X found/founded" or "who founded" → entity_a is the founder.
# The semantic parser maps "found/founded" → "founded_by", but the overlay
# stores (Founder, founded, Company) with entity_a as SUBJECT.
_FOUNDER_AS_ACTOR_RE = re.compile(
    r"\b(?:did|has|have|does)\s+\S.+?\s+(?:found|founded)\b"
    r"|\bwho\s+(?:found|founded)\b",
    re.IGNORECASE,
)

_SECONDARY_VERB_MAP: dict[str, str] = {
    r"\bdevelop\b|\bdevelops\b": "develops",
    r"\bproduce\b|\bproduces\b": "produces",
    r"\bmanufacture\b|\bmanufactures\b": "produces",
    r"\bpublish\b|\bpublishes\b": "publishes",
    r"\bown\b|\bowns\b": "owned_by",
    r"\bbuild\b|\bbuilds\b": "develops",
}

_PRIMARY_VERB_MAP: dict[str, str] = {
    r"\bfound\b|\bfounded\b": "founded",
    r"\bstart\b|\bstarted\b": "founded",
    r"\bcreate\b|\bcreated\b": "founded",
    r"\bown\b|\bowned\b": "owned_by",
}


def build(question: str, semantic: SemanticQuery) -> QueryPlan:
    """Build a QueryPlan from a raw question and its SemanticQuery."""

    # ── "How many X did Y [verb]?" → Find + Count ─────────────────────
    if _HOW_MANY_RE.search(question) and semantic.entity_a and semantic.relation_intent:
        pred = _coerce_relation(question, semantic.relation_intent)
        if semantic.unknown_position == "subject" and pred == semantic.relation_intent:
            find: Find = Find(
                subject=None,
                predicate=pred,
                target="subject",
                known_value=semantic.entity_a,
                result_key="find",
            )
        else:
            find = Find(
                subject=semantic.entity_a,
                predicate=pred,
                target="object",
                result_key="find",
            )
        return QueryPlan(
            steps=[find, Count(source_key="find", result_key="count")],
            confidence=semantic.confidence,
            question_type_hint="count",
        )

    # ── "Who/what did X or Y [verb] more?" → Find + Find + Compare ────
    if (
        _MORE_COMPARE_RE.search(question)
        and semantic.entity_a
        and semantic.entity_b
        and semantic.relation_intent
    ):
        pred = _coerce_relation(question, semantic.relation_intent)
        steps: list[Primitive] = [
            Find(
                subject=semantic.entity_a,
                predicate=pred,
                target="object",
                result_key="find_a",
            ),
            Find(
                subject=semantic.entity_b,
                predicate=pred,
                target="object",
                result_key="find_b",
            ),
            Compare(
                key_a="find_a",
                key_b="find_b",
                mode="size_compare",
                label_a=semantic.entity_a,
                label_b=semantic.entity_b,
                result_key="compare",
            ),
        ]
        return QueryPlan(
            steps=steps,
            confidence=semantic.confidence,
            question_type_hint="size_compare",
        )

    # ── "Which X that Y [verb1] [verb2]?" → Find + Filter ─────────────
    if _FILTER_THAT_RE.search(question) and semantic.entity_a and semantic.relation_intent:
        primary_pred, secondary_pred = _extract_filter_preds(question, semantic.relation_intent)
        if primary_pred and secondary_pred:
            filter_steps: list[Primitive] = [
                Find(
                    subject=semantic.entity_a,
                    predicate=primary_pred,
                    target="object",
                    result_key="find",
                ),
                Filter(
                    source_key="find",
                    condition={
                        "predicate": secondary_pred,
                        **({"object": semantic.filter_object} if semantic.filter_object else {}),
                    },
                    result_key="filter",
                ),
            ]
            return QueryPlan(
                steps=filter_steps,
                confidence=semantic.confidence * 0.9,
                question_type_hint="filtered_lookup",
            )

    # ── Intersection / comparative ("what do X and Y have in common?") ─
    if semantic.query_type == "comparative" and semantic.unknown_position == "intersection":
        if semantic.entity_a and semantic.entity_b:
            intersection_steps: list[Primitive] = [
                Find(
                    subject=semantic.entity_a,
                    predicate=None,
                    target="object",
                    result_key="find_a",
                ),
                Find(
                    subject=semantic.entity_b,
                    predicate=None,
                    target="object",
                    result_key="find_b",
                ),
                Compare(
                    key_a="find_a",
                    key_b="find_b",
                    mode="intersection",
                    label_a=semantic.entity_a,
                    label_b=semantic.entity_b,
                    result_key="compare",
                ),
            ]
            return QueryPlan(
                steps=intersection_steps,
                confidence=semantic.confidence,
                question_type_hint="intersection",
            )

    # ── is_a check ─────────────────────────────────────────────────────
    if semantic.query_type == "is_a" and semantic.entity_a and semantic.entity_b:
        return QueryPlan(
            steps=[Classify(
                entity=semantic.entity_a,
                target_class=semantic.entity_b,
                result_key="classify",
            )],
            confidence=semantic.confidence,
            question_type_hint="is_a",
        )

    # ── Standard object lookup ─────────────────────────────────────────
    if (
        semantic.entity_a
        and semantic.relation_intent
        and semantic.unknown_position == "object"
    ):
        return QueryPlan(
            steps=[Find(
                subject=semantic.entity_a,
                predicate=semantic.relation_intent,
                target="object",
                result_key="find",
            )],
            confidence=semantic.confidence,
            question_type_hint="lookup",
        )

    # ── Inverse lookup (unknown subject) ──────────────────────────────
    if (
        semantic.entity_a
        and semantic.relation_intent
        and semantic.unknown_position == "subject"
    ):
        return QueryPlan(
            steps=[Find(
                subject=None,
                predicate=semantic.relation_intent,
                target="subject",
                known_value=semantic.entity_a,
                result_key="find",
            )],
            confidence=semantic.confidence,
            question_type_hint="inverse_lookup",
        )

    # ── Definition (no relation, just entity) ─────────────────────────
    if semantic.query_type == "definition" and semantic.entity_a:
        return QueryPlan(
            steps=[Find(
                subject=semantic.entity_a,
                predicate=None,
                target="object",
                result_key="find",
            )],
            confidence=semantic.confidence * 0.8,
            question_type_hint="definition",
        )

    # ── Unrecognized → audit ───────────────────────────────────────────
    return QueryPlan(steps=[], confidence=0.0, question_type_hint="unknown")


# ── Helpers ────────────────────────────────────────────────────────────────

def _coerce_relation(question: str, relation: str) -> str:
    """Normalize relations that the semantic parser mis-classifies.

    The relation_policy maps "found/founded" → "founded_by", but when the entity
    is acting as founder (e.g. "did X found") the overlay uses (X, founded, Y).
    Flip to "founded" so Find hits the correct overlay structure.
    """
    if relation == "founded_by" and _FOUNDER_AS_ACTOR_RE.search(question):
        return "founded"
    return relation


def _match_first(text: str, mapping: dict[str, str]) -> str | None:
    for pattern, value in mapping.items():
        if re.search(pattern, text, re.IGNORECASE):
            return value
    return None


def _extract_filter_preds(
    question: str,
    semantic_relation: str,
) -> tuple[str | None, str | None]:
    """Return (primary_pred, secondary_pred) for a filtered-lookup question.

    Primary is the relation used to gather the initial set (e.g. "founded").
    Secondary is the filter applied to each result object (e.g. "develops").
    """
    primary = _match_first(question, _PRIMARY_VERB_MAP) or semantic_relation
    secondary = _match_first(question, _SECONDARY_VERB_MAP)
    if secondary == primary:
        secondary = None
    return primary, secondary
