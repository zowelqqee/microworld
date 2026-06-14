"""Deterministic question analyzer for Entity QA v1.

Parses controlled entity-level questions into AnalyzedEntityQuestion objects.
Rule-based only. No ML. No embeddings. No network access.

Recognized intents:
  define_entity       — Who/What is X?
  relation_lookup     — known_for, leader_of, produces, develops, publishes, founded
  link_explanation    — Why is Y linked to X?
  source_fact_lookup  — Forbes estimate, net worth stable, recheck queries
  unknown_or_unsupported — current CEO, stock price, personal info
"""

from __future__ import annotations

import re
from typing import Optional

from worldpgt.entity_qa.types import AnalyzedEntityQuestion, EntityQAIntent

# ---- current / volatile query signals ---------------------------------
_CURRENT_SIGNALS = re.compile(
    r"\b(current\s+ceo|current\s+stock\s+price|current\s+valuation|"
    r"current\s+market\s+cap|current\s+price|stock\s+price|"
    r"current\s+president|current\s+office)\b",
    re.IGNORECASE,
)

_PERSONAL_SIGNALS = re.compile(
    r"\b(favorite\s+food|favorite\s+color|favorite\s+hobby|personal\s+life|"
    r"home\s+address|phone\s+number|date\s+of\s+birth|age\s+of)\b",
    re.IGNORECASE,
)

# ---- source fact patterns --------------------------------------------
_ESTIMATE_RE = re.compile(
    r"(?:what does|what did)\s+(\w+)\s+estimate\s+about\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)
_STABLE_FACT_RE = re.compile(
    r"is\s+(.+?)'s?\s+net\s+worth\s+a\s+stable\s+fact",
    re.IGNORECASE,
)
_RECHECK_RE = re.compile(
    r"why\s+should\s+(.+?)'s?\s+net\s+worth\s+be\s+rechecked",
    re.IGNORECASE,
)

# ---- link explanation -----------------------------------------------
_LINK_EXPL_RE = re.compile(
    r"why\s+(?:is|are)\s+(.+?)\s+linked\s+to\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)

# ---- relation lookup ------------------------------------------------
_KNOWN_FOR_RE = re.compile(
    r"what\s+is\s+(.+?)\s+known\s+for\b",
    re.IGNORECASE,
)
_LEADER_RE = re.compile(
    r"what\s+companies\s+is\s+(.+?)\s+linked\s+to\s+by\s+leadership\b",
    re.IGNORECASE,
)
_PRODUCES_RE = re.compile(
    r"what\s+does\s+(.+?)\s+(produce|develop|publish|manufacture)s?\b",
    re.IGNORECASE,
)
_FOUNDED_BY_RE = re.compile(
    r"who\s+founded\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)

# ---- define entity --------------------------------------------------
_WHO_IS_RE = re.compile(
    r"^who\s+is\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)
_WHAT_IS_RE = re.compile(
    r"^what\s+is\s+(?:a\s+|an\s+)?(.+?)[\?.]?$",
    re.IGNORECASE,
)

# verb -> predicate mapping
_VERB_PREDICATE: dict[str, str] = {
    "produce": "produces",
    "produces": "produces",
    "develop": "develops",
    "develops": "develops",
    "publish": "publishes",
    "publishes": "publishes",
    "manufacture": "produces",
    "manufactures": "produces",
}


def analyze(question: str) -> AnalyzedEntityQuestion:
    """Parse one entity QA question deterministically."""
    q = question.strip()

    # 1. Current / volatile queries → unsupported
    if _CURRENT_SIGNALS.search(q):
        return AnalyzedEntityQuestion(
            question=q,
            intent="unknown_or_unsupported",
            subject=_extract_subject_from_current(q),
            predicate_hint=None,
            secondary_entity=None,
            source_hint=None,
            is_current_query=True,
            is_unsupported=True,
        )

    # 2. Personal unsupported
    if _PERSONAL_SIGNALS.search(q):
        return AnalyzedEntityQuestion(
            question=q,
            intent="unknown_or_unsupported",
            subject=None,
            predicate_hint=None,
            secondary_entity=None,
            source_hint=None,
            is_current_query=False,
            is_unsupported=True,
        )

    # 3. Source fact — estimate
    m = _ESTIMATE_RE.search(q)
    if m:
        return AnalyzedEntityQuestion(
            question=q,
            intent="source_fact_lookup",
            subject=_clean(m.group(2)),
            predicate_hint="estimated_net_worth",
            secondary_entity=None,
            source_hint=_clean(m.group(1)),
            is_current_query=False,
            is_unsupported=False,
        )

    # 4. Source fact — is net worth stable?
    m = _STABLE_FACT_RE.search(q)
    if m:
        return AnalyzedEntityQuestion(
            question=q,
            intent="source_fact_lookup",
            subject=_clean(m.group(1)),
            predicate_hint="stability_check",
            secondary_entity=None,
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
        )

    # 5. Source fact — recheck
    m = _RECHECK_RE.search(q)
    if m:
        return AnalyzedEntityQuestion(
            question=q,
            intent="source_fact_lookup",
            subject=_clean(m.group(1)),
            predicate_hint="recheck_reason",
            secondary_entity=None,
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
        )

    # 6. Link explanation
    m = _LINK_EXPL_RE.search(q)
    if m:
        return AnalyzedEntityQuestion(
            question=q,
            intent="link_explanation",
            subject=_clean(m.group(2)),
            predicate_hint=None,
            secondary_entity=_clean(m.group(1)),
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
        )

    # 7. Relation — known_for
    m = _KNOWN_FOR_RE.search(q)
    if m:
        return AnalyzedEntityQuestion(
            question=q,
            intent="relation_lookup",
            subject=_clean(m.group(1)),
            predicate_hint="known_for",
            secondary_entity=None,
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
        )

    # 8. Relation — leader
    m = _LEADER_RE.search(q)
    if m:
        return AnalyzedEntityQuestion(
            question=q,
            intent="relation_lookup",
            subject=_clean(m.group(1)),
            predicate_hint="leader_of",
            secondary_entity=None,
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
        )

    # 9. Relation — produces/develops/publishes
    m = _PRODUCES_RE.search(q)
    if m:
        verb = m.group(2).lower()
        predicate = _VERB_PREDICATE.get(verb, "produces")
        return AnalyzedEntityQuestion(
            question=q,
            intent="relation_lookup",
            subject=_clean(m.group(1)),
            predicate_hint=predicate,
            secondary_entity=None,
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
        )

    # 10. Relation — founded by
    m = _FOUNDED_BY_RE.search(q)
    if m:
        return AnalyzedEntityQuestion(
            question=q,
            intent="relation_lookup",
            subject=_clean(m.group(1)),
            predicate_hint="founded",
            secondary_entity=None,
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
        )

    # 11. Define entity — who/what is X
    m = _WHO_IS_RE.match(q)
    if m:
        return AnalyzedEntityQuestion(
            question=q,
            intent="define_entity",
            subject=_clean(m.group(1)),
            predicate_hint=None,
            secondary_entity=None,
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
        )

    m = _WHAT_IS_RE.match(q)
    if m:
        return AnalyzedEntityQuestion(
            question=q,
            intent="define_entity",
            subject=_clean(m.group(1)),
            predicate_hint=None,
            secondary_entity=None,
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
        )

    # Default: unsupported
    return AnalyzedEntityQuestion(
        question=q,
        intent="unknown_or_unsupported",
        subject=None,
        predicate_hint=None,
        secondary_entity=None,
        source_hint=None,
        is_current_query=False,
        is_unsupported=True,
    )


def _clean(s: str) -> str:
    return s.strip().rstrip("?.").strip()


def _extract_subject_from_current(q: str) -> Optional[str]:
    m = re.search(r"(?:of|for|about)\s+(.+?)(?:\?|$)", q, re.IGNORECASE)
    if m:
        return _clean(m.group(1))
    return None
