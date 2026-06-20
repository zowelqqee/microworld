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

from worldpgt.entity_qa.semantic_question_parser import parse_semantic_query
from worldpgt.entity_qa.types import AnalyzedEntityQuestion, EntityQAIntent, SemanticQuery

# ---- current / volatile query signals ---------------------------------
_CURRENT_SIGNALS = re.compile(
    r"\b(current\s+ceo|current\s+stock\s+price|current\s+valuation|"
    r"current\s+market\s+cap|current\s+price|stock\s+price|"
    r"current\s+president|current\s+office|current\s+net\s+worth)\b",
    re.IGNORECASE,
)

_PERSONAL_SIGNALS = re.compile(
    r"\b(favorite\s+food|favorite\s+color|favorite\s+hobby|personal\s+life|"
    r"home\s+address|phone\s+number|date\s+of\s+birth|age\s+of|"
    r"password|private\s+email|personal\s+email|social\s+security)\b",
    re.IGNORECASE,
)

# ---- volatile / time-relative query signals --------------------------
# These force an audit: the overlay has no live/real-time data, and these
# phrasings ask for "now/today/latest" values not present as accepted facts.
_VOLATILE_SIGNALS = re.compile(
    r"\b(right\s+now|today|latest|this\s+year|this\s+quarter|"
    r"latest\s+quarterly|latest\s+ranking|worth\s+right\s+now|"
    r"doing\s+today|live\s+right\s+now)\b",
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

# NEW source-fact paraphrases ------------------------------------------
_NET_WORTH_ESTIMATE_RE = re.compile(
    r"(?:what is|what's)\s+(.+?)'s\s+(?:source-qualified\s+)?net\s+worth\s+estimate",
    re.IGNORECASE,
)
_ACCORDING_ESTIMATE_RE = re.compile(
    r"according\s+to\s+(\w+),?\s+what\s+is\s+(.+?)'s\s+estimated\s+net\s+worth",
    re.IGNORECASE,
)
_IS_VOLATILE_RE = re.compile(
    r"is\s+(.+?)'s\s+net\s+worth\s+volatile",
    re.IGNORECASE,
)
_REQUIRE_RECHECK_RE = re.compile(
    r"(?:does\s+the\s+overlay\s+)?require[s]?\s+recheck(?:ing)?\s+(.+?)'s\s+net\s+worth",
    re.IGNORECASE,
)

# ---- weak-link policy questions --------------------------------------
# Meta questions about how the overlay treats wiki / weak context links.
_WEAK_LINK_POLICY_RE = re.compile(
    r"(is\s+a\s+wiki\s+link\s+treated\s+as\s+a\s+fact|"
    r"does\s+the\s+overlay\s+treat\s+weak\s+context\s+links\s+as\s+facts|"
    r"does\s+a\s+weak\s+link\s+mean\s+the\s+claim\s+is\s+true|"
    r"does\s+a\s+wiki\s+link\s+mean\s+the\s+claim\s+is\s+true|"
    r"are\s+weak\s+context\s+links\s+treated\s+as\s+facts|"
    r"are\s+weak\s+context\s+links\s+facts|"
    r"is\s+a\s+weak\s+link\s+a\s+fact)",
    re.IGNORECASE,
)

# ---- source-qualified volatility meta-questions (adversarial) ---------
# Narrow patterns about THE Forbes net-worth estimate already in the overlay.
# They must yield the volatile / source-qualified / recheck caveat — never a
# "stable" or "exact/current" claim.
_TREAT_AS_STABLE_RE = re.compile(
    r"(?:can|may|should)\s+i\s+treat\s+the\s+forbes\s+estimate\s+as\s+a\s+stable\s+fact",
    re.IGNORECASE,
)
_RECHECK_ESTIMATE_RE = re.compile(
    r"(?:does\s+the\s+overlay\s+)?need\s+to\s+recheck\s+the\s+forbes\s+estimate",
    re.IGNORECASE,
)
_IS_SOURCE_QUALIFIED_RE = re.compile(
    r"is\s+the\s+forbes\s+estimate\s+source-qualified",
    re.IGNORECASE,
)

# ---- link explanation -----------------------------------------------
_LINK_EXPL_RE = re.compile(
    r"why\s+(?:is|are)\s+(.+?)\s+linked\s+to\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)
_LINK_APPEAR_RE = re.compile(
    r"why\s+does\s+(.+?)\s+appear\s+on\s+the\s+(.+?)\s+page\b",
    re.IGNORECASE,
)
_IS_STABLE_REL_RE = re.compile(
    r"is\s+(.+?)\s+a\s+stable\s+factual\s+relation\s+for\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)

_RELATION_ASSERTION_RE = re.compile(
    r"^(?:did|does|is|was|were)\s+.+?\b("
    r"found|founded|founder\s+of|creator\s+of|created|lead|leads|led|"
    r"own|owns|owned\s+by|develop|develops|produce|produces|publish|publishes|"
    r"build|builds|make|makes"
    r")\b.+?[\?\.]?$",
    re.IGNORECASE,
)
_CONDITIONAL_RELATION_ASSERTION_RE = re.compile(
    r"^\s*if\b.+?\b(found|founded|lead|leads|own|owns|develops?|produces?|"
    r"publishes?|makes?|builds?|known\s+for)\b",
    re.IGNORECASE,
)
_WEAK_LINK_ASSERTION_RE = re.compile(
    r"^\s*because\b.+?\blinked\b.+?\b(lead|leads|own|owns|found|founded|company)\b",
    re.IGNORECASE,
)
_DECLARATIVE_RELATION_ASSERTION_RE = re.compile(
    r"^(?!\s*(?:who|what|which|how|why|is|was|were|does|did)\b).+?\b("
    r"found|founded|lead|leads|own|owns|develop|develops|produce|produces|"
    r"publish|publishes|make|makes|build|builds"
    r")\b.+[\?\.]?$",
    re.IGNORECASE,
)

# ---- relation lookup ------------------------------------------------
_KNOWN_FOR_RE = re.compile(
    r"what\s+is\s+(.+?)\s+known\s+for\b",
    re.IGNORECASE,
)
_KNOWN_FOR2_RE = re.compile(
    r"what\s+(?:companies|organizations|things|products)\s+(?:is|are)\s+(.+?)\s+known\s+for\b",
    re.IGNORECASE,
)
_LEADER_RE = re.compile(
    r"what\s+companies\s+is\s+(.+?)\s+linked\s+to\s+by\s+leadership\b",
    re.IGNORECASE,
)
_LEADER2_RE = re.compile(
    r"(?:what|which)\s+(?:companies|organizations)\s+(?:is|are)\s+(.+?)\s+linked\s+to\s+(?:by|through)\s+leadership\b",
    re.IGNORECASE,
)
_CONNECTED_RE = re.compile(
    r"how\s+(?:is|are)\s+(.+?)\s+(?:connected|related|linked)\s+to\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)
_PRODUCES_RE = re.compile(
    r"what\s+does\s+(.+?)\s+(produce|develop|publish|manufacture)s?\b",
    re.IGNORECASE,
)
_REPORT_PUBLISH_RE = re.compile(
    r"(?:which|what)\s+reports?\s+does\s+(.+?)\s+publish(?:\s+annually)?[\?.]?$",
    re.IGNORECASE,
)
_PRODUCTS_MAKE_RE = re.compile(
    r"what\s+products?\s+does\s+(.+?)\s+(?:make|produce|manufacture|build)s?\b",
    re.IGNORECASE,
)
_FOUNDED_BY_RE = re.compile(
    r"who\s+founded\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)
_WAS_FOUNDED_BY_RE = re.compile(
    r"who\s+was\s+(.+?)\s+founded\s+by[\?.]?$",
    re.IGNORECASE,
)
_FOUNDER_OF_RE = re.compile(
    r"who\s+(?:is|are|was|were)\s+the\s+founders?\s+of\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)
_FOUNDERS_OF_RE = re.compile(
    r"who\s+(?:were|are|was)\s+(.+?)'s\s+founders?\b",
    re.IGNORECASE,
)
_FOUND_SUBJECT_RE = re.compile(
    r"(?:what|which)\s+(?:company\s+)?did\s+(.+?)\s+found[\?.]?$",
    re.IGNORECASE,
)
_OWNS_RE = re.compile(
    r"(?:who|what\s+company)\s+owns\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)
_OWNED_BY_RE = re.compile(
    r"who\s+is\s+(.+?)\s+owned\s+by[\?.]?$",
    re.IGNORECASE,
)
_IS_A_CLASS_RE = re.compile(
    r"^is\s+(.+?)\s+an?\s+(.+?)[\?.]?$",
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
_DEFINE_TELL_RE = re.compile(
    r"^tell\s+me\s+(?:about|who)\s+(.+?)(?:\s+is)?[\?.]?$",
    re.IGNORECASE,
)
_DEFINE_KIND_RE = re.compile(
    r"^what\s+(?:kind|type|sort)\s+of\s+.+?\s+is\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)
_DEFINE_RE = re.compile(
    r"^define\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)

# ---- passive develops ("What is developed by X?") -------------------
_PASSIVE_DEVELOPS_RE = re.compile(
    r"^what\s+(?:is|was)\s+(developed|produced|published)\s+by\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)

_PASSIVE_VERB_PREDICATE: dict[str, str] = {
    "developed": "develops",
    "produced": "produces",
    "published": "publishes",
}

_SEMANTIC_HIGH_CONFIDENCE = 0.75

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

    # 1b. Volatile / time-relative queries (now/today/latest) → unsupported
    if _VOLATILE_SIGNALS.search(q):
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

    # 5b. Source fact — "X's (source-qualified) net worth estimate"
    m = _NET_WORTH_ESTIMATE_RE.search(q)
    if m:
        return AnalyzedEntityQuestion(
            question=q,
            intent="source_fact_lookup",
            subject=_clean(m.group(1)),
            predicate_hint="estimated_net_worth",
            secondary_entity=None,
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
        )

    # 5c. Source fact — "According to <source>, what is X's estimated net worth"
    m = _ACCORDING_ESTIMATE_RE.search(q)
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

    # 5d. Source fact — "Is X's net worth volatile?"
    m = _IS_VOLATILE_RE.search(q)
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

    # 5e. Source fact — "require rechecking X's net worth"
    m = _REQUIRE_RECHECK_RE.search(q)
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

    # 5g. Source-qualified volatility — "treat the Forbes estimate as a stable fact?"
    if _TREAT_AS_STABLE_RE.search(q):
        return AnalyzedEntityQuestion(
            question=q,
            intent="source_fact_lookup",
            subject="Elon Musk",
            predicate_hint="stability_check",
            secondary_entity=None,
            source_hint="Forbes",
            is_current_query=False,
            is_unsupported=False,
        )

    # 5h. Source-qualified volatility — "need to recheck the Forbes estimate?"
    if _RECHECK_ESTIMATE_RE.search(q):
        return AnalyzedEntityQuestion(
            question=q,
            intent="source_fact_lookup",
            subject="Elon Musk",
            predicate_hint="recheck_reason",
            secondary_entity=None,
            source_hint="Forbes",
            is_current_query=False,
            is_unsupported=False,
        )

    # 5i. Source-qualified volatility — "is the Forbes estimate source-qualified?"
    if _IS_SOURCE_QUALIFIED_RE.search(q):
        return AnalyzedEntityQuestion(
            question=q,
            intent="source_fact_lookup",
            subject="Elon Musk",
            predicate_hint="source_qualified_confirm",
            secondary_entity=None,
            source_hint="Forbes",
            is_current_query=False,
            is_unsupported=False,
        )

    # 5f. Weak-link policy meta-question
    if _WEAK_LINK_POLICY_RE.search(q):
        return AnalyzedEntityQuestion(
            question=q,
            intent="link_explanation",
            subject=None,
            predicate_hint="weak_link_policy",
            secondary_entity=None,
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
        )

    m = _IS_STABLE_REL_RE.search(q)
    if m:
        return AnalyzedEntityQuestion(
            question=q,
            intent="link_explanation",
            subject=_clean(m.group(2)),
            predicate_hint="stability_relation_check",
            secondary_entity=_clean(m.group(1)),
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
        )

    if (
        _RELATION_ASSERTION_RE.search(q)
        or _CONDITIONAL_RELATION_ASSERTION_RE.search(q)
        or _WEAK_LINK_ASSERTION_RE.search(q)
        or _DECLARATIVE_RELATION_ASSERTION_RE.search(q)
    ):
        return AnalyzedEntityQuestion(
            question=q,
            intent="unknown_or_unsupported",
            subject=None,
            predicate_hint="relation_assertion_not_lookup",
            secondary_entity=None,
            source_hint=None,
            is_current_query=False,
            is_unsupported=True,
        )

    semantic = parse_semantic_query(q)
    semantic_analyzed = _analyze_semantic(q, semantic)
    if semantic_analyzed is not None:
        return semantic_analyzed

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

    # 6b. Link explanation — "Why does Y appear on the X page?"
    m = _LINK_APPEAR_RE.search(q)
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

    # 6c. Link explanation — "Is Y a stable factual relation for X?"
    m = _IS_STABLE_REL_RE.search(q)
    if m:
        return AnalyzedEntityQuestion(
            question=q,
            intent="link_explanation",
            subject=_clean(m.group(2)),
            predicate_hint="stability_relation_check",
            secondary_entity=_clean(m.group(1)),
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
        )

    # 7. Relation — known_for
    m = _KNOWN_FOR2_RE.search(q) or _KNOWN_FOR_RE.search(q)
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
    m = _LEADER2_RE.search(q) or _LEADER_RE.search(q)
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

    # 8b. Relation — "How is X connected to Y?"
    m = _CONNECTED_RE.search(q)
    if m:
        return AnalyzedEntityQuestion(
            question=q,
            intent="relation_lookup",
            subject=_clean(m.group(1)),
            predicate_hint="connection",
            secondary_entity=_clean(m.group(2)),
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
        )

    # 9. Relation — produces/develops/publishes
    m = _REPORT_PUBLISH_RE.search(q)
    if m:
        return AnalyzedEntityQuestion(
            question=q,
            intent="relation_lookup",
            subject=_clean(m.group(1)),
            predicate_hint="publishes",
            secondary_entity=None,
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
        )

    m = _PRODUCTS_MAKE_RE.search(q)
    if m:
        return AnalyzedEntityQuestion(
            question=q,
            intent="relation_lookup",
            subject=_clean(m.group(1)),
            predicate_hint="produces",
            secondary_entity=None,
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
        )

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

    # 10. Relation — founded by / founder of / X's founders
    m = (
        _FOUNDER_OF_RE.search(q)
        or _FOUNDERS_OF_RE.search(q)
        or _FOUNDED_BY_RE.search(q)
        or _WAS_FOUNDED_BY_RE.search(q)
    )
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

    m = _FOUND_SUBJECT_RE.search(q)
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

    m = _OWNS_RE.search(q) or _OWNED_BY_RE.search(q)
    if m:
        return AnalyzedEntityQuestion(
            question=q,
            intent="relation_lookup",
            subject=_clean(m.group(1)),
            predicate_hint="owned_by",
            secondary_entity=None,
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
        )

    # 10a. Class membership — "Is X a/an Y?"
    m = _IS_A_CLASS_RE.match(q)
    if m:
        return AnalyzedEntityQuestion(
            question=q,
            intent="relation_lookup",
            subject=_clean(m.group(1)),
            predicate_hint="is_a",
            secondary_entity=_clean(m.group(2)),
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
        )

    # 10b. Define entity — paraphrases ("Define X", "Tell me about X", "What kind of ... is X")
    m = _DEFINE_RE.match(q) or _DEFINE_KIND_RE.match(q) or _DEFINE_TELL_RE.match(q)
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

    # 10c. Passive develops — "What is developed by X?" (must precede _WHAT_IS_RE)
    m = _PASSIVE_DEVELOPS_RE.match(q)
    if m:
        verb = m.group(1).lower()
        predicate = _PASSIVE_VERB_PREDICATE.get(verb, "develops")
        return AnalyzedEntityQuestion(
            question=q,
            intent="relation_lookup",
            subject=_clean(m.group(2)),
            predicate_hint=predicate,
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
        predicate_hint="question_not_understood",
        secondary_entity=None,
        source_hint=None,
        is_current_query=False,
        is_unsupported=True,
        semantic_query=semantic,
    )


def _clean(s: str) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", s.strip().rstrip("?.").strip(), flags=re.IGNORECASE)


def _extract_subject_from_current(q: str) -> Optional[str]:
    m = re.search(r"(?:of|for|about)\s+(.+?)(?:\?|$)", q, re.IGNORECASE)
    if m:
        return _clean(m.group(1))
    return None


def _analyze_semantic(
    question: str,
    semantic: SemanticQuery,
) -> Optional[AnalyzedEntityQuestion]:
    if semantic.confidence < _SEMANTIC_HIGH_CONFIDENCE:
        return None

    if semantic.query_type == "definition" and semantic.entity_a:
        return AnalyzedEntityQuestion(
            question=question,
            intent="define_entity",
            subject=semantic.entity_a,
            predicate_hint=None,
            secondary_entity=None,
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
            semantic_query=semantic,
        )

    if semantic.query_type == "is_a" and semantic.entity_a and semantic.entity_b:
        return AnalyzedEntityQuestion(
            question=question,
            intent="relation_lookup",
            subject=semantic.entity_a,
            predicate_hint="is_a",
            secondary_entity=semantic.entity_b,
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
            semantic_query=semantic,
        )

    if semantic.unknown_position == "path" and semantic.entity_a and semantic.entity_b:
        return AnalyzedEntityQuestion(
            question=question,
            intent="relation_lookup",
            subject=semantic.entity_a,
            predicate_hint="connection",
            secondary_entity=semantic.entity_b,
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
            semantic_query=semantic,
        )

    if semantic.unknown_position == "intersection":
        return AnalyzedEntityQuestion(
            question=question,
            intent="relation_lookup",
            subject=semantic.entity_a,
            predicate_hint="intersection",
            secondary_entity=semantic.entity_b,
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
            semantic_query=semantic,
        )

    if semantic.relation_intent and semantic.entity_a:
        predicate_hint = semantic.relation_intent
        if predicate_hint == "founded_by" and question.lower().startswith("who founded "):
            predicate_hint = "founded"
        if predicate_hint == "founded_by" and re.match(
            r"^(?:what|which)\b.*\bdid\s+.+?\s+found\b",
            question,
            re.IGNORECASE,
        ):
            predicate_hint = "founded"
        return AnalyzedEntityQuestion(
            question=question,
            intent="relation_lookup",
            subject=semantic.entity_a,
            predicate_hint=predicate_hint,
            secondary_entity=semantic.entity_b,
            source_hint=None,
            is_current_query=False,
            is_unsupported=False,
            semantic_query=semantic,
        )

    return None
