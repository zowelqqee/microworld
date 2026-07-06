"""Deterministic, conservative question router for the Assistant Surface v1.

Routes a user question into exactly one assistant intent. Rule-based regex only.
No ML, no embeddings, no network. The router is the FIRST safety gate: it
classifies hard-safety questions (current/live, private, relation inversion,
unsupported universal) before any answerable intent, so these can never reach an
answer path.

Examples::

    Who is Elon Musk?                                   -> entity_definition
    What does SpaceX develop?                           -> entity_relation
    How is Elon Musk connected to rockets?              -> connection_path
    What does Forbes estimate about Elon Musk?          -> source_qualified_fact
    Why is Forbes linked to Elon Musk?                  -> weak_link_policy
    What is Tesla's current stock price?                -> current_live_request
    What is Elon Musk's private email?                  -> private_sensitive_request
    Did SpaceX found Elon Musk?                         -> relation_inversion
    If SpaceX makes rockets, are all rockets SpaceX...? -> unsupported_universal
"""

from __future__ import annotations

import re

from worldpgt.assistant_surface.types import AssistantRoute
from worldpgt.entity_qa.semantic_question_parser import parse_semantic_query
from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex

# --------------------------------------------------------------------------- #
# Hard-safety signals (checked first, in priority order).
# --------------------------------------------------------------------------- #
_PRIVATE_RE = re.compile(
    r"\b(private\s+email|personal\s+email|phone\s+number|home\s+address|"
    r"private\s+address|date\s+of\s+birth|social\s+security|passport|"
    r"favou?rite\s+(?:food|color|colour|hobby)|personal\s+life)\b",
    re.IGNORECASE,
)

_CURRENT_LIVE_RE = re.compile(
    r"\b(current\s+stock\s+price|stock\s+price|share\s+price|current\s+price|"
    r"current\s+valuation|current\s+market\s+cap|live\s+market\s+cap|"
    r"market\s+cap(?:italization)?|worth\s+right\s+now|right\s+now|today|"
    r"current\s+net\s+worth|"
    r"latest\s+quarterly\s+revenue|latest\s+revenue|quarterly\s+revenue|"
    r"current\s+ceo|current\s+president|current\s+prime\s+minister|"
    r"current\s+mayor|currently|latest\s+ranking|live\s+right\s+now|"
    r"permanently\b.*\bnet\s+worth|net\s+worth\b.*\bpermanently)\b",
    re.IGNORECASE,
)

# "Tell me the current CEO of every company" — universal + current. The
# universal screen below also catches it; current takes priority.
_UNIVERSAL_RE = re.compile(
    r"(\bif\b.+?\b(?:are|is|does|makes?)\b.+?\b(?:all|every|any|each)\b|"
    r"\bare\s+all\b|\bis\s+every\b|\bdoes\s+every\b|\bare\s+any\b|"
    r"\bof\s+every\s+\w+\b|\beach\s+(?:company|entity|product)\b)",
    re.IGNORECASE,
)

# Relation inversion: an action verb applied in the wrong direction, or an
# unsupported ownership assertion. e.g. "Did SpaceX found Elon Musk?",
# "Is Forbes owned by Elon Musk?".
_INVERSION_RE = re.compile(
    r"\b(did|does|is|was|were)\s+(.+?)\s+"
    r"(found|founded|the\s+founder\s+of|create[ds]?|own(?:ed|s)?\s+by|"
    r"owned\s+by|lead|leads|led)\s+(.+?)[\?\.]*\s*$",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Policy / meta questions.
# --------------------------------------------------------------------------- #
_WEAK_POLICY_DEF_RE = re.compile(
    r"(what\s+does\s+(?:a\s+)?weak\s+(?:context\s+)?link\s+mean|"
    r"what\s+is\s+a\s+weak\s+(?:context\s+)?link|"
    r"what\s+does\s+source[\s-]qualified\s+mean|"
    r"what\s+is\s+a\s+source[\s-]qualified\s+fact|"
    r"are\s+weak\s+(?:context\s+)?links?\s+(?:treated\s+as\s+)?facts?|"
    r"does\s+a\s+weak\s+link\s+mean)",
    re.IGNORECASE,
)
_LINK_EXPLAIN_RE = re.compile(
    r"(why\s+(?:is|are)\s+(.+?)\s+linked\s+to\s+(.+?)[\?\.]?$|"
    r"why\s+does\s+(.+?)\s+appear\s+on\s+the\s+(.+?)\s+page)",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Source-qualified facts (Forbes estimate, estimated net worth).
# --------------------------------------------------------------------------- #
_SOURCE_FACT_RE = re.compile(
    r"(according\s+to\s+(\w+)|what\s+does\s+(\w+)\s+estimate|"
    r"\bestimate[ds]?\s+about\b|estimated\s+net\s+worth|"
    r"source[\s-]qualified\s+net\s+worth|net\s+worth\s+estimate)",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Connection / path.
# --------------------------------------------------------------------------- #
_CONNECTION_RE = re.compile(
    r"(how\s+(?:is|are|was|were)\s+(.+?)\s+(?:connected|related|linked)\s+to\s+(.+?)[\?\.]?$|"
    r"what\s+path\s+connects\s+(.+?)\s+(?:to|and|with)\s+(.+?)[\?\.]?$|"
    r"what\s+connects\s+(.+?)\s+(?:to|and|with)\s+(.+?)[\?\.]?$)",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Entity relation lookups.
# --------------------------------------------------------------------------- #
_RELATION_RE = re.compile(
    r"(what\s+does\s+(.+?)\s+(?:develop|produce|publish|manufacture|make|build|"
    r"require|need|allow|permit|prohibit|forbid)s?\b|"
    r"where\s+(?:is|are|was|were)\s+(.+?)(?:\s+located)?[\?\.]?$|"
    r"(?:which|what)\s+reports?\s+does\s+(.+?)\s+publish(?:\s+annually)?[\?\.]?$|"
    r"what\s+products?\s+does\s+(.+?)\s+(?:make|produce|manufacture|build)s?\b|"
    r"what\s+(?:is|was)\s+developed\s+by\s+(.+?)[\?\.]?$|"
    r"what\s+(?:is|was)\s+produced\s+by\s+(.+?)[\?\.]?$|"
    r"what\s+(?:is|was)\s+published\s+by\s+(.+?)[\?\.]?$|"
    r"what\s+(?:did|does)\s+(.+?)\s+develop[\?\.]?$|"
    r"what\s+(?:did|does)\s+(.+?)\s+produce[\?\.]?$|"
    r"who\s+founded\s+(.+?)[\?\.]?$|"
    r"who\s+was\s+(.+?)\s+founded\s+by[\?\.]?$|"
    r"who\s+(?:is|are|was|were)\s+the\s+founders?\s+of\s+(.+?)[\?\.]?$|"
    r"(?:who|what\s+company)\s+owns\s+(.+?)[\?\.]?$|"
    r"who\s+is\s+(.+?)\s+owned\s+by[\?\.]?$|"
    r"(?:what|which)\s+(?:company\s+)?did\s+(.+?)\s+found[\?\.]?$|"
    r"what\s+(?:is|was)\s+(.+?)\s+(?:known|famous)\s+for\b|"
    r"what\s+companies\s+is\s+(.+?)\s+linked\s+to\s+by\s+leadership\b)",
    re.IGNORECASE,
)
_IS_A_CLASS_RE = re.compile(
    r"^is\s+(.+?)\s+an?\s+(.+?)[\?\.]?$",
    re.IGNORECASE,
)
_UNSUPPORTED_RELATION_TAIL_RE = re.compile(
    r"^(?:who|what)\s+(?:is|are|was|were)\s+.+?\s+"
    r"(?:married\s+to|succeeded\s+by|preceded\s+by|born\s+(?:in|on)|"
    r"died\s+(?:in|on))[\?\.]?$",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Entity definition.
# --------------------------------------------------------------------------- #
_DEFINITION_RE = re.compile(
    r"(who\s+qualifies\s+for\s+(.+?)[\?\.]?$|"
    r"who\s+(?:is|are)\s+eligible\s+for\s+(.+?)[\?\.]?$|"
    r"who\s+(?:is|are|was|were)\s+(.+?)[\?\.]?$|"
    r"what\s+(?:is|are)\s+(.+?)[\?\.]?$|"
    r"tell\s+me\s+about\s+(.+?)[\?\.]?$|"
    r"define\s+(.+?)[\?\.]?$|"
    r"what\s+(?:kind|type|sort)\s+of\s+\S+.*?\s+is\s+(.+?)[\?\.]?$)",
    re.IGNORECASE,
)


def _first_group(m: re.Match, *idxs: int) -> str | None:
    for i in idxs:
        try:
            g = m.group(i)
        except (IndexError, re.error):
            g = None
        if g:
            return g.strip()
    return None


def route(question: str, index: EntitySurfaceIndex | None = None) -> AssistantRoute:
    """Classify ``question`` into exactly one conservative assistant intent."""

    q = (question or "").strip()
    risk: list[str] = []

    # ---- 1. Hard-safety screens (highest priority) -------------------- #
    if _PRIVATE_RE.search(q):
        risk.append("private_sensitive")
        return AssistantRoute(
            question=q,
            intent="private_sensitive_request",
            is_hard_safety=True,
            risk_flags=risk,
            notes="private/sensitive information request",
        )

    if _CURRENT_LIVE_RE.search(q):
        risk.append("current_live")
        return AssistantRoute(
            question=q,
            intent="current_live_request",
            is_hard_safety=True,
            risk_flags=risk,
            notes="current/live/volatile data request",
        )

    if _UNIVERSAL_RE.search(q):
        risk.append("unsupported_universal")
        return AssistantRoute(
            question=q,
            intent="unsupported_universal",
            is_hard_safety=True,
            risk_flags=risk,
            notes="unsupported universal generalization",
        )

    passive_relation_question = re.match(
        r"who\s+(?:is|was|were)\s+.+?\s+(?:founded|owned)\s+by[\?\.]?$",
        q,
        re.IGNORECASE,
    )
    inv = None if passive_relation_question else _INVERSION_RE.search(q)
    if inv:
        risk.append("relation_inversion")
        return AssistantRoute(
            question=q,
            intent="relation_inversion",
            subject=inv.group(2).strip(),
            secondary=inv.group(4).strip(),
            is_hard_safety=True,
            risk_flags=risk,
            notes="reversed/unsupported directional relation",
        )

    # ---- 2. Policy / meta questions ---------------------------------- #
    if _WEAK_POLICY_DEF_RE.search(q):
        return AssistantRoute(
            question=q,
            intent="weak_link_policy",
            notes="weak-link / source-qualified policy definition",
        )

    le = _LINK_EXPLAIN_RE.search(q)
    if le:
        subj = _first_group(le, 2, 4)
        sec = _first_group(le, 3, 5)
        return AssistantRoute(
            question=q,
            intent="weak_link_policy",
            subject=subj,
            secondary=sec,
            notes="weak-link explanation request",
        )

    # ---- 3. Source-qualified facts ----------------------------------- #
    sf = _SOURCE_FACT_RE.search(q)
    if sf:
        src = _first_group(sf, 2, 3)
        return AssistantRoute(
            question=q,
            intent="source_qualified_fact",
            source_hint=src,
            notes="source-qualified (volatile) fact lookup",
        )

    semantic = parse_semantic_query(q, index)
    if semantic.confidence >= 0.75:
        if semantic.unknown_position == "path" and semantic.entity_a and semantic.entity_b:
            return AssistantRoute(
                question=q,
                intent="connection_path",
                subject=semantic.entity_a,
                secondary=semantic.entity_b,
                notes="semantic path query",
            )
        if semantic.query_type == "definition" and semantic.entity_a:
            return AssistantRoute(
                question=q,
                intent="entity_definition",
                subject=semantic.entity_a,
                notes="semantic definition query",
            )
        if semantic.query_type == "open_synthesis" and semantic.entity_a:
            return AssistantRoute(
                question=q,
                intent="entity_definition",
                subject=semantic.entity_a,
                notes="semantic open synthesis query",
            )
        if semantic.query_type == "comparative":
            return AssistantRoute(
                question=q,
                intent="entity_relation",
                subject=semantic.entity_a,
                secondary=semantic.entity_b,
                notes="semantic intersection query",
            )
        if semantic.relation_intent and semantic.entity_a:
            return AssistantRoute(
                question=q,
                intent="entity_relation",
                subject=semantic.entity_a,
                secondary=semantic.entity_b,
                notes="semantic relation query",
            )

    # ---- 4. Connection / path ---------------------------------------- #
    cn = _CONNECTION_RE.search(q)
    if cn:
        subj = _first_group(cn, 2, 4, 6)
        sec = _first_group(cn, 3, 5, 7)
        return AssistantRoute(
            question=q,
            intent="connection_path",
            subject=subj,
            secondary=sec,
            notes="connection / path question",
        )

    # ---- 5. Entity relation ------------------------------------------ #
    isa = _IS_A_CLASS_RE.match(q)
    if isa:
        return AssistantRoute(
            question=q,
            intent="entity_relation",
            subject=isa.group(1).strip(),
            secondary=isa.group(2).strip(),
            notes="entity class-membership lookup",
        )

    rl = _RELATION_RE.search(q)
    if rl:
        subj = _first_group(rl, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
        return AssistantRoute(
            question=q,
            intent="entity_relation",
            subject=subj,
            notes="entity relation lookup",
        )

    # ---- 6. Entity definition ---------------------------------------- #
    if _UNSUPPORTED_RELATION_TAIL_RE.search(q):
        return AssistantRoute(
            question=q,
            intent="unknown_or_unsupported",
            notes="unsupported relation lookup",
        )

    df = _DEFINITION_RE.search(q)
    if df:
        subj = _first_group(df, 2, 3, 4, 5, 6, 7, 8)
        return AssistantRoute(
            question=q,
            intent="entity_definition",
            subject=subj,
            notes="entity definition",
        )

    # ---- 7. Fallback ------------------------------------------------- #
    fallback_notes = "question not understood" if semantic.confidence < 0.35 else "no conservative intent matched"
    return AssistantRoute(
        question=q,
        intent="unknown_or_unsupported",
        notes=fallback_notes,
    )
