"""Deterministic question analyzer for Cross-page Entity QA v1.

Routes a controlled cross-page question to one of five intents and extracts the
two endpoints for path/connection questions. Rule-based regex only. No ML, no
embeddings, no network.

Intents:
  connection_path          — "How is A connected to B?"
  relation_chain           — "What path connects A to B?"
  source_qualified_list    — "Which claims require rechecking?"
  weak_link_list_or_policy — "What does weak_context_only mean?"
  unsupported_or_too_far   — current/live, universal, "prove", or no path
"""

from __future__ import annotations

import re

from worldpgt.cross_page_qa.types import CrossPageQuestion

# ---- current / live signals (never answerable from the overlay) -------
_CURRENT_RE = re.compile(
    r"\b(current\s+stock\s+price|current\s+price|stock\s+price|current\s+valuation|"
    r"current\s+market\s+cap|market\s+cap(?:italization)?|latest\s+quarterly\s+revenue|"
    r"latest\s+revenue|current\s+ceo|right\s+now|today|latest\s+ranking|"
    r"stock\s+(?:price\s+)?today|worth\s+right\s+now)\b",
    re.IGNORECASE,
)

# ---- universal / generalization traps --------------------------------
_UNIVERSAL_RE = re.compile(
    r"(\bif\b.+?\b(?:are|is|does)\s+(?:all|every|any|each)\b|"
    r"\bare\s+all\b|\bis\s+every\b|\bdoes\s+every\b|\bare\s+any\b)",
    re.IGNORECASE,
)

# ---- weak-link policy / list -----------------------------------------
_WEAK_LINK_RE = re.compile(
    r"(weak\s+context\s+only|weak[_\s]context[_\s]only|weak\s+context\s+links?|"
    r"weak\s+links?|weak\s+contextual|context\s+links?\s+(?:are\s+)?facts|"
    r"are\s+wiki\s+context\s+links\s+facts|can\s+weak\s+links\s+prove|"
    r"examples?\s+of\s+weak\s+links|what\s+does\s+weak)",
    re.IGNORECASE,
)

# ---- source-qualified / volatile list --------------------------------
_SOURCE_QUALIFIED_RE = re.compile(
    r"(recheck|source[\s-]qualified|volatile\s+facts?|are\s+volatile|"
    r"which\s+.*\bvolatile\b|list\s+the\s+volatile|net\s+worth\s+(?:claims?|estimates?)|"
    r"which\s+facts\s+are|which\s+claims|which\s+overlay\s+facts|"
    r"which\s+net\s+worth)",
    re.IGNORECASE,
)

# ---- "prove" over-inference traps ------------------------------------
_PROVE_RE = re.compile(r"\bprove(?:s|d|n)?\b", re.IGNORECASE)

# ---- connection / relation-chain endpoint patterns -------------------
_CONNECTION_RE = re.compile(
    r"how\s+(?:is|are|was|were)\s+(.+?)\s+(?:connected|related|linked)\s+to\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)
_PATH_RE = re.compile(
    r"(?:what(?:\s+is)?\s+(?:the\s+)?path\s+(?:that\s+)?connect(?:s|ing)?|"
    r"what\s+connects|show\s+(?:me\s+)?the\s+path\s+(?:that\s+)?connect(?:s|ing)?|"
    r"trace\s+the\s+path\s+from)\s+(.+?)\s+(?:to|and|with)\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)
_PATH_FROM_RE = re.compile(
    r"(?:what(?:\s+is)?\s+(?:the\s+)?path\s+from)\s+(.+?)\s+to\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)


def _clean(s: str) -> str:
    return s.strip().rstrip("?.").strip()


def analyze(question: str) -> CrossPageQuestion:
    q = question.strip()

    # 1. Current / live data -> never answerable.
    if _CURRENT_RE.search(q):
        return CrossPageQuestion(q, "unsupported_or_too_far", None, None,
                                 is_current_query=True)

    # 2. Universal / generalization traps -> audit.
    if _UNIVERSAL_RE.search(q):
        return CrossPageQuestion(q, "unsupported_or_too_far", None, None,
                                 is_universal_query=True)

    # 3. Weak-link policy / list (checked before generic "prove").
    if _WEAK_LINK_RE.search(q):
        return CrossPageQuestion(q, "weak_link_list_or_policy", None, None)

    # 4. "prove" over-inference -> audit.
    if _PROVE_RE.search(q):
        return CrossPageQuestion(q, "unsupported_or_too_far", None, None)

    # 5. Source-qualified / volatile list.
    if _SOURCE_QUALIFIED_RE.search(q):
        return CrossPageQuestion(q, "source_qualified_list", None, None)

    # 6. Relation chain ("What path connects A to B?").
    m = _PATH_RE.search(q) or _PATH_FROM_RE.search(q)
    if m:
        return CrossPageQuestion(q, "relation_chain", _clean(m.group(1)), _clean(m.group(2)))

    # 7. Connection path ("How is A connected to B?").
    m = _CONNECTION_RE.search(q)
    if m:
        return CrossPageQuestion(q, "connection_path", _clean(m.group(1)), _clean(m.group(2)))

    # 8. Default: unsupported.
    return CrossPageQuestion(q, "unsupported_or_too_far", None, None)
