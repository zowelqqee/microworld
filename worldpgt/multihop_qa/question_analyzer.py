"""Multi-hop question analyzer for Multi-hop QA v1.

Recognizes explicit 2-hop connection question forms. Direct-relation forms
("Who founded X?", "Who owns X?") are classified as direct_only and must
audit immediately — they must never be answered via a transitive chain.

Rule-based regex only. No ML. No network.
"""
from __future__ import annotations

import re

from worldpgt.multihop_qa.types import MultihopQuestion

# ---- Develops + is_a chain ------------------------------------------------- #
_DEVELOPS_ISA_RE = re.compile(
    r"what\s+does\s+(.+?)\s+develop\s+that\s+is\s+(?:a|an)\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)

# ---- Through-node: "Is A connected to C through B?" ----------------------- #
_IS_CONNECTED_THROUGH_RE = re.compile(
    r"is\s+(.+?)\s+connected\s+to\s+(.+?)\s+through\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)

# ---- Through-node: "Who is A linked to through B?" ------------------------ #
_WHO_THROUGH_RE = re.compile(
    r"who\s+is\s+(.+?)\s+linked\s+to\s+through\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)

# ---- Connection forms ------------------------------------------------------ #
_CONNECTION_HOW_RE = re.compile(
    r"how\s+(?:is|are|was|were)\s+(.+?)\s+(?:connected|related|linked)\s+to\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)
_WHAT_LINKS_RE = re.compile(
    r"what\s+links\s+(.+?)\s+to\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)
_RELATIONSHIP_RE = re.compile(
    r"what\s+is\s+(.+?)\'?s\s+relationship\s+to\s+(.+?)[\?.]?$",
    re.IGNORECASE,
)

# ---- Direct-relation guards (must audit — transitive leak prevention) ------ #
# "Who founded X?" — single-hop entity QA must handle this, not multi-hop.
_DIRECT_FOUNDER_RE = re.compile(
    r"who\s+(?:founded|was\s+.+?\s+founded\s+by)\b",
    re.IGNORECASE,
)
# "Who was X founded by?" alternate form.
_DIRECT_FOUNDED_BY_RE = re.compile(
    r"who\s+was\s+.+?\s+founded\s+by[\?.]?$",
    re.IGNORECASE,
)
# "Who/what company owns X?"
_DIRECT_OWNER_RE = re.compile(
    r"(?:who|what(?:\s+company)?)\s+owns\s+.+?[\?.]?$",
    re.IGNORECASE,
)
# "Who is X owned by?"
_DIRECT_OWNED_BY_RE = re.compile(
    r"who\s+is\s+.+?\s+owned\s+by[\?.]?$",
    re.IGNORECASE,
)


def _clean(s: str) -> str:
    return s.strip().rstrip("?.").strip()


def analyze(question: str) -> MultihopQuestion:
    """Classify a question into a MultihopQuestion."""
    q = (question or "").strip()

    # 1. Direct-relation guards — must audit immediately; never use transitive
    #    chain to answer founder/owner questions.
    if _DIRECT_FOUNDER_RE.search(q) or _DIRECT_FOUNDED_BY_RE.search(q):
        return MultihopQuestion(q, "direct_only", is_direct_relation=True)
    if _DIRECT_OWNER_RE.search(q) or _DIRECT_OWNED_BY_RE.search(q):
        return MultihopQuestion(q, "direct_only", is_direct_relation=True)

    # 2. Develops + is_a chain.
    m = _DEVELOPS_ISA_RE.search(q)
    if m:
        return MultihopQuestion(
            q, "develops_isa",
            endpoint_a=_clean(m.group(1)),
            endpoint_c=_clean(m.group(2)),
        )

    # 3. Through-node: explicit B node.
    m = _IS_CONNECTED_THROUGH_RE.search(q)
    if m:
        return MultihopQuestion(
            q, "through_node",
            endpoint_a=_clean(m.group(1)),
            endpoint_c=_clean(m.group(2)),
            through_b=_clean(m.group(3)),
        )

    # 4. Who-through.
    m = _WHO_THROUGH_RE.search(q)
    if m:
        return MultihopQuestion(
            q, "who_through",
            endpoint_a=_clean(m.group(1)),
            through_b=_clean(m.group(2)),
        )

    # 5. Connection forms (checked after through-node so "through" isn't swallowed).
    m = _CONNECTION_HOW_RE.search(q)
    if m:
        return MultihopQuestion(
            q, "connection",
            endpoint_a=_clean(m.group(1)),
            endpoint_c=_clean(m.group(2)),
        )

    m = _WHAT_LINKS_RE.search(q)
    if m:
        return MultihopQuestion(
            q, "connection",
            endpoint_a=_clean(m.group(1)),
            endpoint_c=_clean(m.group(2)),
        )

    m = _RELATIONSHIP_RE.search(q)
    if m:
        return MultihopQuestion(
            q, "connection",
            endpoint_a=_clean(m.group(1)),
            endpoint_c=_clean(m.group(2)),
        )

    # 6. Unsupported — no recognized multi-hop pattern.
    return MultihopQuestion(q, "unsupported")
