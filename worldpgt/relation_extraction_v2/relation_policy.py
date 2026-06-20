"""Relation stability and current-sensitivity policy.

Single source of truth for:

1. CURRENT_SENSITIVE_PREDICATES — predicates that must never be used as a
   bridge in a multi-hop inference chain.  Leadership, valuation, and ranking
   predicates reflect real-time organisational state; baking them into a chain
   answer is unsafe because the state may have changed by the time the user
   reads the result.

2. RENDERING_HEDGE_PREDICATES — predicates whose overlay values must be
   presented with explicit hedging in user-facing text (e.g. "linked through
   leadership" rather than a bare factual assertion).

3. VOLATILE_STABILITY_PREDICATES — relations whose *stability* field should
   be "volatile" when they appear in PatternSpec entries.  These are distinct
   from CURRENT_SENSITIVE_PREDICATES: a predicate can be volatile (changes
   frequently) without being current-sensitive for multi-hop purposes, though
   in practice the sets largely overlap.

Callers
-------
- worldpgt.multihop_qa.path_validator  uses CURRENT_SENSITIVE_PREDICATES
- worldpgt.entity_qa.entity_answer_renderer  uses RENDERING_HEDGE_PREDICATES

Policy rationale per predicate
-------------------------------
All relation types in the system are classified below. "stable", "semi-stable",
and "volatile" refer to how quickly the fact can change in the real world.
"current-sensitive" means the predicate is unsafe as a multi-hop bridge.

  founded_by          stable       NOT current-sensitive  (historical event)
  founded             stable       NOT current-sensitive  (historical event)
  develops            semi-stable  NOT current-sensitive  (core business activity)
  developed_by        semi-stable  NOT current-sensitive  (historical/ongoing)
  manufactures        semi-stable  NOT current-sensitive  (core business)
  produces            semi-stable  NOT current-sensitive  (core business)
  product_of          semi-stable  NOT current-sensitive  (stable product relation)
  owned_by            semi-stable  NOT current-sensitive  (changes via M&A over years)
  subsidiary_of       semi-stable  NOT current-sensitive  (changes via restructuring)
  parent_company_of   semi-stable  NOT current-sensitive  (same as owned_by)
  headquartered_in    semi-stable  NOT current-sensitive  (very rarely changes)
  industry            stable       NOT current-sensitive  (stable categorisation)
  operates            semi-stable  NOT current-sensitive  (long-term activity)
  created_by          stable       NOT current-sensitive  (historical event)
  published_by        semi-stable  NOT current-sensitive  (stable for established works)
  service_of          semi-stable  NOT current-sensitive  (stable platform relation)
  platform_of         semi-stable  NOT current-sensitive  (stable platform relation)
  uses                semi-stable  NOT current-sensitive  (changes slowly)
  part_of             semi-stable  NOT current-sensitive  (structural, changes rarely)
  is_a                stable       NOT current-sensitive  (definitional)
  type_of             stable       NOT current-sensitive  (definitional)
  leader_of           volatile     CURRENT-SENSITIVE      (CEO/head changes suddenly)
  valued_at           volatile     CURRENT-SENSITIVE      (financial valuation daily)
"""

from __future__ import annotations

import math
import re

# ---------------------------------------------------------------------------
# Current-sensitive predicates
# ---------------------------------------------------------------------------

# These predicates reflect real-time or near-real-time state.  They must not
# appear as a HOP inside a multi-hop inference chain.  The answer produced by
# such a chain would assert a fact derived from this predicate as if it were
# stable, but the underlying state may already have changed.
CURRENT_SENSITIVE_PREDICATES: frozenset[str] = frozenset({
    "leader_of",    # Who leads an org changes without notice (CEO departures, elections)
    "valued_at",    # Financial valuations fluctuate daily with market conditions
})

# ---------------------------------------------------------------------------
# Rendering-hedge predicates
# ---------------------------------------------------------------------------

# When rendering a user-facing answer, relations in this set must receive
# hedged/cautious language that signals the fact is volatile rather than stable.
# Example: "leader_of" → "linked to X through leadership" (not "leads X").
RENDERING_HEDGE_PREDICATES: frozenset[str] = frozenset({
    "leader_of",
    "valued_at",
})

# ---------------------------------------------------------------------------
# Volatile stability predicates
# ---------------------------------------------------------------------------

# Predicates that must carry stability="volatile" in any PatternSpec that
# extracts them.  This list is a superset policy guarantee: any new pattern
# for these predicates must use volatile stability.
VOLATILE_STABILITY_PREDICATES: frozenset[str] = frozenset({
    "leader_of",
    "valued_at",
})

# ---------------------------------------------------------------------------
# Freshness windows
# ---------------------------------------------------------------------------

INFINITE_FRESHNESS_DAYS = math.inf

FRESHNESS_WINDOWS_BY_RELATION: dict[str, float] = {
    "estimated_net_worth": 30,
    "net_worth": 30,
    "valued_at": 30,
    "current_price": 30,
    "market_capitalization": 30,
    "ranking": 90,
    "wealth_rank": 90,
    "population": 365,
    "current_population": 365,
    "owned_by": 180,
}

FRESHNESS_WINDOWS_BY_TEMPORAL_CLASS: dict[str, float] = {
    "historical": INFINITE_FRESHNESS_DAYS,
    "derived": INFINITE_FRESHNESS_DAYS,
    "snapshot": 90,
    "aggregate": 60,
}

# ---------------------------------------------------------------------------
# Semantic question parser relation keywords
# ---------------------------------------------------------------------------

# Short word/phrase lookup for mapping user question wording to canonical
# relation intents. Longest phrase wins; this is deliberately not a full
# sentence regex parser.
RELATION_KEYWORD_MAP: dict[str, str] = {
    # founding / creation
    "founded by": "founded_by",
    "founders of": "founded_by",
    "founder of": "founded_by",
    "founders": "founded_by",
    "co-founder": "founded_by",
    "cofounder": "founded_by",
    "started by": "founded_by",
    "started": "founded_by",
    "founded": "founded_by",
    "found": "founded_by",
    "created by": "created_by",
    "create": "founded_by",
    "created": "created_by",
    # ownership / corporate structure
    "owned by": "owned_by",
    "belongs to": "owned_by",
    "belong to": "owned_by",
    "acquired by": "owned_by",
    "owns": "owned_by",
    "own": "owned_by",
    "parent company of": "parent_company_of",
    "subsidiary of": "subsidiary_of",
    "subsidiary": "subsidiary_of",
    # development / production
    "developed by": "developed_by",
    "built by": "developed_by",
    "made by": "developed_by",
    "designed by": "developed_by",
    "manufactured by": "developed_by",
    "develop": "develops",
    "develops": "develops",
    "build": "develops",
    "builds": "develops",
    "make": "develops",
    "makes": "develops",
    "manufacture": "manufactures",
    "manufactures": "manufactures",
    "produce": "produces",
    "produces": "produces",
    "product of": "product_of",
    # publishing / media
    "published by": "published_by",
    "publish": "publishes",
    "publishes": "publishes",
    # leadership / volatile state
    "leader of": "leader_of",
    "led by": "leader_of",
    "leads": "leader_of",
    "lead": "leader_of",
    "run by": "leader_of",
    "runs": "leader_of",
    "run": "leader_of",
    "headed by": "leader_of",
    "heads": "leader_of",
    "head": "leader_of",
    "ceo of": "leader_of",
    "chief executive": "leader_of",
    # operations / location / categorisation
    "operated by": "operated_by",
    "operates": "operates",
    "operate": "operates",
    "headquartered in": "headquartered_in",
    "based in": "headquartered_in",
    "industry": "industry",
    "known for": "known_for",
    "service of": "service_of",
    "platform of": "platform_of",
    "uses": "uses",
    "use": "uses",
    "part of": "part_of",
    # type / class
    "type of": "type_of",
    "kind of": "type_of",
    "is a": "is_a",
    "is an": "is_a",
    # source-qualified snapshots
    "estimated net worth": "estimated_net_worth",
    "net worth estimate": "estimated_net_worth",
    "valued at": "valued_at",
    "valuation": "valued_at",
}

_RELATION_KEYWORDS_BY_LENGTH = tuple(
    sorted(RELATION_KEYWORD_MAP, key=lambda keyword: (-len(keyword), keyword))
)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_current_sensitive(predicate: str) -> bool:
    """Return True if this predicate is unsafe as a multi-hop bridge."""
    return predicate in CURRENT_SENSITIVE_PREDICATES


def should_hedge_render(predicate: str) -> bool:
    """Return True if this predicate's value must be rendered with hedged language."""
    return predicate in RENDERING_HEDGE_PREDICATES


def is_volatile_predicate(predicate: str) -> bool:
    """Return True if pattern specs for this predicate must use volatile stability."""
    return predicate in VOLATILE_STABILITY_PREDICATES


def freshness_window_days(predicate: str, temporal_class: str | None = None) -> float:
    """Return the freshness window in days for a relation/temporal class."""
    if temporal_class == "historical":
        return INFINITE_FRESHNESS_DAYS
    if predicate in FRESHNESS_WINDOWS_BY_RELATION:
        return FRESHNESS_WINDOWS_BY_RELATION[predicate]
    return FRESHNESS_WINDOWS_BY_TEMPORAL_CLASS.get(
        temporal_class or "",
        FRESHNESS_WINDOWS_BY_TEMPORAL_CLASS["snapshot"],
    )


def relation_intent_from_text(text: str) -> str | None:
    """Return a relation intent found in *text* via keyword/short-phrase lookup."""

    normalized = re.sub(r"\s+", " ", (text or "").lower().strip())
    if not normalized:
        return None
    for keyword in _RELATION_KEYWORDS_BY_LENGTH:
        pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
        if re.search(pattern, normalized):
            return RELATION_KEYWORD_MAP[keyword]
    return None
