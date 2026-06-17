"""Rank/filter frontier titles for Knowledge Pump v1."""

from __future__ import annotations

import re

from worldpgt.knowledge_pump.types import FrontierTitle

_HIGH_VALUE = (
    "model", "rocket", "engine", "company", "corporation", "program",
    "vehicle", "satellite", "space", "tesla", "spacex", "neuralink",
    "battery", "software", "cloud", "technology", "launch", "station",
)
_LOW_VALUE = ("net worth", "stock price", "market capitalization", "revenue", "latest", "current")
_GENERIC = {"Company", "Product", "Founder", "Revenue", "Organization", "Leadership"}


def normalize_title(title: str) -> str:
    title = re.sub(r"\s+", " ", (title or "").replace("_", " ")).strip(" .,:;()[]{}")
    return title


def risk_hint(title: str) -> str:
    low = title.lower()
    if any(term in low for term in _LOW_VALUE):
        return "volatile_or_current"
    if len(title) < 4 or title in _GENERIC:
        return "generic"
    return "low"


def priority_for(frontier: FrontierTitle) -> int:
    title = normalize_title(frontier.title)
    score = frontier.weight
    low = title.lower()
    if any(term in low for term in _HIGH_VALUE):
        score += 20
    if frontier.source in {"relation_v2", "overlay_relation", "overlay_entity"}:
        score += 10
    if risk_hint(title) == "volatile_or_current":
        score -= 40
    if risk_hint(title) == "generic":
        score -= 20
    if title.lower().startswith("list of "):
        score -= 30
    return score


def rank_titles(frontier: list[FrontierTitle]) -> list[tuple[FrontierTitle, int, str]]:
    seen: set[str] = set()
    ranked: list[tuple[FrontierTitle, int, str]] = []
    for item in frontier:
        title = normalize_title(item.title)
        key = title.casefold()
        if not title or key in seen:
            continue
        seen.add(key)
        ranked.append((FrontierTitle(title, item.source, item.reason, item.weight), priority_for(item), risk_hint(title)))
    return sorted(ranked, key=lambda x: (-x[1], x[0].title.casefold()))

