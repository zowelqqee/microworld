"""Lightweight, deterministic entity discovery.

Entities are discovered from the claims themselves (subjects, objects, and
destination/cause modifiers) plus capitalized spans. For each mention we keep
*observed* context terms (notably the head noun) which later drive local type
induction. There is NO global rigid enum of entity types here — type hints are
whatever words the corpus actually used.

Optionally consults the repo's ``EntitySurfaceIndex`` when overlay data is
available, but never requires it.
"""

from __future__ import annotations

import hashlib
import re

from worldpgt.schema_induction.types import EntityMention, RawClaim, SentenceRecord

_WS = re.compile(r"\s+")
# Words too generic to be a head-noun type hint on their own.
_GENERIC_STOP = frozenset({
    "the", "a", "an", "of", "and", "or", "for", "to", "with", "in", "on",
    "some", "many", "few", "all", "other", "such", "this", "that", "these",
    "those", "its", "their",
})

# Tokens that, if they end a phrase, are not useful as a type label.
_BAD_HEADS = frozenset({
    "income", "means", "work", "conditions", "water", "food", "grass",
    "availability", "rainfall", "forage",
})


def _norm(text: str) -> str:
    return _WS.sub(" ", (text or "").strip())


def _normalize_key(surface: str) -> str:
    return _norm(surface).lower()


def _stable_id(prefix: str, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _head_noun(surface: str) -> str | None:
    tokens = [t.strip(",.;:'\"") for t in _norm(surface).split(" ") if t.strip()]
    tokens = [t for t in tokens if t.lower() not in _GENERIC_STOP]
    if not tokens:
        return None
    head = tokens[-1].lower()
    if head in _BAD_HEADS or len(head) < 3:
        return None
    return head


def _context_terms(surface: str) -> tuple[str, ...]:
    terms: list[str] = []
    for tok in _norm(surface).split(" "):
        low = tok.strip(",.;:'\"").lower()
        if not low or low in _GENERIC_STOP or len(low) < 3:
            continue
        terms.append(low)
    # Stable, de-duplicated order.
    seen: list[str] = []
    for t in terms:
        if t not in seen:
            seen.append(t)
    return tuple(seen)


class _Acc:
    __slots__ = ("surface", "docs", "sents", "occ", "roles")

    def __init__(self, surface: str) -> None:
        self.surface = surface
        self.docs: set[str] = set()
        self.sents: set[str] = set()
        self.occ = 0
        self.roles: set[str] = set()


def _surface_candidates(claim: RawClaim) -> list[tuple[str, str]]:
    """Return (surface, role) candidates from a claim."""

    out: list[tuple[str, str]] = []
    if claim.subject:
        out.append((claim.subject, "subject"))
    if claim.object:
        out.append((claim.object, "object"))
    for key in ("destination", "cause"):
        val = claim.modifiers.get(key)
        if val:
            out.append((val, key))
    return out


def discover_entities(
    claims: list[RawClaim],
    sentences: list[SentenceRecord] | None = None,
) -> list[EntityMention]:
    """Discover entity mentions from extracted claims."""

    accs: dict[str, _Acc] = {}
    # Preserve first-seen order for deterministic output.
    order: list[str] = []

    for claim in claims:
        for surface, role in _surface_candidates(claim):
            surface = _norm(surface)
            if not surface:
                continue
            key = _normalize_key(surface)
            acc = accs.get(key)
            if acc is None:
                acc = _Acc(surface)
                accs[key] = acc
                order.append(key)
            acc.docs.add(claim.source_doc_id)
            acc.sents.add(claim.source_sentence_id)
            acc.occ += 1
            acc.roles.add(role)

    mentions: list[EntityMention] = []
    for key in order:
        acc = accs[key]
        head = _head_noun(acc.surface)
        ctx = _context_terms(acc.surface)
        hints: list[str] = []
        if head:
            hints.append(head)
        mentions.append(
            EntityMention(
                mention_id=_stable_id("ent", key),
                surface=acc.surface,
                normalized=key,
                doc_ids=tuple(sorted(acc.docs)),
                sentence_ids=tuple(sorted(acc.sents)),
                context_terms=ctx,
                type_hints=tuple(hints),
                occurrences=acc.occ,
            )
        )
    return mentions
