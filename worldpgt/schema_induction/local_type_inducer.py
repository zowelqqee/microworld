"""Induce local types from entities and frames.

A local type clusters entity mentions that share an observed context term
(typically a head noun like "visa") and the roles they play. The type label is
an *observed* word from the corpus — never drawn from a global rigid enum. A
type only forms when at least two members share the term, so single mentions do
not manufacture spurious types.
"""

from __future__ import annotations

import hashlib

from worldpgt.schema_induction.types import (
    ArgumentFrame,
    EntityMention,
    LocalType,
)

_MIN_MEMBERS = 2


def _stable_id(prefix: str, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def induce_local_types(
    entities: list[EntityMention],
    frames: list[ArgumentFrame] | None = None,
) -> list[LocalType]:
    """Induce local types from shared context terms across entities."""

    # Map entity normalized surface -> roles it appears in (from frames).
    roles_by_entity: dict[str, set[str]] = {}
    for frame in frames or []:
        for role, surface in frame.roles.items():
            key = surface.strip().lower()
            roles_by_entity.setdefault(key, set()).add(role)

    # Group entities by each of their type-hint terms.
    by_term: dict[str, list[EntityMention]] = {}
    term_order: list[str] = []
    for ent in entities:
        for term in ent.type_hints:
            if term not in by_term:
                by_term[term] = []
                term_order.append(term)
            by_term[term].append(ent)

    local_types: list[LocalType] = []
    for term in term_order:
        members = by_term[term]
        # Need >= 2 distinct members, and skip when the term IS the only member
        # (e.g. the head noun equals the whole surface).
        distinct = [m for m in members if m.normalized != term]
        if len(distinct) < _MIN_MEMBERS:
            continue

        member_surfaces = sorted({m.surface for m in distinct})
        induced_roles: set[str] = set()
        context: set[str] = set()
        for m in distinct:
            context.update(m.context_terms)
            induced_roles.update(roles_by_entity.get(m.normalized, set()))

        confidence = round(min(0.95, 0.5 + 0.1 * len(distinct)), 4)
        local_types.append(
            LocalType(
                type_id=_stable_id("ltype", term),
                label=term,
                members=tuple(member_surfaces),
                context_terms=tuple(sorted(context)),
                induced_from_roles=tuple(sorted(induced_roles)),
                confidence=confidence,
            )
        )
    return local_types
