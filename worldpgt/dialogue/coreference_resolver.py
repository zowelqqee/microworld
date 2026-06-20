"""Deterministic coreference resolver for interactive Microworld QA.

Resolution is deliberately conservative: it only substitutes references with
canonical entity labels already present in the in-memory conversation context.
It never asserts a relation or creates a new fact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import re

from worldpgt.dialogue.conversation_context import ConversationContext
from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoreferenceReplacement:
    reference: str
    entities: tuple[str, ...]

    @property
    def display_target(self) -> str:
        return ", ".join(self.entities)


@dataclass(frozen=True)
class CoreferenceResolution:
    original_question: str
    resolved_question: str
    replacements: tuple[CoreferenceReplacement, ...] = field(default_factory=tuple)
    unresolved_reference: str | None = None

    @property
    def changed(self) -> bool:
        return self.original_question != self.resolved_question


_DESCRIPTOR_RE = re.compile(
    r"\b(the\s+same\s+person|the\s+organization|the\s+company|the\s+founder|this\s+entity)\b",
    re.IGNORECASE,
)
_PRONOUN_RE = re.compile(r"\b(he|she|it|they|his|her|its|their)\b", re.IGNORECASE)
_START_DEMONSTRATIVE_RE = re.compile(r"^\s*(that|this)\b", re.IGNORECASE)

_PERSON_TYPES = frozenset({"person"})
_THING_TYPES = frozenset({"organization", "vehicle", "program", "product"})
_ORGANIZATION_TYPES = frozenset({"organization"})
_POSSESSIVE_REFERENCES = frozenset({"his", "her", "its", "their"})


def resolve_coreferences(
    question: str,
    context: ConversationContext,
    index: EntitySurfaceIndex,
) -> CoreferenceResolution:
    """Return *question* with resolvable dialogue references substituted."""

    matches = _reference_matches(question)
    if not matches:
        return CoreferenceResolution(original_question=question, resolved_question=question)

    if not context.turns:
        return CoreferenceResolution(
            original_question=question,
            resolved_question=question,
            unresolved_reference=matches[0][2],
        )

    resolved = question
    replacements: list[CoreferenceReplacement] = []
    offset = 0
    for start, end, reference in matches:
        entities = _resolve_reference(reference, context, index)
        if not entities:
            return CoreferenceResolution(
                original_question=question,
                resolved_question=question,
                unresolved_reference=reference,
            )
        replacement = _replacement_text(reference, entities)
        adj_start = start + offset
        adj_end = end + offset
        resolved = resolved[:adj_start] + replacement + resolved[adj_end:]
        offset += len(replacement) - (end - start)
        replacements.append(CoreferenceReplacement(reference=reference, entities=tuple(entities)))

    return CoreferenceResolution(
        original_question=question,
        resolved_question=resolved,
        replacements=tuple(replacements),
    )


def _reference_matches(question: str) -> list[tuple[int, int, str]]:
    candidates: list[tuple[int, int, str]] = []
    for pattern in (_DESCRIPTOR_RE, _PRONOUN_RE, _START_DEMONSTRATIVE_RE):
        for match in pattern.finditer(question):
            candidates.append((match.start(1), match.end(1), match.group(1)))

    candidates.sort(key=lambda row: (row[0], -(row[1] - row[0])))
    out: list[tuple[int, int, str]] = []
    covered_until = -1
    for start, end, reference in candidates:
        if start < covered_until:
            continue
        out.append((start, end, reference))
        covered_until = end
    return out


def _resolve_reference(
    reference: str,
    context: ConversationContext,
    index: EntitySurfaceIndex,
) -> list[str]:
    ref = reference.lower()
    normalized = re.sub(r"\s+", " ", ref.strip())

    if normalized in {"he", "she", "his", "her", "the founder", "the same person"}:
        entity = _latest_entity_matching(context, index, _PERSON_TYPES)
        _log.debug("resolve %r (person) → %r  (turns=%d)", reference, entity,
                   len(context.turns))
        for i, t in enumerate(context.turns):
            _log.debug("  turn[%d] decision=%s primary=%r mentioned=%r",
                       i, t.decision, t.primary_entity, t.mentioned_entities)
        return [entity] if entity else []

    if normalized in {"it", "its"}:
        entity = _latest_entity_matching(context, index, _THING_TYPES)
        return [entity] if entity else []

    if normalized in {"the company", "the organization"}:
        entity = _latest_primary_matching(context, index, _ORGANIZATION_TYPES)
        if entity is None:
            entity = _latest_entity_matching(context, index, _ORGANIZATION_TYPES)
        return [entity] if entity else []

    if normalized in {"they", "their"}:
        return _latest_plural_entities(context, index)

    if normalized in {"that", "this", "this entity"}:
        entity = context.last_entity()
        if entity:
            return [entity]
        mentioned = context.last_mentioned_entities()
        return [mentioned[0]] if mentioned else []

    return []


def _replacement_text(reference: str, entities: list[str]) -> str:
    joined = " and ".join(entities) if len(entities) > 1 else entities[0]
    if reference.lower() in _POSSESSIVE_REFERENCES:
        return f"{joined}'s"
    return joined


def _latest_entity_matching(
    context: ConversationContext,
    index: EntitySurfaceIndex,
    allowed_types: frozenset[str],
) -> str | None:
    primary = _latest_primary_matching(context, index, allowed_types)
    if primary is not None:
        return primary

    for turn in reversed(context.turns):
        matches = _unique_entities_of_type(turn.mentioned_entities, index, allowed_types)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None
    return None


def _latest_primary_matching(
    context: ConversationContext,
    index: EntitySurfaceIndex,
    allowed_types: frozenset[str],
) -> str | None:
    for turn in reversed(context.turns):
        entity = turn.primary_entity
        if entity and _entity_type_matches(entity, index, allowed_types):
            return entity
    return None


def _latest_plural_entities(
    context: ConversationContext,
    index: EntitySurfaceIndex,
) -> list[str]:
    primaries: list[str] = []
    seen: set[str] = set()
    for turn in reversed(context.turns[-3:]):
        entity = turn.primary_entity
        if not entity or entity in seen or index.entity_type(entity) is None:
            continue
        seen.add(entity)
        primaries.append(entity)
        if len(primaries) == 2:
            break
    if len(primaries) == 2:
        return list(reversed(primaries))

    mentioned = context.last_mentioned_entities()
    typed = [entity for entity in mentioned if index.entity_type(entity) is not None]
    deduped = _dedupe(typed)
    return deduped if len(deduped) == 2 else []


def _unique_entities_of_type(
    entities: list[str],
    index: EntitySurfaceIndex,
    allowed_types: frozenset[str],
) -> list[str]:
    return _dedupe(
        [entity for entity in entities if _entity_type_matches(entity, index, allowed_types)]
    )


def _entity_type_matches(
    entity: str,
    index: EntitySurfaceIndex,
    allowed_types: frozenset[str],
) -> bool:
    etype = index.entity_type(entity)
    return etype in allowed_types


def _dedupe(entities: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        if entity in seen:
            continue
        seen.add(entity)
        out.append(entity)
    return out
