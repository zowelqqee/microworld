"""Semantic SPO-style question parser for Entity QA.

This parser is intentionally deterministic. It resolves known entities through
the same ``EntitySurfaceIndex`` used by extraction, maps short relation phrases
through ``relation_policy``, and emits a small structured query that downstream
QA code can either use directly or fall back from.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

from worldpgt.entity_qa.types import SemanticQuery
from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex
from worldpgt.relation_extraction_v2.relation_policy import relation_intent_from_text

_EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
_ACCEPTED_OVERLAY_PATH = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_PROMOTED_OVERLAY_PATH = (
    _EXPERIMENTS / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json"
)
_SNAPSHOT_OVERLAY_PATH = _EXPERIMENTS / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json"

_IS_A_RE = re.compile(r"^is\s+(.+?)\s+(?:a|an|the)\s+(.+?)[\?\.]?$", re.IGNORECASE)
_DEFINITION_RE = re.compile(
    r"^(?:who|what)\s+(?:is|are|was|were)\s+(.+?)[\?\.]?$|"
    r"^(?:define|tell\s+me\s+about)\s+(.+?)[\?\.]?$|"
    r"^what\s+(?:kind|type|sort)\s+of\s+.+?\s+is\s+(.+?)[\?\.]?$",
    re.IGNORECASE,
)
_PRODUCTS_MAKE_RE = re.compile(
    r"what\s+products?\s+does\s+.+?\s+(?:make|produce|manufacture|build)s?\b",
    re.IGNORECASE,
)
_PASSIVE_OPEN_RELATION_RE = re.compile(
    r"^what\s+(?:is|was)\s+(developed|produced|published)\s+by\s+.+?[\?\.]?$",
    re.IGNORECASE,
)
_DEFINE_KIND_PREFIX_RE = re.compile(r"^what\s+(?:kind|type|sort)\s+of\b", re.IGNORECASE)
_WHAT_IS_DEFINITION_PREFIX_RE = re.compile(r"^what\s+is\s+(?:a|an|the)\s+", re.IGNORECASE)
_OPEN_MANUFACTURE_RE = re.compile(r"^what\s+does\s+.+?\s+manufactures?\b", re.IGNORECASE)
_TITLE_TOKEN_RE = re.compile(r"[A-Z][A-Za-z0-9&.-]*")
_PATH_RE = re.compile(
    r"\b(?:how\s+(?:is|are|was|were)|what\s+path)\b.+\b(?:connected|related|linked|connects)\b",
    re.IGNORECASE,
)
_INTERSECTION_RE = re.compile(
    r"\b(?:have\s+in\s+common|in\s+common|share|common\s+between)\b",
    re.IGNORECASE,
)
_SUBJECT_WH_RE = re.compile(r"^(?:who|which|what)\b", re.IGNORECASE)
_PASSIVE_BY_RE = re.compile(r"\b(?:by|belong\s+to|belongs\s+to)\b", re.IGNORECASE)

_INVERSE_CANONICAL_RELATIONS = frozenset(
    {
        "developed_by",
        "created_by",
        "published_by",
        "product_of",
        "service_of",
        "platform_of",
        "subsidiary_of",
        "part_of",
        "leader_of",
    }
)
_ACTIVE_OWNER_RE = re.compile(r"\b(?:does|did)?\s*.+?\b(?:own|owns)\b", re.IGNORECASE)
_OWNER_ACTIVE_ENTITY_RE = re.compile(r"^\s*(?:which|what)\s+\w+.+\bdoes\s+.+?\s+own\b", re.IGNORECASE)
_PASSIVE_OWNED_BY_ENTITY_RE = re.compile(
    r"^\s*(?:which|what)\s+.+?\bowned\s+by\b",
    re.IGNORECASE,
)
_ACTIVE_LEADER_RE = re.compile(r"^(?:who|which\s+person)\s+(?:leads?|runs?|heads?)\b", re.IGNORECASE)
_ACTIVE_RELATION_SUBJECT_RE = re.compile(
    r"^(?:who|which\s+.+?)\s+"
    r"(?:develops?|builds?|makes?|produces?|manufactures?|publishes?)\b",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def default_surface_index() -> EntitySurfaceIndex:
    return EntitySurfaceIndex(
        accepted_overlay_path=_ACCEPTED_OVERLAY_PATH,
        promoted_overlay_path=_PROMOTED_OVERLAY_PATH,
        snapshot_overlay_path=_SNAPSHOT_OVERLAY_PATH,
    )


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip(" ?.\t\n\r"))


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _entity_mentions(
    question: str,
    index: EntitySurfaceIndex,
) -> list[tuple[str, str, int, int]]:
    mentions = index.find_in_text(question)
    out: list[tuple[str, str, int, int]] = []
    seen: set[str] = set()
    for surface, canonical, start, end in sorted(mentions, key=lambda row: row[2]):
        if _looks_like_partial_title_match(question, surface, start, end):
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append((surface, canonical, start, end))
    return out


def _looks_like_partial_title_match(text: str, surface: str, start: int, end: int) -> bool:
    before = text[:start].rstrip()
    after = text[end:].lstrip()
    prev = re.search(r"([A-Z][A-Za-z0-9&.-]*)\s*$", before)
    nxt = _TITLE_TOKEN_RE.match(after)
    return bool(prev or nxt)


def _definition_subject(match: re.Match[str]) -> str:
    for group in match.groups():
        if group:
            return re.sub(r"^(?:a|an|the)\s+", "", _clean(group), flags=re.IGNORECASE)
    return ""


def _exact_definition_entity(
    raw_subject: str,
    mentions: list[tuple[str, str, int, int]],
) -> str | None:
    raw_norm = _norm_text(raw_subject)
    for surface, canonical, _start, _end in mentions:
        if _norm_text(canonical) == raw_norm:
            return canonical
        if " " in raw_norm and _norm_text(surface) == raw_norm:
            return canonical
    return None


def _is_inverse_relation_question(question: str, relation: str) -> bool:
    q = question.lower()
    if relation == "owned_by":
        return bool(_OWNER_ACTIVE_ENTITY_RE.search(q) or _PASSIVE_OWNED_BY_ENTITY_RE.search(q))
    if relation == "leader_of" and _ACTIVE_LEADER_RE.search(q):
        return True
    if relation in {"develops", "produces", "publishes"} and _ACTIVE_RELATION_SUBJECT_RE.search(q):
        return True
    if relation in _INVERSE_CANONICAL_RELATIONS and _SUBJECT_WH_RE.search(q):
        return True
    if relation in {"developed_by", "created_by", "published_by"} and _PASSIVE_BY_RE.search(q):
        return True
    return False


def _unknown_position(question: str, relation: str | None) -> str:
    if _PATH_RE.search(question):
        return "path"
    if _INTERSECTION_RE.search(question):
        return "intersection"
    if relation and _is_inverse_relation_question(question, relation):
        return "subject"
    if relation:
        return "object"
    return "relation"


def parse_semantic_query(
    question: str,
    index: EntitySurfaceIndex | None = None,
) -> SemanticQuery:
    """Parse *question* into a structured semantic query."""

    q = _clean(question)
    surface_index = index or default_surface_index()
    mentions = _entity_mentions(q, surface_index)
    entities = [canonical for _surface, canonical, _start, _end in mentions]
    relation = relation_intent_from_text(q)
    if relation == "develops" and _PRODUCTS_MAKE_RE.search(q):
        relation = "produces"
    if relation == "manufactures" and _OPEN_MANUFACTURE_RE.search(q):
        relation = "produces"

    is_a_match = _IS_A_RE.match(q)
    if is_a_match:
        left_subject = _clean(is_a_match.group(1))
        left_end = is_a_match.start(2)
        left_entities = [canonical for _surface, canonical, _start, end in mentions if end <= left_end]
        subject = left_entities[0] if left_entities else surface_index.resolve(left_subject)
        if subject is None and left_subject:
            subject = left_subject
        target = _clean(is_a_match.group(2))
        confidence = 0.95 if subject and target else 0.45
        return SemanticQuery(
            entity_a=subject,
            entity_b=target or None,
            relation_intent="is_a",
            unknown_position="relation",
            query_type="is_a",
            confidence=confidence,
        )

    passive_open = _PASSIVE_OPEN_RELATION_RE.match(q)
    if passive_open and entities:
        predicate = {
            "developed": "develops",
            "produced": "produces",
            "published": "publishes",
        }.get(passive_open.group(1).lower(), "develops")
        return SemanticQuery(
            entity_a=entities[0],
            entity_b=None,
            relation_intent=predicate,
            unknown_position="object",
            query_type="lookup",
            confidence=0.9,
        )

    if _INTERSECTION_RE.search(q):
        return SemanticQuery(
            entity_a=entities[0] if entities else None,
            entity_b=entities[1] if len(entities) > 1 else None,
            relation_intent=relation,
            unknown_position="intersection",
            query_type="comparative",
            confidence=0.95 if len(entities) >= 2 else 0.4,
        )

    if _PATH_RE.search(q):
        return SemanticQuery(
            entity_a=entities[0] if entities else None,
            entity_b=entities[1] if len(entities) > 1 else None,
            relation_intent=relation,
            unknown_position="path",
            query_type="lookup",
            confidence=0.95 if len(entities) >= 2 else 0.4,
        )

    definition_match = _DEFINITION_RE.match(q)
    if definition_match:
        raw_subject = _definition_subject(definition_match)
        exact_entity = _exact_definition_entity(raw_subject, mentions)
        if exact_entity and (
            relation is None
            or _DEFINE_KIND_PREFIX_RE.match(q)
            or _WHAT_IS_DEFINITION_PREFIX_RE.match(q)
        ):
            return SemanticQuery(
                entity_a=exact_entity,
                entity_b=None,
                relation_intent=None,
                unknown_position="relation",
                query_type="definition",
                confidence=0.9,
            )
        if relation is None:
            return SemanticQuery(
                entity_a=None,
                entity_b=None,
                relation_intent=None,
                unknown_position="relation",
                query_type="definition",
                confidence=0.1,
            )

    if relation and entities:
        unknown = _unknown_position(q, relation)
        return SemanticQuery(
            entity_a=entities[0],
            entity_b=entities[1] if len(entities) > 1 else None,
            relation_intent=relation,
            unknown_position=unknown,  # type: ignore[arg-type]
            query_type="inverse" if unknown == "subject" else "lookup",
            confidence=0.9,
        )

    if entities:
        return SemanticQuery(
            entity_a=entities[0],
            entity_b=entities[1] if len(entities) > 1 else None,
            relation_intent=None,
            unknown_position="relation",
            query_type="lookup",
            confidence=0.55,
        )

    return SemanticQuery(
        entity_a=None,
        entity_b=None,
        relation_intent=relation,
        unknown_position="relation",
        query_type="lookup",
        confidence=0.35 if relation else 0.1,
    )
