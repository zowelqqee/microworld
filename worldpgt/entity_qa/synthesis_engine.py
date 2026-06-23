"""Synthesis engine for Entity QA layer 3.

Given an open question about an entity ("Tell me about SpaceX"), gather *every*
relevant fact about that entity from the overlay graph and group it by type and
confidence tier. The engine is purely a retrieval-and-grouping step: it never
invents a fact. Rendering of the grouped facts into prose lives in the renderer.

Three confidence tiers (mirroring the overlay's own trust model):
    VERIFIED — facts from stable / semi_stable relations, definitions, is_a.
    SNAPSHOT — source-qualified, dated estimates (as_of present, volatile).
    UNKNOWN  — parts of the question the graph has no answer for.

Deterministic, rule-based, offline. No ML. No embeddings. No network.
"""

from __future__ import annotations

import re

from worldpgt.entity_qa.types import SynthesisAnswer, SynthesisFactGroup
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider
from worldpgt.relation_extraction_v2.relation_policy import is_current_sensitive

_ANSWERABLE_STABILITIES = frozenset({"stable", "semi_stable"})

# Inverse predicates that produce noisy / duplicative clauses when synthesized
# from the object's perspective — dropped to keep the answer clean.
_INVERSE_PREDICATE_SKIP = frozenset({"known_for"})

# Tiny / generic tokens ignored when scoring keyword overlap with known labels.
_OVERLAP_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "for", "to", "in", "on", "company",
    "corporation", "inc", "ltd", "llc", "group", "about", "tell", "me",
    "what", "how", "does", "do", "is", "are", "work", "works", "know",
})


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split()).removeprefix("the ")


def _tokens(s: str) -> set[str]:
    raw = re.findall(r"[a-z0-9]+", (s or "").lower())
    return {t for t in raw if t not in _OVERLAP_STOPWORDS and len(t) > 1}


def _is_answerable_relation(item: dict) -> bool:
    predicate = str(item.get("predicate") or "")
    stability = str(item.get("stability") or "")
    return (
        stability in _ANSWERABLE_STABILITIES
        and not is_current_sensitive(predicate)
        and str(item.get("risk") or "").lower() != "high"
    )


def _dedup(items: list[str]) -> list[str]:
    """Case-insensitive de-duplication preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        key = _norm(it)
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


def _resolve_entity(
    provider: WikiMemoryOverlayProvider,
    subject_raw: str,
    surface_index=None,
) -> tuple[dict | None, str]:
    """Return (entity_dict, match_kind). match_kind in {exact, none}."""
    if not subject_raw:
        return None, "none"
    entity = provider.get_entity(subject_raw)
    if entity:
        return entity, "exact"
    if surface_index is not None:
        canonical = surface_index.resolve(subject_raw)
        if canonical:
            entity = provider.get_entity(canonical)
            if entity:
                return entity, "exact"
    return None, "none"


def _keyword_candidates(
    provider: WikiMemoryOverlayProvider,
    subject_raw: str,
) -> tuple[dict | None, list[str]]:
    """Find the closest known entity by keyword overlap with its label.

    Returns (best_entity_or_None, ranked_candidate_labels). Used only when exact
    resolution fails — this is a *suggestion* mechanism, never a fabrication: any
    facts still come from the matched entity's own overlay items.
    """
    want = _tokens(subject_raw)
    if not want:
        return None, []
    scored: list[tuple[int, dict]] = []
    for entity in provider.all_entities():
        label = str(entity.get("label") or "")
        have = _tokens(label)
        overlap = len(want & have)
        if overlap:
            scored.append((overlap, entity))
    if not scored:
        return None, []
    scored.sort(key=lambda x: (-x[0], str(x[1].get("label") or "")))
    candidates = [str(e.get("label") or "") for _s, e in scored]
    best = scored[0][1]
    return best, candidates


def synthesize(
    provider: WikiMemoryOverlayProvider,
    subject_raw: str,
    question: str = "",
    surface_index=None,
) -> SynthesisAnswer:
    """Gather and group everything the overlay knows about *subject_raw*."""

    entity, match_kind = _resolve_entity(provider, subject_raw, surface_index)
    candidate_entities: list[str] = []

    if entity is None:
        best, candidate_entities = _keyword_candidates(provider, subject_raw)
        if best is not None:
            entity = best
            match_kind = "keyword"

    if entity is None:
        return SynthesisAnswer(
            subject=subject_raw or None,
            matched=False,
            match_kind="none",
            definition=None,
            entity_type=None,
            candidate_entities=candidate_entities,
        )

    label = str(entity.get("label") or subject_raw)
    definition_item = provider.get_definition(label)
    definition = (
        str(definition_item["definition"])
        if definition_item and definition_item.get("definition")
        else None
    )
    entity_type = str(entity.get("entity_type") or "") or None

    groups: list[SynthesisFactGroup] = []

    # ── Forward relations: subject does/relates-to object ────────────────────
    forward: dict[str, list[str]] = {}
    for rel in provider.get_relations(label):
        if not _is_answerable_relation(rel):
            continue
        pred = str(rel.get("predicate") or "")
        obj = str(rel.get("object") or "")
        if pred and obj:
            forward.setdefault(pred, []).append(obj)
    for pred, objs in forward.items():
        groups.append(
            SynthesisFactGroup(
                kind="forward_relation",
                predicate=pred,
                objects=_dedup(objs),
                tier="VERIFIED",
            )
        )

    # ── Inverse relations: other entity does/relates-to subject ──────────────
    label_norm = _norm(label)
    inverse: dict[str, list[str]] = {}
    for rel in provider.all_relations():
        if _norm(rel.get("object", "")) != label_norm:
            continue
        if not _is_answerable_relation(rel):
            continue
        pred = str(rel.get("predicate") or "")
        subj = str(rel.get("subject") or "")
        if not pred or not subj or pred in _INVERSE_PREDICATE_SKIP:
            continue
        inverse.setdefault(pred, []).append(subj)
    for pred, subs in inverse.items():
        groups.append(
            SynthesisFactGroup(
                kind="inverse_relation",
                predicate=pred,
                objects=_dedup(subs),
                tier="VERIFIED",
            )
        )

    # ── Snapshot facts: dated, source-qualified estimates ────────────────────
    for fact in provider.get_source_facts(subject=label):
        obj = str(fact.get("object") or "")
        pred = str(fact.get("predicate") or "")
        if not obj or not pred:
            continue
        groups.append(
            SynthesisFactGroup(
                kind="snapshot",
                predicate=pred,
                objects=[obj],
                tier="SNAPSHOT",
                source_name=str(fact.get("source_name") or "") or None,
                as_of=str(fact.get("as_of") or "") or None,
            )
        )

    unknown_notes = _unknown_notes(question, forward, definition)

    return SynthesisAnswer(
        subject=label,
        matched=True,
        match_kind=match_kind,
        definition=definition,
        entity_type=entity_type,
        groups=groups,
        unknown_notes=unknown_notes,
        candidate_entities=candidate_entities if match_kind == "keyword" else [],
    )


_HOW_WORKS_RE = re.compile(
    r"\bhow\s+(?:does|do|is|are)\b.*\b(work|works|operate|operates|function|"
    r"functions|made|built)\b",
    re.IGNORECASE,
)
_HOW_TO_RE = re.compile(r"\bhow\s+(?:to|can\s+i|do\s+i)\b", re.IGNORECASE)

# Predicates that describe what an entity *does* operationally.
_OPERATIONAL_PREDICATES = frozenset(
    {"develops", "produces", "publishes", "operates", "provides"}
)


def _unknown_notes(
    question: str,
    forward: dict[str, list[str]],
    definition: str | None,
) -> list[str]:
    """Flag parts of the question the graph cannot answer — never silently drop."""
    notes: list[str] = []
    q = question or ""
    if _HOW_WORKS_RE.search(q) and not (forward.keys() & _OPERATIONAL_PREDICATES):
        notes.append("how it works (operational or process details)")
    elif _HOW_TO_RE.search(q):
        notes.append("the process or steps involved")
    return notes
