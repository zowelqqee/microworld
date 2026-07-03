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

from worldpgt.entity_qa.types import (
    SynthesisAnswer,
    SynthesisEnrichment,
    SynthesisFactGroup,
)
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider
from worldpgt.relation_extraction_v2.relation_policy import is_current_sensitive, predicate_in_class

_ANSWERABLE_STABILITIES = frozenset({"stable", "semi_stable"})

# Founding / ownership / leadership predicates whose object is worth one extra
# fact: "X founded Y, an aerospace company". Both directions count — the subject
# may be the founder (forward ``founded``) or the foundee (inverse ``founded_by``).
_ENRICHMENT_RELATIONS = frozenset({
    "founded",
    "founded_by",
    "owned_by",
    "owns",
    "subsidiary_of",
    "part_of",
    "service_of",
    "platform_of",
})

# Tokens that mark a definition as a ranking — preferred enrichment because a
# rank ("ranked 5th on the Fortune 500") is more telling than a bare class.
_RANKING_TOKENS = ("ranked", "fortune", "largest", "biggest")

# Inverse predicates that produce noisy / duplicative clauses when synthesized
# from the object's perspective — dropped to keep the answer clean.
_INVERSE_PREDICATE_SKIP = frozenset({"known_for"})

_PERSON_FORWARD_PREDICATES = frozenset({
    "founded",
    "founded_by",
    "known_for",
    "is_a",
    "develops",
    "created_by",
    "published_by",
    "leads",
    "operates",
})

_PERSON_DEFINITION_TOKENS = (
    "actor",
    "artist",
    "author",
    "businessman",
    "businesswoman",
    "entrepreneur",
    "engineer",
    "founder",
    "inventor",
    "musician",
    "person",
    "politician",
    "scientist",
    "writer",
)

# Tiny / generic tokens ignored when scoring keyword overlap with known labels.
_OVERLAP_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "for", "to", "in", "on", "company",
    "corporation", "inc", "ltd", "llc", "group", "about", "tell", "me",
    "what", "how", "does", "do", "is", "are", "work", "works", "know",
})


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split()).removeprefix("the ")


def _surface_norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[''`]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^(?:the|a|an)\s+", "", s)
    return s


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


def _is_person_entity(entity_type: str | None, definition: str | None) -> bool:
    if entity_type == "person":
        return True
    if entity_type in {"organization", "place", "product", "work"}:
        return False
    text = (definition or "").lower()
    return any(
        re.search(rf"\b{re.escape(token)}\b", text)
        for token in _PERSON_DEFINITION_TOKENS
    )


def _person_forward_relation_ok(predicate: str) -> bool:
    return predicate in _PERSON_FORWARD_PREDICATES


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


def _provider_items(provider: WikiMemoryOverlayProvider, *method_names: str) -> list[dict]:
    for method_name in method_names:
        method = getattr(provider, method_name, None)
        if method is None:
            continue
        return list(method())
    return []


def _all_entities(provider: WikiMemoryOverlayProvider) -> list[dict]:
    return _provider_items(provider, "all_entities", "entities")


def _all_definitions(provider: WikiMemoryOverlayProvider) -> list[dict]:
    return _provider_items(provider, "all_definitions", "definitions")


def _all_relations(provider: WikiMemoryOverlayProvider) -> list[dict]:
    return _provider_items(provider, "all_relations", "relations")


def _all_source_facts(provider: WikiMemoryOverlayProvider) -> list[dict]:
    return _provider_items(provider, "all_source_facts", "source_facts")


def _overlay_items_for_subject_inference(
    provider: WikiMemoryOverlayProvider,
    label: str,
    definition: str | None,
) -> list[dict]:
    """Small subject-neighborhood overlay for profile inference.

    Running the full rule set over a pump overlay is expensive, especially for
    shared-capability competitor detection. For synthesis we only need inferred
    facts whose subject is *label*, so keep entities plus relations that can
    participate in rules around that subject.
    """
    entities = _all_entities(provider)
    labels = {str(e.get("label") or "") for e in entities if e.get("label")}
    subject_norm = _surface_norm(label)
    seed_subjects = {subject_norm}
    seed_objects: set[str] = set()
    capability_objects: set[str] = set()

    if definition:
        seed_subjects.update(_surface_norm(name) for name in _entity_labels_in_text(definition, labels))

    subject_definition = _definition_for_subject(provider, label)
    definitions = [subject_definition] if subject_definition else []

    direct_relations: list[dict] = []
    all_relations = _all_relations(provider)
    for rel in all_relations:
        subj = _surface_norm(rel.get("subject", ""))
        obj = _surface_norm(rel.get("object", ""))
        pred = str(rel.get("predicate") or "")
        if subj == subject_norm or obj == subject_norm:
            direct_relations.append(rel)
            if subj == subject_norm:
                seed_objects.add(obj)
                if predicate_in_class(pred, "capability"):
                    capability_objects.add(obj)
            if obj == subject_norm:
                seed_subjects.add(subj)

    related_relations: list[dict] = []
    for rel in all_relations:
        subj = _surface_norm(rel.get("subject", ""))
        obj = _surface_norm(rel.get("object", ""))
        pred = str(rel.get("predicate") or "")
        if rel in direct_relations:
            continue
        if subj in seed_subjects:
            related_relations.append(rel)
            if predicate_in_class(pred, "capability"):
                capability_objects.add(obj)
        elif subj in seed_objects and pred == "requires":
            related_relations.append(rel)
        elif predicate_in_class(pred, "capability") and obj in capability_objects:
            related_relations.append(rel)

    return _dedupe_overlay_items([*entities, *definitions, *direct_relations, *related_relations])


def _entity_labels_in_text(text: str, labels: set[str]) -> list[str]:
    text_norm = f" {_surface_norm(text)} "
    out: list[str] = []
    for label in labels:
        label_norm = _surface_norm(label)
        if label_norm and f" {label_norm} " in text_norm:
            out.append(label)
    return out


def _dedupe_overlay_items(items: list[dict]) -> list[dict]:
    seen: set[int] = set()
    out: list[dict] = []
    for item in items:
        marker = id(item)
        if marker in seen or item is None:
            continue
        seen.add(marker)
        out.append(item)
    return out


def _entity_surfaces(entity: dict) -> list[str]:
    return [
        str(entity.get("label") or ""),
        *(str(alias) for alias in (entity.get("aliases") or [])),
        str(entity.get("source_page") or ""),
    ]


def _resolve_entity(
    provider: WikiMemoryOverlayProvider,
    subject_raw: str,
    surface_index=None,
) -> tuple[dict | None, str]:
    """Return (entity_dict, match_kind). match_kind in {exact, none}."""
    if not subject_raw:
        return None, "none"

    entities = _all_entities(provider)
    subject_norm = _surface_norm(subject_raw)
    for entity in entities:
        if entity.get("label") == subject_raw:
            return entity, "exact"
    for entity in entities:
        if subject_raw in (entity.get("aliases") or []):
            return entity, "exact"
    for entity in entities:
        if _surface_norm(entity.get("label", "")) == subject_norm:
            return entity, "exact"
    for entity in entities:
        if any(_surface_norm(alias) == subject_norm for alias in (entity.get("aliases") or [])):
            return entity, "exact"
    for entity in entities:
        if _surface_norm(entity.get("source_page", "")) == subject_norm:
            return entity, "exact"

    if surface_index is not None:
        canonical = surface_index.resolve(subject_raw)
        if canonical:
            canonical_norm = _surface_norm(canonical)
            for entity in entities:
                if any(_surface_norm(surface) == canonical_norm for surface in _entity_surfaces(entity)):
                    return entity, "exact"
    return None, "none"


def _definition_for_subject(
    provider: WikiMemoryOverlayProvider,
    subject: str,
    entity: dict | None = None,
) -> dict | None:
    if entity is not None and hasattr(provider, "get_definition_for_entity"):
        definition = provider.get_definition_for_entity(entity)
        if definition is not None:
            return definition
    if hasattr(provider, "get_definition"):
        definition = provider.get_definition(subject)
        if definition is not None:
            return definition
    subject_norms = {_surface_norm(subject)}
    if entity is not None:
        subject_norms.update(
            _surface_norm(surface) for surface in _entity_surfaces(entity)
        )
    subject_norms = {norm for norm in subject_norms if norm}
    for definition in _all_definitions(provider):
        if _surface_norm(definition.get("subject", "")) in subject_norms:
            return definition
    return None


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
    for entity in _all_entities(provider):
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


def _subject_surface_norms(entity: dict, definition_item: dict | None) -> set[str]:
    """Every normalized surface the subject's facts may be filed under.

    Facts are not always stored under an entity's canonical label — they may use
    an alias ("Buffett"), the source page, or the surface a definition was
    extracted from ("Warren Edward Buffett"). Collecting all of them is what lets
    multi-fact aggregation pull in relations that single-label matching misses.
    """
    norms = {_surface_norm(surface) for surface in _entity_surfaces(entity)}
    if definition_item:
        norms.add(_surface_norm(definition_item.get("subject", "")))
    return {norm for norm in norms if norm}


def _ranking_note(provider: WikiMemoryOverlayProvider, obj: str) -> str | None:
    """A ranking definition for *obj* if one exists ("ranked 5th on the Fortune 500")."""
    obj_norm = _surface_norm(obj)
    for definition in _all_definitions(provider):
        if _surface_norm(definition.get("subject", "")) != obj_norm:
            continue
        text = str(definition.get("definition") or "")
        low = text.lower()
        if any(token in low for token in _RANKING_TOKENS):
            return text
    return None


def _compute_enrichment(
    provider: WikiMemoryOverlayProvider,
    subject_label: str,
    groups: list[SynthesisFactGroup],
    surface_index=None,
) -> SynthesisEnrichment | None:
    """Pull one key fact about the first entity the subject founds / owns / leads.

    Walks the founding/ownership groups in render order, takes the first object
    that resolves to an entity with a verified descriptor (a ranking if present,
    else its definition) and returns it as an enrichment. Never fabricates: the
    descriptor is an overlay fact about the related entity, and the connecting
    relation already exists on the subject.
    """
    subject_norm = _surface_norm(subject_label)
    for group in groups:
        if str(getattr(group, "predicate", "") or "") not in _ENRICHMENT_RELATIONS:
            continue
        for obj in getattr(group, "objects", []):
            obj = str(obj or "").strip()
            if not obj or _surface_norm(obj) == subject_norm:
                continue
            entity, _kind = _resolve_entity(provider, obj, surface_index)
            ranking = _ranking_note(provider, obj)
            note = ranking
            kind = "ranking"
            if note is None:
                definition_item = _definition_for_subject(provider, obj, entity)
                note = (
                    str(definition_item["definition"])
                    if definition_item and definition_item.get("definition")
                    else None
                )
                kind = "definition"
            if note:
                entity_type = (
                    str(entity.get("entity_type") or "") or None if entity else None
                )
                return SynthesisEnrichment(
                    object=obj,
                    note=note,
                    relation=str(getattr(group, "predicate", "") or ""),
                    kind=kind,
                    entity_type=entity_type,
                )
    return None


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
    definition_item = _definition_for_subject(provider, label, entity)
    definition = (
        str(definition_item["definition"])
        if definition_item and definition_item.get("definition")
        else None
    )
    entity_type = str(entity.get("entity_type") or "") or None

    # Facts about the subject may be filed under any of its surfaces (aliases,
    # source page, the definition's extracted surface). Collect them all so
    # aggregation does not silently drop, say, a relation stored under "Buffett"
    # when the canonical label is "Warren Buffett".
    subject_surfaces = _subject_surface_norms(entity, definition_item)
    is_person = _is_person_entity(entity_type, definition)

    groups: list[SynthesisFactGroup] = []
    direct_keys: set[tuple[str, str, str]] = set()

    # ── Forward relations: subject does/relates-to object ────────────────────
    forward: dict[str, list[str]] = {}
    for rel in _all_relations(provider):
        if _surface_norm(rel.get("subject", "")) not in subject_surfaces:
            continue
        if not _is_answerable_relation(rel):
            continue
        pred = str(rel.get("predicate") or "")
        obj = str(rel.get("object") or "")
        if is_person and not _person_forward_relation_ok(pred):
            continue
        if pred and obj:
            forward.setdefault(pred, []).append(obj)
            direct_keys.add((_norm(label), pred, _norm(obj)))
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
    inverse: dict[str, list[str]] = {}
    for rel in _all_relations(provider):
        if _surface_norm(rel.get("object", "")) not in subject_surfaces:
            continue
        if not _is_answerable_relation(rel):
            continue
        pred = str(rel.get("predicate") or "")
        subj = str(rel.get("subject") or "")
        if not pred or not subj or pred in _INVERSE_PREDICATE_SKIP:
            continue
        inverse.setdefault(pred, []).append(subj)
        direct_keys.add((_norm(label), pred, _norm(subj)))
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
    seen_snapshots: set[tuple[str, str, str, str]] = set()
    for fact in _all_source_facts(provider):
        if _surface_norm(fact.get("subject", "")) not in subject_surfaces:
            continue
        obj = str(fact.get("object") or "")
        pred = str(fact.get("predicate") or "")
        if not obj or not pred:
            continue
        # The same estimate can be filed under several surfaces; count it once.
        snapshot_key = (
            pred,
            _norm(obj),
            str(fact.get("source_name") or ""),
            str(fact.get("as_of") or ""),
        )
        if snapshot_key in seen_snapshots:
            continue
        seen_snapshots.add(snapshot_key)
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

    # ── Inferred facts: ephemeral rule conclusions, explicitly tiered ────────
    # The inference workspace never writes to the overlay. These facts are kept
    # separate so renderers can say "Based on reasoning" instead of presenting
    # them as direct verified facts.
    try:
        from worldpgt.cognition.inference_engine import run_inference

        workspace = run_inference(
            _overlay_items_for_subject_inference(provider, label, definition)
        )
        for fact in workspace.for_subject(label):
            key = (_norm(label), fact.predicate, _norm(fact.object))
            if key in direct_keys:
                continue
            groups.append(
                SynthesisFactGroup(
                    kind="inferred_relation",
                    predicate=fact.predicate,
                    objects=[fact.object],
                    tier="INFERRED",
                    rule=fact.rule,
                    confidence=fact.confidence,
                    chain=fact.chain,
                )
            )
    except Exception:
        # Synthesis is still useful without inference; do not turn a supported
        # direct profile into an audit because an optional reasoning pass failed.
        pass

    unknown_notes = _unknown_notes(question, forward, definition)

    # 2-hop cross-entity enrichment: one key fact about what the subject
    # founds / owns / leads, so the narrative can connect the two entities.
    enrichment = _compute_enrichment(provider, label, groups, surface_index)

    return SynthesisAnswer(
        subject=label,
        matched=True,
        match_kind=match_kind,
        definition=definition,
        entity_type=entity_type,
        groups=groups,
        unknown_notes=unknown_notes,
        candidate_entities=candidate_entities if match_kind == "keyword" else [],
        enrichment=enrichment,
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
