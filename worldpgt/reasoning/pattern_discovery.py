"""Pattern discovery: find unnamed structural regularities in the fact graph.

The discoverer analyzes the system's own graph (intended to run on a nightly
cycle, see ``run_pattern_discovery``) and produces ``GraphPattern`` objects —
observations about graph structure, never new facts. Each pattern carries:

  supporting_evidence — the concrete verified triples that instantiate it;
  confidence          — the share of condition-matching entities that also
                        satisfy the consequent;
  counter_examples    — entities that match the condition but violate the
                        consequent.

Two universal pattern families:

  class_implication — "entities that are C tend to have predicate p":
      members(is_a C) → share holding p; e.g. "launch vehicle companies
      develop launch vehicles" when the objects share a common class.

  cooccurrence — "entities whose p1-object is a D tend to have p2":
      e.g. "entities founded_by someone who is_a entrepreneur also develop
      rockets". The condition is a typed 2-hop property, the consequent a
      single predicate.

Deterministic: sorted iteration everywhere, stable pattern ids, no randomness.
No ML. No network.
"""

from __future__ import annotations

import re

from worldpgt.reasoning.fact_graph import FactGraph, norm
from worldpgt.reasoning.types import (
    GraphPattern,
    PatternCounterExample,
)
from worldpgt.relation_extraction_v2.relation_policy import PREDICATE_CLASSES

_CLASSIFICATION = PREDICATE_CLASSES["classification"]

_MAX_EVIDENCE = 25
_MAX_COUNTER_EXAMPLES = 25
_MAX_MATCHED_ENTITIES = 50

# Class strings longer than this are lifted definition sentences, not usable
# class labels — they can never generalize across entities anyway.
_MAX_CLASS_LABEL_WORDS = 6


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", norm(text)).strip("_")


def _base_verb(predicate: str) -> str:
    """Naive third-person → base form for readable descriptions
    (develops → develop, publishes → publish). Compound predicates
    (founded_by, known_for) are not verbs and are kept as-is by the caller."""
    for suffix in ("shes", "ches", "xes", "sses"):
        if predicate.endswith(suffix):
            return predicate[:-2]
    if predicate.endswith("s") and not predicate.endswith("ss"):
        return predicate[:-1]
    return predicate


def _consequent_phrase(predicate: str, object_class: str | None, single_object: str | None) -> str:
    """Readable, grammar-safe phrase for a pattern's consequent."""
    if object_class is not None:
        obj_desc = f'something that is a "{object_class}"'
    elif single_object is not None:
        obj_desc = f'"{single_object}"'
    else:
        return f"have the relation '{predicate}'"
    if "_" in predicate:
        return f"have '{predicate}' pointing to {obj_desc}"
    return f"{_base_verb(predicate)} {obj_desc}"


def _is_class_label(text: str) -> bool:
    return 0 < len(text.split()) <= _MAX_CLASS_LABEL_WORDS


def _shared_object_class(graph: FactGraph, objects: list[str]) -> str | None:
    """A class every object belongs to, if one exists (sorted-first tie-break)."""
    shared: set[str] | None = None
    for obj in objects:
        classes = {norm(c) for c in graph.classes_of(obj) if _is_class_label(c)}
        if not classes:
            return None
        shared = classes if shared is None else (shared & classes)
        if not shared:
            return None
    if not shared:
        return None
    winner_norm = sorted(shared)[0]
    # Recover a display form from any object's class list.
    for obj in objects:
        for cls in graph.classes_of(obj):
            if norm(cls) == winner_norm:
                return cls
    return winner_norm


def _entity_predicates(graph: FactGraph, entity_norm: str) -> dict[str, list]:
    """predicate -> facts for one entity, excluding classification edges."""
    out: dict[str, list] = {}
    for fact in graph.outgoing(entity_norm):
        if fact.predicate in _CLASSIFICATION:
            continue
        out.setdefault(fact.predicate, []).append(fact)
    return out


def discover_patterns(
    overlay_items: list[dict],
    min_support: int = 2,
    min_confidence: float = 0.5,
    max_patterns: int = 200,
    as_of: str = "",
) -> list[GraphPattern]:
    """Analyze the graph and return discovered patterns, best-first."""
    graph = FactGraph(overlay_items)

    # class label (norm) -> sorted member entity norms
    class_members: dict[str, list[str]] = {}
    class_display: dict[str, str] = {}
    # entity norm -> {predicate -> facts} (non-classification)
    entity_preds: dict[str, dict[str, list]] = {}

    entity_norms = sorted({norm(f.subject) for f in graph.facts})
    for entity in entity_norms:
        entity_preds[entity] = _entity_predicates(graph, entity)
        for cls in graph.classes_of(entity):
            if not _is_class_label(cls):
                continue
            key = norm(cls)
            class_display.setdefault(key, cls)
            class_members.setdefault(key, []).append(entity)

    patterns: list[GraphPattern] = []
    patterns.extend(
        _class_implication_patterns(
            graph, class_members, class_display, entity_preds,
            min_support, min_confidence, as_of,
        )
    )
    patterns.extend(
        _cooccurrence_patterns(
            graph, entity_preds, min_support, min_confidence, as_of,
        )
    )

    patterns.sort(key=lambda p: (-p.confidence, -p.support, p.pattern_id))
    return patterns[:max_patterns]


def _build_pattern(
    graph: FactGraph,
    kind: str,
    condition: dict,
    consequent: dict,
    description: str,
    pattern_id: str,
    matched: list[str],
    population: list[str],
    evidence_facts: list,
    counter_notes: dict[str, str],
    as_of: str,
) -> GraphPattern:
    counter_examples = [
        PatternCounterExample(entity=graph.display(e), note=counter_notes[e])
        for e in sorted(counter_notes)
    ][:_MAX_COUNTER_EXAMPLES]
    evidence = [
        [f.subject, f.predicate, f.object] for f in evidence_facts
    ]
    evidence_sorted = sorted(evidence, key=lambda t: (norm(t[0]), norm(t[1]), norm(t[2])))
    # Deduplicate while keeping sorted order.
    seen: set[tuple[str, str, str]] = set()
    unique_evidence: list[list[str]] = []
    for triple in evidence_sorted:
        key = (norm(triple[0]), norm(triple[1]), norm(triple[2]))
        if key not in seen:
            seen.add(key)
            unique_evidence.append(triple)
    return GraphPattern(
        pattern_id=pattern_id,
        kind=kind,  # type: ignore[arg-type]
        description=description,
        condition=condition,
        consequent=consequent,
        support=len(matched),
        population=len(population),
        confidence=len(matched) / len(population),
        matched_entities=[graph.display(e) for e in sorted(matched)][:_MAX_MATCHED_ENTITIES],
        counter_examples=counter_examples,
        supporting_evidence=unique_evidence[:_MAX_EVIDENCE],
        as_of=as_of,
    )


def _class_implication_patterns(
    graph: FactGraph,
    class_members: dict[str, list[str]],
    class_display: dict[str, str],
    entity_preds: dict[str, dict[str, list]],
    min_support: int,
    min_confidence: float,
    as_of: str,
) -> list[GraphPattern]:
    out: list[GraphPattern] = []
    for cls_norm in sorted(class_members):
        members = class_members[cls_norm]
        if len(members) < min_support:
            continue
        cls = class_display[cls_norm]
        predicates = sorted({
            p for m in members for p in entity_preds[m]
        })
        for predicate in predicates:
            matched = [m for m in members if predicate in entity_preds[m]]
            if len(matched) < min_support:
                continue
            confidence = len(matched) / len(members)
            if confidence < min_confidence:
                continue
            evidence_facts = [
                f for m in matched for f in entity_preds[m][predicate]
            ]
            objects = sorted(
                {f.object for f in evidence_facts}, key=norm
            )
            object_class = _shared_object_class(graph, objects)
            consequent: dict = {"predicate": predicate}
            single_object = objects[0] if len(objects) == 1 else None
            if object_class:
                consequent["object_class"] = object_class
            elif single_object is not None:
                consequent["object"] = single_object
            target_phrase = _consequent_phrase(predicate, object_class, single_object)
            description = (
                f"{len(matched)} of {len(members)} entities that are "
                f"\"{cls}\" also {target_phrase}"
            )
            counter_notes = {
                m: f"is \"{cls}\" but has no '{predicate}' fact"
                for m in members
                if m not in set(matched)
            }
            out.append(
                _build_pattern(
                    graph,
                    kind="class_implication",
                    condition={"class": cls},
                    consequent=consequent,
                    description=description,
                    pattern_id=f"class_implication:{_slug(cls)}=>{_slug(predicate)}",
                    matched=matched,
                    population=members,
                    evidence_facts=evidence_facts,
                    counter_notes=counter_notes,
                    as_of=as_of,
                )
            )
    return out


def _cooccurrence_patterns(
    graph: FactGraph,
    entity_preds: dict[str, dict[str, list]],
    min_support: int,
    min_confidence: float,
    as_of: str,
) -> list[GraphPattern]:
    # (p1, object_class_norm) -> {entity_norm: [condition facts]}
    condition_groups: dict[tuple[str, str], dict[str, list]] = {}
    class_display: dict[str, str] = {}

    for entity in sorted(entity_preds):
        for p1 in sorted(entity_preds[entity]):
            for fact in entity_preds[entity][p1]:
                for obj_class in graph.classes_of(fact.object):
                    if not _is_class_label(obj_class):
                        continue
                    key = (p1, norm(obj_class))
                    class_display.setdefault(norm(obj_class), obj_class)
                    condition_groups.setdefault(key, {}).setdefault(entity, []).append(fact)

    out: list[GraphPattern] = []
    for (p1, obj_class_norm) in sorted(condition_groups):
        members_facts = condition_groups[(p1, obj_class_norm)]
        members = sorted(members_facts)
        if len(members) < min_support:
            continue
        obj_class = class_display[obj_class_norm]
        consequent_predicates = sorted({
            p for m in members for p in entity_preds[m] if p != p1
        })
        for p2 in consequent_predicates:
            matched = [m for m in members if p2 in entity_preds[m]]
            if len(matched) < min_support:
                continue
            confidence = len(matched) / len(members)
            if confidence < min_confidence:
                continue
            evidence_facts = []
            for m in matched:
                evidence_facts.extend(members_facts[m])       # condition facts
                evidence_facts.extend(entity_preds[m][p2])    # consequent facts
            # The class facts that type the condition objects are evidence too.
            for m in matched:
                for cond_fact in members_facts[m]:
                    for class_fact in graph.find(cond_fact.object):
                        if (
                            class_fact.predicate in _CLASSIFICATION
                            and norm(class_fact.object) == obj_class_norm
                        ):
                            evidence_facts.append(class_fact)
            objects2 = sorted({
                f.object for m in matched for f in entity_preds[m][p2]
            }, key=norm)
            single_object2 = objects2[0] if len(objects2) == 1 else None
            object_class2 = _shared_object_class(graph, objects2)
            consequent: dict = {"predicate": p2}
            if object_class2:
                consequent["object_class"] = object_class2
            elif single_object2 is not None:
                consequent["object"] = single_object2
            target_phrase = _consequent_phrase(p2, object_class2, single_object2)
            description = (
                f"{len(matched)} of {len(members)} entities whose "
                f"'{p1}' points to a \"{obj_class}\" also {target_phrase}"
            )
            counter_notes = {
                m: f"'{p1}' points to a \"{obj_class}\" but has no '{p2}' fact"
                for m in members
                if m not in set(matched)
            }
            out.append(
                _build_pattern(
                    graph,
                    kind="cooccurrence",
                    condition={"predicate": p1, "object_class": obj_class},
                    consequent=consequent,
                    description=description,
                    pattern_id=(
                        f"cooccurrence:{_slug(p1)}@{_slug(obj_class)}=>{_slug(p2)}"
                    ),
                    matched=matched,
                    population=members,
                    evidence_facts=evidence_facts,
                    counter_notes=counter_notes,
                    as_of=as_of,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Using patterns in reasoning
# ---------------------------------------------------------------------------

def relevant_patterns(
    patterns: list[GraphPattern],
    entity: str,
    predicate: str | None = None,
    limit: int = 3,
) -> list[GraphPattern]:
    """Patterns that mention *entity* among their matched entities (optionally
    filtered to a consequent predicate). Order preserved (best-first)."""
    entity_norm = norm(entity)
    out: list[GraphPattern] = []
    for pattern in patterns:
        if predicate is not None and (
            norm(pattern.consequent.get("predicate", "")) != norm(predicate)
        ):
            continue
        matched_norms = {norm(e) for e in pattern.matched_entities}
        counter_norms = {norm(c.entity) for c in pattern.counter_examples}
        if entity_norm in matched_norms or entity_norm in counter_norms:
            out.append(pattern)
        if len(out) >= limit:
            break
    return out


def render_pattern_note(pattern: GraphPattern) -> str:
    """One-line, evidence-qualified remark suitable as answer context."""
    return (
        f"By the way, in my graph {pattern.description} "
        f"({pattern.confidence:.0%} of {pattern.population}; "
        f"{len(pattern.counter_examples)} counter-example(s))."
    )
