"""Data-driven inference rule interpreter.

A miniature, deterministic, non-recursive Datalog-style engine. Rules are plain
data (``overlay_inference_rule`` items); this module unifies their patterns
against a normalized fact store and produces ``InferredFact`` results.

Adding a new inference rule requires *no code change* — only a new rule object
(see ``worldpgt/knowledge/base_inference_rules.json``).

Pipeline
--------
1. ``normalize_overlay`` flattens overlay items into uniform ``(subject,
   predicate, object, stability)`` facts. This is where two non-triple shapes in
   the overlay are lifted into triples:
     - ``overlay_definition`` → an ``is_a`` fact, plus a derived ``owned_by`` fact
       when the definition text names a creator ("operated by SpaceX",
       "division of Tesla").
     - configured relations get an inverse twin (``founded`` → ``founded_by``,
       ``leader_of`` → ``led_by``) so rules can be written in the natural
       subject direction.
2. ``apply_rules`` joins each rule's ``if_pattern`` against the facts, binding
   ``?variables``, enforcing ``type_guards`` and ``predicate_class`` filters, and
   instantiating ``then_pattern`` for every consistent binding.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from worldpgt.cognition.inference_engine import InferredFact
from worldpgt.relation_extraction_v2.relation_policy import (
    PREDICATE_CLASSES,
    predicate_in_class,
)


# ---------------------------------------------------------------------------
# Rule representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pattern:
    """One triple pattern; terms beginning with ``?`` are unification variables."""

    subject: str
    predicate: str
    object: str
    predicate_class: str | None = None


@dataclass(frozen=True)
class InferenceRule:
    """A parsed, data-defined inference rule."""

    rule_id: str
    name: str
    description: str
    if_pattern: tuple[Pattern, ...]
    then_pattern: Pattern
    confidence_rule: str = "min_stability"
    type_guards: dict[str, tuple[str, ...]] = field(default_factory=dict)
    constant_confidence: float = 0.5
    stability: str = "stable"
    enabled: bool = True
    # Negative guards: if ANY of these patterns matches under the current
    # bindings, the rule does not fire. Lets a rule say "infer X unless …".
    unless_pattern: tuple[Pattern, ...] = ()


def parse_rule(data: dict) -> InferenceRule:
    """Parse one ``overlay_inference_rule`` dict into an ``InferenceRule``."""
    return InferenceRule(
        rule_id=str(data.get("rule_id") or ""),
        name=str(data.get("name") or ""),
        description=str(data.get("description") or ""),
        if_pattern=tuple(_parse_pattern(p) for p in data.get("if_pattern", [])),
        then_pattern=_parse_pattern(data["then_pattern"]),
        confidence_rule=str(data.get("confidence_rule") or "min_stability"),
        type_guards={
            var: tuple(types) for var, types in (data.get("type_guards") or {}).items()
        },
        constant_confidence=float(data.get("constant_confidence", 0.5)),
        stability=str(data.get("stability") or "stable"),
        enabled=bool(data.get("enabled", True)),
        unless_pattern=tuple(_parse_pattern(p) for p in data.get("unless_pattern", [])),
    )


def _parse_pattern(data: dict) -> Pattern:
    return Pattern(
        subject=str(data.get("subject") or ""),
        predicate=str(data.get("predicate") or ""),
        object=str(data.get("object") or ""),
        predicate_class=data.get("predicate_class"),
    )


_BASE_RULES_PATH = (
    Path(__file__).resolve().parent.parent / "knowledge" / "base_inference_rules.json"
)


def load_base_rules(path: str | Path | None = None) -> list[InferenceRule]:
    """Load the base inference rule set from JSON."""
    rules_path = Path(path) if path is not None else _BASE_RULES_PATH
    data = json.loads(rules_path.read_text())
    return [parse_rule(item) for item in data]


# ---------------------------------------------------------------------------
# Normalized fact store
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Fact:
    subject: str
    predicate: str
    object: str
    stability: str = "semi_stable"


@dataclass(frozen=True)
class _FactIndex:
    by_predicate: dict[str, tuple[_Fact, ...]]
    by_subject: dict[str, tuple[_Fact, ...]]
    by_object: dict[str, tuple[_Fact, ...]]
    by_class: dict[str, tuple[_Fact, ...]]


# Relations that get an inverse twin so rules read in the natural direction.
_INVERSE_PREDICATES: dict[str, str] = {
    "founded": "founded_by",
    "leader_of": "led_by",
}

_STABILITY_SCORE: dict[str, float] = {
    "stable": 1.0,
    "semi_stable": 0.7,
    "volatile": 0.3,
}

# Creator phrases inside is_a definitions → a derived owned_by edge.
_CREATOR_RE = re.compile(
    r"\b(?:developed|produced|operated|published|manufactured)\s+by\s+([A-Z][^,.]*)"
    r"|\b(?:division|subsidiary|unit)\s+of\s+([A-Z][^,.]*)",
    re.IGNORECASE,
)
_QUALIFIER_RE = re.compile(r"\s+(?:for|to|in|at|with)\b.*$", re.IGNORECASE)


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


@dataclass(frozen=True)
class _NormalizedOverlay:
    facts: tuple[_Fact, ...]
    entity_types: dict[str, str]
    labels: frozenset[str]
    index: _FactIndex


def normalize_overlay(overlay_items: list[dict]) -> _NormalizedOverlay:
    """Flatten overlay items into a uniform triple store + an entity-type map."""
    entity_types: dict[str, str] = {}
    labels: set[str] = set()
    for item in overlay_items:
        if not isinstance(item, dict):
            continue
        otype = item.get("overlay_type")
        if otype == "overlay_entity":
            label = str(item.get("label") or "").strip()
            if label:
                labels.add(label)
                etype = str(item.get("entity_type") or "").strip()
                if etype:
                    entity_types[_norm(label)] = etype
        elif otype == "overlay_definition":
            subject = str(item.get("subject") or "").strip()
            if subject:
                labels.add(subject)
    label_by_norm = {_norm(label): label for label in labels}

    facts: list[_Fact] = []
    for item in overlay_items:
        if not isinstance(item, dict):
            continue
        otype = item.get("overlay_type")
        if otype == "overlay_relation":
            facts.extend(_facts_from_relation(item))
        elif otype == "overlay_definition":
            facts.extend(_facts_from_definition(item, label_by_norm))

    fact_tuple = tuple(facts)
    return _NormalizedOverlay(
        facts=fact_tuple,
        entity_types=entity_types,
        labels=frozenset(labels),
        index=_build_fact_index(fact_tuple),
    )


def _build_fact_index(facts: tuple[_Fact, ...]) -> _FactIndex:
    by_predicate: dict[str, list[_Fact]] = {}
    by_subject: dict[str, list[_Fact]] = {}
    by_object: dict[str, list[_Fact]] = {}
    by_class: dict[str, list[_Fact]] = {name: [] for name in PREDICATE_CLASSES}
    for fact in facts:
        by_predicate.setdefault(_norm(fact.predicate), []).append(fact)
        by_subject.setdefault(_norm(fact.subject), []).append(fact)
        by_object.setdefault(_norm(fact.object), []).append(fact)
        for class_name, members in PREDICATE_CLASSES.items():
            if fact.predicate in members:
                by_class.setdefault(class_name, []).append(fact)
    return _FactIndex(
        by_predicate={k: tuple(v) for k, v in by_predicate.items()},
        by_subject={k: tuple(v) for k, v in by_subject.items()},
        by_object={k: tuple(v) for k, v in by_object.items()},
        by_class={k: tuple(v) for k, v in by_class.items()},
    )


def _facts_from_relation(item: dict) -> list[_Fact]:
    subject = str(item.get("subject") or "").strip()
    predicate = str(item.get("predicate") or "").strip()
    obj = str(item.get("object") or "").strip()
    stability = str(item.get("stability") or "semi_stable")
    if not (subject and predicate and obj):
        return []
    out = [_Fact(subject, predicate, obj, stability)]
    inverse = _INVERSE_PREDICATES.get(predicate)
    if inverse:
        out.append(_Fact(obj, inverse, subject, stability))
    return out


def _facts_from_definition(item: dict, label_by_norm: dict[str, str]) -> list[_Fact]:
    subject = str(item.get("subject") or "").strip()
    definition = str(item.get("definition") or "").strip()
    stability = str(item.get("stability") or "stable")
    if not (subject and definition):
        return []
    out = [_Fact(subject, "is_a", definition, stability)]
    parent, is_subsidiary = _extract_creator(definition, label_by_norm)
    if parent and _norm(parent) != _norm(subject):
        out.append(_Fact(subject, "owned_by", parent, stability))
        # A division/subsidiary/unit is a *structural* part of its parent — it
        # inherits ownership chains but NOT the parent's capabilities (a finance
        # division doesn't build the parent's products). Tag it so capability
        # inheritance can veto on this marker while ownership rules still fire.
        if is_subsidiary:
            out.append(_Fact(subject, "subsidiary_of", parent, stability))
    return out


def _extract_creator(
    definition: str,
    label_by_norm: dict[str, str],
) -> tuple[str | None, bool]:
    """Return (parent_label, is_subsidiary) for a creator/parent phrase.

    ``is_subsidiary`` is True for "division/subsidiary/unit of X" phrasing (a
    structural part), False for "developed/operated by X" (a maker relationship).
    """
    m = _CREATOR_RE.search(definition)
    if not m:
        return None, False
    # Group 1 = "developed/operated by X"; group 2 = "division/subsidiary/unit of X".
    is_subsidiary = m.group(2) is not None
    raw = next(g for g in m.groups() if g is not None)
    raw = _QUALIFIER_RE.sub("", raw).strip().rstrip(".")
    raw_norm = _norm(raw)
    return label_by_norm.get(raw_norm), is_subsidiary


# ---------------------------------------------------------------------------
# Unification + rule application
# ---------------------------------------------------------------------------

def _is_var(term: str) -> bool:
    return term.startswith("?")


def _unify(pattern_term: str, value: str, bindings: dict[str, str]) -> bool:
    """Bind a variable or check a literal against a fact term."""
    if _is_var(pattern_term):
        existing = bindings.get(pattern_term)
        if existing is not None:
            return _norm(existing) == _norm(value)
        bindings[pattern_term] = value
        return True
    return _norm(pattern_term) == _norm(value)


def _match_pattern(
    pattern: Pattern,
    fact: _Fact,
    bindings: dict[str, str],
) -> dict[str, str] | None:
    """Return extended bindings if *fact* matches *pattern*, else None."""
    if pattern.predicate_class and not predicate_in_class(
        fact.predicate, pattern.predicate_class
    ):
        return None
    new = dict(bindings)
    if not _unify(pattern.predicate, fact.predicate, new):
        return None
    if not _unify(pattern.subject, fact.subject, new):
        return None
    if not _unify(pattern.object, fact.object, new):
        return None
    return new


def _resolve_term(term: str, bindings: dict[str, str]) -> str:
    return bindings.get(term, term) if _is_var(term) else term


def _bound_value(term: str, bindings: dict[str, str]) -> str | None:
    if _is_var(term):
        return bindings.get(term)
    return term


def _candidate_facts(
    pattern: Pattern,
    bindings: dict[str, str],
    normalized: _NormalizedOverlay,
) -> tuple[_Fact, ...]:
    """Return the smallest indexed fact subset that can match this pattern."""
    candidate_groups: list[tuple[_Fact, ...]] = []

    if pattern.predicate_class:
        candidate_groups.append(normalized.index.by_class.get(pattern.predicate_class, ()))
    else:
        predicate = _bound_value(pattern.predicate, bindings)
        if predicate is not None:
            candidate_groups.append(normalized.index.by_predicate.get(_norm(predicate), ()))

    subject = _bound_value(pattern.subject, bindings)
    if subject is not None:
        candidate_groups.append(normalized.index.by_subject.get(_norm(subject), ()))

    obj = _bound_value(pattern.object, bindings)
    if obj is not None:
        candidate_groups.append(normalized.index.by_object.get(_norm(obj), ()))

    if not candidate_groups:
        return normalized.facts
    return min(candidate_groups, key=len)


def _check_type_guards(
    type_guards: dict[str, tuple[str, ...]],
    bindings: dict[str, str],
    entity_types: dict[str, str],
) -> bool:
    """Every guarded variable must resolve to an entity of an allowed type."""
    for var, allowed in type_guards.items():
        value = bindings.get(var)
        if value is None:
            continue  # variable not part of this binding; cannot constrain
        etype = entity_types.get(_norm(value))
        if etype is None or etype not in allowed:
            return False
    return True


def _unless_violated(
    rule: InferenceRule,
    bindings: dict[str, str],
    normalized: _NormalizedOverlay,
) -> bool:
    """True if any negative-guard pattern matches a fact under *bindings*."""
    for pattern in rule.unless_pattern:
        for fact in _candidate_facts(pattern, bindings, normalized):
            if _match_pattern(pattern, fact, bindings) is not None:
                return True
    return False


def _confidence(rule: InferenceRule, stabilities: list[str]) -> float:
    if rule.confidence_rule == "constant":
        return rule.constant_confidence
    scores = [_STABILITY_SCORE.get(s, 0.5) for s in stabilities] or [0.5]
    if rule.confidence_rule == "max_stability":
        return max(scores)
    return min(scores)  # default: min_stability


def apply_rule(
    rule: InferenceRule,
    normalized: _NormalizedOverlay,
) -> list[InferredFact]:
    """Join *rule*'s if_pattern against the facts and instantiate then_pattern."""
    results: list[InferredFact] = []
    facts = normalized.facts

    def recurse(
        idx: int,
        bindings: dict[str, str],
        chain: list[tuple[str, str, str]],
        stabilities: list[str],
    ) -> None:
        if idx == len(rule.if_pattern):
            if not _check_type_guards(rule.type_guards, bindings, normalized.entity_types):
                return
            if _unless_violated(rule, bindings, normalized):
                return
            subject = _resolve_term(rule.then_pattern.subject, bindings)
            predicate = _resolve_term(rule.then_pattern.predicate, bindings)
            obj = _resolve_term(rule.then_pattern.object, bindings)
            if not (subject and predicate and obj):
                return
            if _norm(subject) == _norm(obj):
                return  # drop reflexive conclusions (cycles, self-pairs)
            results.append(InferredFact(
                subject=subject,
                predicate=predicate,
                object=obj,
                rule=rule.rule_id,
                chain=tuple(chain),
                confidence=_confidence(rule, stabilities),
            ))
            return
        pattern = rule.if_pattern[idx]
        for fact in _candidate_facts(pattern, bindings, normalized):
            extended = _match_pattern(pattern, fact, bindings)
            if extended is not None:
                recurse(
                    idx + 1,
                    extended,
                    [*chain, (fact.subject, fact.predicate, fact.object)],
                    [*stabilities, fact.stability],
                )

    recurse(0, {}, [], [])
    return results


def apply_rules(
    overlay_items: list[dict],
    rules: list[InferenceRule],
) -> list[InferredFact]:
    """Apply every enabled rule to the overlay; return deduplicated facts."""
    normalized = normalize_overlay(overlay_items)
    out: list[InferredFact] = []
    for rule in rules:
        if not rule.enabled:
            continue
        out.extend(apply_rule(rule, normalized))
    return _deduplicate(out)


def _deduplicate(facts: list[InferredFact]) -> list[InferredFact]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[InferredFact] = []
    for f in facts:
        key = (_norm(f.subject), _norm(f.predicate), _norm(f.object), f.rule)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out
