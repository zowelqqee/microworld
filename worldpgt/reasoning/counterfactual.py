"""Counterfactual reasoning as structural dependency analysis.

"What if X had not founded Y?" is not answerable as a factual question — and
the system never pretends it is. What it *can* do deterministically is show
what in its world model rests on that fact:

  removed_facts     — the concrete overlay facts the hypothesis removes;
  lost_inferences   — inferred facts that stop being derivable (computed by
                      re-running the inference engine on the reduced overlay
                      and diffing, with the removed links highlighted in each
                      lost proof chain);
  affected_patterns — discovered patterns whose supporting evidence includes
                      a removed fact, with confidence recomputed on the
                      reduced graph;
  dependent_entities — entities that appear in lost inferences.

If the target fact is not in the graph, the decision is an honest audit:
nothing can depend on a fact the system does not hold.

Universal, additive, deterministic. No ML. No network.
"""

from __future__ import annotations

from worldpgt.cognition.inference_engine import run_inference
from worldpgt.reasoning.fact_graph import norm
from worldpgt.reasoning.pattern_discovery import discover_patterns
from worldpgt.reasoning.types import (
    AffectedPattern,
    CounterfactualTrace,
    GraphPattern,
    LostInference,
    RemovedFact,
)

# Predicate pairs treated as the same underlying fact in either direction, so
# "What if SpaceX had not been founded by Musk?" finds "Musk founded SpaceX".
# Mirrors the inverse twins the rule interpreter derives.
_INVERSE_PAIRS: dict[str, str] = {
    "founded": "founded_by",
    "founded_by": "founded",
    "leader_of": "led_by",
    "led_by": "leader_of",
}


def _matches_target(
    item: dict,
    subject_norm: str,
    predicate: str | None,
    object_norm: str | None,
) -> bool:
    if item.get("overlay_type") != "overlay_relation":
        return False
    s = norm(str(item.get("subject") or ""))
    p = str(item.get("predicate") or "")
    o = norm(str(item.get("object") or ""))
    if not (s and p and o):
        return False

    endpoints = {s, o}
    if object_norm is None:
        # Entity-wide hypothesis: any relation touching the subject.
        if subject_norm not in endpoints:
            return False
        return predicate is None or norm(p) == norm(predicate)

    # Fact-level hypothesis: match in the stated direction, or in the inverse
    # direction for known inverse predicate pairs.
    if predicate is None:
        return endpoints == {subject_norm, object_norm}
    pn = norm(predicate)
    if s == subject_norm and o == object_norm and norm(p) == pn:
        return True
    inverse = _INVERSE_PAIRS.get(predicate)
    if inverse and s == object_norm and o == subject_norm and norm(p) == norm(inverse):
        return True
    return False


def find_target_items(
    overlay_items: list[dict],
    subject: str,
    predicate: str | None = None,
    obj: str | None = None,
) -> list[dict]:
    """Overlay relation items the counterfactual hypothesis removes."""
    subject_norm = norm(subject)
    object_norm = norm(obj) if obj else None
    return [
        item
        for item in overlay_items
        if _matches_target(item, subject_norm, predicate, object_norm)
    ]


def _fact_key(subject: str, predicate: str, obj: str) -> tuple[str, str, str]:
    return (norm(subject), norm(predicate), norm(obj))


def analyze_counterfactual(
    overlay_items: list[dict],
    subject: str,
    predicate: str | None = None,
    obj: str | None = None,
    patterns: list[GraphPattern] | None = None,
    min_support: int = 2,
    min_confidence: float = 0.5,
) -> CounterfactualTrace:
    """Compute the structural dependency trace for a hypothetical removal.

    When *patterns* is None, patterns are discovered on the fly from the full
    overlay (so affected-pattern analysis always has a baseline to diff).
    """
    targets = find_target_items(overlay_items, subject, predicate, obj)
    if not targets:
        return CounterfactualTrace(
            target_subject=subject,
            target_predicate=predicate,
            target_object=obj,
            decision="audit",
            audit_reason=(
                "that fact is not in the graph, so nothing in the world model "
                "depends on it"
            ),
        )

    removed_facts = [
        RemovedFact(
            subject=str(t.get("subject") or ""),
            predicate=str(t.get("predicate") or ""),
            object=str(t.get("object") or ""),
        )
        for t in targets
    ]
    removed_facts.sort(key=lambda f: (norm(f.subject), norm(f.predicate), norm(f.object)))
    removed_ids = {id(t) for t in targets}
    reduced_items = [item for item in overlay_items if id(item) not in removed_ids]

    # Direct removed triples, plus their inverse twins — proof chains may cite
    # either direction of an inverse pair.
    removed_keys: set[tuple[str, str, str]] = set()
    for f in removed_facts:
        removed_keys.add(_fact_key(f.subject, f.predicate, f.object))
        inverse = _INVERSE_PAIRS.get(f.predicate)
        if inverse:
            removed_keys.add(_fact_key(f.object, inverse, f.subject))

    # --- Inference diff -----------------------------------------------------
    baseline = run_inference(overlay_items)
    reduced = run_inference(reduced_items)
    reduced_keys = {
        (norm(f.subject), norm(f.predicate), norm(f.object), f.rule)
        for f in reduced.facts
    }
    lost_inferences: list[LostInference] = []
    for fact in baseline.facts:
        key = (norm(fact.subject), norm(fact.predicate), norm(fact.object), fact.rule)
        if key in reduced_keys:
            continue
        removed_links = [
            [s, p, o]
            for (s, p, o) in fact.chain
            if _fact_key(s, p, o) in removed_keys
        ]
        lost_inferences.append(
            LostInference(
                subject=fact.subject,
                predicate=fact.predicate,
                object=fact.object,
                rule=fact.rule,
                chain=[list(step) for step in fact.chain],
                removed_links=removed_links,
            )
        )
    lost_inferences.sort(
        key=lambda f: (norm(f.subject), norm(f.predicate), norm(f.object), f.rule)
    )

    # --- Pattern diff ---------------------------------------------------------
    if patterns is None:
        patterns = discover_patterns(
            overlay_items, min_support=min_support, min_confidence=min_confidence
        )
    reduced_patterns = {
        p.pattern_id: p
        for p in discover_patterns(
            reduced_items, min_support=min_support, min_confidence=min_confidence
        )
    }
    affected_patterns: list[AffectedPattern] = []
    for pattern in patterns:
        removed_evidence = [
            triple
            for triple in pattern.supporting_evidence
            if _fact_key(triple[0], triple[1], triple[2]) in removed_keys
        ]
        if not removed_evidence:
            continue
        after = reduced_patterns.get(pattern.pattern_id)
        affected_patterns.append(
            AffectedPattern(
                pattern_id=pattern.pattern_id,
                description=pattern.description,
                old_confidence=pattern.confidence,
                new_confidence=after.confidence if after is not None else None,
                old_support=pattern.support,
                new_support=after.support if after is not None else 0,
                removed_evidence=[list(t) for t in removed_evidence],
            )
        )
    affected_patterns.sort(key=lambda p: p.pattern_id)

    dependent_entities = sorted(
        {f.subject for f in lost_inferences} | {f.object for f in lost_inferences},
        key=norm,
    )

    return CounterfactualTrace(
        target_subject=subject,
        target_predicate=predicate,
        target_object=obj,
        decision="analysis",
        removed_facts=removed_facts,
        lost_inferences=lost_inferences,
        affected_patterns=affected_patterns,
        dependent_entities=dependent_entities,
    )


def render(trace: CounterfactualTrace) -> str:
    """Render a counterfactual trace as an honest structural analysis."""
    if trace.decision == "audit":
        return (
            f"[audit] I can't analyze that counterfactual: "
            f"{trace.audit_reason or 'the fact is not in the graph'}.\n"
            "Decision: audit."
        )

    hypothesis = trace.target_subject
    if trace.target_predicate and trace.target_object:
        hypothesis = (
            f"{trace.target_subject} | {trace.target_predicate} | {trace.target_object}"
        )

    lines: list[str] = [
        "I can't answer a counterfactual as a fact — but I can show what in "
        "my world model rests on it.",
        "",
        f"Hypothetically removed: {hypothesis}",
    ]
    lines.append("")
    lines.append(f"Facts removed ({len(trace.removed_facts)}):")
    for f in trace.removed_facts:
        lines.append(f"  - {f.display()}")

    lines.append("")
    if trace.lost_inferences:
        lines.append(
            f"Inferences that stop being derivable ({len(trace.lost_inferences)}):"
        )
        for f in trace.lost_inferences:
            lines.append(f"  - {f.display()}")
    else:
        lines.append("No inferred facts depend on it.")

    lines.append("")
    if trace.affected_patterns:
        lines.append(f"Patterns losing evidence ({len(trace.affected_patterns)}):")
        for p in trace.affected_patterns:
            if p.new_confidence is None:
                change = "no longer meets discovery thresholds"
            else:
                change = f"confidence {p.old_confidence:.0%} → {p.new_confidence:.0%}"
            lines.append(f"  - {p.description} ({change})")
    else:
        lines.append("No discovered patterns lose evidence.")

    if trace.dependent_entities:
        lines.append("")
        lines.append(
            f"Entities affected: {', '.join(trace.dependent_entities)}."
        )

    lines.append("")
    lines.append("Support: structural dependency analysis over verified facts.")
    lines.append("Decision: analysis.")
    return "\n".join(lines)
