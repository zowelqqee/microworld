"""Explanatory chains: answer *why* a fact exists, not just *what* is true.

Given a target fact (subject, predicate, object), the builder produces an
``ExplanationChain`` whose every link is verified:

  1. The fact itself must be in the graph — direct, or derivable by an
     inference rule. A fact the system does not assert is never "explained"
     (decision = audit).
  2. For an *inferred* fact, the explanation is its proof chain: the base
     facts plus the rule that connects them. That chain closes by definition.
  3. For a *direct* fact, the builder grounds it in the rest of the graph:
       - the subject's classification (``X is_a C``);
       - optionally a discovered ``GraphPattern`` linking that class to the
         asked predicate (class-level context, marked as an observation);
       - a deterministic breadth-first walk from the object over explanatory
         predicates (classification, capability, activity, causal edges),
         looking for a node the subject itself links back to.
     If the walk reconnects to the subject the chain closes (decision =
     answer). If not, the system honestly reports a partial chain and shows
     the frontier it reached (decision = partial).

Universal: nothing here is entity- or domain-specific. Deterministic: sorted
expansion, stable tie-breaks, no randomness. No ML. No network.
"""

from __future__ import annotations

from worldpgt.cognition.inference_engine import InferenceWorkspace, run_inference
from worldpgt.reasoning.fact_graph import FactGraph, norm
from worldpgt.reasoning.types import (
    ExplanationChain,
    ExplanationStep,
    GraphPattern,
)
from worldpgt.relation_extraction_v2.relation_policy import PREDICATE_CLASSES

# Causal / outcome predicates worth following when tracing "why this makes
# sense". Domain-agnostic verbs only; concrete predicates ride along via the
# capability / activity predicate classes below.
_CAUSAL_PREDICATES = frozenset({
    "enables",
    "reduces",
    "lowers",
    "increases",
    "improves",
    "leads_to",
    "causes",
    "results_in",
    "supports",
    "allows",
    "drives",
    "powers",
    "advances",
    "makes_possible",
    "contributes_to",
    "accelerates",
    "requires",
})

_WALK_PREDICATES = (
    _CAUSAL_PREDICATES
    | PREDICATE_CLASSES["capability"]
    | PREDICATE_CLASSES["activity"]
    | PREDICATE_CLASSES["classification"]
)

_MAX_WALK_DEPTH = 4
_MAX_FRONTIER = 5
_MAX_CLASS_STEPS = 2


def _fact_step(fact) -> ExplanationStep:
    return ExplanationStep(
        kind="fact",
        subject=fact.subject,
        predicate=fact.predicate,
        object=fact.object,
    )


def _pattern_step(pattern: GraphPattern) -> ExplanationStep:
    return ExplanationStep(
        kind="pattern",
        text=pattern.description,
        pattern_id=pattern.pattern_id,
        confidence=pattern.confidence,
    )


def _dedupe_steps(steps: list[ExplanationStep]) -> list[ExplanationStep]:
    seen: set[tuple] = set()
    out: list[ExplanationStep] = []
    for step in steps:
        key = (
            step.kind,
            norm(step.subject),
            norm(step.predicate),
            norm(step.object),
            step.text,
            step.rule,
            step.pattern_id,
        )
        if key not in seen:
            seen.add(key)
            out.append(step)
    return out


def _find_inferred(
    overlay_items: list[dict],
    subject: str,
    predicate: str,
    obj: str,
    workspace: InferenceWorkspace | None = None,
):
    """Return the InferredFact matching the target triple, if any.

    *workspace* lets callers pass a workspace already computed for the full
    overlay (e.g. once at server startup) instead of paying for a fresh
    ``run_inference`` pass on every call.
    """
    if workspace is None:
        workspace = run_inference(overlay_items)
    for fact in workspace.facts:
        if (
            norm(fact.subject) == norm(subject)
            and norm(fact.predicate) == norm(predicate)
            and norm(fact.object) == norm(obj)
        ):
            return fact
    return None


def _class_pattern_for(
    patterns: list[GraphPattern],
    subject_classes: list[str],
    predicate: str,
) -> GraphPattern | None:
    """First pattern whose condition class matches the subject and whose
    consequent predicate matches the asked predicate."""
    class_norms = {norm(c) for c in subject_classes}
    for pattern in patterns:
        if pattern.kind != "class_implication":
            continue
        if norm(pattern.condition.get("class", "")) not in class_norms:
            continue
        if norm(pattern.consequent.get("predicate", "")) == norm(predicate):
            return pattern
    return None


def _explanatory_walk(
    graph: FactGraph,
    subject: str,
    predicate: str,
    obj: str,
) -> tuple[list, str | None, list[str]]:
    """Deterministic BFS from *obj* over explanatory predicates.

    Returns (path_facts, closing_fact_or_None, frontier_nodes).

    The walk closes when it reaches a node the subject links back to with any
    fact other than the target fact itself — that link is the closing fact.
    """
    subject_norm = norm(subject)
    target_key = (subject_norm, norm(predicate), norm(obj))

    def closing_fact(node_norm: str):
        for fact in graph.outgoing(subject_norm):
            key = (norm(fact.subject), norm(fact.predicate), norm(fact.object))
            if key == target_key:
                continue
            if norm(fact.object) == node_norm:
                return fact
        return None

    start = norm(obj)
    visited: set[str] = {start, subject_norm}
    # parent[node] = fact that reached it (for path reconstruction)
    parent: dict[str, object] = {}
    queue: list[tuple[str, int]] = [(start, 0)]
    frontier: list[str] = []

    while queue:
        node, depth = queue.pop(0)
        if depth >= _MAX_WALK_DEPTH:
            frontier.append(graph.display(node))
            continue
        advanced = False
        for fact in graph.outgoing(node):
            if fact.predicate not in _WALK_PREDICATES:
                continue
            nxt = norm(fact.object)
            if nxt in visited:
                continue
            visited.add(nxt)
            parent[nxt] = fact
            advanced = True

            # Closure check: does the subject link back to this node?
            close = closing_fact(nxt)
            if close is not None:
                path = []
                cursor = nxt
                while cursor in parent:
                    step_fact = parent[cursor]
                    path.append(step_fact)
                    cursor = norm(step_fact.subject)
                path.reverse()
                return path, close, []
            queue.append((nxt, depth + 1))
        if not advanced and depth > 0:
            frontier.append(graph.display(node))

    frontier_sorted = sorted(dict.fromkeys(frontier), key=norm)[:_MAX_FRONTIER]
    return [], None, frontier_sorted


def explain_fact(
    overlay_items: list[dict],
    subject: str,
    predicate: str,
    obj: str,
    patterns: list[GraphPattern] | None = None,
    workspace: InferenceWorkspace | None = None,
    graph: FactGraph | None = None,
) -> ExplanationChain:
    """Build a deterministic explanation chain for the target fact.

    *workspace* is an optional pre-computed :class:`InferenceWorkspace` (e.g.
    cached once at server startup) reused instead of re-running inference.
    *graph* is likewise an optional pre-built :class:`FactGraph` over the
    full overlay (building one is as expensive as inference itself) — pass
    the same instance already built for the cached pattern index to skip
    rebuilding it on every explanation request.
    """
    if graph is None:
        graph = FactGraph(overlay_items)

    direct_facts = graph.find(subject, predicate, obj)
    if direct_facts:
        return _explain_direct(
            graph, overlay_items, direct_facts[0], patterns or [], workspace
        )

    inferred = _find_inferred(overlay_items, subject, predicate, obj, workspace)
    if inferred is not None:
        steps: list[ExplanationStep] = []
        for s, p, o in inferred.chain:
            steps.append(ExplanationStep(kind="fact", subject=s, predicate=p, object=o))
        steps.append(
            ExplanationStep(
                kind="rule",
                subject=inferred.subject,
                predicate=inferred.predicate,
                object=inferred.object,
                rule=inferred.rule,
                confidence=inferred.confidence,
                text=(
                    f"rule {inferred.rule} derives: "
                    f"{inferred.subject} | {inferred.predicate} | {inferred.object}"
                ),
            )
        )
        return ExplanationChain(
            subject=inferred.subject,
            predicate=inferred.predicate,
            object=inferred.object,
            fact_status="inferred",
            decision="answer",
            steps=_dedupe_steps(steps),
        )

    return ExplanationChain(
        subject=subject,
        predicate=predicate,
        object=obj,
        fact_status="absent",
        decision="audit",
        audit_reason=(
            "that statement is not a fact in the graph (neither direct nor "
            "derivable by an inference rule), so there is nothing to explain"
        ),
    )


def _explain_direct(
    graph: FactGraph,
    overlay_items: list[dict],
    fact,
    patterns: list[GraphPattern],
    workspace: InferenceWorkspace | None = None,
) -> ExplanationChain:
    subject, predicate, obj = fact.subject, fact.predicate, fact.object
    steps: list[ExplanationStep] = [_fact_step(fact)]

    # Subject classification: X is_a C.
    subject_classes = graph.classes_of(subject)[:_MAX_CLASS_STEPS]
    for cls in subject_classes:
        for class_fact in graph.find(subject, None, cls):
            if class_fact.predicate in PREDICATE_CLASSES["classification"]:
                steps.append(_fact_step(class_fact))
                break

    # Class-level context from a discovered pattern (observation, not a fact).
    pattern = _class_pattern_for(patterns, subject_classes, predicate)
    if pattern is not None:
        steps.append(_pattern_step(pattern))

    # Rule grounding: the direct fact is *also* derivable — its proof chain is
    # a closed explanation on its own.
    inferred = _find_inferred(overlay_items, subject, predicate, obj, workspace)
    if inferred is not None:
        for s, p, o in inferred.chain:
            steps.append(ExplanationStep(kind="fact", subject=s, predicate=p, object=o))
        steps.append(
            ExplanationStep(
                kind="rule",
                subject=subject,
                predicate=predicate,
                object=obj,
                rule=inferred.rule,
                confidence=inferred.confidence,
                text=(
                    f"rule {inferred.rule} independently derives: "
                    f"{subject} | {predicate} | {obj}"
                ),
            )
        )
        return ExplanationChain(
            subject=subject,
            predicate=predicate,
            object=obj,
            fact_status="direct",
            decision="answer",
            steps=_dedupe_steps(steps),
        )

    # Explanatory walk from the object back toward the subject.
    path, closing, frontier = _explanatory_walk(graph, subject, predicate, obj)
    if closing is not None:
        for step_fact in path:
            steps.append(_fact_step(step_fact))
        steps.append(_fact_step(closing))
        return ExplanationChain(
            subject=subject,
            predicate=predicate,
            object=obj,
            fact_status="direct",
            decision="answer",
            steps=_dedupe_steps(steps),
        )

    # Honest partial: the fact is verified, context did not close the loop.
    context_found = len(steps) > 1 or bool(frontier)
    steps.append(
        ExplanationStep(
            kind="note",
            text=(
                "the fact is verified, but the explanatory chain does not close "
                "back to the subject"
                + (f"; exploration stopped at: {', '.join(frontier)}" if frontier else "")
            ),
        )
    )
    return ExplanationChain(
        subject=subject,
        predicate=predicate,
        object=obj,
        fact_status="direct",
        decision="partial",
        steps=_dedupe_steps(steps),
        frontier=frontier,
        audit_reason=None if context_found else "no explanatory context found in the graph",
    )
