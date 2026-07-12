"""Graph-native cognitive loop over a working semantic state.

This module makes cognitive moves act as graph operators, not answer
templates. Each move has a precondition over the working semantic state and
an effect that mutates that state (activation, evidence boundaries, notes,
pending questions). The loop iterates move selection until a stop condition
emerges from the graph itself, then emits an ``AnswerPlan`` that a renderer
can execute downstream.

Trust rules:
- ``ReasoningTrace`` is the only factual boundary. ``AnswerPlan.facts_to_say``
  can only contain evidence already admitted into the trace workspace.
- Community cognitive patterns are compiled into pattern -> move edges; they
  add behavioral pressure only. Any pattern with
  ``factual_support_allowed=True`` is discarded before graph construction.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Literal

from worldpgt.cognition.semantic_thought_graph import (
    SemanticThoughtEdge,
    SemanticThoughtNode,
    _coerce_pattern,
    _slug,
    _tokens,
)
from worldpgt.cognition.types import ActionPlan, ReasoningTrace
from worldpgt.community_context.types import CognitivePatternEvent

# Mirrors entity_qa.symbolic_text_generator._CONTENT_BUCKETS. Duplicated
# rather than imported to avoid a cross-package import edge at cognition
# package init time; both sides are the same closed set of EvidenceRole
# values that the existing renderer treats as content buckets.
_CONTENT_BUCKET_ROLES = frozenset(
    {
        "mechanism",
        "purpose",
        "activity",
        "origin",
        "ownership",
        "classification",
        "recognition",
        "other",
    }
)

# The loop reuses the semantic graph node/edge shapes; the aliases document
# that these are the cognitive-graph concepts, not new parallel types.
CognitiveGraphNode = SemanticThoughtNode
CognitiveGraphEdge = SemanticThoughtEdge


CognitiveMoveKind = Literal[
    "decompose_question",
    "activate_related_concepts",
    "inspect_evidence_boundary",
    "separate_fact_from_interpretation",
    "check_missing_evidence",
    "compare_concepts",
    "ground_in_supported_example",
    "reduce_to_minimal_repro",
    "detect_likely_mistake",
    "ask_missing_constraint",
    "choose_explanation_depth",
    "choose_tone",
    "verify_before_answering",
    "stop_when_unsupported",
]

CognitiveLoopStopReason = Literal[
    "answer_plan_ready",
    "blocked_unsupported",
    "no_applicable_moves",
    "max_iterations",
]

_ALL_MOVE_KINDS: tuple[CognitiveMoveKind, ...] = (
    "decompose_question",
    "activate_related_concepts",
    "inspect_evidence_boundary",
    "separate_fact_from_interpretation",
    "check_missing_evidence",
    "compare_concepts",
    "ground_in_supported_example",
    "reduce_to_minimal_repro",
    "detect_likely_mistake",
    "ask_missing_constraint",
    "choose_explanation_depth",
    "choose_tone",
    "verify_before_answering",
    "stop_when_unsupported",
)

# Pattern events compile into edges toward these moves. The edge is the only
# channel: a pattern that never activates cannot pull its moves in.
_PATTERN_KIND_TO_MOVES: dict[str, tuple[CognitiveMoveKind, ...]] = {
    "explanation_pattern": (
        "choose_explanation_depth",
        "ground_in_supported_example",
        "separate_fact_from_interpretation",
    ),
    "debugging_pattern": ("reduce_to_minimal_repro", "check_missing_evidence"),
    "mistake_pattern": ("detect_likely_mistake",),
    "analogy_pattern": ("compare_concepts", "ground_in_supported_example"),
    "question_pattern": ("ask_missing_constraint", "decompose_question"),
    "uncertainty_pattern": (
        "separate_fact_from_interpretation",
        "check_missing_evidence",
        "stop_when_unsupported",
    ),
    "style_tone_pattern": ("choose_tone", "choose_explanation_depth"),
    "procedure_pattern": ("decompose_question", "reduce_to_minimal_repro"),
    "concept_pattern": ("activate_related_concepts",),
    "answer_shape_pattern": ("choose_explanation_depth",),
}

_EXPLANATION_SHAPING_MOVES = frozenset(
    {
        "decompose_question",
        "compare_concepts",
        "ground_in_supported_example",
        "reduce_to_minimal_repro",
        "separate_fact_from_interpretation",
        "choose_explanation_depth",
    }
)

_GROUNDING_ROLE_PREFERENCE = ("mechanism", "activity", "purpose", "definition", "ownership")


@dataclass(frozen=True)
class CognitiveMove:
    """Catalog identity of one reusable cognitive operator."""

    kind: CognitiveMoveKind
    description: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "description": self.description}


MOVE_CATALOG: tuple[CognitiveMove, ...] = (
    CognitiveMove("decompose_question", "split the task into its subgoal nodes and activate them"),
    CognitiveMove("activate_related_concepts", "spread one extra activation round from admitted evidence"),
    CognitiveMove("inspect_evidence_boundary", "split roles into supported and missing before saying anything"),
    CognitiveMove("separate_fact_from_interpretation", "forbid interpretation from leaking into factual claims"),
    CognitiveMove("check_missing_evidence", "inspect active gap nodes and turn them into stated uncertainty"),
    CognitiveMove("compare_concepts", "contrast two grounded concept nodes instead of describing one"),
    CognitiveMove("ground_in_supported_example", "attach one admitted evidence item as the working example"),
    CognitiveMove("reduce_to_minimal_repro", "organize the answer around a minimal reproduction"),
    CognitiveMove("detect_likely_mistake", "surface a community-observed mistake as a check, not a fact"),
    CognitiveMove("ask_missing_constraint", "convert inspected gaps into one focused follow-up question"),
    CognitiveMove("choose_explanation_depth", "pick explanation depth from the supported evidence shape"),
    CognitiveMove("choose_tone", "adopt tone pressure from an activated style pattern"),
    CognitiveMove("verify_before_answering", "re-check every planned claim against the admitted workspace"),
    CognitiveMove("stop_when_unsupported", "refuse to plan an answer when no admitted evidence supports one"),
)


@dataclass(frozen=True)
class MoveCandidate:
    """One applicable move with its graph-derived score."""

    kind: CognitiveMoveKind
    score: float
    reason: str
    activated_by: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "score": round(self.score, 4),
            "reason": self.reason,
            "activated_by": list(self.activated_by),
        }


@dataclass(frozen=True)
class RejectedMove:
    """One move whose precondition failed, with the graph-state reason."""

    kind: CognitiveMoveKind
    reason: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "reason": self.reason}


@dataclass(frozen=True)
class MoveApplication:
    """One selected move and the state changes it produced."""

    kind: CognitiveMoveKind
    score: float
    reason: str
    effects: tuple[str, ...] = ()
    activated_by: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "score": round(self.score, 4),
            "reason": self.reason,
            "effects": list(self.effects),
            "activated_by": list(self.activated_by),
        }


@dataclass(frozen=True)
class LoopIteration:
    """One selection step: candidates, rejections, and the applied move."""

    index: int
    candidates: tuple[MoveCandidate, ...] = ()
    rejected: tuple[RejectedMove, ...] = ()
    selected: MoveApplication | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "rejected": [rejection.to_dict() for rejection in self.rejected],
            "selected": self.selected.to_dict() if self.selected else None,
        }


@dataclass(frozen=True)
class AnswerPlan:
    """What the renderer may and may not do, decided by the loop."""

    facts_to_say: tuple[tuple[str, str], ...] = ()
    facts_not_allowed: tuple[str, ...] = ()
    explanation_strategy: tuple[str, ...] = ()
    uncertainty_to_state: tuple[str, ...] = ()
    examples: tuple[tuple[str, str], ...] = ()
    next_useful_question: str | None = None
    explanation_depth: str | None = None
    tone: str | None = None
    verified: bool = False
    mistake_checks: tuple[str, ...] = ()
    repro_checklist: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "facts_to_say": [{"role": role, "text": text} for role, text in self.facts_to_say],
            "facts_not_allowed": list(self.facts_not_allowed),
            "explanation_strategy": list(self.explanation_strategy),
            "uncertainty_to_state": list(self.uncertainty_to_state),
            "examples": [{"kind": kind, "text": text} for kind, text in self.examples],
            "next_useful_question": self.next_useful_question,
            "explanation_depth": self.explanation_depth,
            "tone": self.tone,
            "verified": self.verified,
            "mistake_checks": list(self.mistake_checks),
            "repro_checklist": list(self.repro_checklist),
        }


@dataclass
class WorkingSemanticState:
    """Mutable graph state the loop reads and each move effect updates."""

    nodes: dict[str, CognitiveGraphNode]
    edges: tuple[CognitiveGraphEdge, ...]
    activation: dict[str, float]
    pattern_events: dict[str, CognitivePatternEvent]
    grounded_concepts: tuple[str, ...]
    open_gap_ids: set[str] = field(default_factory=set)
    inspected_gap_ids: set[str] = field(default_factory=set)
    applied: list[MoveApplication] = field(default_factory=list)
    applied_kinds: set[str] = field(default_factory=set)
    boundary_inspected: bool = False
    facts_allowed: list[tuple[str, str]] = field(default_factory=list)
    facts_forbidden: list[str] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)
    examples: list[tuple[str, str]] = field(default_factory=list)
    pending_questions: list[str] = field(default_factory=list)
    explanation_moves: list[str] = field(default_factory=list)
    explanation_depth: str | None = None
    tone: str | None = None
    verified: bool = False
    stop_reason: CognitiveLoopStopReason | None = None
    mistake_checks: list[str] = field(default_factory=list)
    repro_checklist: list[str] = field(default_factory=list)

    def pattern_sources(self, move_kind: CognitiveMoveKind) -> tuple[str, ...]:
        """Pattern nodes with an active edge into the given move node."""

        target = f"move:{move_kind}"
        sources = sorted(
            edge.source
            for edge in self.edges
            if edge.target == target
            and edge.source.startswith("pattern:")
            and self.activation.get(edge.source, 0.0) > 1.0
        )
        return tuple(dict.fromkeys(sources))

    def gap_ids_by_activation(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                self.open_gap_ids,
                key=lambda node_id: (-self.activation.get(node_id, 0.0), node_id),
            )
        )


@dataclass(frozen=True)
class CognitiveLoopTrace:
    """Full inspectable record of one cognitive graph loop."""

    subject: str
    task_intent: str
    question: str
    nodes: tuple[CognitiveGraphNode, ...] = ()
    edges: tuple[CognitiveGraphEdge, ...] = ()
    initial_activation: dict[str, float] = field(default_factory=dict)
    iterations: tuple[LoopIteration, ...] = ()
    applied_moves: tuple[MoveApplication, ...] = ()
    facts_allowed_roles: tuple[str, ...] = ()
    missing_roles: tuple[str, ...] = ()
    stop_reason: CognitiveLoopStopReason = "no_applicable_moves"
    answer_plan: AnswerPlan = field(default_factory=AnswerPlan)
    factual_support_allowed_from_patterns: bool = False

    @property
    def applied_kinds(self) -> tuple[str, ...]:
        return tuple(application.kind for application in self.applied_moves)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "task_intent": self.task_intent,
            "question": self.question,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "initial_activation": {
                node_id: round(value, 4)
                for node_id, value in sorted(
                    self.initial_activation.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            },
            "iterations": [iteration.to_dict() for iteration in self.iterations],
            "applied_moves": [application.to_dict() for application in self.applied_moves],
            "facts_allowed_roles": list(self.facts_allowed_roles),
            "missing_roles": list(self.missing_roles),
            "stop_reason": self.stop_reason,
            "answer_plan": self.answer_plan.to_dict(),
            "factual_support_allowed_from_patterns": self.factual_support_allowed_from_patterns,
        }


def run_cognitive_graph_loop(
    trace: ReasoningTrace,
    *,
    question: str = "",
    cognitive_patterns: Iterable[CognitivePatternEvent | dict] = (),
    max_iterations: int = 12,
) -> CognitiveLoopTrace:
    """Iterate cognitive move selection over a working semantic graph.

    Moves are selected from graph activation and working-state predicates,
    never from a fixed sequence. Each applied move mutates the working state,
    so the next selection sees a different graph.
    """

    safe_patterns = tuple(
        event
        for event in (_coerce_pattern(raw) for raw in cognitive_patterns)
        if event is not None and event.factual_support_allowed is False
    )
    state = _initial_state(trace, question=question, patterns=safe_patterns)
    initial_activation = dict(state.activation)

    iterations: list[LoopIteration] = []
    while len(iterations) < max_iterations and state.stop_reason is None:
        candidates: list[MoveCandidate] = []
        rejected: list[RejectedMove] = []
        for kind in _ALL_MOVE_KINDS:
            applicable, reason = _precondition(kind, state, trace)
            if not applicable:
                rejected.append(RejectedMove(kind=kind, reason=reason))
                continue
            score, score_reason, activated_by = _score(kind, state, trace)
            candidates.append(
                MoveCandidate(
                    kind=kind,
                    score=score,
                    reason=score_reason,
                    activated_by=activated_by,
                )
            )
        candidates.sort(key=lambda candidate: (-candidate.score, candidate.kind))

        if not candidates:
            state.stop_reason = "answer_plan_ready" if state.verified else "no_applicable_moves"
            iterations.append(
                LoopIteration(index=len(iterations), rejected=tuple(rejected))
            )
            break

        best = candidates[0]
        application = _apply(best, state, trace)
        state.applied.append(application)
        state.applied_kinds.add(best.kind)
        iterations.append(
            LoopIteration(
                index=len(iterations),
                candidates=tuple(candidates),
                rejected=tuple(rejected),
                selected=application,
            )
        )

    if state.stop_reason is None:
        state.stop_reason = "answer_plan_ready" if state.verified else "max_iterations"

    return CognitiveLoopTrace(
        subject=trace.task.subject,
        task_intent=trace.task.intent,
        question=question,
        nodes=tuple(state.nodes.values()),
        edges=state.edges,
        initial_activation=initial_activation,
        iterations=tuple(iterations),
        applied_moves=tuple(state.applied),
        facts_allowed_roles=tuple(dict.fromkeys(role for role, _text in state.facts_allowed)),
        missing_roles=tuple(
            dict.fromkeys(
                node_id.removeprefix("gap:")
                for node_id in sorted(state.open_gap_ids | state.inspected_gap_ids)
            )
        ),
        stop_reason=state.stop_reason,
        answer_plan=_answer_plan(state),
        factual_support_allowed_from_patterns=False,
    )


# -- graph construction ------------------------------------------------------


class _Builder:
    def __init__(self) -> None:
        self.nodes: dict[str, CognitiveGraphNode] = {}
        self.edges: list[CognitiveGraphEdge] = []

    def node(
        self,
        node_id: str,
        kind: str,
        label: str,
        *,
        weight: float = 1.0,
        factual_support_allowed: bool = False,
    ) -> str:
        existing = self.nodes.get(node_id)
        if existing is None or weight > existing.weight:
            self.nodes[node_id] = CognitiveGraphNode(
                node_id=node_id,
                kind=kind,  # type: ignore[arg-type]
                label=label,
                weight=weight,
                factual_support_allowed=factual_support_allowed,
            )
        return node_id

    def edge(self, source: str, target: str, relation: str, *, weight: float = 1.0) -> None:
        if source in self.nodes and target in self.nodes:
            self.edges.append(CognitiveGraphEdge(source, target, relation, weight))


def _initial_state(
    trace: ReasoningTrace,
    *,
    question: str,
    patterns: tuple[CognitivePatternEvent, ...],
) -> WorkingSemanticState:
    builder = _Builder()
    task_id = builder.node("task:current", "task", trace.task.intent, weight=2.0)
    subject_id = builder.node(
        f"subject:{_slug(trace.task.subject)}", "subject", trace.task.subject, weight=2.0
    )
    builder.edge(task_id, subject_id, "about", weight=1.0)

    for kind in _ALL_MOVE_KINDS:
        builder.node(f"move:{kind}", "move", kind, weight=0.5)
    builder.edge(task_id, "move:verify_before_answering", "always_relevant", weight=0.3)

    gap_ids: set[str] = set()
    for subgoal in trace.subgoals:
        subgoal_id = builder.node(
            f"subgoal:{_slug(subgoal.name)}",
            "subgoal",
            subgoal.name,
            weight={"missing": 2.0, "partial": 1.6}.get(subgoal.status, 1.0),
        )
        builder.edge(task_id, subgoal_id, "needs", weight=1.0)
        for role in subgoal.required_roles:
            role_id = builder.node(f"role:{role}", "evidence_role", str(role), weight=1.2)
            builder.edge(subgoal_id, role_id, "requires_role", weight=0.8)
        for role in subgoal.missing_roles:
            gap_id = builder.node(f"gap:{role}", "gap", f"missing {role}", weight=2.4)
            gap_ids.add(gap_id)
            builder.edge(subgoal_id, gap_id, "blocked_by", weight=1.4)

    evidence_roles: set[str] = set()
    evidence_ids: list[str] = []
    for item in trace.workspace.items:
        role_id = builder.node(f"role:{item.role}", "evidence_role", str(item.role), weight=1.4)
        evidence_roles.add(str(item.role))
        item_id = builder.node(
            f"evidence:{item.role}:{_slug(item.text)[:48]}",
            "evidence_item",
            item.text,
            weight=1.0,
            factual_support_allowed=True,
        )
        evidence_ids.append(item_id)
        builder.edge(subject_id, role_id, "has_evidence_role", weight=0.9)
        builder.edge(role_id, item_id, "supported_by", weight=0.7)

    for missing in trace.missing_evidence:
        gap_id = builder.node(
            f"gap:{missing.role}",
            "gap",
            missing.reason or f"missing {missing.role}",
            weight=2.8,
        )
        gap_ids.add(gap_id)
        builder.edge(task_id, gap_id, "blocked_by", weight=1.5)

    grounded_concepts = _concept_nodes(builder, trace, subject_id, question)
    pattern_events = _pattern_nodes(builder, trace, task_id, question, patterns)

    for gap_id in sorted(gap_ids):
        builder.edge(gap_id, "move:check_missing_evidence", "activates_move", weight=1.5)
        builder.edge(gap_id, "move:ask_missing_constraint", "suggests_followup", weight=1.0)
        builder.edge(gap_id, "move:separate_fact_from_interpretation", "activates_move", weight=0.8)
        builder.edge(gap_id, "move:stop_when_unsupported", "activates_move", weight=0.6)
    for role in sorted(evidence_roles):
        role_id = f"role:{role}"
        builder.edge(role_id, "move:inspect_evidence_boundary", "activates_move", weight=0.8)
        builder.edge(role_id, "move:ground_in_supported_example", "activates_move", weight=0.6)
        builder.edge(role_id, "move:choose_explanation_depth", "activates_move", weight=0.5)
    if len(trace.subgoals) >= 2:
        for subgoal in trace.subgoals:
            builder.edge(
                f"subgoal:{_slug(subgoal.name)}",
                "move:decompose_question",
                "activates_move",
                weight=0.9,
            )
    for item_id in evidence_ids:
        builder.edge(item_id, "move:activate_related_concepts", "activates_move", weight=0.4)
    for concept_id in grounded_concepts:
        builder.edge(concept_id, "move:compare_concepts", "activates_move", weight=1.2)

    activation = _propagate(builder.nodes, builder.edges)
    return WorkingSemanticState(
        nodes=builder.nodes,
        edges=tuple(builder.edges),
        activation=activation,
        pattern_events=pattern_events,
        grounded_concepts=grounded_concepts,
        open_gap_ids=set(gap_ids),
    )


def _concept_nodes(
    builder: _Builder,
    trace: ReasoningTrace,
    subject_id: str,
    question: str,
) -> tuple[str, ...]:
    subject_tokens = _tokens(trace.task.subject)
    evidence_tokens: set[str] = set()
    for item in trace.workspace.items:
        evidence_tokens |= _tokens(item.text)
    grounded: list[str] = []
    for token in sorted(_tokens(question)):
        if token in subject_tokens:
            continue
        is_grounded = token in evidence_tokens
        concept_id = builder.node(
            f"concept:{token}",
            "concept",
            token,
            weight=1.5 if is_grounded else 0.5,
        )
        builder.edge(concept_id, subject_id, "mentioned_with", weight=0.5)
        if is_grounded:
            grounded.append(concept_id)
    return tuple(grounded)


def _pattern_nodes(
    builder: _Builder,
    trace: ReasoningTrace,
    task_id: str,
    question: str,
    patterns: tuple[CognitivePatternEvent, ...],
) -> dict[str, CognitivePatternEvent]:
    query_terms = _tokens(question or trace.task.question_style)
    pattern_events: dict[str, CognitivePatternEvent] = {}
    for event in patterns:
        event_terms = _tokens(
            " ".join([event.topic, event.kind, event.pattern, " ".join(event.steps)])
        )
        matched = query_terms & event_terms
        if not matched and event.topic and _slug(event.topic) not in _slug(trace.task.subject):
            continue
        pattern_id = builder.node(
            f"pattern:{event.event_id}",
            "pattern",
            event.pattern,
            weight=1.0 + min(event.signal_count, 5) * 0.15 + len(matched) * 0.2,
            factual_support_allowed=False,
        )
        pattern_events[pattern_id] = event
        builder.edge(task_id, pattern_id, "may_use_pattern", weight=0.7 + len(matched) * 0.1)
        for move in _PATTERN_KIND_TO_MOVES.get(event.kind, ()):
            builder.edge(pattern_id, f"move:{move}", "suggests_move", weight=1.2)
    return pattern_events


def _propagate(
    nodes: dict[str, CognitiveGraphNode],
    edges: list[CognitiveGraphEdge],
    *,
    rounds: int = 3,
    damping: float = 0.4,
    cap: float = 12.0,
) -> dict[str, float]:
    activation = {node_id: node.weight for node_id, node in nodes.items()}
    outgoing: dict[str, list[CognitiveGraphEdge]] = defaultdict(list)
    for edge in edges:
        outgoing[edge.source].append(edge)
    for _ in range(rounds):
        delta: dict[str, float] = defaultdict(float)
        for source in sorted(activation):
            for edge in outgoing.get(source, ()):
                delta[edge.target] += activation[source] * edge.weight * damping
        for node_id in sorted(delta):
            activation[node_id] = min(cap, activation.get(node_id, 0.0) + delta[node_id])
    return activation


# -- move preconditions ------------------------------------------------------


def _precondition(
    kind: CognitiveMoveKind,
    state: WorkingSemanticState,
    trace: ReasoningTrace,
) -> tuple[bool, str]:
    if kind in state.applied_kinds:
        return False, "already applied in this loop"

    if kind == "decompose_question":
        if len(trace.subgoals) >= 2 or state.pattern_sources(kind):
            return True, ""
        return False, "fewer than two subgoal nodes and no question/procedure pattern activation"
    if kind == "activate_related_concepts":
        if trace.workspace.items:
            return True, ""
        return False, "no admitted evidence nodes to spread activation from"
    if kind == "inspect_evidence_boundary":
        if trace.workspace.items or state.open_gap_ids:
            return True, ""
        return False, "no evidence or gap nodes to bound"
    if kind == "separate_fact_from_interpretation":
        if not state.boundary_inspected:
            return False, "evidence boundary has not been inspected yet"
        if state.facts_forbidden or state.pattern_events:
            return True, ""
        return False, "no missing roles and no low-trust patterns to separate from facts"
    if kind == "check_missing_evidence":
        if state.open_gap_ids:
            return True, ""
        return False, "no active gap nodes in the working graph"
    if kind == "compare_concepts":
        if len(state.grounded_concepts) >= 2:
            return True, ""
        return False, "fewer than two grounded concept nodes activated by the question"
    if kind == "ground_in_supported_example":
        if state.boundary_inspected and state.facts_allowed:
            return True, ""
        return False, "no inspected supported evidence to ground an example in"
    if kind == "reduce_to_minimal_repro":
        if state.pattern_sources(kind):
            return True, ""
        return False, "no activated debugging/procedure pattern node feeds this move"
    if kind == "detect_likely_mistake":
        if state.pattern_sources(kind):
            return True, ""
        return False, "no activated mistake pattern node feeds this move"
    if kind == "ask_missing_constraint":
        if "check_missing_evidence" in state.applied_kinds and state.inspected_gap_ids:
            return True, ""
        return False, "missing-evidence check has not surfaced an inspected gap yet"
    if kind == "choose_explanation_depth":
        if state.boundary_inspected and state.facts_allowed:
            return True, ""
        return False, "explanation depth needs an inspected non-empty evidence boundary"
    if kind == "choose_tone":
        if state.pattern_sources(kind):
            return True, ""
        return False, "no activated style/tone pattern node feeds this move"
    if kind == "verify_before_answering":
        if state.boundary_inspected and (state.facts_allowed or state.uncertainty_notes):
            return True, ""
        return False, "nothing planned yet to verify against the workspace"
    if kind == "stop_when_unsupported":
        if (
            state.boundary_inspected
            and not state.facts_allowed
            and (state.open_gap_ids or state.inspected_gap_ids)
        ):
            return True, ""
        return False, "supported evidence exists, so stopping is not warranted"
    return False, "unknown move"


# -- move scoring ------------------------------------------------------------


def _score(
    kind: CognitiveMoveKind,
    state: WorkingSemanticState,
    trace: ReasoningTrace,
) -> tuple[float, str, tuple[str, ...]]:
    base = state.activation.get(f"move:{kind}", 0.0)
    pattern_sources = state.pattern_sources(kind)

    if kind == "check_missing_evidence":
        gaps = state.gap_ids_by_activation()
        gap_activation = sum(state.activation.get(gap_id, 0.0) for gap_id in gaps)
        return (
            base + 0.5 * gap_activation,
            f"{len(gaps)} active gap node(s) with total activation {gap_activation:.2f}",
            gaps,
        )
    if kind == "inspect_evidence_boundary":
        roles = tuple(f"role:{role}" for role in trace.workspace.roles())
        return (
            base + 0.6 + 0.15 * len(roles),
            f"{len(roles)} evidence role(s) admitted into the workspace",
            roles,
        )
    if kind == "decompose_question":
        subgoal_ids = tuple(f"subgoal:{_slug(subgoal.name)}" for subgoal in trace.subgoals)
        return (
            base + 0.3 * len(subgoal_ids),
            f"{len(subgoal_ids)} subgoal node(s) attached to the task",
            subgoal_ids + pattern_sources,
        )
    if kind == "separate_fact_from_interpretation":
        return (
            base + 0.5 + 0.3 * len(state.facts_forbidden),
            f"{len(state.facts_forbidden)} forbidden claim boundary(ies) and "
            f"{len(state.pattern_events)} low-trust pattern node(s)",
            tuple(sorted(state.pattern_events)) or ("task:current",),
        )
    if kind == "ground_in_supported_example":
        return (
            base + 0.25 * len(state.facts_allowed),
            f"{len(state.facts_allowed)} supported fact(s) inside the boundary",
            pattern_sources,
        )
    if kind == "ask_missing_constraint":
        inspected = tuple(sorted(state.inspected_gap_ids))
        return (
            base + 0.4 * len(inspected),
            f"{len(inspected)} inspected gap(s) can become a follow-up constraint",
            inspected,
        )
    if kind == "reduce_to_minimal_repro":
        return (
            base + 0.6 * len(pattern_sources),
            f"{len(pattern_sources)} activated debugging pattern node(s)",
            pattern_sources,
        )
    if kind == "detect_likely_mistake":
        return (
            base + 0.4 * len(pattern_sources),
            f"{len(pattern_sources)} activated mistake pattern node(s)",
            pattern_sources,
        )
    if kind == "compare_concepts":
        return (
            base + 0.5 * len(state.grounded_concepts),
            f"{len(state.grounded_concepts)} grounded concept node(s) to contrast",
            state.grounded_concepts,
        )
    if kind == "activate_related_concepts":
        return (
            base + 0.1 * len(trace.workspace.items),
            f"{len(trace.workspace.items)} evidence node(s) can spread activation",
            (),
        )
    if kind == "choose_explanation_depth":
        return (base + 0.4, "supported boundary is ready for a depth decision", pattern_sources)
    if kind == "choose_tone":
        return (base, "style pattern activation only", pattern_sources)
    if kind == "verify_before_answering":
        return (
            base + 0.25 * len(state.applied),
            f"{len(state.applied)} applied move(s) accumulated state to verify",
            (),
        )
    if kind == "stop_when_unsupported":
        return (
            base + 4.0,
            "boundary shows no supported facts while gaps are active",
            tuple(sorted(state.open_gap_ids | state.inspected_gap_ids)),
        )
    return (base, "graph activation only", ())


# -- move effects ------------------------------------------------------------


def _apply(
    candidate: MoveCandidate,
    state: WorkingSemanticState,
    trace: ReasoningTrace,
) -> MoveApplication:
    effects: list[str] = []
    kind = candidate.kind

    if kind == "decompose_question":
        for subgoal in trace.subgoals:
            subgoal_id = f"subgoal:{_slug(subgoal.name)}"
            state.activation[subgoal_id] = state.activation.get(subgoal_id, 0.0) + 0.5
            for role in subgoal.required_roles:
                role_id = f"role:{role}"
                state.activation[role_id] = state.activation.get(role_id, 0.0) + 0.3
            effects.append(f"activated subgoal {subgoal.name!r} ({subgoal.status})")
        state.explanation_moves.append(kind)

    elif kind == "activate_related_concepts":
        risers = _extra_propagation_round(state)
        effects.append(
            "spread one extra activation round; top risers: " + ", ".join(risers)
            if risers
            else "spread one extra activation round; no nodes rose"
        )

    elif kind == "inspect_evidence_boundary":
        for item in trace.workspace.items:
            state.facts_allowed.append((str(item.role), item.text))
        missing_roles = _missing_roles(trace)
        for role in missing_roles:
            state.facts_forbidden.append(f"{role}: no admitted evidence")
        state.boundary_inspected = True
        effects.append(
            f"boundary set: {len(state.facts_allowed)} supported fact(s), "
            f"{len(missing_roles)} missing role(s)"
        )

    elif kind == "separate_fact_from_interpretation":
        state.facts_forbidden.append(
            "interpretation beyond admitted evidence must be labeled as interpretation"
        )
        if state.pattern_events:
            state.uncertainty_notes.append(
                "community patterns shape behavior only; they are not factual support"
            )
            effects.append(
                f"quarantined {len(state.pattern_events)} low-trust pattern node(s) from facts"
            )
        state.explanation_moves.append(kind)
        effects.append("added interpretation boundary to forbidden claims")

    elif kind == "check_missing_evidence":
        for gap_id in state.gap_ids_by_activation():
            node = state.nodes.get(gap_id)
            label = node.label if node else gap_id
            state.uncertainty_notes.append(label)
            state.inspected_gap_ids.add(gap_id)
            effects.append(f"inspected {gap_id}: {label}")
        state.open_gap_ids -= state.inspected_gap_ids
        follow_up = "move:ask_missing_constraint"
        state.activation[follow_up] = state.activation.get(follow_up, 0.0) + 1.0 * len(
            state.inspected_gap_ids
        )
        effects.append("boosted ask_missing_constraint activation from inspected gaps")

    elif kind == "compare_concepts":
        labels = [
            state.nodes[concept_id].label
            for concept_id in state.grounded_concepts
            if concept_id in state.nodes
        ]
        state.explanation_moves.append(kind)
        effects.append("planned contrast across grounded concepts: " + ", ".join(labels))

    elif kind == "ground_in_supported_example":
        example = _pick_grounded_example(state)
        if example is not None:
            state.examples.append(("grounded", example))
            effects.append(f"grounded example selected from admitted evidence: {example!r}")
        analogy_sources = dict.fromkeys(
            state.pattern_sources("compare_concepts") + state.pattern_sources(kind)
        )
        for pattern_id in analogy_sources:
            event = state.pattern_events.get(pattern_id)
            if event is not None and event.kind == "analogy_pattern":
                analogy = event.example_shape or event.pattern
                state.examples.append(("analogy", analogy))
                effects.append(f"analogy allowed only as marked analogy from {pattern_id}")
        state.explanation_moves.append(kind)

    elif kind == "reduce_to_minimal_repro":
        state.explanation_moves.append(kind)
        for pattern_id in candidate.activated_by:
            event = state.pattern_events.get(pattern_id)
            if event is not None and event.steps:
                for step in event.steps:
                    if step not in state.repro_checklist:
                        state.repro_checklist.append(step)
                effects.append(
                    f"answer will be organized around a minimal reproduction "
                    f"({pattern_id}, {len(event.steps)} behavioral step(s))"
                )

    elif kind == "detect_likely_mistake":
        for pattern_id in candidate.activated_by:
            event = state.pattern_events.get(pattern_id)
            if event is not None:
                note = f"likely mistake to check (community pattern, non-factual): {event.pattern}"
                state.uncertainty_notes.append(note)
                if event.pattern not in state.mistake_checks:
                    state.mistake_checks.append(event.pattern)
                effects.append(f"surfaced mistake check from {pattern_id}")

    elif kind == "ask_missing_constraint":
        questions: list[str] = []
        for missing in trace.missing_evidence:
            questions.extend(missing.next_questions)
        if not questions:
            for gap_id in sorted(state.inspected_gap_ids):
                role = gap_id.removeprefix("gap:")
                questions.append(
                    f"What {role} information is available for {trace.task.subject}?"
                )
        for question_text in questions:
            if question_text not in state.pending_questions:
                state.pending_questions.append(question_text)
        effects.append(f"queued {len(state.pending_questions)} follow-up constraint question(s)")

    elif kind == "choose_explanation_depth":
        allowed_roles = {role for role, _text in state.facts_allowed}
        if "mechanism" in allowed_roles:
            state.explanation_depth = "mechanism_first"
        elif len(state.facts_allowed) >= 3:
            state.explanation_depth = "overview_then_detail"
        else:
            state.explanation_depth = "single_fact_direct"
        state.explanation_moves.append(kind)
        effects.append(f"explanation depth chosen from evidence shape: {state.explanation_depth}")

    elif kind == "choose_tone":
        for pattern_id in candidate.activated_by:
            event = state.pattern_events.get(pattern_id)
            if event is not None:
                state.tone = event.pattern
                effects.append(f"tone adopted from {pattern_id} (behavioral only)")
                break

    elif kind == "verify_before_answering":
        admitted = {item.text for item in trace.workspace.items}
        kept: list[tuple[str, str]] = []
        for role, text in state.facts_allowed:
            if text in admitted:
                kept.append((role, text))
            else:
                effects.append(f"dropped unverifiable claim: {text!r}")
        state.facts_allowed = kept
        state.verified = True
        effects.append(f"verified {len(kept)} planned claim(s) against the admitted workspace")

    elif kind == "stop_when_unsupported":
        state.uncertainty_notes.append(
            "no admitted evidence supports an answer; refusing to plan unsupported claims"
        )
        state.stop_reason = "blocked_unsupported"
        effects.append("stopped: answer planning blocked without supported evidence")

    return MoveApplication(
        kind=kind,
        score=candidate.score,
        reason=candidate.reason,
        effects=tuple(effects),
        activated_by=candidate.activated_by,
    )


def _extra_propagation_round(state: WorkingSemanticState) -> tuple[str, ...]:
    outgoing: dict[str, list[CognitiveGraphEdge]] = defaultdict(list)
    for edge in state.edges:
        outgoing[edge.source].append(edge)
    delta: dict[str, float] = defaultdict(float)
    for source in sorted(state.activation):
        for edge in outgoing.get(source, ()):
            delta[edge.target] += state.activation[source] * edge.weight * 0.4
    risers = sorted(delta.items(), key=lambda item: (-item[1], item[0]))[:3]
    for node_id in sorted(delta):
        state.activation[node_id] = min(12.0, state.activation.get(node_id, 0.0) + delta[node_id])
    return tuple(node_id for node_id, _value in risers)


def _pick_grounded_example(state: WorkingSemanticState) -> str | None:
    by_role = {role: text for role, text in reversed(state.facts_allowed)}
    for role in _GROUNDING_ROLE_PREFERENCE:
        if role in by_role:
            return by_role[role]
    if state.facts_allowed:
        return state.facts_allowed[0][1]
    return None


def _missing_roles(trace: ReasoningTrace) -> tuple[str, ...]:
    roles: list[str] = []
    for subgoal in trace.subgoals:
        roles.extend(str(role) for role in subgoal.missing_roles)
    roles.extend(str(missing.role) for missing in trace.missing_evidence)
    return tuple(dict.fromkeys(roles))


def _answer_plan(state: WorkingSemanticState) -> AnswerPlan:
    return AnswerPlan(
        facts_to_say=tuple(state.facts_allowed),
        facts_not_allowed=tuple(dict.fromkeys(state.facts_forbidden)),
        explanation_strategy=tuple(dict.fromkeys(state.explanation_moves)),
        uncertainty_to_state=tuple(dict.fromkeys(state.uncertainty_notes)),
        examples=tuple(state.examples),
        next_useful_question=state.pending_questions[0] if state.pending_questions else None,
        explanation_depth=state.explanation_depth,
        tone=state.tone,
        verified=state.verified,
        mistake_checks=tuple(dict.fromkeys(state.mistake_checks)),
        repro_checklist=tuple(dict.fromkeys(state.repro_checklist)),
    )


def render_plan_addendum(plan: AnswerPlan) -> tuple[str, ...]:
    """Realize the non-factual parts of an ``AnswerPlan`` as labeled sentences.

    This is the downstream rendering step for cognitive moves that have no
    slot in the existing fact-bucket renderer (``reduce_to_minimal_repro``,
    ``detect_likely_mistake``, analogy-marked examples from
    ``ground_in_supported_example`` / ``compare_concepts``). Every sentence
    is built from data the loop already selected — never a canned string
    chosen independent of graph state — and is only emitted when the
    corresponding move actually fired.
    """

    lines: list[str] = []
    for check in plan.mistake_checks:
        lines.append(
            f"Worth double-checking (community-observed pattern, not a verified fact): {check}."
        )
    if plan.repro_checklist:
        steps = "; ".join(plan.repro_checklist)
        lines.append(f"If this needs debugging, the useful next steps are: {steps}.")
    for kind, text in plan.examples:
        if kind == "analogy":
            lines.append(f"As an analogy only, not a factual claim: {text}")
    return tuple(_clean_addendum_line(line) for line in lines)


def _clean_addendum_line(line: str) -> str:
    line = line.strip()
    if line.endswith(".."):
        line = line[:-1]
    return line


def action_plan_from_cognitive_loop(loop: CognitiveLoopTrace, *, fallback: ActionPlan) -> ActionPlan:
    """Let the graph-derived plan tighten the renderer's action, safely.

    This is the rendering-layer bridge: it does not reorder the existing
    style-driven bucket priorities (``how`` questions must still lead with
    mechanism when it is present), but it applies what the loop discovered
    that the fallback action could not know on its own:
    - refuse to answer when the loop found no supported facts at all,
    - suppress any additional role the loop marked as missing,
    - shorten the answer when the evidence boundary supports only one fact,
    - offer a next question when the fallback did not already have one.
    """

    plan = loop.answer_plan
    next_action = fallback.next_action
    if loop.stop_reason == "blocked_unsupported" or not plan.facts_to_say:
        next_action = "ask_clarification"

    extra_suppressed = {
        role for role in loop.missing_roles if role in _CONTENT_BUCKET_ROLES
    }
    suppressed_buckets = tuple(
        dict.fromkeys(tuple(fallback.suppressed_buckets) + tuple(sorted(extra_suppressed)))
    )

    detail_unit_limit = fallback.detail_unit_limit
    if plan.explanation_depth == "single_fact_direct":
        detail_unit_limit = 1 if detail_unit_limit is None else min(detail_unit_limit, 1)

    next_questions = fallback.next_questions
    if not next_questions and plan.next_useful_question:
        next_questions = (plan.next_useful_question,)

    forbidden_claims = tuple(
        dict.fromkeys(tuple(fallback.forbidden_claims) + plan.facts_not_allowed)
    )

    return ActionPlan(
        next_action=next_action,
        surface_goal=fallback.surface_goal,
        forbidden_claims=forbidden_claims,
        preferred_buckets=fallback.preferred_buckets,
        suppressed_buckets=suppressed_buckets,
        detail_unit_limit=detail_unit_limit,
        next_questions=next_questions,
    )
