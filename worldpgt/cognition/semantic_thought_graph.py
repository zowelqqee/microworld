"""Semantic graph operations for non-template cognitive planning.

This module does not render answers and does not create facts. It turns the
already admitted reasoning workspace plus optional cognitive-pattern events
into a small activation graph, then selects reusable thinking moves from the
graph state.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import re
from typing import Iterable, Literal

from worldpgt.cognition.types import EvidenceRole, ReasoningTrace
from worldpgt.community_context.types import CognitivePatternEvent


ThoughtNodeKind = Literal[
    "task",
    "subject",
    "subgoal",
    "evidence_role",
    "evidence_item",
    "gap",
    "pattern",
    "move",
    "concept",
]

ThoughtMoveKind = Literal[
    "decompose_question",
    "check_missing_evidence",
    "separate_fact_from_interpretation",
    "ground_in_supported_example",
    "reduce_to_minimal_repro",
    "compare_concepts",
    "ask_missing_constraint",
    "explain_mechanism_before_jargon",
]

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "what",
        "why",
        "with",
        "а",
        "в",
        "и",
        "как",
        "на",
        "не",
        "по",
        "про",
        "с",
        "что",
        "это",
    }
)

_MOVE_TO_PATTERN_KINDS: dict[ThoughtMoveKind, frozenset[str]] = {
    "decompose_question": frozenset({"question_pattern", "procedure_pattern"}),
    "check_missing_evidence": frozenset({"uncertainty_pattern", "debugging_pattern"}),
    "separate_fact_from_interpretation": frozenset(
        {"uncertainty_pattern", "explanation_pattern", "style_tone_pattern"}
    ),
    "ground_in_supported_example": frozenset({"explanation_pattern", "analogy_pattern"}),
    "reduce_to_minimal_repro": frozenset({"debugging_pattern", "procedure_pattern"}),
    "compare_concepts": frozenset({"analogy_pattern", "explanation_pattern"}),
    "ask_missing_constraint": frozenset({"question_pattern", "uncertainty_pattern"}),
    "explain_mechanism_before_jargon": frozenset({"explanation_pattern", "style_tone_pattern"}),
}


@dataclass(frozen=True)
class SemanticThoughtNode:
    node_id: str
    kind: ThoughtNodeKind
    label: str
    weight: float = 1.0
    factual_support_allowed: bool = False

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "weight": round(self.weight, 4),
            "factual_support_allowed": self.factual_support_allowed,
        }


@dataclass(frozen=True)
class SemanticThoughtEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": round(self.weight, 4),
        }


@dataclass(frozen=True)
class SemanticThoughtMove:
    kind: ThoughtMoveKind
    score: float
    reason: str
    activated_by: tuple[str, ...] = ()
    factual_support_allowed: bool = False

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "score": round(self.score, 4),
            "reason": self.reason,
            "activated_by": list(self.activated_by),
            "factual_support_allowed": self.factual_support_allowed,
        }


@dataclass(frozen=True)
class SemanticThoughtGraphTrace:
    """A graph-state explanation of selected cognitive moves."""

    subject: str
    task_intent: str
    nodes: tuple[SemanticThoughtNode, ...] = ()
    edges: tuple[SemanticThoughtEdge, ...] = ()
    activation: dict[str, float] = field(default_factory=dict)
    candidate_moves: tuple[SemanticThoughtMove, ...] = ()
    selected_moves: tuple[SemanticThoughtMove, ...] = ()
    factual_support_allowed_from_patterns: bool = False

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "task_intent": self.task_intent,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "activation": {
                node_id: round(value, 4)
                for node_id, value in sorted(
                    self.activation.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            },
            "candidate_moves": [move.to_dict() for move in self.candidate_moves],
            "selected_moves": [move.to_dict() for move in self.selected_moves],
            "factual_support_allowed_from_patterns": self.factual_support_allowed_from_patterns,
        }


class _GraphBuilder:
    def __init__(self) -> None:
        self.nodes: dict[str, SemanticThoughtNode] = {}
        self.edges: list[SemanticThoughtEdge] = []

    def node(
        self,
        node_id: str,
        kind: ThoughtNodeKind,
        label: str,
        *,
        weight: float = 1.0,
        factual_support_allowed: bool = False,
    ) -> str:
        existing = self.nodes.get(node_id)
        if existing is None or weight > existing.weight:
            self.nodes[node_id] = SemanticThoughtNode(
                node_id=node_id,
                kind=kind,
                label=label,
                weight=weight,
                factual_support_allowed=factual_support_allowed,
            )
        return node_id

    def edge(self, source: str, target: str, relation: str, *, weight: float = 1.0) -> None:
        if source in self.nodes and target in self.nodes:
            self.edges.append(SemanticThoughtEdge(source, target, relation, weight))


def run_semantic_thought_graph(
    trace: ReasoningTrace,
    *,
    question: str = "",
    cognitive_patterns: Iterable[CognitivePatternEvent | dict] = (),
    max_moves: int = 4,
) -> SemanticThoughtGraphTrace:
    """Select cognitive moves by activation over semantic links.

    ``ReasoningTrace`` remains the factual boundary. Pattern events may add
    behavior/style pressure, but any pattern that allows factual support is
    ignored by this graph.
    """

    safe_patterns = tuple(
        event
        for event in (_coerce_pattern(raw) for raw in cognitive_patterns)
        if event is not None and event.factual_support_allowed is False
    )
    builder = _build_graph(trace, question=question, patterns=safe_patterns)
    activation = _activate(builder.nodes, builder.edges)
    candidates = _candidate_moves(trace, question, safe_patterns, builder.nodes, activation)
    selected = tuple(candidates[:max_moves])
    return SemanticThoughtGraphTrace(
        subject=trace.task.subject,
        task_intent=trace.task.intent,
        nodes=tuple(builder.nodes.values()),
        edges=tuple(builder.edges),
        activation=activation,
        candidate_moves=tuple(candidates),
        selected_moves=selected,
        factual_support_allowed_from_patterns=False,
    )


def _build_graph(
    trace: ReasoningTrace,
    *,
    question: str,
    patterns: tuple[CognitivePatternEvent, ...],
) -> _GraphBuilder:
    graph = _GraphBuilder()
    task_id = graph.node("task:current", "task", trace.task.intent, weight=2.0)
    subject_id = graph.node(f"subject:{_slug(trace.task.subject)}", "subject", trace.task.subject, weight=2.0)
    graph.edge(task_id, subject_id, "about", weight=1.0)

    for subgoal in trace.subgoals:
        subgoal_id = graph.node(
            f"subgoal:{_slug(subgoal.name)}",
            "subgoal",
            subgoal.name,
            weight=_status_weight(subgoal.status),
        )
        graph.edge(task_id, subgoal_id, "needs", weight=1.0)
        for role in subgoal.required_roles:
            role_id = _role_node(graph, role, weight=1.2)
            graph.edge(subgoal_id, role_id, "requires_role", weight=0.8)
        for role in subgoal.missing_roles:
            gap_id = graph.node(f"gap:{role}", "gap", f"missing {role}", weight=2.4)
            role_id = _role_node(graph, role, weight=1.4)
            graph.edge(gap_id, role_id, "missing_role", weight=1.2)
            graph.edge(subgoal_id, gap_id, "blocked_by", weight=1.4)

    for item in trace.workspace.items:
        role_id = _role_node(graph, item.role, weight=1.4)
        item_id = graph.node(
            f"evidence:{item.role}:{_slug(item.text)[:64]}",
            "evidence_item",
            item.text,
            weight=1.0,
            factual_support_allowed=True,
        )
        graph.edge(subject_id, role_id, "has_evidence_role", weight=0.9)
        graph.edge(role_id, item_id, "supported_by", weight=0.7)

    for missing in trace.missing_evidence:
        gap_id = graph.node(
            f"gap:{missing.role}",
            "gap",
            missing.reason or f"missing {missing.role}",
            weight=2.8,
        )
        graph.edge(task_id, gap_id, "blocked_by", weight=1.5)

    query_terms = _tokens(question or trace.task.question_style)
    for event in patterns:
        event_terms = _tokens(" ".join([event.topic, event.kind, event.pattern, " ".join(event.steps)]))
        matched = query_terms & event_terms
        if not matched and event.topic and _slug(event.topic) not in _slug(trace.task.subject):
            continue
        pattern_id = graph.node(
            f"pattern:{event.event_id}",
            "pattern",
            event.pattern,
            weight=1.0 + min(event.signal_count, 5) * 0.15 + len(matched) * 0.2,
            factual_support_allowed=False,
        )
        graph.edge(task_id, pattern_id, "may_use_pattern", weight=0.7 + len(matched) * 0.1)
        for move in _moves_for_pattern(event.kind):
            move_id = graph.node(f"move:{move}", "move", move, weight=1.0)
            graph.edge(pattern_id, move_id, "suggests_move", weight=1.2)

    for move in _all_move_kinds():
        move_id = graph.node(f"move:{move}", "move", move, weight=0.7)
        for role in _roles_for_move(move):
            role_id = f"role:{role}"
            if role_id in graph.nodes:
                graph.edge(role_id, move_id, "activates_move", weight=0.6)
        if move == "check_missing_evidence":
            for node_id, node in tuple(graph.nodes.items()):
                if node.kind == "gap":
                    graph.edge(node_id, move_id, "activates_move", weight=1.5)
        if move == "ask_missing_constraint":
            for node_id, node in tuple(graph.nodes.items()):
                if node.kind == "gap":
                    graph.edge(node_id, move_id, "suggests_followup", weight=1.2)

    return graph


def _activate(
    nodes: dict[str, SemanticThoughtNode],
    edges: list[SemanticThoughtEdge],
    *,
    rounds: int = 3,
) -> dict[str, float]:
    activation = {node_id: node.weight for node_id, node in nodes.items()}
    outgoing: dict[str, list[SemanticThoughtEdge]] = defaultdict(list)
    for edge in edges:
        outgoing[edge.source].append(edge)
    for _ in range(rounds):
        next_activation = dict(activation)
        for source, source_value in activation.items():
            for edge in outgoing.get(source, ()):
                spread = source_value * edge.weight * 0.35
                if spread > next_activation.get(edge.target, 0.0):
                    next_activation[edge.target] = spread
        activation = next_activation
    return activation


def _candidate_moves(
    trace: ReasoningTrace,
    question: str,
    patterns: tuple[CognitivePatternEvent, ...],
    nodes: dict[str, SemanticThoughtNode],
    activation: dict[str, float],
) -> list[SemanticThoughtMove]:
    pattern_kinds = {event.kind for event in patterns}
    candidates: list[SemanticThoughtMove] = []
    for move in _all_move_kinds():
        move_id = f"move:{move}"
        score = activation.get(move_id, 0.0)
        activated_by = _incoming_activated_nodes(move_id, nodes, activation)
        reason_parts: list[str] = []

        if move == "check_missing_evidence" and trace.missing_evidence:
            score += 3.0
            reason_parts.append("the current reasoning trace has explicit missing evidence")
        if move == "ask_missing_constraint" and (trace.missing_evidence or trace.workspace.gaps):
            score += 1.8
            reason_parts.append("the graph contains a blocked or incomplete role")
        if move == "decompose_question" and len(trace.subgoals) > 1:
            score += 1.2
            reason_parts.append("the task has multiple subgoals")
        if move == "ground_in_supported_example" and len(trace.workspace.items) > 1:
            score += 1.0
            reason_parts.append("there is source-admitted evidence to ground the explanation")
        if move == "explain_mechanism_before_jargon" and trace.task.intent == "mechanism_explanation":
            score += 1.4
            reason_parts.append("the user is asking for a mechanism")
        if move == "separate_fact_from_interpretation" and patterns:
            score += 1.4
            reason_parts.append("low-trust cognitive patterns are present and must stay non-factual")
        if move == "reduce_to_minimal_repro" and (
            "debugging_pattern" in pattern_kinds or _has_any(question, ("debug", "bug", "error", "traceback"))
        ):
            score += 2.4
            reason_parts.append("debugging signals activate the minimal-reproduction move")
        if move == "compare_concepts" and _has_any(question, ("compare", "versus", " vs ", "разница", "сравни")):
            score += 2.0
            reason_parts.append("comparison wording activates a contrastive move")

        if pattern_kinds & _MOVE_TO_PATTERN_KINDS[move]:
            score += 0.8
            reason_parts.append("retrieved cognitive patterns point at this move")

        if score <= 0.05:
            continue
        candidates.append(
            SemanticThoughtMove(
                kind=move,
                score=score,
                reason="; ".join(reason_parts) or "activated by semantic graph links",
                activated_by=activated_by,
                factual_support_allowed=False,
            )
        )

    candidates.sort(key=lambda move: (-move.score, move.kind))
    return candidates


def _role_node(graph: _GraphBuilder, role: EvidenceRole, *, weight: float) -> str:
    return graph.node(f"role:{role}", "evidence_role", str(role), weight=weight)


def _status_weight(status: str) -> float:
    if status == "missing":
        return 2.0
    if status == "partial":
        return 1.6
    return 1.0


def _moves_for_pattern(kind: str) -> tuple[ThoughtMoveKind, ...]:
    return tuple(move for move, kinds in _MOVE_TO_PATTERN_KINDS.items() if kind in kinds)


def _roles_for_move(move: ThoughtMoveKind) -> tuple[EvidenceRole, ...]:
    if move == "check_missing_evidence":
        return ("mechanism", "other", "gap")
    if move == "ground_in_supported_example":
        return ("definition", "activity", "purpose", "mechanism")
    if move == "explain_mechanism_before_jargon":
        return ("mechanism", "purpose")
    return ("definition",)


def _incoming_activated_nodes(
    move_id: str,
    nodes: dict[str, SemanticThoughtNode],
    activation: dict[str, float],
) -> tuple[str, ...]:
    prefix = move_id.removeprefix("move:")
    out: list[str] = []
    for node_id, node in nodes.items():
        if node.kind in {"gap", "pattern", "evidence_role", "subgoal"} and activation.get(node_id, 0.0) > 1.0:
            out.append(node_id)
        if len(out) >= 6:
            break
    if prefix and move_id in activation and not out:
        out.append("task:current")
    return tuple(out)


def _coerce_pattern(raw: CognitivePatternEvent | dict) -> CognitivePatternEvent | None:
    if isinstance(raw, CognitivePatternEvent):
        return raw
    if not isinstance(raw, dict):
        return None
    return CognitivePatternEvent(
        event_id=str(raw.get("event_id") or ""),
        kind=str(raw.get("kind") or ""),
        topic=str(raw.get("topic") or ""),
        pattern=str(raw.get("pattern") or ""),
        use_when=tuple(str(x) for x in raw.get("use_when") or ()),
        avoid_when=tuple(str(x) for x in raw.get("avoid_when") or ()),
        steps=tuple(str(x) for x in raw.get("steps") or ()),
        example_shape=str(raw.get("example_shape") or ""),
        confidence=str(raw.get("confidence") or "medium"),
        source=str(raw.get("source") or "community_context"),
        trust=str(raw.get("trust") or "low_for_facts_high_for_style"),
        graph_layers=tuple(str(x) for x in raw.get("graph_layers") or ()),
        source_item_ids=tuple(str(x) for x in raw.get("source_item_ids") or ()),
        evidence_refs=tuple(str(x) for x in raw.get("evidence_refs") or ()),
        signal_count=int(raw.get("signal_count") or 1),
        factual_support_allowed=raw.get("factual_support_allowed") is True,
    )


def _all_move_kinds() -> tuple[ThoughtMoveKind, ...]:
    return (
        "decompose_question",
        "check_missing_evidence",
        "separate_fact_from_interpretation",
        "ground_in_supported_example",
        "reduce_to_minimal_repro",
        "compare_concepts",
        "ask_missing_constraint",
        "explain_mechanism_before_jargon",
    )


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text or "")
        if len(token) > 1 and token.lower() not in _STOP_WORDS
    }


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    low = (text or "").lower()
    return any(needle in low for needle in needles)


def _slug(text: str) -> str:
    value = "-".join(_TOKEN_RE.findall((text or "").lower()))
    return value or "unknown"
