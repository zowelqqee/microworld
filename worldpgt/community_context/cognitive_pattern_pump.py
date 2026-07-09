"""Cognitive pattern extraction from low-trust community context.

This module intentionally learns reusable reasoning and communication shapes,
not facts. Community text can suggest how people explain, debug, ask, worry, or
make mistakes, but it is never factual authority.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import re
from typing import Iterable

from worldpgt.community_context.types import (
    COMMUNITY_CONTEXT_TRUST,
    CognitivePatternEvent,
    CognitivePatternSearchResult,
    CommunityContextItem,
)


_STOP_WORDS = frozenset(
    {
        "about", "after", "again", "also", "and", "are", "because", "been",
        "being", "but", "can", "could", "does", "for", "from", "had", "has",
        "have", "how", "into", "its", "just", "like", "more", "most", "not",
        "one", "only", "our", "out", "over", "really", "some", "than", "that",
        "the", "their", "them", "then", "there", "these", "they", "this",
        "those", "through", "was", "were", "what", "when", "where", "which",
        "while", "who", "why", "with", "would", "you", "your",
    }
)

_KIND_TO_GRAPH_LAYER = {
    "concept_pattern": "concept_graph",
    "procedure_pattern": "procedure_graph",
    "explanation_pattern": "explanation_graph",
    "debugging_pattern": "procedure_graph",
    "mistake_pattern": "error_mistake_graph",
    "analogy_pattern": "analogy_graph",
    "question_pattern": "question_intent_graph",
    "answer_shape_pattern": "question_intent_graph",
    "style_tone_pattern": "style_tone_graph",
    "uncertainty_pattern": "explanation_graph",
}

_ALL_GRAPH_LAYERS = (
    "concept_graph",
    "procedure_graph",
    "explanation_graph",
    "error_mistake_graph",
    "analogy_graph",
    "question_intent_graph",
    "style_tone_graph",
    "evidence_graph",
)


def extract_cognitive_pattern_events(
    items: Iterable[CommunityContextItem | dict],
    *,
    max_events: int | None = None,
) -> list[CognitivePatternEvent]:
    """Extract reusable cognitive pattern events from community-context items."""

    proposals: list[CognitivePatternEvent] = []
    for raw_item in items:
        item = _coerce_item(raw_item)
        proposals.extend(_events_for_item(item))

    consolidated = _consolidate_events(proposals)
    consolidated.sort(
        key=lambda event: (
            -event.signal_count,
            _confidence_rank(event.confidence),
            event.kind,
            event.topic,
            event.pattern,
        )
    )
    if max_events is not None:
        return consolidated[:max_events]
    return consolidated


def build_cognitive_pattern_graph(events: Iterable[CognitivePatternEvent | dict]) -> dict:
    """Build graph projections from pattern events without creating facts."""

    graphs = {
        layer: {"nodes": [], "edges": []}
        for layer in _ALL_GRAPH_LAYERS
    }
    node_sets: dict[str, set[str]] = {layer: set() for layer in _ALL_GRAPH_LAYERS}
    edge_sets: dict[str, set[tuple[str, str, str]]] = {layer: set() for layer in _ALL_GRAPH_LAYERS}

    for raw_event in events:
        event = _coerce_event(raw_event)
        pattern_node = f"pattern:{event.event_id}"
        topic_node = f"topic:{_slug(event.topic)}"
        source_nodes = [f"source:{source_id}" for source_id in event.source_item_ids]
        layers = event.graph_layers or (_KIND_TO_GRAPH_LAYER.get(event.kind, "concept_graph"),)

        for layer in layers:
            _add_node(graphs, node_sets, layer, topic_node, "topic", event.topic)
            _add_node(graphs, node_sets, layer, pattern_node, event.kind, event.pattern)
            _add_edge(graphs, edge_sets, layer, topic_node, pattern_node, "uses_pattern")
            for step in event.steps:
                step_node = f"step:{_stable_id(event.event_id, step)}"
                _add_node(graphs, node_sets, layer, step_node, "step", step)
                _add_edge(graphs, edge_sets, layer, pattern_node, step_node, "has_step")

        for source_node in source_nodes:
            _add_node(graphs, node_sets, "evidence_graph", source_node, event.source, source_node.removeprefix("source:"))
            _add_node(graphs, node_sets, "evidence_graph", pattern_node, event.kind, event.pattern)
            _add_edge(graphs, edge_sets, "evidence_graph", source_node, pattern_node, "suggests_pattern")

    return graphs


def query_cognitive_patterns(
    events: Iterable[CognitivePatternEvent | dict],
    question: str,
    *,
    max_results: int = 5,
) -> list[CognitivePatternSearchResult]:
    """Retrieve cognitive patterns relevant to a question or planned answer."""

    query_terms = _expanded_terms(question)
    if not query_terms:
        return []
    results: list[CognitivePatternSearchResult] = []
    for raw_event in events:
        event = _coerce_event(raw_event)
        searchable = " ".join(
            [
                event.kind,
                event.topic,
                event.pattern,
                " ".join(event.use_when),
                " ".join(event.steps),
                event.example_shape,
            ]
        )
        event_terms = _expanded_terms(searchable)
        matched = tuple(sorted(query_terms & event_terms))
        if not matched:
            continue
        score = len(matched) * 12.0 + event.signal_count * 3.0
        if event.confidence == "high":
            score += 5.0
        elif event.confidence == "low":
            score -= 2.0
        results.append(CognitivePatternSearchResult(event=event, score=score, matched_terms=matched))
    results.sort(key=lambda result: (-result.score, result.event.kind, result.event.pattern))
    return results[:max_results]


def plan_answer_with_cognitive_patterns(
    question: str,
    events: Iterable[CognitivePatternEvent | dict],
    *,
    max_patterns: int = 5,
) -> dict:
    """Return a source-separated answer-planning scaffold."""

    matches = query_cognitive_patterns(events, question, max_results=max_patterns)
    selected = [match.event for match in matches]
    return {
        "question": question,
        "user_intent_guess": _intent_guess(question, selected),
        "facts_required": _facts_required(question),
        "known_facts_source": "factual_memory_or_live_search_required",
        "cognitive_patterns": [event.to_dict() for event in selected],
        "reasoning_pattern": _first_pattern(selected, {"explanation_pattern", "debugging_pattern", "procedure_pattern"}),
        "examples_or_analogies": _first_pattern(selected, {"analogy_pattern"}),
        "uncertainty_to_state": _uncertainty_note(selected),
        "checks_before_answering": [
            "Do not treat community pattern text as factual support.",
            "Verify source-backed claims in factual memory or live search.",
            "Separate cited claims from phrasing, examples, and tone choices.",
        ],
        "tone": _first_pattern(selected, {"style_tone_pattern"}) or "Use a clear, concrete, helpful tone.",
        "helpful_next_move": _helpful_next_move(question, selected),
        "factual_support_allowed_from_patterns": False,
    }


def _events_for_item(item: CommunityContextItem) -> list[CognitivePatternEvent]:
    text = f"{item.title} {item.text}"
    low = text.lower()
    terms = _expanded_terms(text)
    topic = _topic_for_item(item, terms)
    events: list[CognitivePatternEvent] = []

    if _has_any(low, ("explain", "explanation", "understand", "simple", "plain", "concrete example")):
        events.append(
            _event(
                item,
                kind="explanation_pattern",
                topic=topic,
                pattern="explain one idea at a time with plain language and a concrete example",
                use_when=("beginner explanation", "confusing concept", "community-style answer"),
                avoid_when=("formal proof required", "source-backed claim needed"),
                example_shape="Start with the simple rule, give one tiny example, then name the limit or caveat.",
                confidence=_confidence(item, low, ("example", "plain", "simple", "understand")),
            )
        )

    if "recursion" in terms:
        events.append(
            _event(
                item,
                kind="explanation_pattern",
                topic="recursion",
                pattern="explain self-reference through a base case and a smaller subproblem",
                use_when=("beginner programming explanation", "recursive algorithm intuition"),
                avoid_when=("formal proof required", "advanced complexity analysis"),
                example_shape="Use an everyday nested structure, then show a tiny code sketch.",
                confidence="medium",
            )
        )

    if _has_any(low, ("debug", "bug", "error", "traceback", "reproduce", "repro", "expected", "actual")):
        events.append(
            _event(
                item,
                kind="debugging_pattern",
                topic="programming",
                pattern="reduce the problem to a minimal reproducible example",
                use_when=("programming bug", "debugging help", "unclear error report"),
                steps=(
                    "state expected behavior",
                    "state actual behavior",
                    "include the exact error message",
                    "show the smallest code or steps that reproduce it",
                    "say what was already tried",
                ),
                confidence=_confidence(item, low, ("debug", "error", "smallest", "expected", "actual")),
                trust="behavioral_pattern",
            )
        )

    if _has_any(low, ("ask", "question", "what should i include", "help faster")):
        events.append(
            _event(
                item,
                kind="question_pattern",
                topic=topic,
                pattern="turn vague confusion into a focused question with context, expectation, result, and attempted fixes",
                use_when=("follow-up question", "debugging request", "learning help"),
                steps=(
                    "name the goal",
                    "show the smallest relevant context",
                    "state where understanding breaks",
                    "include what was already tried",
                ),
                example_shape="Goal -> expected result -> actual result -> smallest example -> attempted fixes.",
                confidence=_confidence(item, low, ("question", "expected", "tried", "help")),
                trust="behavioral_pattern",
            )
        )

    if _has_any(low, ("mistake", "pitfall", "trap", "wrong", "confused", "stuck", "struggle")):
        events.append(
            _event(
                item,
                kind="mistake_pattern",
                topic=topic,
                pattern="surface the likely mistake, then convert it into a small check or corrective habit",
                use_when=("beginner guidance", "debugging", "reviewing failed reasoning"),
                avoid_when=("blaming the user", "unsupported diagnosis"),
                example_shape="Name the common mistake gently, explain why it happens, then give one check.",
                confidence=_confidence(item, low, ("mistake", "stuck", "confused", "struggle")),
            )
        )

    if _has_any(low, ("analogy", "compare", "comparing", "like ", "as if", "think of it as")):
        events.append(
            _event(
                item,
                kind="analogy_pattern",
                topic=topic,
                pattern="use an analogy only as a bridge, then return to the precise concept",
                use_when=("first explanation", "abstract concept", "beginner framing"),
                avoid_when=("analogy hides a critical distinction", "legal or medical precision required"),
                example_shape="Analogy -> exact mapping -> where the analogy stops working.",
                confidence=_confidence(item, low, ("analogy", "compare", "like")),
            )
        )

    if _has_any(low, ("worry", "concern", "tradeoff", "caveat", "uncertain", "depends")):
        events.append(
            _event(
                item,
                kind="uncertainty_pattern",
                topic=topic,
                pattern="state the tradeoff or caveat before giving practical guidance",
                use_when=("ambiguous advice", "planning", "high-variance experience reports"),
                avoid_when=("stable sourced fact is available", "false balance"),
                example_shape="Short answer, key caveat, then what to check next.",
                confidence=_confidence(item, low, ("tradeoff", "caveat", "depends", "concern")),
            )
        )

    if _has_any(low, ("plain", "clear", "concrete", "short steps", "less formal", "no jargon", "simple words")):
        events.append(
            _event(
                item,
                kind="style_tone_pattern",
                topic=topic,
                pattern="prefer clear, concrete, low-jargon phrasing over sounding clever",
                use_when=("user-facing explanation", "teaching", "follow-up answer"),
                avoid_when=("expert shorthand explicitly requested"),
                example_shape="Use simple words, then add terminology only after the idea is grounded.",
                confidence=_confidence(item, low, ("plain", "clear", "concrete", "jargon")),
                trust="low_for_facts_high_for_style",
            )
        )

    if _has_any(low, ("step by step", "workflow", "process", "method", "checklist")):
        events.append(
            _event(
                item,
                kind="procedure_pattern",
                topic=topic,
                pattern="turn advice into an ordered checklist with one decision per step",
                use_when=("planning", "how-to answer", "debugging workflow"),
                steps=(
                    "state the goal",
                    "list the smallest next checks",
                    "identify the first blocking unknown",
                    "stop when evidence changes the plan",
                ),
                confidence=_confidence(item, low, ("step", "workflow", "process", "checklist")),
                trust="behavioral_pattern",
            )
        )

    return events


def _event(
    item: CommunityContextItem,
    *,
    kind: str,
    topic: str,
    pattern: str,
    use_when: tuple[str, ...] = (),
    avoid_when: tuple[str, ...] = (),
    steps: tuple[str, ...] = (),
    example_shape: str = "",
    confidence: str = "medium",
    trust: str = "low_for_facts_high_for_style",
) -> CognitivePatternEvent:
    event_id = _stable_id(kind, topic, pattern)
    layer = _KIND_TO_GRAPH_LAYER.get(kind, "concept_graph")
    source_ref = item.url or item.item_id
    return CognitivePatternEvent(
        event_id=event_id,
        kind=kind,
        topic=topic,
        pattern=pattern,
        use_when=use_when,
        avoid_when=avoid_when,
        steps=steps,
        example_shape=example_shape,
        confidence=confidence,
        source="community_context",
        trust=trust,
        graph_layers=(layer,),
        source_item_ids=(item.item_id,),
        evidence_refs=(source_ref,),
        signal_count=1,
        factual_support_allowed=False,
    )


def _consolidate_events(events: list[CognitivePatternEvent]) -> list[CognitivePatternEvent]:
    grouped: dict[tuple[str, str, str], list[CognitivePatternEvent]] = defaultdict(list)
    for event in events:
        grouped[(event.kind, event.topic, event.pattern)].append(event)

    consolidated: list[CognitivePatternEvent] = []
    for group in grouped.values():
        first = group[0]
        confidence = first.confidence
        if len(group) >= 2 and any(event.confidence == "high" for event in group):
            confidence = "high"
        elif len(group) >= 3:
            confidence = "high"
        source_ids = _unique(part for event in group for part in event.source_item_ids)
        evidence_refs = _unique(part for event in group for part in event.evidence_refs)
        consolidated.append(
            CognitivePatternEvent(
                event_id=first.event_id,
                kind=first.kind,
                topic=first.topic,
                pattern=first.pattern,
                use_when=first.use_when,
                avoid_when=first.avoid_when,
                steps=first.steps,
                example_shape=first.example_shape,
                confidence=confidence,
                source=first.source,
                trust=first.trust,
                graph_layers=first.graph_layers,
                source_item_ids=tuple(source_ids),
                evidence_refs=tuple(evidence_refs),
                signal_count=len(group),
                factual_support_allowed=False,
            )
        )
    return consolidated


def _coerce_item(raw_item: CommunityContextItem | dict) -> CommunityContextItem:
    if isinstance(raw_item, CommunityContextItem):
        return raw_item
    return CommunityContextItem(
        item_id=str(raw_item.get("item_id") or raw_item.get("source_id") or ""),
        source_system=str(raw_item.get("source_system") or "community"),
        source_kind=str(raw_item.get("source_kind") or ""),
        trust=str(raw_item.get("trust") or COMMUNITY_CONTEXT_TRUST),
        subreddit=str(raw_item.get("subreddit") or ""),
        title=str(raw_item.get("title") or ""),
        text=str(raw_item.get("text") or raw_item.get("body") or ""),
        url=str(raw_item.get("url") or raw_item.get("permalink") or ""),
        score=_to_int(raw_item.get("score")),
        created_utc=str(raw_item.get("created_utc") or ""),
        topic_terms=tuple(raw_item.get("topic_terms") or ()),
        flags=tuple(raw_item.get("flags") or ()),
        risk=str(raw_item.get("risk") or "medium"),
        stability=str(raw_item.get("stability") or "volatile"),
    )


def _coerce_event(raw_event: CognitivePatternEvent | dict) -> CognitivePatternEvent:
    if isinstance(raw_event, CognitivePatternEvent):
        return raw_event
    return CognitivePatternEvent(
        event_id=str(raw_event.get("event_id") or _stable_id(str(raw_event.get("kind")), str(raw_event.get("pattern")))),
        kind=str(raw_event.get("kind") or "pattern"),
        topic=str(raw_event.get("topic") or "general"),
        pattern=str(raw_event.get("pattern") or ""),
        use_when=tuple(raw_event.get("use_when") or ()),
        avoid_when=tuple(raw_event.get("avoid_when") or ()),
        steps=tuple(raw_event.get("steps") or ()),
        example_shape=str(raw_event.get("example_shape") or ""),
        confidence=str(raw_event.get("confidence") or "medium"),
        source=str(raw_event.get("source") or "community_context"),
        trust=str(raw_event.get("trust") or "low_for_facts_high_for_style"),
        graph_layers=tuple(raw_event.get("graph_layers") or ()),
        source_item_ids=tuple(raw_event.get("source_item_ids") or ()),
        evidence_refs=tuple(raw_event.get("evidence_refs") or ()),
        signal_count=int(raw_event.get("signal_count") or 1),
        factual_support_allowed=bool(raw_event.get("factual_support_allowed", False)),
    )


def _topic_for_item(item: CommunityContextItem, terms: set[str]) -> str:
    for candidate in item.topic_terms:
        clean = _clean_token(candidate)
        if clean and clean not in _STOP_WORDS:
            return clean
    preferred = (
        "recursion", "programming", "python", "api", "debugging", "explanation",
        "learning", "planning", "conversation",
    )
    for candidate in preferred:
        if candidate in terms:
            return "debugging" if candidate == "debug" else candidate
    return "general"


def _intent_guess(question: str, events: list[CognitivePatternEvent]) -> str:
    low = question.lower()
    if "debug" in low or "error" in low or any(event.kind == "debugging_pattern" for event in events):
        return "debugging_help"
    if "explain" in low or any(event.kind == "explanation_pattern" for event in events):
        return "explanation"
    if "compare" in low or "tradeoff" in low:
        return "comparison_or_tradeoff"
    if "how" in low or any(event.kind == "procedure_pattern" for event in events):
        return "procedure"
    return "answer_or_clarify"


def _facts_required(question: str) -> list[str]:
    low = question.lower()
    if any(word in low for word in ("latest", "today", "current", "price", "law", "ceo")):
        return ["fresh source-backed live information"]
    if any(word in low for word in ("what is", "who is", "when did", "where is")):
        return ["source-backed factual memory or cited lookup"]
    return ["only factual claims that can be backed by factual memory or live sources"]


def _first_pattern(events: list[CognitivePatternEvent], kinds: set[str]) -> str:
    for event in events:
        if event.kind in kinds:
            return event.pattern
    return ""


def _uncertainty_note(events: list[CognitivePatternEvent]) -> str:
    if any(event.kind == "uncertainty_pattern" for event in events):
        return "State the tradeoff, caveat, or missing evidence explicitly."
    return "State uncertainty when factual support is absent, stale, low-trust, or source-separated."


def _helpful_next_move(question: str, events: list[CognitivePatternEvent]) -> str:
    del question
    if any(event.kind == "debugging_pattern" for event in events):
        return "Ask for the smallest reproducible example or the exact error if it is missing."
    if any(event.kind == "explanation_pattern" for event in events):
        return "Give a simple explanation first, then one example and a caveat."
    return "Answer what is supported, then ask one focused follow-up if the task is underspecified."


def _confidence(item: CommunityContextItem, low: str, markers: tuple[str, ...]) -> str:
    marker_hits = sum(1 for marker in markers if marker in low)
    score = item.score or 0
    if marker_hits >= 3 or score >= 15:
        return "high"
    if marker_hits <= 1 and score < 2:
        return "low"
    return "medium"


def _add_node(
    graphs: dict,
    node_sets: dict[str, set[str]],
    layer: str,
    node_id: str,
    node_type: str,
    label: str,
) -> None:
    if node_id in node_sets[layer]:
        return
    node_sets[layer].add(node_id)
    graphs[layer]["nodes"].append({"id": node_id, "type": node_type, "label": label})


def _add_edge(
    graphs: dict,
    edge_sets: dict[str, set[tuple[str, str, str]]],
    layer: str,
    source: str,
    target: str,
    relation: str,
) -> None:
    key = (source, target, relation)
    if key in edge_sets[layer]:
        return
    edge_sets[layer].add(key)
    graphs[layer]["edges"].append({"source": source, "target": target, "relation": relation})


def _expanded_terms(text: str) -> set[str]:
    terms = set(_tokenize(text))
    if "bug" in terms or "error" in terms or "traceback" in terms:
        terms.add("debug")
    if "explain" in terms or "understand" in terms:
        terms.add("explanation")
    if "mistake" in terms or "pitfall" in terms:
        terms.add("mistake")
    if "question" in terms or "ask" in terms:
        terms.add("question")
    return terms


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in (_clean_token(match.group(0)) for match in re.finditer(r"[A-Za-z0-9][A-Za-z0-9_'-]*", text.lower()))
        if token and token not in _STOP_WORDS and len(token) > 2
    ]


def _clean_token(token: str) -> str:
    token = re.sub(r"(^[^a-z0-9]+|[^a-z0-9]+$)", "", token.lower())
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "general"


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _confidence_rank(confidence: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(confidence, 3)


def _to_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
