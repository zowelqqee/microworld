"""Read-only graph for attaching question language to relation predicates.

The graph has phrase nodes connected by ``denotes`` edges to canonical
predicate nodes.  It deliberately stores no domain facts and does not create
aliases: it is an input interpretation layer, analogous to the graph-backed
entity input layer.  New wording is added as graph data, not as a parser
branch.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable


_GRAPH_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "relation_input_graph_v1.json"
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", re.IGNORECASE)
_QUESTION_FUNCTION_WORDS = frozenset({
    "a", "an", "are", "be", "does", "do", "did", "for", "how", "is",
    "of", "the", "to", "was", "were", "what", "which", "who", "whom",
})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(text))


def _without_spans(text: str, spans: Iterable[tuple[int, int]]) -> str:
    """Replace resolved entity spans with spaces, preserving question order."""

    pieces: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        if start < cursor:
            continue
        pieces.append(text[cursor:start])
        pieces.append(" ")
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def question_frame_tokens(
    question: str,
    *,
    entity_spans: Iterable[tuple[int, int]] = (),
) -> tuple[str, ...]:
    """Return the content-bearing question frame after removing entity spans."""

    frame = _without_spans(question, entity_spans)
    return tuple(token for token in _tokens(frame) if token not in _QUESTION_FUNCTION_WORDS)


@dataclass(frozen=True)
class RelationInputGraph:
    """A compact phrase-to-predicate graph loaded from repository data."""

    phrase_edges: tuple[tuple[tuple[str, ...], str], ...]

    @classmethod
    def from_path(cls, path: Path) -> "RelationInputGraph":
        payload = json.loads(path.read_text(encoding="utf-8"))
        nodes = {
            node["id"]: node
            for node in payload.get("nodes", [])
            if isinstance(node, dict) and isinstance(node.get("id"), str)
        }
        edges: list[tuple[tuple[str, ...], str]] = []
        for edge in payload.get("edges", []):
            if not isinstance(edge, dict) or edge.get("predicate") != "denotes":
                continue
            source, target = nodes.get(edge.get("source")), nodes.get(edge.get("target"))
            if not source or not target or source.get("kind") != "phrase" or target.get("kind") != "predicate":
                continue
            phrase = _tokens(str(source.get("surface", "")))
            predicate = target.get("predicate")
            if phrase and isinstance(predicate, str) and predicate:
                edges.append((phrase, predicate))
        return cls(phrase_edges=tuple(sorted(edges, key=lambda row: (-len(row[0]), row[0], row[1]))))

    def resolve(self, question: str, *, entity_spans: Iterable[tuple[int, int]] = ()) -> str | None:
        """Return the best predicate denoted by the entity-free question frame.

        Resolution is graph traversal over ``phrase --denotes--> predicate``.
        Function words are discarded so the parser need not contain a special
        syntactic template for each question form.
        """

        matches = self.resolve_all(question, entity_spans=entity_spans)
        return next(iter(matches), None)

    def resolve_all(self, question: str, *, entity_spans: Iterable[tuple[int, int]] = ()) -> tuple[str, ...]:
        """Return every distinct predicate denoted by the question frame.

        A coordinated question can name more than one relation.  This remains
        graph traversal over phrase ``denotes`` edges: the parser receives the
        graph result instead of carrying a new syntactic template per wording.
        """

        content = question_frame_tokens(question, entity_spans=entity_spans)
        matches: list[tuple[int, str]] = []
        seen: set[str] = set()
        for phrase, predicate in self.phrase_edges:
            if predicate in seen or len(phrase) > len(content):
                continue
            offsets = [
                offset for offset in range(len(content) - len(phrase) + 1)
                if content[offset : offset + len(phrase)] == phrase
            ]
            if offsets:
                seen.add(predicate)
                matches.append((offsets[0], predicate))
        return tuple(predicate for _offset, predicate in sorted(matches))


@lru_cache(maxsize=1)
def default_relation_input_graph() -> RelationInputGraph:
    return RelationInputGraph.from_path(_GRAPH_PATH)
