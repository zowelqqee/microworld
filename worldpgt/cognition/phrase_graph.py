"""Typed phrase graph for deterministic synthesis prose.

This module is a tiny local language model: it learns phrase fragments and
predicate-to-predicate transitions from local overlay/artifact text, then uses
deterministic graph traversal to render a ``SynthesisAnswer`` as connected
prose. It is offline, rule-bounded, and never introduces facts that are not
already present in the supplied synthesis result.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from worldpgt.entity_qa.types import SynthesisAnswer

PhraseNode = tuple[str, str]

_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTS = _ROOT / "worldpgt" / "experiments"
_DEFAULT_OVERLAYS = (
    _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json",
    _EXPERIMENTS / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json",
)
_DEFAULT_ARTIFACTS = (
    _EXPERIMENTS / "knowledge_pump_v1" / "assistant" / "assistant_surface_outputs.json",
    _EXPERIMENTS / "knowledge_pump_v1" / "pump_fact_qa_v1" / "pump_fact_qa_outputs.json",
)
_DEFAULT_SNAPSHOT_DIR = _EXPERIMENTS / "wiki_snapshots_v1" / "normalized_docs"

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
# "... is a businessman who ...", "... is an automobile that ...": captures the
# definition head noun and the relative pronoun that subordinates the next clause.
_SUBORDINATOR_RE = re.compile(
    r"\bis\s+(?:a|an|the)\s+([a-z][a-z -]*?)\s+(who|that|which)\b",
    re.IGNORECASE,
)

_CAPABILITY_PREDICATES = frozenset({
    "develops",
    "produces",
    "manufactures",
    "operates",
    "publishes",
    "provides",
    "enables",
    "uses",
    "used_for",
    "works_by",
    "supports",
    "runs_on",
    "offers",
})
_RELATION_PREDICATES = frozenset({
    "founded",
    "founded_by",
    "owned_by",
    "owns",
    "subsidiary_of",
    "part_of",
    "service_of",
    "platform_of",
    "located_in",
    "based_at",
    "headquartered_in",
})
_META_PREDICATES = frozenset({"known_for"})
_SNAPSHOT_KIND = "snapshot"
_INFERRED_KIND = "inferred"

_PREDICATE_ORDER = {
    "is_a": 0,
    "definition": 0,
    "develops": 10,
    "produces": 11,
    "manufactures": 12,
    "operates": 13,
    "publishes": 14,
    "provides": 15,
    "enables": 16,
    "uses": 17,
    "used_for": 18,
    "works_by": 19,
    "supports": 20,
    "runs_on": 21,
    "offers": 22,
    "founded": 40,
    "founded_by": 41,
    "owned_by": 42,
    "owns": 43,
    "subsidiary_of": 44,
    "part_of": 45,
    "service_of": 46,
    "platform_of": 47,
    "located_in": 48,
    "based_at": 49,
    "headquartered_in": 50,
    "known_for": 70,
    _SNAPSHOT_KIND: 90,
    _INFERRED_KIND: 100,
}

_PERSON_ROLE_TOKENS = frozenset({
    "actor",
    "artist",
    "author",
    "banker",
    "businessman",
    "businesswoman",
    "ceo",
    "chairman",
    "co-founder",
    "cofounder",
    "engineer",
    "entrepreneur",
    "executive",
    "founder",
    "inventor",
    "investor",
    "journalist",
    "lawyer",
    "magnate",
    "musician",
    "person",
    "philanthropist",
    "politician",
    "scientist",
    "writer",
})

_PHRASE_MARKERS: tuple[tuple[str, str], ...] = (
    ("was founded by", "founded_by"),
    ("founded", "founded"),
    ("is known for", "known_for"),
    ("known for", "known_for"),
    ("develops", "develops"),
    ("produces", "produces"),
    ("manufactures", "manufactures"),
    ("operates", "operates"),
    ("publishes", "publishes"),
    ("provides", "provides"),
    ("enables", "enables"),
    ("is used for", "used_for"),
    ("used for", "used_for"),
    ("uses", "uses"),
    ("works by", "works_by"),
    ("supports", "supports"),
    ("runs on", "runs_on"),
    ("offers", "offers"),
    ("is owned by", "owned_by"),
    ("owns", "owns"),
    ("is a subsidiary of", "subsidiary_of"),
    ("is part of", "part_of"),
    ("is based in", "located_in"),
    ("is based at", "based_at"),
    ("is headquartered in", "headquartered_in"),
    ("estimated net worth", "estimated_net_worth"),
)


@dataclass(frozen=True)
class TypedPhraseFact:
    """Renderable fact assigned to a graph node."""

    entity_type: str
    predicate: str
    kind: str
    subject: str
    objects: tuple[str, ...] = ()
    definition: str = ""
    source_name: str = ""
    as_of: str = ""
    rule: str = ""
    confidence: float | None = None

    @property
    def node(self) -> PhraseNode:
        return (self.entity_type or "entity", self.predicate)


@dataclass
class PhraseGraph:
    """Frequency graph over typed phrase fragments and connectors."""

    fragments: dict[PhraseNode, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    edges: dict[tuple[PhraseNode, PhraseNode], Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    # Relative-clause subordinators observed right after a copular definition
    # ("... is a businessman who ...", "... is a company that ..."), bucketed by
    # a coarse ``person`` / ``entity`` key and counted by frequency so the most
    # common learned pronoun can weave the first fact into the definition.
    subordinators: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))

    def add_fragment(self, entity_type: str, predicate: str, fragment: str) -> None:
        entity_type = entity_type or "entity"
        predicate = predicate or "related_to"
        fragment = _clean_fragment(fragment)
        if fragment:
            self.fragments[(entity_type, predicate)][fragment] += 1
            if entity_type != "entity":
                self.fragments[("entity", predicate)][fragment] += 1

    def add_transition(
        self,
        from_node: PhraseNode,
        to_node: PhraseNode,
        connector_phrase: str,
    ) -> None:
        if not from_node or not to_node:
            return
        connector_phrase = _clean_connector(connector_phrase)
        self.edges[(from_node, to_node)][connector_phrase] += 1
        generic_from = ("entity", from_node[1])
        generic_to = ("entity", to_node[1])
        self.edges[(generic_from, generic_to)][connector_phrase] += 1

    def best_fragment(self, node: PhraseNode) -> str | None:
        counters = [
            self.fragments.get(node),
            self.fragments.get(("entity", node[1])),
        ]
        if node[0] != "entity":
            counters.extend(counter for n, counter in self.fragments.items() if n[1] == node[1])
        for counter in counters:
            if counter:
                return _counter_best(counter)
        return None

    def best_connector(self, from_node: PhraseNode, to_node: PhraseNode) -> str | None:
        counters = [
            self.edges.get((from_node, to_node)),
            self.edges.get((("entity", from_node[1]), ("entity", to_node[1]))),
        ]
        for counter in counters:
            if counter:
                return _counter_best(counter)
        return None

    def learn_subordinator(self, entity_type: str, pronoun: str) -> None:
        pronoun = (pronoun or "").strip().lower()
        if pronoun not in {"who", "that", "which"}:
            return
        self.subordinators[_subordinator_key(entity_type)][pronoun] += 1

    def best_subordinator(self, entity_type: str) -> str | None:
        counter = self.subordinators.get(_subordinator_key(entity_type))
        if counter:
            return _counter_best(counter)
        return None

    def covers(self, facts: Iterable[TypedPhraseFact]) -> bool:
        return all(
            fact.predicate in {_SNAPSHOT_KIND, _INFERRED_KIND}
            or self.best_fragment(fact.node) is not None
            for fact in facts
        )


def build_phrase_graph(
    overlay_paths: Iterable[str | Path] | None = None,
    artifact_paths: Iterable[str | Path] | None = None,
    snapshot_dir: str | Path | None = None,
) -> PhraseGraph:
    """Build a phrase graph from local overlay, answer, and lead-paragraph data."""

    graph = PhraseGraph()
    for overlay_path in overlay_paths or _DEFAULT_OVERLAYS:
        _train_from_overlay(graph, Path(overlay_path))
    for artifact_path in artifact_paths or _DEFAULT_ARTIFACTS:
        _train_from_answer_artifact(graph, Path(artifact_path))
    _train_from_snapshot_leads(graph, Path(snapshot_dir) if snapshot_dir else _DEFAULT_SNAPSHOT_DIR)
    return graph


@lru_cache(maxsize=1)
def default_phrase_graph() -> PhraseGraph:
    return build_phrase_graph()


def generate(
    result: SynthesisAnswer,
    *,
    answer_style: str = "normal",
    graph: PhraseGraph | None = None,
) -> str | None:
    """Render a synthesis result through deterministic best-path traversal.

    Returns ``None`` when the learned graph lacks coverage for a predicate type;
    callers should then use their existing fallback renderer.
    """

    if result.unknown_notes:
        return None
    facts = facts_from_synthesis(result)
    if not facts:
        return None
    graph = graph or default_phrase_graph()
    if not graph.covers(facts):
        return None

    ordered = _order_facts(facts)
    if answer_style in {"brief", "followup"}:
        ordered = [f for f in ordered if f.predicate != _INFERRED_KIND][:2]

    sentences: list[str] = []
    inferred: list[str] = []
    start = 0

    # Fold the first fact into the definition through a learned relative clause
    # ("X is a Y that develops Z.") instead of emitting two choppy clauses. The
    # subordinator is chosen by frequency from real lead paragraphs, so this is
    # the graph's learned connector, not a hardcoded template.
    if ordered and ordered[0].predicate == "is_a":
        def_fact = ordered[0]
        clause = _definition_clause(def_fact.subject, def_fact.definition)
        start = 1
        pronoun = graph.best_subordinator(def_fact.entity_type)
        woven = None
        trailing = ""
        if pronoun and start < len(ordered):
            relative = _relative_clause_body(graph, ordered[start], result)
            if relative:
                body_text, trailing = relative
                woven = f"{clause} {pronoun} {body_text}."
                start += 1
        sentences.append(woven or f"{clause}.")
        if woven and trailing:
            sentences.append(trailing)

    for fact in ordered[start:]:
        sentence = _render_fact(graph, fact, result)
        if not sentence:
            return None
        if fact.predicate == _INFERRED_KIND:
            inferred.append(sentence)
        else:
            sentences.append(sentence)

    body = " ".join(sentences).strip()
    if inferred and answer_style not in {"brief", "followup"}:
        body = f"{body}\n\nBased on reasoning:\n{' '.join(inferred)}".strip()
    return body


def facts_from_synthesis(result: SynthesisAnswer) -> list[TypedPhraseFact]:
    subject = result.subject or ""
    entity_type = _entity_type(result.entity_type, result.definition)
    facts: list[TypedPhraseFact] = []
    if subject and result.definition:
        facts.append(
            TypedPhraseFact(
                entity_type=entity_type,
                predicate="is_a",
                kind="definition",
                subject=subject,
                definition=result.definition,
            )
        )
    elif subject and result.entity_type:
        facts.append(
            TypedPhraseFact(
                entity_type=entity_type,
                predicate="is_a",
                kind="definition",
                subject=subject,
                definition=result.entity_type,
            )
        )

    for group in result.groups:
        tier = str(getattr(group, "tier", "") or "")
        pred = str(getattr(group, "predicate", "") or "")
        if not pred:
            continue
        objects = tuple(str(obj) for obj in getattr(group, "objects", ()) if str(obj))
        if not objects:
            continue
        kind = str(getattr(group, "kind", "") or "")
        if tier == "SNAPSHOT":
            facts.append(
                TypedPhraseFact(
                    entity_type=entity_type,
                    predicate=_SNAPSHOT_KIND,
                    kind=pred,
                    subject=subject,
                    objects=objects,
                    source_name=str(getattr(group, "source_name", "") or ""),
                    as_of=str(getattr(group, "as_of", "") or ""),
                )
            )
            continue
        if tier == "INFERRED":
            facts.append(
                TypedPhraseFact(
                    entity_type=entity_type,
                    predicate=_INFERRED_KIND,
                    kind=pred,
                    subject=subject,
                    objects=objects,
                    rule=str(getattr(group, "rule", "") or ""),
                    confidence=getattr(group, "confidence", None),
                )
            )
            continue
        if pred == "is_a":
            continue
        render_pred = _render_predicate(kind, pred)
        facts.append(
            TypedPhraseFact(
                entity_type=entity_type,
                predicate=render_pred,
                kind=kind,
                subject=subject,
                objects=objects,
            )
        )

    return facts


def _train_from_overlay(graph: PhraseGraph, path: Path) -> None:
    if not path.is_file():
        return
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(rows, list):
        return

    entity_types: dict[str, str] = {}
    for item in rows:
        if item.get("overlay_type") != "overlay_entity":
            continue
        label = str(item.get("label") or "")
        entity_type = str(item.get("entity_type") or "entity")
        for surface in (label, item.get("source_page"), *(item.get("aliases") or [])):
            if surface:
                entity_types[_norm(surface)] = entity_type

    source_sequences: dict[str, list[PhraseNode]] = defaultdict(list)
    for item in rows:
        overlay_type = item.get("overlay_type")
        source_page = str(item.get("source_page") or "")
        if overlay_type == "overlay_definition":
            subject = str(item.get("subject") or source_page)
            entity_type = _entity_type(entity_types.get(_norm(subject)), item.get("definition"))
            graph.add_fragment(entity_type, "is_a", "is {definition}")
            source_sequences[source_page].append((entity_type, "is_a"))
            _train_from_sentences(graph, [str(item.get("evidence_text") or "")], entity_type)
        elif overlay_type == "overlay_relation":
            subject = str(item.get("subject") or source_page)
            pred = str(item.get("predicate") or "")
            entity_type = _entity_type(entity_types.get(_norm(subject)), None)
            render_pred = _render_predicate("forward_relation", pred)
            phrase = _fragment_from_evidence(str(item.get("evidence_text") or ""), pred)
            if phrase:
                graph.add_fragment(entity_type, render_pred, phrase)
            source_sequences[source_page].append((entity_type, render_pred))
            _train_from_sentences(graph, [str(item.get("evidence_text") or "")], entity_type)

    for nodes in source_sequences.values():
        _add_ordered_transitions(graph, _dedupe_adjacent(nodes))


def _train_from_answer_artifact(graph: PhraseGraph, path: Path) -> None:
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    rows = data if isinstance(data, list) else data.get("outputs", [])
    for row in rows:
        answer = row.get("answer", row) if isinstance(row, dict) else {}
        if not isinstance(answer, dict):
            continue
        answer_text = str(answer.get("answer_text") or answer.get("answer") or "")
        entity_type = _entity_type_from_answer(answer)
        _train_from_sentences(graph, _sentences(answer_text), entity_type)


def _train_from_snapshot_leads(graph: PhraseGraph, snapshot_dir: Path) -> None:
    if not snapshot_dir.is_dir():
        return
    for path in sorted(snapshot_dir.glob("*.md"))[:250]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        body = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
        lead = ""
        for line in body:
            if ":" in line and line.split(":", 1)[0] in {
                "Source",
                "Retrieved at",
                "Revision ID",
                "Raw text SHA256",
                "Status",
                "Safe for accepted memory",
                "Requires ingestion/quarantine/promotion/regression",
            }:
                continue
            lead = line
            break
        if lead:
            _train_from_sentences(graph, _sentences(lead), "entity")


def _train_from_sentences(graph: PhraseGraph, sentences: Iterable[str], entity_type: str) -> None:
    previous: PhraseNode | None = None
    for sentence in sentences:
        pronoun = _subordinator_from_sentence(sentence)
        if pronoun:
            graph.learn_subordinator(_entity_type(entity_type, sentence), pronoun)
        for node in _nodes_from_sentence(sentence, entity_type):
            fragment = _fragment_from_sentence(sentence, node[1])
            if fragment:
                graph.add_fragment(node[0], node[1], fragment)
            if previous is not None:
                graph.add_transition(previous, node, _connector_between_sentences(sentence))
            previous = node


def _add_ordered_transitions(graph: PhraseGraph, nodes: list[PhraseNode]) -> None:
    ordered = sorted(nodes, key=lambda n: (_PREDICATE_ORDER.get(n[1], 80), n[1]))
    for left, right in zip(ordered, ordered[1:]):
        graph.add_transition(left, right, "")


def _node_from_sentence(sentence: str, entity_type: str) -> PhraseNode | None:
    nodes = _nodes_from_sentence(sentence, entity_type)
    return nodes[0] if nodes else None


def _nodes_from_sentence(sentence: str, entity_type: str) -> list[PhraseNode]:
    low = sentence.lower()
    found: list[tuple[int, PhraseNode]] = []
    if re.search(r"\bis\s+(?:an?|the)\s+", low):
        found.append((low.find(" is "), (_entity_type(entity_type, sentence), "is_a")))
    seen_predicates: set[str] = set()
    for marker, predicate in _PHRASE_MARKERS:
        idx = low.find(marker)
        if idx >= 0 and predicate not in seen_predicates:
            found.append((idx, (_entity_type(entity_type, sentence), predicate)))
            seen_predicates.add(predicate)
    found.sort(key=lambda item: item[0])
    return [node for _idx, node in found]


def _fragment_from_sentence(sentence: str, predicate: str) -> str:
    text = " ".join(sentence.strip().rstrip(".").split())
    low = text.lower()
    if predicate == "is_a":
        match = re.search(r"\bis\s+((?:an?|the)\s+)?", text, re.IGNORECASE)
        if match:
            return "is {definition}"
        return ""
    for marker, pred in _PHRASE_MARKERS:
        if pred != predicate:
            continue
        idx = low.find(marker)
        if idx >= 0:
            return f"{text[idx:idx + len(marker)]} {{object_list}}"
    return ""


def _fragment_from_evidence(evidence_text: str, predicate: str) -> str:
    evidence_text = " ".join((evidence_text or "").strip().rstrip(".").split())
    if not evidence_text:
        return ""
    render_pred = _render_predicate("forward_relation", predicate)
    fragment = _fragment_from_sentence(evidence_text, render_pred)
    if fragment:
        return fragment
    phrase = {
        "founded": "founded {object_list}",
        "founded_by": "was founded by {object_list}",
        "develops": "develops {object_list}",
        "produces": "produces {object_list}",
        "known_for": "is known for {object_list}",
        "leader_of": "leads {object_list}",
        "provides": "provides {object_list}",
        "enables": "enables {object_list}",
        "uses": "uses {object_list}",
        "used_for": "is used for {object_list}",
    }.get(predicate)
    return phrase or ""


def _render_fact(graph: PhraseGraph, fact: TypedPhraseFact, result: SynthesisAnswer) -> str:
    if fact.predicate == _SNAPSHOT_KIND:
        return _render_snapshot(fact, result)
    if fact.predicate == _INFERRED_KIND:
        return _render_inferred(fact, result)
    fragment = graph.best_fragment(fact.node)
    if fragment is None:
        return ""
    if fact.predicate == "is_a":
        return _definition_sentence(fact.subject, fact.definition)
    reference = _reference(fact.subject, fact.entity_type, result.definition)
    obj_list = _join_limited(list(fact.objects))
    if not obj_list:
        return ""
    sentence = _sentence_from_fragment(reference, fragment, obj_list)
    enrichment = getattr(result, "enrichment", None)
    if (
        fact.predicate in {"founded", "founded_by", "owned_by", "owns"}
        and enrichment
        and str(getattr(enrichment, "object", "") or "") in fact.objects
    ):
        sentence, extra = _apply_enrichment(sentence, enrichment)
        if extra:
            return f"{sentence} {extra}"
    return sentence


def _sentence_from_fragment(reference: str, fragment: str, obj_list: str) -> str:
    phrase = fragment.replace("{object_list}", obj_list).strip()
    if phrase.startswith("known for "):
        phrase = "is " + phrase
    if phrase.startswith("used for "):
        phrase = "is " + phrase
    if reference == "They":
        if phrase.startswith("is "):
            phrase = "are " + phrase.removeprefix("is ")
        elif phrase.startswith("was "):
            phrase = "were " + phrase.removeprefix("was ")
    return f"{reference} {phrase}."


def _relative_clause_body(
    graph: PhraseGraph, fact: TypedPhraseFact, result: SynthesisAnswer
) -> tuple[str, str] | None:
    """Build the referent-less clause used after a learned subordinator.

    Returns ``(clause_body, trailing_sentence)`` where ``trailing_sentence`` is
    an optional standalone enrichment sentence (e.g. defining a founded object)
    that follows the woven sentence. Returns ``None`` for tiers that do not read
    cleanly inside a relative clause, leaving the fact to render on its own.
    """

    if fact.predicate in {_SNAPSHOT_KIND, _INFERRED_KIND, "is_a"}:
        return None
    fragment = graph.best_fragment(fact.node)
    if fragment is None:
        return None
    obj_list = _join_limited(list(fact.objects))
    if not obj_list:
        return None
    phrase = fragment.replace("{object_list}", obj_list).strip()
    if phrase.startswith(("known for ", "used for ")):
        phrase = "is " + phrase
    if not phrase:
        return None

    trailing = ""
    enrichment = getattr(result, "enrichment", None)
    if (
        fact.predicate in {"founded", "founded_by", "owned_by", "owns"}
        and enrichment
        and str(getattr(enrichment, "object", "") or "") in fact.objects
    ):
        note = str(getattr(enrichment, "note", "") or "").strip().rstrip(".")
        obj = str(getattr(enrichment, "object", "") or "").strip()
        if note and obj:
            if phrase.endswith(obj):
                phrase = f"{phrase}, {_article_phrase(note)}"
            else:
                trailing = _definition_sentence(obj, note)
    return phrase, trailing


def _render_snapshot(fact: TypedPhraseFact, result: SynthesisAnswer) -> str:
    obj = fact.objects[0] if fact.objects else ""
    if not obj:
        return ""
    source = fact.source_name or "source"
    as_of = _human_as_of(fact.as_of)
    possessive = _possessive(fact.subject, fact.entity_type, result.definition)
    predicate = fact.kind.replace("_", " ") if fact.kind else "estimate"
    if as_of:
        return f"As of {as_of}, {possessive} {predicate} is {obj} ({source})."
    return f"{fact.subject}'s {predicate} was {obj} ({source} - may be outdated)."


def _render_inferred(fact: TypedPhraseFact, result: SynthesisAnswer) -> str:
    phrase = {
        "competes_with": "competes with",
        "share_founder": "shares a founder with",
        "share_leader": "shares a leader with",
        "associated_with_expertise": "is associated with expertise around",
        "indirectly_requires": "indirectly requires",
    }.get(fact.kind, "")
    if not phrase:
        phrase = {
            "competitor_detection_v1": "competes with",
            "shared_founder_v1": "shares a founder with",
            "shared_leader_v1": "shares a leader with",
            "expertise_association_v1": "is associated with expertise around",
        }.get(fact.rule or "", "")
    if not phrase:
        phrase = _fallback_phrase("inferred_relation", fact.kind)
    reference = _reference(fact.subject, fact.entity_type, result.definition)
    return _sentence_from_fragment(reference, f"{phrase} {{object_list}}", _join_limited(list(fact.objects)))


def _apply_enrichment(sentence: str, enrichment) -> tuple[str, str]:
    note = str(getattr(enrichment, "note", "") or "").strip().rstrip(".")
    obj = str(getattr(enrichment, "object", "") or "").strip()
    if not note or not obj:
        return sentence, ""
    note_phrase = _article_phrase(note)
    if sentence.endswith(f"{obj}."):
        return f"{sentence[:-1]}, {note_phrase}.", ""
    return sentence, _definition_sentence(obj, note)


def _order_facts(facts: list[TypedPhraseFact]) -> list[TypedPhraseFact]:
    return sorted(
        facts,
        key=lambda fact: (
            _PREDICATE_ORDER.get(fact.predicate, 80),
            fact.predicate,
            fact.kind,
            ",".join(fact.objects),
        ),
    )


def _render_predicate(kind: str, predicate: str) -> str:
    if kind == "inverse_relation":
        return {
            "founded": "founded_by",
            "owned_by": "owns",
            "develops": "developed_by",
            "produces": "produced_by",
            "publishes": "published_by",
            "leader_of": "led_by",
            "subsidiary_of": "parent_company_of",
        }.get(predicate, predicate)
    return predicate


def _fallback_phrase(kind: str, predicate: str) -> str:
    if kind == "inverse_relation":
        return {
            "founded": "was founded by",
            "develops": "is developed by",
            "produces": "is produced by",
            "leader_of": "is led by",
        }.get(predicate, f"is linked to via {predicate}")
    return {
        "founded": "founded",
        "founded_by": "was founded by",
        "develops": "develops",
        "produces": "produces",
        "known_for": "is known for",
        "owned_by": "is owned by",
        "owns": "owns",
        "located_in": "is based in",
        "based_at": "is based at",
        "headquartered_in": "is headquartered in",
    }.get(predicate, f"is linked to via {predicate}")


def _entity_type(entity_type: str | None, definition: str | None) -> str:
    entity_type = (entity_type or "").strip().lower()
    if entity_type and entity_type not in {"other", "unknown", "entity"}:
        return entity_type
    tokens = set(re.findall(r"[a-z-]+", (definition or "").lower()))
    return "person" if tokens & _PERSON_ROLE_TOKENS else "entity"


def _entity_type_from_answer(answer: dict) -> str:
    trace = answer.get("trace") or {}
    ctx = trace.get("context_summary") or {}
    definitions = ctx.get("definitions") or []
    if definitions:
        joined = " ".join(str(d) for d in definitions)
        return _entity_type(None, joined)
    return "entity"


def _reference(subject: str, entity_type: str, definition: str | None) -> str:
    if _entity_type(entity_type, definition) == "person":
        if subject in {"Elon Musk", "Jeff Bezos", "Michael Bloomberg", "Ray Kroc"}:
            return "He"
        return "They"
    return "It"


def _possessive(subject: str, entity_type: str, definition: str | None) -> str:
    if _entity_type(entity_type, definition) == "person":
        if subject in {"Elon Musk", "Jeff Bezos", "Michael Bloomberg", "Ray Kroc"}:
            return "his"
        return "their"
    return "its"


def _definition_clause(subject: str, definition: str) -> str:
    definition = (definition or "").strip().rstrip(".")
    article = _article_for_definition(definition)
    if article:
        return f"{subject} is {article} {definition}"
    return f"{subject} is {definition}"


def _definition_sentence(subject: str, definition: str) -> str:
    return f"{_definition_clause(subject, definition)}."


def _article_phrase(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    if _ARTICLE_RE.match(text):
        return text
    article = _article_for_definition(text)
    return f"{article} {text}" if article else text


def _article_for_definition(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "a"
    if _ARTICLE_RE.match(text):
        return ""
    lower = text.lower()
    if lower.startswith(("one of ", "part of ")):
        return ""
    if lower.startswith(("uni", "use", "user", "euro", "eu ")):
        return "a"
    return "an" if lower[:1] in {"a", "e", "i", "o", "u"} else "a"


def _join_limited(items: list[str], *, max_items: int = 4) -> str:
    deduped = _dedupe_ci(items)
    if len(deduped) > max_items:
        shown = deduped[:max_items]
        shown.append(f"{len(deduped) - max_items} more")
        return _join_list(shown)
    return _join_list(deduped)


def _join_list(items: list[str]) -> str:
    items = [str(item).strip() for item in items if str(item).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _dedupe_ci(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        cleaned = str(item).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            out.append(cleaned)
    return out


def _dedupe_adjacent(nodes: list[PhraseNode]) -> list[PhraseNode]:
    out: list[PhraseNode] = []
    for node in nodes:
        if not out or out[-1] != node:
            out.append(node)
    return out


def _sentences(text: str) -> list[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _clean_fragment(fragment: str) -> str:
    return " ".join((fragment or "").strip().rstrip(".").split())


def _clean_connector(connector: str) -> str:
    return " ".join((connector or "").strip().split())


def _connector_between_sentences(sentence: str) -> str:
    text = sentence.strip()
    if re.match(r"^(?:it|he|she|they|as of)\b", text, re.IGNORECASE):
        return ""
    return ""


def _subordinator_from_sentence(sentence: str) -> str | None:
    match = _SUBORDINATOR_RE.search(sentence or "")
    if not match:
        return None
    head = match.group(1).lower()
    # Guard against the pronoun binding to a trailing modifier rather than the
    # definition head (e.g. "is a person, who ..."): keep only short heads.
    if len(head.split()) > 6:
        return None
    return match.group(2).lower()


def _subordinator_key(entity_type: str) -> str:
    return "person" if (entity_type or "").strip().lower() == "person" else "entity"


def _counter_best(counter: Counter[str]) -> str:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _norm(s: str) -> str:
    return " ".join(_TOKEN_RE.findall((s or "").lower())).removeprefix("the ")


def _human_as_of(value: str) -> str:
    match = re.fullmatch(r"(\d{4})-(\d{2})", value.strip())
    if not match:
        return value
    months = {
        "01": "January",
        "02": "February",
        "03": "March",
        "04": "April",
        "05": "May",
        "06": "June",
        "07": "July",
        "08": "August",
        "09": "September",
        "10": "October",
        "11": "November",
        "12": "December",
    }
    return f"{months.get(match.group(2), match.group(2))} {match.group(1)}"
