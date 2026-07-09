"""Typed phrase graph for deterministic synthesis prose.

This module is a tiny local language model: it learns phrase fragments and
predicate-to-predicate transitions from local overlay/artifact text, then uses
deterministic graph traversal to render a ``SynthesisAnswer`` as connected
prose. It is offline, rule-bounded, and never introduces facts that are not
already present in the supplied synthesis result.

Generic verb learning (2026-07-07): ``_PHRASE_MARKERS`` below is a fixed,
hand-enumerated list of ~26 surface phrases. This is the SPEECH side of the
system (learning *how things are phrased*), NOT the facts side, so it must
not be restricted to a curated relation vocabulary: any content verb
("make", "have", "include", "win", "build", ...) is legitimate language to
learn a phrasing for. ``_generic_verb_fragment`` learns a fragment for any
active-voice SVO verb -- keyed by its canonical predicate when the verb is in
``relation_extraction_v2.spacy_extractor``'s ``_VERB_RELATIONS`` table
("lead" -> "leader_of", so it lines up with overlay facts), else by the
verb's own surface lemma (matching how schema_induction keeps surface
relations). Copular "be" is excluded (the is_a definition path owns it). The
noise that matters for fact *extraction* ("is have a real relation?") is
irrelevant here -- a fragment for an unused predicate is simply never looked
up, since lookup is driven by the predicates that actually appear in facts.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Iterable

from worldpgt.entity_qa.semantic_speech_planner import _join_clauses
from worldpgt.entity_qa.types import SynthesisAnswer
from worldpgt.relation_extraction_v2.spacy_extractor import _VERB_RELATIONS

# spaCy is optional -- see raw_claim_extractor.py / query_compiler.py's
# identical lazy-load pattern. Only used by _generic_verb_fragment below;
# everything else in this module is pure string/frequency-table logic and
# works without it.
_NLP = None
try:
    import spacy as _spacy_mod  # noqa: F401
    _SPACY_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without spaCy
    _SPACY_AVAILABLE = False


def _get_nlp():
    global _NLP
    if _NLP is None:
        if not _SPACY_AVAILABLE:
            return None
        try:
            _NLP = _spacy_mod.load("en_core_web_sm")
        except Exception:  # pragma: no cover - model missing
            return None
    return _NLP

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
_DEFAULT_COMMUNITY_CONTEXT_PATHS = (
    _EXPERIMENTS / "community_context_v1" / "reddit_community_context.json",
)

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

# When a node's most frequent eligible phrasing is observed at least this
# often, once-seen sibling phrasings are treated as noise and excluded from
# variant selection (see PhraseGraph.fragment_variant).
_SINGLETON_DENOISE_FLOOR = 3

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

# Closed-class function words never carry type signal ("the", "and", "of");
# excluding them keeps ``word_types`` from being dominated by whichever
# entity_type happens to be most numerous in the corpus.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "is", "was", "are", "were", "that", "which",
    "who", "its", "it", "this", "these", "those", "also", "known",
})

# Publicly exported: the dialogue reference grammar reuses this set to decide
# whether a noun is eligible for role-descriptor resolution ("the founder"),
# instead of keeping a separate copy. It's the same speech-layer vocabulary
# that already drives He/They pronoun choice below.
PERSON_ROLE_TOKENS = frozenset({
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
_PERSON_ROLE_TOKENS = PERSON_ROLE_TOKENS  # local alias, kept for brevity below

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
    # Reverse index learned from the same training pass: which entity_type a
    # content word from a definition tends to co-occur with ("satellite" seen
    # in Starlink's "satellite internet constellation" definition -> whatever
    # canonical type Starlink is actually tagged with in the overlay). This is
    # the dialogue reference grammar's understanding of nouns -- speech-layer
    # data, trained the same way fragments/transitions are, never read from
    # live fact lookups at resolution time.
    word_types: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    # Which learned fragments are grammatical to drop straight into the render
    # frame "It {fragment}." / "They {fragment}.". Only these are eligible for
    # seeded variant selection. Auto-learned generic-verb fragments qualify
    # only when their leading verb is finite (VBZ/VBD/VBN "develops/founded"),
    # never a bare infinitive or gerund ("create"/"producing" -> "It create X"
    # is ungrammatical); hand-verified _PHRASE_MARKERS / is_a fragments are
    # always eligible. This is what lets variant selection add variety without
    # ever degrading grammar -- the noise top-1 currently sidesteps is instead
    # excluded from the candidate set outright.
    fragment_eligible: dict[PhraseNode, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_fragment(
        self,
        entity_type: str,
        predicate: str,
        fragment: str,
        *,
        frame_eligible: bool = True,
    ) -> None:
        entity_type = entity_type or "entity"
        predicate = predicate or "related_to"
        fragment = _clean_fragment(fragment)
        if fragment:
            self.fragments[(entity_type, predicate)][fragment] += 1
            if frame_eligible:
                self.fragment_eligible[(entity_type, predicate)].add(fragment)
            if entity_type != "entity":
                self.fragments[("entity", predicate)][fragment] += 1
                if frame_eligible:
                    self.fragment_eligible[("entity", predicate)].add(fragment)

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

    def _resolve_fragment_node(self, node: PhraseNode) -> tuple[PhraseNode | None, Counter[str] | None]:
        """Return the (node_key, counter) ``best_fragment`` would read, in the
        same priority order: exact node, then the generic ``("entity", pred)``
        node, then any same-predicate node. Shared by best_fragment and
        fragment_variant so both resolve identically."""

        exact = self.fragments.get(node)
        if exact:
            return node, exact
        generic = self.fragments.get(("entity", node[1]))
        if generic:
            return ("entity", node[1]), generic
        if node[0] != "entity":
            for n, counter in self.fragments.items():
                if n[1] == node[1] and counter:
                    return n, counter
        return None, None

    def best_fragment(self, node: PhraseNode) -> str | None:
        _key, counter = self._resolve_fragment_node(node)
        return _counter_best(counter) if counter else None

    def fragment_variant(self, node: PhraseNode, seed: str) -> str | None:
        """Seeded choice among the *frame-eligible* learned fragments for
        ``node`` (frequency-weighted, deterministic in ``seed`` so the same
        entity always renders identically). Falls back to ``best_fragment``
        when the resolved node has no eligibility record — coverage and the
        exact-match tests never regress, and variety appears only where the
        graph actually learned two or more grammatical phrasings."""

        key, counter = self._resolve_fragment_node(node)
        if not counter:
            return None
        eligible = self.fragment_eligible.get(key, set())
        choices = [(frag, count) for frag, count in counter.items() if frag in eligible]
        # Singleton denoise: a once-seen phrasing next to a well-supported one
        # is training noise (a mis-tensed "produced" beside "produces"@54, a
        # "useS" typo the tagger mistook for finite) — exclude it so variety is
        # drawn only from repeatedly-observed phrasings. This is the hygiene
        # step done at selection time, leaving the learned counts intact for
        # inspection and the exact-match tests.
        if choices:
            top = max(count for _frag, count in choices)
            if top >= _SINGLETON_DENOISE_FLOOR:
                choices = [(frag, count) for frag, count in choices if count > 1]
        if len(choices) <= 1:
            return _counter_best(counter)
        return _seeded_weighted_pick(seed, f"frag:{key}", choices)

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

    def learn_word_type(self, word: str, entity_type: str) -> None:
        normalized = (word or "").strip().lower()
        if len(normalized) < 3 or normalized in _STOPWORDS:
            return
        if not entity_type or entity_type in {"entity", "other", "unknown"}:
            return
        self.word_types[normalized][entity_type] += 1

    def type_for_word(self, word: str, min_count: int = 2) -> str | None:
        """Canonical type most often associated with *word* in training data,
        or None below ``min_count`` observations (a single co-occurrence is
        noise, same threshold schema_induction uses for local types)."""

        normalized = (word or "").strip().lower()
        counter = self.word_types.get(normalized)
        if not counter:
            return None
        etype, count = counter.most_common(1)[0]
        return etype if count >= min_count else None

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
    community_context_paths: Iterable[str | Path] | None = None,
) -> PhraseGraph:
    """Build a phrase graph from local overlay, answer, and lead-paragraph data.

    ``community_context_paths`` adds a low-trust, style-only source: accepted
    Reddit ``CommunityContextItem`` records (see
    ``worldpgt.community_context``). Training from this source can only ever
    add phrase *fragments* and *connectors* (see ``_fragment_from_sentence``,
    which stores templated wording like ``"is known for {object_list}"``,
    never a literal subject/object) — it cannot make Reddit text a source of
    facts, because facts still come exclusively from the ``SynthesisAnswer``
    passed to ``generate``.
    """

    graph = PhraseGraph()
    resolved_overlays = _DEFAULT_OVERLAYS if overlay_paths is None else overlay_paths
    for overlay_path in resolved_overlays:
        _train_from_overlay(graph, Path(overlay_path))
    resolved_artifacts = _DEFAULT_ARTIFACTS if artifact_paths is None else artifact_paths
    for artifact_path in resolved_artifacts:
        _train_from_answer_artifact(graph, Path(artifact_path))
    _train_from_snapshot_leads(
        graph, Path(snapshot_dir) if snapshot_dir is not None else _DEFAULT_SNAPSHOT_DIR
    )
    resolved_community = (
        _DEFAULT_COMMUNITY_CONTEXT_PATHS if community_context_paths is None else community_context_paths
    )
    for community_path in resolved_community:
        _train_from_community_context_file(graph, Path(community_path))
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

    # Inferred (derived, non-stated) facts are left out of the answer — it
    # states the directly supported profile, not a reasoning dump.
    ordered = [f for f in _order_facts(facts) if f.predicate != _INFERRED_KIND]
    if answer_style in {"brief", "followup"}:
        ordered = ordered[:2]

    # Per-entity phrasing seed: the same entity always renders the same way
    # (deterministic, replayable), different entities draw different eligible
    # phrasings where the graph learned more than one.
    seed = _render_seed(result)

    sentences: list[str] = []
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
            weave_end = start + 1
            if ordered[start].predicate in _CAPABILITY_PREDICATES:
                while (
                    weave_end < len(ordered)
                    and ordered[weave_end].predicate in _CAPABILITY_PREDICATES
                ):
                    weave_end += 1
            relative = _relative_clause_body(graph, ordered[start:weave_end], result, seed)
            if relative:
                body_text, trailing = relative
                woven = f"{clause} {pronoun} {body_text}."
                start = weave_end
        sentences.append(woven or f"{clause}.")
        if woven and trailing:
            sentences.append(trailing)

    # Consecutive capability facts (develops/produces/operates/...) read as one
    # developed sentence with a shared subject ("It develops X, and produces
    # Y.") instead of a flat "It X. It Y." run -- the same comma-plus-"and"
    # joining semantic_speech_planner already uses for its own clause lists,
    # applied here to facts instead of pre-built clause strings.
    i = start
    while i < len(ordered):
        fact = ordered[i]
        if fact.predicate in _CAPABILITY_PREDICATES:
            j = i + 1
            while j < len(ordered) and ordered[j].predicate in _CAPABILITY_PREDICATES:
                j += 1
            sentence = _render_capability_run(graph, ordered[i:j], result, seed)
            if not sentence:
                return None
            sentences.append(sentence)
            i = j
        else:
            sentence = _render_fact(graph, fact, result, seed)
            if not sentence:
                return None
            sentences.append(sentence)
            i += 1

    return " ".join(sentences).strip()


def _render_capability_run(
    graph: PhraseGraph,
    facts: list[TypedPhraseFact],
    result: SynthesisAnswer,
    seed: str,
) -> str:
    """Render one or more consecutive capability facts as a single sentence
    sharing one subject reference. A run of one behaves exactly like
    ``_render_fact`` did before (no regression for the common single-fact
    case); a run of two or more is where the elaboration happens."""

    reference = _reference(facts[0].subject, facts[0].entity_type, result.definition)
    phrases: list[str] = []
    for fact in facts:
        fragment = graph.fragment_variant(fact.node, seed)
        if fragment is None:
            return ""
        obj_list = _join_limited(list(fact.objects))
        if not obj_list:
            continue
        phrases.append(_fragment_phrase(fragment, obj_list, reference))
    if not phrases:
        return ""
    return f"{reference} {_join_clauses(phrases)}."


def _render_seed(result: SynthesisAnswer) -> str:
    """Stable per-entity seed for phrasing variety. Keyed on the subject (and
    its definition) so a given entity always renders identically across runs
    and sessions, while different entities diverge."""

    return f"{result.subject or ''}|{result.definition or ''}"


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

    return _merge_same_predicate_facts(facts)


def _merge_same_predicate_facts(facts: list[TypedPhraseFact]) -> list[TypedPhraseFact]:
    """Combine facts that render under the same predicate into one, so two
    co-founders recorded on opposite sides of the overlay (a forward
    "founded_by" fact and an inverse "founded" fact that ``_render_predicate``
    both map to "founded_by") read as "was founded by X and Y." instead of two
    separate "was founded by X. was founded by Y." sentences -- the same
    combination ``semantic_speech_planner``'s origin bucket already does.
    Snapshot/inferred facts keep the shared ``_SNAPSHOT_KIND``/``_INFERRED_KIND``
    predicate constant for a different purpose (their real predicate lives in
    ``kind``), so they -- and ``is_a``, always singular -- are left alone.
    """

    skip = {"is_a", _SNAPSHOT_KIND, _INFERRED_KIND}
    merged: dict[str, int] = {}
    out: list[TypedPhraseFact] = []
    for fact in facts:
        if fact.predicate in skip:
            out.append(fact)
            continue
        existing_idx = merged.get(fact.predicate)
        if existing_idx is None:
            merged[fact.predicate] = len(out)
            out.append(fact)
            continue
        existing = out[existing_idx]
        combined = existing.objects + tuple(o for o in fact.objects if o not in existing.objects)
        out[existing_idx] = replace(existing, objects=combined)
    return out


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
            definition = item.get("definition")
            entity_type = _entity_type(entity_types.get(_norm(subject)), definition)
            graph.add_fragment(entity_type, "is_a", "is {definition}")
            source_sequences[source_page].append((entity_type, "is_a"))
            _train_from_sentences(graph, [str(item.get("evidence_text") or "")], entity_type)
            # Only learn word->type from an *explicit* overlay type (not the
            # role-token fallback _entity_type just computed above) -- else a
            # definition would teach the graph to associate its own words with
            # a type that was itself guessed from those same words.
            explicit_type = entity_types.get(_norm(subject))
            if explicit_type and definition:
                for token in _TOKEN_RE.findall(str(definition).lower()):
                    graph.learn_word_type(token, explicit_type)
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


def _train_from_community_context_file(graph: PhraseGraph, path: Path) -> None:
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    rows = data if isinstance(data, list) else data.get("items", [])
    if not isinstance(rows, list):
        return
    _train_from_community_context(graph, (row for row in rows if isinstance(row, dict)))


def _train_from_community_context(
    graph: PhraseGraph,
    items: Iterable[dict],
    *,
    max_items: int = 2000,
) -> None:
    """Learn phrasing (never facts) from accepted low-trust Reddit items.

    Only sentences that already match a known predicate marker or a
    copular "is a/an/the" pattern (see ``_nodes_from_sentence``) contribute
    anything; everything else is silently inert. What is learned is always a
    fragment template or a connector phrase, keyed by predicate — never the
    literal subject/object content of the Reddit post.

    Relative-pronoun learning (``who``/``that``/``which``) is deliberately
    skipped for this source: casual Reddit phrasing leans on "that" for
    people more than the curated wiki lead paragraphs do, and that one
    grammar choice is kept anchored to the cleaner source rather than
    swayed by informal text.
    """

    for item in itertools.islice(items, max_items):
        if str(item.get("trust") or "") != "community_context_only":
            continue
        text = " ".join(
            part for part in (str(item.get("title") or ""), str(item.get("text") or "")) if part
        )
        _train_from_sentences(graph, _sentences(text), "entity", learn_subordinators=False)


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


def _train_from_sentences(
    graph: PhraseGraph,
    sentences: Iterable[str],
    entity_type: str,
    *,
    learn_subordinators: bool = True,
) -> None:
    previous: PhraseNode | None = None
    for sentence in sentences:
        if learn_subordinators:
            pronoun = _subordinator_from_sentence(sentence)
            if pronoun:
                graph.learn_subordinator(_entity_type(entity_type, sentence), pronoun)
        # A fragment is frame-eligible unless it is the auto-learned generic
        # verb whose surface disagrees with the "It ___" frame.
        generic_pred, generic_eligible = _generic_fragment_eligibility(sentence)
        for node in _nodes_from_sentence(sentence, entity_type):
            fragment = _fragment_from_sentence(sentence, node[1])
            if fragment:
                eligible = generic_eligible if node[1] == generic_pred else True
                graph.add_fragment(node[0], node[1], fragment, frame_eligible=eligible)
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
    generic = _generic_verb_fragment(sentence)
    if generic is not None:
        predicate, _fragment, position, _eligible = generic
        if predicate not in seen_predicates:
            found.append((position, (_entity_type(entity_type, sentence), predicate)))
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
    generic = _generic_verb_fragment(sentence)
    if generic is not None and generic[0] == predicate:
        return generic[1]
    return ""


def _generic_fragment_eligibility(sentence: str) -> tuple[str | None, bool]:
    """``(generic_predicate, frame_eligible)`` for a sentence, so the trainer
    can tag just the auto-learned generic-verb fragment; canonical marker /
    is_a fragments are always frame-eligible."""

    generic = _generic_verb_fragment(sentence)
    if generic is None:
        return None, True
    return generic[0], generic[3]


# Copular "be" is handled separately by the is_a definition path, so it must
# never be learned as an ordinary relation phrasing here.
_NON_RELATION_VERB_LEMMAS = frozenset({"be"})


# Fine-grained POS tags whose surface form drops cleanly into the render frame
# "It {fragment}." / "They {fragment}.": 3rd-person present ("develops"), past
# ("founded"), past participle ("known"). A bare infinitive (VB "create") or
# gerund (VBG "producing") does not ("It create X" / "It producing X"), so a
# fragment led by one is learnable (the graph still stores it) but excluded
# from seeded variant selection.
_FRAME_ELIGIBLE_VERB_TAGS = frozenset({"VBZ", "VBD", "VBN"})


def _generic_verb_fragment(sentence: str) -> tuple[str, str, int, bool] | None:
    """``(predicate, fragment_template, verb_char_offset, frame_eligible)`` for
    an active-voice SVO sentence whose main verb has no ``_PHRASE_MARKERS``
    entry. ``frame_eligible`` is True only when the verb's surface form agrees
    with the "It ___" render frame (see ``_FRAME_ELIGIBLE_VERB_TAGS``).

    This is the SPEECH side of the system, not the facts side -- its job is
    to learn *how things are phrased*, so it deliberately does NOT filter to
    a curated relation vocabulary. Any content verb ("make", "have",
    "include", "win", "build", ...) is a legitimate piece of language to
    learn a phrasing for; the noise that would matter for *fact* extraction
    (is "have" a real relation?) is irrelevant here.

    Predicate keying:
    - verb in ``_VERB_RELATIONS`` -> its canonical predicate ("lead" ->
      "leader_of"), so learned fragments line up with the canonical
      predicates the overlay's facts already use.
    - otherwise -> the verb's own lemma as the predicate (surface-verb key),
      matching how schema_induction keeps surface relations. The typed
      ``(entity_type, verb)`` key is what lets a person's "have" and an
      organization's "have" occupy distinct buckets -- different senses of
      the same verb are naturally separated by subject type.

    Passive voice ("X was founded by Y") is intentionally left to the
    existing ``_PHRASE_MARKERS`` entries that already cover it explicitly.
    """
    nlp = _get_nlp()
    if nlp is None:
        return None
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ != "VERB":
            continue
        lemma = token.lemma_.lower()
        if lemma in _NON_RELATION_VERB_LEMMAS:
            continue
        has_nsubj = any(c.dep_ == "nsubj" for c in token.children)
        has_obj = any(c.dep_ in ("dobj", "prep") for c in token.children)
        if not (has_nsubj and has_obj):
            continue
        verb_config = _VERB_RELATIONS.get(lemma)
        predicate = verb_config[0] if verb_config is not None else lemma
        frame_eligible = token.tag_ in _FRAME_ELIGIBLE_VERB_TAGS
        return predicate, f"{token.text} {{object_list}}", token.idx, frame_eligible
    return None


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


def _render_fact(
    graph: PhraseGraph, fact: TypedPhraseFact, result: SynthesisAnswer, seed: str = ""
) -> str:
    if fact.predicate == _SNAPSHOT_KIND:
        return _render_snapshot(fact, result)
    fragment = graph.fragment_variant(fact.node, seed)
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


def _fragment_phrase(fragment: str, obj_list: str, reference: str) -> str:
    """Verb phrase for one fact, tense-agreed with *reference* -- no leading
    subject, no trailing period, so it can stand alone in a full sentence or
    sit beside sibling phrases in one combined sentence (see
    ``_render_capability_run``)."""

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
    return phrase


def _sentence_from_fragment(reference: str, fragment: str, obj_list: str) -> str:
    return f"{reference} {_fragment_phrase(fragment, obj_list, reference)}."


def _relative_clause_body(
    graph: PhraseGraph, facts: list[TypedPhraseFact], result: SynthesisAnswer, seed: str = ""
) -> tuple[str, str] | None:
    """Build the referent-less clause used after a learned subordinator, for
    one fact or a whole run of consecutive capability facts ("...company
    that develops X and produces Y.", not just the first of them).

    Returns ``(clause_body, trailing_sentence)`` where ``trailing_sentence`` is
    an optional standalone enrichment sentence (e.g. defining a founded object)
    that follows the woven sentence -- only ever attempted for a single
    founding/ownership fact, never a multi-fact capability run. Returns
    ``None`` for tiers that do not read cleanly inside a relative clause,
    leaving the fact(s) to render on their own.
    """

    phrases: list[str] = []
    for fact in facts:
        if fact.predicate in {_SNAPSHOT_KIND, _INFERRED_KIND, "is_a"}:
            return None
        fragment = graph.fragment_variant(fact.node, seed)
        if fragment is None:
            return None
        obj_list = _join_limited(list(fact.objects))
        if not obj_list:
            continue
        phrase = fragment.replace("{object_list}", obj_list).strip()
        if phrase.startswith(("known for ", "used for ")):
            phrase = "is " + phrase
        if phrase:
            phrases.append(phrase)
    if not phrases:
        return None

    trailing = ""
    if len(facts) == 1:
        fact = facts[0]
        phrase = phrases[0]
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
    return _join_clauses(phrases), trailing


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


def _seeded_weighted_pick(seed: str, node: str, choices: list[tuple[str, int]]) -> str:
    """Deterministic frequency-weighted choice among ``choices`` ((value,
    weight) pairs). Same seeded-sha256 primitive the speech planner
    (`_weighted_choice`) and symbolic generator (`_stable_index`) already use,
    so per-entity phrasing is stable, replayable, and spread proportionally to
    how often each phrasing was observed. Iteration order is sorted first so
    the result never depends on dict insertion order."""

    ordered = sorted(choices, key=lambda item: (-item[1], item[0]))
    total = sum(weight for _value, weight in ordered)
    if total <= 0:
        return ordered[0][0]
    value = int(sha256(f"{seed}:{node}".encode("utf-8")).hexdigest()[:12], 16)
    pick = value % total
    acc = 0
    for option, weight in ordered:
        acc += weight
        if pick < acc:
            return option
    return ordered[-1][0]


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
