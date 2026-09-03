"""Scale-correct retrieval index for the legal QA lane.

The first version of this lane scanned every stored item on every question,
which is fine at a hundred relations and dead at a million. This module gives
the lane the same shape of index the evidence graph already uses
(``prepare_evidence_graph``): an inverted token index over compact ``array('I')``
postings, built once at load time.

Two properties matter at statutory scale specifically:

* **Rare tokens drive candidate generation.** In a legal corpus, words like
  "section", "invention", "patent", "united", "states" occur in a large share of
  provisions. Letting them generate candidates costs a near-full scan *and*
  drowns precision. Candidate generation therefore walks postings in ascending
  document frequency and stops once a budget is spent, so cost is bounded by the
  query's most selective terms rather than by graph size.
* **Definitional synonymy comes from the graph.** A statute's definitions
  section is a synonym dictionary the corpus ships with itself ("this country"
  means "United States"). Expansion follows those stored definition edges only,
  at depth 1 with a cap, so it stays bounded and fully inspectable.

Deterministic. No ML, no network, no overlay writes.
"""

from __future__ import annotations

import re
from array import array
from dataclasses import dataclass, field

_STOP = frozenset("""a an the of to in on for by with as is are was were be been being that which
who whom whose this these those such it its any no not shall may under does do did what when where
how or and but if then than there here about into over more most some all each every upon""".split())

# Candidate generation stops after visiting this many postings. It bounds work
# per question independently of graph size; the budget is spent on the query's
# rarest — most selective — terms first, so the terms that actually identify a
# provision are always the ones consulted.
_POSTING_BUDGET = 4000

# A token appearing in more than this share of items identifies nothing in a
# legal corpus and is never used to generate candidates (it may still score).
_MAX_DOC_FRACTION = 0.25

# Bounded, depth-1 expansion through the graph's own definition edges.
_MAX_EXPANSION_TERMS = 8


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9()]+", (text or "").lower())
            if t not in _STOP and len(t) > 2}


def _compact(values: list[int]) -> array:
    return array("I", values)


@dataclass
class LegalIndex:
    """Inverted index over stored statutory items.

    Per-item token sets are computed once here rather than on every question.
    Re-tokenizing each candidate per query is what makes an otherwise indexed
    retriever grow with corpus size: candidate *generation* is bounded, but
    candidate *scoring* was not.
    """

    items: list[dict]
    token_index: dict[str, array] = field(default_factory=dict)
    doc_freq: dict[str, int] = field(default_factory=dict)
    # Precomputed, parallel to ``items``.
    item_tokens: list[set[str]] = field(default_factory=list)
    subject_tokens: list[set[str]] = field(default_factory=list)
    heading_tokens: list[set[str]] = field(default_factory=list)
    citation_tokens: list[set[str]] = field(default_factory=list)
    # The statute's own synonym table, indexed so expansion is a lookup rather
    # than a scan: token -> ids of definitions whose term contains that token.
    definition_terms: list[tuple[set[str], set[str]]] = field(default_factory=list)
    definition_index: dict[str, array] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.items)

    # -- candidate generation ------------------------------------------------

    def candidates(self, focus: set[str]) -> set[int]:
        """Return item ids worth scoring for ``focus``, within a fixed budget."""
        if not focus:
            return set()
        total = max(1, len(self.items))
        usable = [
            t for t in focus
            if t in self.token_index and self.doc_freq.get(t, 0) / total <= _MAX_DOC_FRACTION
        ]
        # Rarest first: the selective terms are the ones that identify a
        # provision, and spending the budget on them keeps cost off graph size.
        usable.sort(key=lambda t: self.doc_freq.get(t, 0))

        out: set[int] = set()
        spent = 0
        for token in usable:
            room = _POSTING_BUDGET - spent
            if room <= 0:
                break
            posting = self.token_index[token]
            # A posting longer than the remaining budget is *truncated*, never
            # taken whole. Without this the budget silently fails to bind on the
            # first term — one huge posting is ingested entirely and per-question
            # cost goes back to linear in corpus size. The cost of truncation is
            # bounded recall on terms that are not selective anyway; the benefit
            # is a hard latency ceiling independent of how large the graph gets.
            out.update(posting if len(posting) <= room else posting[:room])
            spent += min(len(posting), room)
        if not out:
            # Every query term was too common to be selective. Fall back to a
            # bounded slice of the rarest term rather than scanning everything.
            common = sorted((t for t in focus if t in self.token_index),
                            key=lambda t: self.doc_freq.get(t, 0))
            if common:
                out.update(self.token_index[common[0]][:_POSTING_BUDGET])
        return out

    # -- graph-native expansion ---------------------------------------------

    def expand(self, focus: set[str]) -> set[str]:
        """Add tokens the statute itself declares equivalent to the question's.

        Depth 1, capped: only terms the graph defines, and only their definition
        tokens. Fully inspectable — every added token is traceable to a stored
        definition edge.

        Only definitions sharing a token with the question are examined, via the
        definition token index. Scanning every stored definition per question is
        linear in corpus size and was, measurably, the dominant cost once the
        posting budget was made to bind.
        """
        added: set[str] = set()
        seen: set[int] = set()
        for token in focus:
            for def_id in self.definition_index.get(token, ()):  # bounded fan-out
                if def_id in seen:
                    continue
                seen.add(def_id)
                term_tokens, definition_tokens = self.definition_terms[def_id]
                if not term_tokens or not term_tokens <= focus:
                    continue
                added |= definition_tokens
                if len(added) >= _MAX_EXPANSION_TERMS * 4:
                    return focus | set(list(added)[: _MAX_EXPANSION_TERMS * 4])
        return focus | set(list(added)[: _MAX_EXPANSION_TERMS * 4])


_CITATION_RE = re.compile(r"\b\d+(?:\([a-z0-9]+\))*\b", re.IGNORECASE)


def _citation_tokens(text: str) -> set[str]:
    """Numeric provision references, e.g. '873', '878(a)'."""
    return {m.group(0).lower() for m in _CITATION_RE.finditer(text or "")}


def _indexable_text(item: dict) -> str:
    parts = [
        item.get("subject", ""),
        item.get("object", "") or item.get("definition", ""),
        item.get("section_heading", ""),
    ]
    for clause in (item.get("conditions") or []) + (item.get("exceptions") or []):
        parts.append(clause.get("text", ""))
    return " ".join(parts)


def build_index(items: list[dict]) -> LegalIndex:
    """Build the inverted index once, at overlay load time."""
    kept = [i for i in items
            if i.get("overlay_type") in ("overlay_relation", "overlay_definition")]
    token_lists: dict[str, list[int]] = {}
    definition_terms: list[tuple[set[str], set[str]]] = []
    definition_lists: dict[str, list[int]] = {}
    for item_id, item in enumerate(kept):
        for token in tokens(_indexable_text(item)):
            token_lists.setdefault(token, []).append(item_id)
        if item.get("overlay_type") == "overlay_definition":
            term_tokens = tokens(str(item.get("subject", "")))
            if not term_tokens:
                continue
            def_id = len(definition_terms)
            definition_terms.append((term_tokens, tokens(item.get("definition", ""))))
            for token in term_tokens:
                definition_lists.setdefault(token, []).append(def_id)
    return LegalIndex(
        items=kept,
        token_index={t: _compact(v) for t, v in token_lists.items()},
        doc_freq={t: len(v) for t, v in token_lists.items()},
        definition_terms=definition_terms,
        definition_index={t: _compact(v) for t, v in definition_lists.items()},
        item_tokens=[tokens(_indexable_text(i)) for i in kept],
        subject_tokens=[tokens(i.get("subject", "")) for i in kept],
        heading_tokens=[tokens(i.get("section_heading", "")) for i in kept],
        citation_tokens=[_citation_tokens(i.get("stated_in", "")) for i in kept],
    )
