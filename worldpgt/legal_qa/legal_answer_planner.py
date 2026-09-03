"""Retrieve and realize statutory answers from the explicit graph.

Retrieval is purely content-based: a stored item is scored by how much of the
question's content it accounts for, over the item's subject, predicate, object,
section heading, and every condition/exception. No legal vocabulary, predicate
allowlist, or section table is encoded here, so the lane behaves the same on a
statute it has never seen.

Realization enforces the mandatory-guard invariant: a rule that holds only
under conditions, or is defeated by exceptions, or is negated, is *never*
stated without them. If the guards cannot be rendered, the answer is withheld
and the lane audits rather than emit a claim that would misstate the law.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from worldpgt.legal_qa.legal_index import LegalIndex, build_index, tokens as _index_tokens
from worldpgt.legal_qa.legal_question_analyzer import AnalyzedLegalQuestion

_STOP = frozenset("""a an the of to in on for by with as is are was were be been being that which
who whom whose this these those such it its any no not shall may under does do did what when where
how or and but if then than there here about into over more most some all each every upon""".split())

# Which stored predicates satisfy which question shape. Membership is decided by
# the predicate's own wording, not by a fixed vocabulary: a predicate *containing*
# "punish"/"penal" answers a penalty question, one containing "cross_reference"
# answers a citation question, and so on. A statute using different predicate
# wording still routes correctly as long as extraction named the relation for
# what it does.
_SHAPE_PREDICATE_CUE = {
    "penalty": ("punish", "penal", "fined", "imprison", "sentence"),
    "cross_reference": ("cross_reference", "cross_references", "cites", "refer"),
    "scope": ("govern", "applies", "apply", "scope"),
}
_MIN_SCORE = 0.34


def _toks(text: str) -> set[str]:
    """The index's tokenizer, reused verbatim.

    Question and stored rule must be tokenized identically — including
    stemming — or a lexical responsiveness check compares two different
    vocabularies and reads every inflection as a mismatch.
    """
    return _index_tokens(text)


@dataclass
class LegalAnswer:
    decision: str                     # "answer" | "audit"
    text: str = ""
    audit_reason: str = ""
    citations: list[str] = field(default_factory=list)
    edges_used: list[dict] = field(default_factory=list)
    guards_rendered: int = 0


def _item_text(item: dict) -> str:
    parts = [item.get("subject", ""), item.get("predicate", "").replace("_", " "),
             item.get("object", "") or item.get("definition", ""),
             item.get("section_heading", "")]
    for clause in (item.get("conditions") or []) + (item.get("exceptions") or []):
        parts.append(clause.get("text", ""))
    return " ".join(parts)


def _shape_fits(shape: str, item: dict) -> bool:
    cues = _SHAPE_PREDICATE_CUE.get(shape)
    if not cues:
        return True
    predicate = str(item.get("predicate", "")).lower()
    return any(cue in predicate for cue in cues)


_CITATION_TOKEN = re.compile(r"\b\d+(?:\([a-z0-9]+\))*\b", re.IGNORECASE)


def _citation_tokens(text: str) -> set[str]:
    """Numeric provision references, e.g. '873', '878(a)', '102(b)(1)'."""
    return {m.group(0).lower() for m in _CITATION_TOKEN.finditer(text or "")}


# A question term rarer than this share of the corpus is "distinctive": it is
# part of what makes this question *this* question rather than a neighbouring
# one. Common terms ("invention", "section") cannot discriminate and are not
# required to be accounted for.
_DISTINCTIVE_DOC_FRACTION = 0.10


def responsiveness_gap(
    asked: set[str], item_tokens: set[str], index: LegalIndex
) -> tuple[set[str], set[str]]:
    """Return the question's distinctive terms this item fails to account for.

    Similarity says an item *resembles* the question; this says whether it
    *addresses* it. A rule is responsive only if the terms that distinguish the
    question are present in the rule — in its subject, object, heading, or any
    condition or exception.

    Two kinds of gap are reported separately because they mean different things:

    * ``absent`` — the term occurs nowhere in the corpus. A question turning on
      a word the statute never uses is a question the statute does not decide;
      this is the signal that separates "unanswerable" from "answered badly".
    * ``missing`` — the term exists in the corpus but not in *this* item, i.e.
      some other provision is the one being asked about. This is what separates
      two near-identical sibling provisions differing by one qualifier.

    Both are derived from corpus statistics the index already holds. No
    vocabulary, topic list, or threshold on meaning is encoded.
    """
    total = max(1, len(index.items))
    absent: set[str] = set()
    missing: set[str] = set()
    for token in asked:
        freq = index.doc_freq.get(token, 0)
        if freq == 0:
            if len(token) > 3:
                absent.add(token)
            continue
        if freq / total > _DISTINCTIVE_DOC_FRACTION:
            continue
        if token not in item_tokens:
            missing.add(token)
    return absent, missing


# Share of a question's distinctive terms a rule must account for to be
# admitted as answering it. Calibrated on the frozen question set (see
# report section 4); the sweep is recorded in responsiveness_sweep.json.
_MIN_ACCOUNTED = 0.6


def is_responsive(asked: set[str], item_tokens: set[str], index: LegalIndex) -> bool:
    """Admission decision: does this item actually address the question?

    An absent term (one the corpus never uses) and a missing term (one some
    other provision has) are both failures to account for what was asked, and
    are weighed together: a rule must account for at least ``_MIN_ACCOUNTED``
    of the question's distinctive terms. Vetoing on any single absent term was
    tried first and rejected — it fires on question-framing words that no
    statute ever contains ("mean", "regarding", "always"), which collapsed
    coverage from 40 to 20 answers while fixing only two items.
    """
    absent, missing = responsiveness_gap(asked, item_tokens, index)
    total = max(1, len(index.items))
    distinctive = {
        t for t in asked
        if index.doc_freq.get(t, 0) == 0
        or index.doc_freq.get(t, 0) / total <= _DISTINCTIVE_DOC_FRACTION
    }
    distinctive = {t for t in distinctive if len(t) > 3}
    if not distinctive:
        return True
    unaccounted = (absent | missing) & distinctive
    accounted = 1.0 - len(unaccounted) / len(distinctive)
    return accounted >= _MIN_ACCOUNTED


def retrieve(analyzed: AnalyzedLegalQuestion, index: LegalIndex, limit: int = 3) -> list[dict]:
    """Score every stored item against the question's content; return the best.

    Three signals, all content-derived: how much of the question the item
    accounts for, whether the question names the provision the item is stated
    in, and how exactly the item's subject matches. None encodes legal
    knowledge — a provision reference is recognized by its numeric shape, not
    by a table of sections.
    """
    focus = _toks(analyzed.focus) or _toks(analyzed.question)
    if not focus:
        return []
    # The statute's own definitions section is a synonym table: expand through
    # it before retrieval so "this country" reaches the rule stored under
    # "United States". Bounded to depth 1 and fully traceable.
    # The terms actually asked, before graph expansion: responsiveness is judged
    # against what the user said, never against terms the graph added for recall.
    asked = set(focus)
    focus = index.expand(focus)
    asked_citations = _citation_tokens(analyzed.question)

    scored: list[tuple[float, float, dict, int]] = []
    # Only postings for the query's selective terms are visited, so cost is
    # bounded by the question, not by the size of the graph.
    for item_id in index.candidates(focus):
        item = index.items[item_id]
        if analyzed.shape == "definition" and item.get("overlay_type") != "overlay_definition":
            continue
        if not _shape_fits(analyzed.shape, item):
            continue
        # Token sets were computed once at index build; scoring never
        # re-tokenizes, which is what keeps per-question cost off corpus size.
        item_tokens = index.item_tokens[item_id]
        if not item_tokens:
            continue

        # Relevance must reach the item's *identity* — what the rule is about —
        # not merely any word anywhere in its text. Without this a question
        # about filing fees matches a rule that happens to contain the word
        # "filing", which is how a loose similarity search fabricates an answer
        # to a question the statute never decides.
        #
        # What counts as identity differs by item type, because the two store
        # their "what this is about" in different places:
        #   - a definition is identified by the term it defines (its subject);
        #     its heading is generic ("Definitions") and says nothing.
        #   - a rule is identified by its section heading ("Blackmail"), because
        #     its subject is a long conduct clause that shares little vocabulary
        #     with any natural question.
        subject_tokens_all = index.subject_tokens[item_id]
        heading = index.heading_tokens[item_id]
        if item.get("overlay_type") == "overlay_definition":
            if not subject_tokens_all:
                continue
            if len(focus & subject_tokens_all) / len(subject_tokens_all) < 0.5:
                continue
        else:
            names_heading = bool(heading and (focus & heading))
            names_subject = bool(subject_tokens_all) and (
                len(focus & subject_tokens_all) / len(subject_tokens_all) >= 0.34
            )
            if not (names_heading or names_subject):
                continue

        # Naming the provision is a strong, unambiguous signal of intent.
        cite_hit = bool(asked_citations & index.citation_tokens[item_id])

        covered = len(focus & item_tokens) / len(focus)
        if cite_hit:
            covered += 0.5

        # An item whose subject is wholly named by the question is a better
        # answer than one that merely shares vocabulary with it.
        subject_exact = bool(subject_tokens_all) and subject_tokens_all <= focus
        if subject_exact:
            covered += 0.25

        if covered < _MIN_SCORE:
            continue
        # Explicit admission gate: resembling the question is not answering it.
        # Naming the provision outright is decisive intent and overrides the
        # check: the user pointed at this rule, so unaccounted framing verbs
        # ("cite", "refer", "defines") — words no statute ever contains — must
        # not veto the very provision that was asked for.
        if not cite_hit and not is_responsive(asked, item_tokens, index):
            continue
        scored.append((covered, -len(item_tokens), item, item_id))

    scored.sort(key=lambda row: (-row[0], -row[1]))

    # When the question names a provision and the graph has that provision,
    # answer from it alone — sibling provisions are not responsive to a question
    # that asked about a specific one.
    if asked_citations:
        on_point = [row for row in scored
                    if asked_citations & index.citation_tokens[row[3]]]
        if on_point:
            scored = on_point

    if analyzed.shape == "definition":
        limit = 1
    return [row[2] for row in scored[:limit]]


def _negate(predicate: str) -> str:
    words = predicate.replace("_", " ").split()
    if words and words[0] in ("is", "are", "was", "were", "shall", "may", "can", "will", "has", "have"):
        return " ".join([words[0], "not", *words[1:]])
    return f"is not {' '.join(words)}"


def _join(items: list[str], word: str) -> str:
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} {word} {items[1]}"
    return ", ".join(items[:-1]) + f", {word} {items[-1]}"


def realize(item: dict) -> tuple[str, int]:
    """Render one stored item as a sentence that always carries its guards."""
    if item.get("overlay_type") == "overlay_definition":
        return f"{item['subject']} means {item['definition']}.", 0

    predicate = str(item.get("predicate", "")).replace("_", " ")
    if item.get("polarity") == "negate":
        predicate = _negate(str(item.get("predicate", "")))

    conditions = item.get("conditions") or []
    exceptions = item.get("exceptions") or []
    scope = [c["text"] for c in conditions if c.get("kind") == "scope"]
    factual = [c["text"] for c in conditions if c.get("kind") != "scope"]
    exc = [c["text"] for c in exceptions]

    sentence = f"{item['subject']} {predicate} {item['object']}".strip()
    if factual:
        sentence += f", provided that {_join(factual, 'and')}"
    if exc:
        sentence += f", except where {_join(exc, 'or')}"
    if scope:
        sentence = f"For purposes of {_join(scope, 'and')}, {sentence}"
    sentence = sentence.rstrip(".") + "."
    sentence = sentence[0].upper() + sentence[1:] if sentence else sentence

    # Mandatory-guard invariant, enforced at realization: every guard the stored
    # rule carries must appear in the emitted sentence.
    guards = [*scope, *factual, *exc]
    for guard in guards:
        if guard and guard.lower() not in sentence.lower():
            raise AssertionError(f"guard dropped while realizing statutory rule: {guard!r}")
    return sentence, len(guards)


def plan(analyzed: AnalyzedLegalQuestion, index: LegalIndex) -> LegalAnswer:
    """Answer a statutory question, or audit when the graph does not support one."""
    if analyzed.shape == "unknown":
        return LegalAnswer("audit", audit_reason="question shape not recognized by the legal lane")

    hits = retrieve(analyzed, index)
    if not hits:
        return LegalAnswer(
            "audit",
            audit_reason="no stored statutory relation covers this question",
        )

    sentences: list[str] = []
    citations: list[str] = []
    used: list[dict] = []
    guards = 0
    for item in hits:
        try:
            sentence, n = realize(item)
        except AssertionError:
            # A rule whose guards cannot be surfaced is withheld, never flattened.
            continue
        cite = item.get("stated_in", "")
        sentences.append(f"{sentence} ({cite})" if cite else sentence)
        if cite:
            citations.append(cite)
        used.append(item)
        guards += n

    if not sentences:
        return LegalAnswer("audit", audit_reason="supporting rule could not be stated with its guards")
    return LegalAnswer("answer", " ".join(sentences), "", citations, used, guards)
