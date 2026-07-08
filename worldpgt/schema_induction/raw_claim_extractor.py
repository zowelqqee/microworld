"""Deterministic raw-claim extractor.

Extracts *surface* relations from sentences. The output ``relation_surface`` is
the observed trigger phrase ("requires", "migrates toward", "was founded by"),
NOT a canonical domain predicate. Generic trigger patterns are allowed; mapping
them to a fixed domain ontology is NOT.

Sentence segmentation uses the repo's deterministic ``split_sentences``. spaCy
is used only when available to refine segmentation; everything degrades
gracefully without it. No ML inference produces the relations themselves.

Generic SVO fallback (2026-07-07): the ``_TRIGGERS`` list above only fires for
a fixed, enumerated set of surface phrases -- verified against a genuinely
novel domain (Fields Medal test corpus, not in this repo) that active-voice
sentences using an unlisted verb ("won", "worked at", "studied", "died") are
silently skipped entirely, even though they are simple, common English SVO
constructions. This directly contradicts the "NOT a canonical domain
predicate" design intent above: a fixed trigger list *is* effectively a small
predicate dictionary, just an informally-named one.

``_extract_generic_svo_claim`` closes that specific gap: when no trigger
matches, it uses spaCy's dependency parse to find ANY verb with a subject and
an object (direct or prepositional), and records the verb's own lemma as
``relation_surface`` -- no verb dictionary lookup, so it does not need a new
entry added by hand for every new verb a fresh domain happens to use. This is
still bounded: it only recognizes the standard SVO/SV-prep-O shape spaCy's
parser assigns, and gracefully returns nothing without spaCy installed.
"""

from __future__ import annotations

import hashlib
import re

from worldpgt.relation_extraction_v2.sentence_splitter import split_sentences
from worldpgt.schema_induction.types import (
    DocumentRecord,
    RawClaim,
    SentenceRecord,
)

# spaCy is optional -- loaded lazily on first use, matching the lazy-load
# pattern already used in entity_bootstrapper.py / relation_extraction_v2's
# spacy_extractor.py.
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

# ---------------------------------------------------------------------------
# Trigger inventory.
#
# These are GENERIC surface verbs/phrases, ordered longest/most-specific first
# so multi-word triggers win over their single-word substrings. Each entry is a
# surface form; the extractor records exactly what it matched (lightly
# normalized) as ``relation_surface``. There is deliberately NO mapping here to
# domain predicates — that grouping happens later and is derived from evidence.
# ---------------------------------------------------------------------------

# (surface, kind) — kind only controls generic priority/confidence, not domain.
_TRIGGERS: tuple[tuple[str, str], ...] = (
    # agentive / passive constructions (high confidence)
    ("was founded by", "explicit"),
    ("were founded by", "explicit"),
    ("is operated by", "explicit"),
    ("are operated by", "explicit"),
    ("was operated by", "explicit"),
    ("is run by", "explicit"),
    ("was created by", "explicit"),
    ("is valid for", "explicit"),
    ("are valid for", "explicit"),
    # directional movement (object becomes a destination later)
    ("migrates toward", "directional"),
    ("migrate toward", "directional"),
    ("migrates towards", "directional"),
    ("migrate towards", "directional"),
    ("moves toward", "directional"),
    ("move toward", "directional"),
    ("moves towards", "directional"),
    ("move towards", "directional"),
    ("migrates to", "directional"),
    ("migrate to", "directional"),
    ("moves to", "directional"),
    ("move to", "directional"),
    ("travels to", "directional"),
    ("travel to", "directional"),
    # dependency / causal
    ("depends on", "explicit"),
    ("depend on", "explicit"),
    ("feeds on", "explicit"),
    ("feed on", "explicit"),
    # modal requirement
    ("must show", "explicit"),
    ("must provide", "explicit"),
    ("must have", "explicit"),
    # single-word relations
    ("requires", "explicit"),
    ("require", "explicit"),
    ("required", "explicit"),
    ("needs", "explicit"),
    ("need", "explicit"),
    ("allows", "explicit"),
    ("allow", "explicit"),
    ("permits", "explicit"),
    ("permit", "explicit"),
    ("prohibits", "explicit"),
    ("prohibit", "explicit"),
    ("forbids", "explicit"),
    ("forbid", "explicit"),
    ("bans", "explicit"),
    ("migrates", "directional"),
    ("migrate", "directional"),
    ("moves", "directional"),
    ("move", "directional"),
    ("shifts", "explicit"),
    ("shift", "explicit"),
    # copula (lowest priority, lower confidence)
    ("is", "copula"),
    ("are", "copula"),
    ("was", "copula"),
    ("were", "copula"),
)

_KIND_CONFIDENCE = {
    "explicit": 0.9,
    "directional": 0.88,
    "copula": 0.6,
}

# Markers that terminate the object phrase and instead open a modifier clause.
# (marker_regex, modifier_key)
_MODIFIER_MARKERS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bin search of\b", re.IGNORECASE), "cause"),
    (re.compile(r"\bbecause of\b", re.IGNORECASE), "cause"),
    (re.compile(r"\bdue to\b", re.IGNORECASE), "cause"),
    (re.compile(r"\bin order to\b", re.IGNORECASE), "purpose"),
    (re.compile(r"\bunder\b", re.IGNORECASE), "condition"),
    (re.compile(r"\bduring\b", re.IGNORECASE), "time"),
)

# Standalone time adverbs.
_TIME_ADVERBS = frozenset({
    "seasonally", "annually", "yearly", "daily", "monthly", "weekly",
    "periodically", "occasionally", "regularly", "constantly",
})

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", (text or "").strip())


def _clean_phrase(text: str) -> str:
    """Trim punctuation and leading filler from a captured phrase."""

    text = _norm(text).strip(" .,;:")
    # Drop a single leading article/determiner — keeps surfaces comparable.
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.IGNORECASE)
    return text.strip()


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def segment_document(doc: DocumentRecord) -> list[SentenceRecord]:
    """Segment a document into ordered ``SentenceRecord`` items."""

    sentences = split_sentences(doc.text)
    if not sentences:
        # Fallback: treat the whole non-empty text as one sentence.
        stripped = _norm(doc.text)
        sentences = [stripped] if stripped else []
    records: list[SentenceRecord] = []
    for i, sent in enumerate(sentences):
        records.append(
            SentenceRecord(
                sentence_id=f"{doc.doc_id}:s{i}",
                doc_id=doc.doc_id,
                index=i,
                text=_norm(sent),
            )
        )
    return records


def _extract_modifiers(tail: str) -> tuple[str, dict[str, str]]:
    """Pull modifier clauses out of an object tail.

    Returns ``(remaining_object, modifiers)``. The earliest modifier marker
    splits the object from the modifier clause; remaining standalone time
    adverbs are also lifted out.
    """

    modifiers: dict[str, str] = {}
    work = tail

    # Find the earliest modifier marker and split there.
    earliest = None
    for pattern, key in _MODIFIER_MARKERS:
        m = pattern.search(work)
        if m and (earliest is None or m.start() < earliest[0]):
            earliest = (m.start(), m.end(), key)
    if earliest is not None:
        start, end, key = earliest
        head = work[:start]
        clause = _clean_phrase(work[end:])
        if clause:
            modifiers.setdefault(key, clause)
        work = head

    # Lift standalone time adverbs out of whatever object remains.
    tokens = [t for t in _norm(work).split(" ") if t]
    kept: list[str] = []
    for tok in tokens:
        if tok.lower().strip(",.;") in _TIME_ADVERBS:
            modifiers.setdefault("time", tok.lower().strip(",.;"))
        else:
            kept.append(tok)
    remaining = _clean_phrase(" ".join(kept))
    return remaining, modifiers


def _find_trigger(sentence: str) -> tuple[str, str, int, int] | None:
    """Return ``(surface, kind, start, end)`` for the best trigger, or None.

    Triggers are tried in priority order (most specific first). The first one
    that occurs with a non-empty subject before it wins.
    """

    low = sentence.lower()
    for surface, kind in _TRIGGERS:
        pattern = re.compile(r"\b" + re.escape(surface) + r"\b", re.IGNORECASE)
        m = pattern.search(low)
        if not m:
            continue
        subject = _clean_phrase(sentence[: m.start()])
        if not subject:
            continue
        return surface, kind, m.start(), m.end()
    return None


def _svo_span_text(token) -> str:
    """Clean surface text for the noun phrase headed by ``token``.

    Same convention as relation_extraction_v2/spacy_extractor.py's
    ``_span_text``: collects ``token.left_edge`` through ``token`` inclusive,
    dropping bare lowercase determiners so "the Fields Medal" doesn't lose
    its capitalized "The" (kept only when part of a proper-noun compound).
    """
    parts: list[str] = []
    doc = token.doc
    for t in doc[token.left_edge.i : token.i + 1]:
        if t.pos_ == "DET":
            if t.text.lower() == "the":
                next_i = t.i + 1
                if next_i <= token.i and doc[next_i].pos_ == "PROPN":
                    parts.append(t.text)
        else:
            parts.append(t.text)
    return " ".join(parts).strip()


def _extract_generic_svo_claim(sent: SentenceRecord) -> RawClaim | None:
    """Fallback when no fixed trigger matches: any verb with a subject and a
    direct or prepositional object, using the verb's own lemma as
    ``relation_surface``. See module docstring for why this exists."""
    nlp = _get_nlp()
    if nlp is None:
        return None
    text = sent.text.rstrip(" .")
    if not text:
        return None
    doc = nlp(text)
    for token in doc:
        if token.pos_ != "VERB":
            continue
        nsubj = next((c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")), None)
        if nsubj is None:
            continue
        dobj = next((c for c in token.children if c.dep_ == "dobj"), None)
        obj_span: str | None = None
        if dobj is not None:
            obj_span = _svo_span_text(dobj)
        else:
            prep = next((c for c in token.children if c.dep_ == "prep"), None)
            if prep is not None:
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if pobj is not None:
                    obj_span = f"{prep.text} {_svo_span_text(pobj)}"
        if not obj_span:
            continue
        subject = _clean_phrase(_svo_span_text(nsubj))
        obj = _clean_phrase(obj_span)
        if not subject or not obj:
            continue
        relation_surface = token.lemma_.lower()
        claim_id = _stable_id("claim", sent.sentence_id, subject, relation_surface, obj)
        return RawClaim(
            claim_id=claim_id,
            subject=subject,
            relation_surface=relation_surface,
            object=obj,
            sentence=sent.text,
            source_doc_id=sent.doc_id,
            source_sentence_id=sent.sentence_id,
            modifiers={},
            extraction_method="spacy_svo",
            confidence=0.5,
        )
    return None


def extract_claims_from_sentence(sent: SentenceRecord) -> list[RawClaim]:
    """Extract zero or one raw claim from a sentence (first matching trigger,
    or the generic spaCy SVO fallback when no trigger matches)."""

    text = sent.text.rstrip(" .")
    found = _find_trigger(text)
    if found is None:
        fallback = _extract_generic_svo_claim(sent)
        return [fallback] if fallback is not None else []
    surface, kind, start, end = found

    subject = _clean_phrase(text[:start])
    tail = text[end:]
    obj, modifiers = _extract_modifiers(tail)

    # Directional triggers expose their object as a destination modifier too,
    # so QA can answer "where/куда" questions without a domain predicate.
    if kind == "directional" and obj:
        modifiers.setdefault("destination", obj)

    relation_surface = _norm(surface).lower()
    if not subject:
        return []

    claim_id = _stable_id(
        "claim", sent.sentence_id, subject, relation_surface, obj or ""
    )
    return [
        RawClaim(
            claim_id=claim_id,
            subject=subject,
            relation_surface=relation_surface,
            object=obj or None,
            sentence=sent.text,
            source_doc_id=sent.doc_id,
            source_sentence_id=sent.sentence_id,
            modifiers=modifiers,
            extraction_method="regex_trigger",
            confidence=_KIND_CONFIDENCE.get(kind, 0.6),
        )
    ]


def extract_claims(
    docs: list[DocumentRecord],
) -> tuple[list[SentenceRecord], list[RawClaim]]:
    """Segment ``docs`` and extract raw claims from every sentence."""

    all_sentences: list[SentenceRecord] = []
    all_claims: list[RawClaim] = []
    for doc in docs:
        sentences = segment_document(doc)
        all_sentences.extend(sentences)
        for sent in sentences:
            all_claims.extend(extract_claims_from_sentence(sent))
    return all_sentences, all_claims
