"""Deterministic raw-claim extractor.

Extracts *surface* relations from sentences. The output ``relation_surface`` is
the observed trigger phrase ("requires", "migrates toward", "was founded by"),
NOT a canonical domain predicate. Generic trigger patterns are allowed; mapping
them to a fixed domain ontology is NOT.

Sentence segmentation uses the repo's deterministic ``split_sentences``. spaCy
is used only when available to refine segmentation; everything degrades
gracefully without it. No ML inference produces the relations themselves.
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


def extract_claims_from_sentence(sent: SentenceRecord) -> list[RawClaim]:
    """Extract zero or one raw claim from a sentence (first matching trigger)."""

    text = sent.text.rstrip(" .")
    found = _find_trigger(text)
    if found is None:
        return []
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
