"""Universal entity bootstrap — find named entities in unfamiliar text.

The existing ``EntitySurfaceIndex`` requires a *prior* list of known entities
(built from accepted/promoted overlays). For a brand-new topic (visas, medicine,
law) that list is empty, so a cold start is impossible.

This module solves the cold start with spaCy NER + deterministic term mining:

    Pass 1a  spaCy NER            -> typed named entities (PERSON/ORG/GPE/...)
    Pass 1b  noun-chunk mining    -> repeated domain terms NER misses
                                     ("O-1A visa", "extraordinary ability")
    Pass 1c  alias clustering     -> merge acronyms + containment variants
                                     ("USCIS" == "U.S. Citizenship and ...")

Output: a list of :class:`BootstrappedEntity` with a canonical label, aliases,
and a canonical entity_type derived from the spaCy label (no prior list needed).

spaCy is loaded lazily and is OPTIONAL: without it, a deterministic
capitalized-span + repeated-noun-phrase fallback is used. No ML training, no
network, no LLM.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from worldpgt.knowledge.entity_types import canonicalize_entity_type

# spaCy is optional — loaded lazily on first use.
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


# spaCy NER label -> canonical entity type (generic, not domain specific).
_SPACY_LABEL_TO_TYPE: dict[str, str] = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "place",
    "LOC": "place",
    "FAC": "place",
    "PRODUCT": "product",
    "WORK_OF_ART": "publication",
    "LAW": "concept",
    "LANGUAGE": "concept",
    "NORP": "concept",
    "EVENT": "other",
}

# Head-noun hints for domain terms that spaCy types weakly. Generic English
# nouns, not a domain ontology — used only to give a term a coarse type.
_HEAD_NOUN_TYPE: dict[str, str] = {
    "visa": "program",
    "permit": "program",
    "program": "program",
    "service": "service",
    "agency": "organization",
    "department": "organization",
    "office": "organization",
    "act": "concept",
    "law": "concept",
    "form": "product",
    "petition": "concept",
    "system": "technology",
    "platform": "technology",
    "software": "technology",
    "vaccine": "product",
    "drug": "product",
    "disease": "concept",
    "species": "concept",
    "treaty": "concept",
}

_WS = re.compile(r"\s+")

# Generic stopwords that should never be a standalone entity.
_STOP = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "it", "they", "he",
    "she", "we", "you", "i", "who", "which", "what", "their", "its", "his",
    "her", "individual", "individuals", "person", "people", "someone",
    "something", "anything", "everyone", "year", "years", "time", "day",
    "example", "examples", "etc", "such",
})

_MIN_LEN = 3
_MIN_OCCURRENCES = 1


@dataclass(frozen=True)
class BootstrappedEntity:
    """A named entity discovered from raw text without a prior list."""

    entity_id: str
    canonical_label: str
    aliases: tuple[str, ...]
    entity_type: str
    spacy_labels: tuple[str, ...]
    occurrences: int
    source_doc_ids: tuple[str, ...]
    context_terms: tuple[str, ...] = field(default_factory=tuple)


def _norm(text: str) -> str:
    return _WS.sub(" ", (text or "").strip())


def _slug(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return s or "entity"


def _head_noun(label: str) -> str | None:
    tokens = [t.strip(",.;:'\"()") for t in _norm(label).split(" ") if t.strip()]
    tokens = [t for t in tokens if t.lower() not in _STOP]
    if not tokens:
        return None
    return tokens[-1].lower()


def _acronym(label: str) -> str:
    """Return the uppercase acronym of a multiword label (skipping stopwords).

    Tokens that are already dotted/all-caps abbreviations contribute ALL their
    letters, so "U.S. Citizenship and Immigration Services" -> "USCIS".
    """
    letters = []
    for tok in _norm(label).split(" "):
        clean = re.sub(r"[^A-Za-z]", "", tok)
        if not clean:
            continue
        if clean.lower() in {"and", "of", "the", "for", "to", "a", "an"}:
            continue
        # Dotted abbreviation (e.g. "U.S.") or already all-caps -> take all.
        if "." in tok or (clean.isupper() and len(clean) <= 4):
            letters.append(clean.upper())
        else:
            letters.append(clean[0].upper())
    return "".join(letters)


def _is_code_like(surface: str) -> bool:
    """True for short identifier tokens like 'O-1A', 'I-129', 'COVID-19'."""
    s = _norm(surface)
    return bool(re.search(r"[A-Za-z]\-?\d", s)) and len(s.split(" ")) == 1


def _is_junk_surface(surface: str) -> bool:
    s = _norm(surface)
    low = s.lower()
    if len(s) < _MIN_LEN:
        return False if s.isupper() and len(s) >= 2 else True
    if low in _STOP:
        return True
    # Pure number / date-ish.
    if re.fullmatch(r"[\d.,%$\-]+", s):
        return True
    # Must contain a letter.
    if not re.search(r"[A-Za-z]", s):
        return True
    return False


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    surface: str
    labels: Counter = field(default_factory=Counter)
    count: int = 0
    docs: set = field(default_factory=set)


def _extract_candidates_spacy(
    nlp, texts: list[tuple[str, str]]
) -> dict[str, _Candidate]:
    """Return normalized-surface -> _Candidate using spaCy NER + noun chunks."""
    cands: dict[str, _Candidate] = {}

    def _add(surface: str, label: str | None, doc_id: str) -> None:
        surface = _norm(surface).strip("’'\".,;:()")
        if _is_junk_surface(surface):
            return
        key = surface.lower()
        c = cands.get(key)
        if c is None:
            c = _Candidate(surface=surface)
            cands[key] = c
        c.count += 1
        c.docs.add(doc_id)
        if label:
            c.labels[label] += 1
        # Keep the most "title-like" surface form.
        if surface[:1].isupper() and not c.surface[:1].isupper():
            c.surface = surface

    for doc_id, text in texts:
        doc = nlp(text)
        for ent in doc.ents:
            _add(ent.text, ent.label_, doc_id)
        # Mine noun chunks for domain terms NER misses. Strip leading
        # determiners; keep multiword or capitalized single-word chunks.
        for chunk in doc.noun_chunks:
            phrase = chunk.text
            phrase = re.sub(r"^(the|a|an|this|that|these|those|its|their|his|her)\s+",
                            "", phrase, flags=re.IGNORECASE)
            phrase = _norm(phrase)
            if not phrase:
                continue
            words = phrase.split(" ")
            is_multiword = len(words) >= 2
            is_capitalized = phrase[:1].isupper()
            looks_codey = bool(re.search(r"[A-Z]\-?\d|\d", phrase)) and len(phrase) <= 30
            if is_multiword or is_capitalized or looks_codey:
                _add(phrase, None, doc_id)
    return cands


def _extract_candidates_fallback(
    texts: list[tuple[str, str]]
) -> dict[str, _Candidate]:
    """Deterministic fallback: capitalized spans + repeated noun-ish phrases."""
    cands: dict[str, _Candidate] = {}
    # Capitalized multi-word spans, acronyms, and code-like tokens.
    span_re = re.compile(
        r"\b([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*)*"
        r"|[A-Z]{2,}|[A-Z]\-?\d[A-Za-z0-9\-]*)\b"
    )
    for doc_id, text in texts:
        for m in span_re.finditer(text):
            surface = _norm(m.group(1)).strip(".,;:")
            if _is_junk_surface(surface):
                continue
            key = surface.lower()
            c = cands.get(key)
            if c is None:
                c = _Candidate(surface=surface)
                cands[key] = c
            c.count += 1
            c.docs.add(doc_id)
    return cands


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _cluster_candidates(cands: dict[str, _Candidate]) -> list[list[_Candidate]]:
    """Group candidates that are aliases of one another.

    Two candidates merge when:
      * one is the uppercase acronym of the other, OR
      * one is contained in the other AND they share a head noun.
    Deterministic: process by descending (count, length).
    """
    items = sorted(
        cands.values(), key=lambda c: (-c.count, -len(c.surface), c.surface.lower())
    )
    clusters: list[list[_Candidate]] = []
    assigned: set[str] = set()

    for c in items:
        if c.surface.lower() in assigned:
            continue
        cluster = [c]
        assigned.add(c.surface.lower())
        c_head = _head_noun(c.surface)
        c_acr = _acronym(c.surface)
        for other in items:
            ol = other.surface.lower()
            if ol in assigned:
                continue
            merged = False
            # Acronym match (either direction).
            if len(other.surface) <= 6 and other.surface.isupper():
                if other.surface.upper() == c_acr and c_acr:
                    merged = True
            elif c.surface.isupper() and len(c.surface) <= 6:
                if c.surface.upper() == _acronym(other.surface) and _acronym(other.surface):
                    merged = True
            # Code-like containment: "O-1A" <-> "O-1A visa" (head nouns differ
            # but the code token uniquely identifies the longer term). The
            # longer phrase must START with the code token (after an optional
            # determiner) so "every O-1A petition" is NOT merged.
            if not merged:
                a, b = c.surface.lower(), ol
                if a in b or b in a:
                    shorter = c.surface if len(c.surface) <= len(other.surface) else other.surface
                    longer = other.surface if shorter == c.surface else c.surface
                    if _is_code_like(shorter):
                        longer_stripped = re.sub(
                            r"^(the|a|an|every|each|this|that)\s+", "",
                            longer.lower(),
                        )
                        if longer_stripped.startswith(shorter.lower() + " ") or \
                           longer_stripped == shorter.lower():
                            merged = True
            # Containment + shared head noun.
            if not merged:
                o_head = _head_noun(other.surface)
                if c_head and o_head and c_head == o_head:
                    a, b = c.surface.lower(), ol
                    if a in b or b in a:
                        merged = True
            if merged:
                cluster.append(other)
                assigned.add(ol)
        clusters.append(cluster)
    return clusters


def _pick_type(labels: Counter, canonical_label: str) -> tuple[str, tuple[str, ...]]:
    """Pick a canonical entity type from spaCy labels / head-noun hints."""
    spacy_labels = tuple(sorted(labels))
    # Most common spaCy label first.
    for label, _cnt in labels.most_common():
        mapped = _SPACY_LABEL_TO_TYPE.get(label)
        if mapped:
            etype = canonicalize_entity_type(mapped) or "other"
            if etype != "other":
                return etype, spacy_labels
    # Head-noun hint.
    head = _head_noun(canonical_label)
    if head and head in _HEAD_NOUN_TYPE:
        etype = canonicalize_entity_type(_HEAD_NOUN_TYPE[head]) or "concept"
        return etype, spacy_labels
    # Default: a domain concept.
    return "concept", spacy_labels


def bootstrap_entities(
    texts: list[str] | list[tuple[str, str]],
    *,
    min_occurrences: int = _MIN_OCCURRENCES,
) -> list[BootstrappedEntity]:
    """Find named entities in raw text with no prior entity list.

    ``texts`` may be a list of strings or a list of ``(doc_id, text)`` pairs.
    """
    pairs: list[tuple[str, str]] = []
    for i, item in enumerate(texts):
        if isinstance(item, tuple):
            pairs.append((str(item[0]), str(item[1])))
        else:
            pairs.append((f"doc{i}", str(item)))

    nlp = _get_nlp()
    if nlp is not None:
        cands = _extract_candidates_spacy(nlp, pairs)
    else:  # pragma: no cover - exercised only without spaCy model
        cands = _extract_candidates_fallback(pairs)

    clusters = _cluster_candidates(cands)

    entities: list[BootstrappedEntity] = []
    for cluster in clusters:
        total = sum(c.count for c in cluster)
        if total < min_occurrences:
            continue
        # Canonical selection:
        #   * If the cluster has a code-like identifier ("O-1A", "I-129"),
        #     that bare code is the canonical the way users reference it
        #     (longer "O-1A visa" becomes an alias). This keeps the canonical
        #     STABLE across corpora regardless of phrase frequency.
        #   * Otherwise prefer the fullest, most frequent surface (so an
        #     acronym's expansion wins, e.g. "U.S. Citizenship ..." over "USCIS").
        code_members = [c for c in cluster if _is_code_like(c.surface)]
        if code_members:
            canonical_c = max(code_members, key=lambda c: (c.count, -len(c.surface)))
        else:
            canonical_c = max(
                cluster,
                key=lambda c: (c.count, len(c.surface.split(" ")), len(c.surface)),
            )
        canonical_label = canonical_c.surface
        labels: Counter = Counter()
        docs: set = set()
        aliases: list[str] = []
        for c in cluster:
            labels.update(c.labels)
            docs.update(c.docs)
            if c.surface != canonical_label and c.surface not in aliases:
                aliases.append(c.surface)
        etype, spacy_labels = _pick_type(labels, canonical_label)
        entities.append(
            BootstrappedEntity(
                entity_id=f"bootstrap:{_slug(canonical_label)}",
                canonical_label=canonical_label,
                aliases=tuple(aliases),
                entity_type=etype,
                spacy_labels=spacy_labels,
                occurrences=total,
                source_doc_ids=tuple(sorted(docs)),
            )
        )

    # Stable order: most frequent first.
    entities.sort(key=lambda e: (-e.occurrences, e.canonical_label.lower()))
    return entities
