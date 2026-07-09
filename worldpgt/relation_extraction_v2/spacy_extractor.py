"""spaCy dependency-parse relation extraction (supplementary path).

Supplements the regex path in relation_candidate_extractor. Handles:
- Passive voice: "SpaceX was founded by Elon Musk"       → founded_by
- Active voice:  "Elon Musk founded SpaceX"               → founded
- Coordination:  "X and Y founded Z", "X founded Y and Z"
- Passive prep:  "SpaceX is headquartered in Hawthorne"   → headquartered_in
- Copula:        "Elon Musk is the CEO of SpaceX"         → leader_of
                 "SpaceX is an aerospace company"          → is_a

No ML training, no embeddings, no network access during extraction.
The spaCy model is loaded lazily on first call when enable_spacy=True.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

from worldpgt.relation_extraction_v2.relation_patterns import (
    CURRENT_LIVE_SCREEN,
    PRIVATE_SCREEN,
    UNIVERSAL_SCREEN,
)
from worldpgt.relation_extraction_v2.types import (
    ExtractedRelationCandidate,
    RelationExtractionEvidence,
)

if TYPE_CHECKING:
    from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex
    from worldpgt.wiki_snapshot_ingestion.types import ReadySnapshotDoc


# Lazy-loaded spaCy pipeline — initialised on first call to _get_nlp().
_NLP = None

try:
    import spacy as _spacy_mod  # optional dependency
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False


def _get_nlp():
    global _NLP
    if _NLP is None:
        if not _SPACY_AVAILABLE:
            raise RuntimeError(
                "spaCy is not installed. Run: pip install spacy && "
                "python -m spacy download en_core_web_sm"
            )
        _NLP = _spacy_mod.load("en_core_web_sm")
    return _NLP


# ---------------------------------------------------------------------------
# Verb-lemma → relation mapping
# ---------------------------------------------------------------------------

# Maps verb lemma to (active_relation, passive_relation, invert_for_active).
# invert_for_active=True: for active voice the grammatical dobj becomes the
# triple subject and nsubj becomes the triple object — used for verbs where
# the canonical fact is stated from the object's perspective (e.g. "X owns Y"
# → Y owned_by X; "X publishes Y" → Y published_by X).
_VERB_RELATIONS: dict[str, tuple[str, str, bool]] = {
    "found":        ("founded",          "founded_by",      False),
    "establish":    ("founded",          "founded_by",      False),
    "create":       ("founded",          "created_by",      False),
    "develop":      ("develops",         "developed_by",    False),
    "manufacture":  ("manufactures",     "developed_by",    False),
    "produce":      ("produces",         "developed_by",    False),
    "publish":      ("published_by",     "published_by",    True),
    "operate":      ("operates",         "owned_by",        False),
    "own":          ("owned_by",         "owned_by",        True),
    "acquire":      ("owned_by",         "owned_by",        True),
    "lead":         ("leader_of",        "leader_of",       False),
    "headquarter":  ("headquartered_in", "headquartered_in", False),
}

# Confidence / stability / risk for spaCy-extracted triples (per relation).
_RELATION_META: dict[str, tuple[str, str, str]] = {
    "founded":          ("high",   "semi_stable", "medium"),
    "founded_by":       ("high",   "semi_stable", "medium"),
    "created_by":       ("high",   "semi_stable", "medium"),
    "develops":         ("high",   "semi_stable", "medium"),
    "developed_by":     ("high",   "semi_stable", "medium"),
    "manufactures":     ("high",   "semi_stable", "medium"),
    "produces":         ("medium", "semi_stable", "medium"),
    "published_by":     ("high",   "semi_stable", "medium"),
    "operates":         ("medium", "semi_stable", "medium"),
    "owned_by":         ("medium", "semi_stable", "medium"),
    "headquartered_in": ("high",   "semi_stable", "low"),
    "leader_of":        ("high",   "volatile",    "high"),
    "is_a":             ("high",   "stable",      "low"),
}

# Noun lemmas that indicate a named organisation type (for is_a copula).
_ORG_TYPE_NOUNS = frozenset({
    "company", "corporation", "manufacturer", "provider", "firm", "startup",
    "organization", "conglomerate", "enterprise", "group", "institution",
    "platform", "agency", "publication", "constellation", "service",
})

# Attr lemmas that signal a leadership title (for leader_of copula).
_LEADERSHIP_TITLES = frozenset({
    "ceo", "president", "chairman", "chairperson", "head", "director",
    "founder", "cofounder", "officer",
})

_MAX_OBJ_WORDS = 8
_MIN_ENTITY_LEN = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _clean(s: str) -> str:
    s = s.strip().rstrip(".,;:)")
    s = re.sub(
        r"\s+(?:and|or|with|for|in|on|at|by|of|to|the|a|an|its|their|which|that)$",
        "", s, flags=re.IGNORECASE,
    ).strip()
    return s


def _too_generic(s: str) -> bool:
    low = s.lower().strip()
    if len(low) < _MIN_ENTITY_LEN:
        return True
    if low in {"it", "he", "she", "they", "its", "the", "a", "an", "this", "that", "which"}:
        return True
    return False


def _span_text(token) -> str:
    """Return clean surface text for the NP headed by *token*.

    Collects tokens from token.left_edge to token (inclusive).
    - Skips lowercase determiners ("a", "an", "the").
    - Preserves capitalised "The" when followed immediately by a proper noun
      so that names like "The Boring Company" are kept intact.
    """
    parts: list[str] = []
    doc = token.doc
    for t in doc[token.left_edge.i: token.i + 1]:
        if t.pos_ == "DET":
            if t.text.lower() == "the":
                next_i = t.i + 1
                # Keep "The" only as part of a proper-noun compound name.
                if next_i <= token.i and doc[next_i].pos_ == "PROPN":
                    parts.append(t.text)
            # Skip "a", "an", and "the" before common nouns.
        else:
            parts.append(t.text)
    return " ".join(parts).strip()


def _resolve(surface: str, index: "EntitySurfaceIndex") -> str:
    """Resolve surface form to canonical entity name.

    Tries direct lookup first, then longest-match among known surfaces.
    Falls back to the cleaned surface if nothing resolves.
    """
    canon = index.resolve(surface)
    if canon:
        return canon
    best_s = ""
    best_c = ""
    for known in sorted(index.all_surfaces(), key=len, reverse=True):
        pattern = r"(?<!\w)" + re.escape(known) + r"(?!\w)"
        if re.search(pattern, surface, re.IGNORECASE):
            if len(known) > len(best_s):
                best_s = known
                best_c = index.resolve(known) or known
    return best_c if best_c else _clean(surface)


def _conj_chain(token) -> list:
    """Return *token* plus all conj (coordination) descendants, recursively."""
    result = [token]
    for child in token.children:
        if child.dep_ == "conj":
            result.extend(_conj_chain(child))
    return result


def _screen(sentence: str) -> list[str]:
    reasons: list[str] = []
    if CURRENT_LIVE_SCREEN.search(sentence):
        reasons.append("current_or_live_claim")
    if PRIVATE_SCREEN.search(sentence):
        reasons.append("private_or_sensitive")
    if UNIVERSAL_SCREEN.search(sentence):
        reasons.append("unsupported_universal")
    return reasons


def _make_candidate(
    subject: str,
    relation: str,
    obj: str,
    sent_text: str,
    screen_reasons: list[str],
    pattern_id: str,
    doc_meta: dict,
) -> ExtractedRelationCandidate | None:
    subject = _clean(subject)
    obj = _clean(obj)
    if not subject or not obj:
        return None
    if _too_generic(subject) or _too_generic(obj):
        return None
    if len(subject.split()) > _MAX_OBJ_WORDS or len(obj.split()) > _MAX_OBJ_WORDS:
        return None
    if relation in ("is_a", "type_of") and subject.lower() == obj.lower():
        return None

    conf, stab, risk = _RELATION_META.get(relation, ("medium", "semi_stable", "medium"))
    if screen_reasons:
        conf = "low"

    cand_id = f"spacy:{doc_meta['sha'][:8]}:{_sha8(subject + relation + obj)}"
    evidence = RelationExtractionEvidence(
        sentence=sent_text,
        pattern_id=pattern_id,
        pattern_description=pattern_id,
        subject_surface=subject,
        object_surface=obj,
        sentence_index=doc_meta.get("sent_idx", 0),
        paragraph_index=doc_meta.get("para_idx", 0),
    )
    return ExtractedRelationCandidate(
        id=cand_id,
        subject=subject,
        relation=relation,
        object=obj,
        confidence=conf,
        stability=stab,
        risk=risk,
        source_title=doc_meta["title"],
        source_url=doc_meta["url"],
        retrieved_at=doc_meta["retrieved_at"],
        raw_text_sha256=doc_meta["sha"],
        evidence_sentence=sent_text,
        pattern_id=pattern_id,
        directionality="forward",
        requires_review=True,
        safe_for_overlay_delta=False,
        evidence=evidence,
        notes="; ".join(screen_reasons) if screen_reasons else "",
    )


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------

def _extract_from_spacy_doc(
    spacy_doc,
    index: "EntitySurfaceIndex",
    doc_meta: dict,
) -> list[ExtractedRelationCandidate]:
    """Extract SPO triples from a single-sentence spaCy Doc."""
    sent_text = spacy_doc.text
    screen_reasons = _screen(sent_text)
    results: list[ExtractedRelationCandidate] = []

    def emit(subj_raw: str, rel: str, obj_raw: str, pid: str) -> None:
        subj = _resolve(subj_raw, index)
        obj = _resolve(obj_raw, index)
        cand = _make_candidate(subj, rel, obj, sent_text, screen_reasons, pid, doc_meta)
        if cand:
            results.append(cand)

    for token in spacy_doc:

        # ------------------------------------------------------------------ #
        # Verb-based patterns (VERB tokens only)
        # ------------------------------------------------------------------ #
        if token.pos_ == "VERB":
            lemma = token.lemma_.lower()
            verb_config = _VERB_RELATIONS.get(lemma)

            if verb_config:
                active_rel, passive_rel, invert_active = verb_config

                nsubjpass_toks = [c for c in token.children if c.dep_ == "nsubjpass"]
                agent_preps   = [c for c in token.children if c.dep_ == "agent"]
                nsubj_toks    = [c for c in token.children if c.dep_ == "nsubj"]
                dobj_toks     = [c for c in token.children if c.dep_ == "dobj"]

                # -- Passive + agent: "X was founded by Y" ----------------- #
                if nsubjpass_toks:
                    agents: list = []
                    for ap in agent_preps:
                        for pobj in ap.children:
                            if pobj.dep_ == "pobj":
                                agents.extend(_conj_chain(pobj))

                    # headquartered_in uses prep("in") instead of an agent.
                    if lemma == "headquarter" and not agents:
                        for prep_ch in token.children:
                            if prep_ch.dep_ == "prep" and prep_ch.text.lower() == "in":
                                for pobj in prep_ch.children:
                                    if pobj.dep_ == "pobj":
                                        agents.append(pobj)

                    for nsubjpass in nsubjpass_toks:
                        for subj_tok in _conj_chain(nsubjpass):
                            subj_text = _span_text(subj_tok)
                            for obj_tok in agents:
                                emit(subj_text, passive_rel, _span_text(obj_tok),
                                     f"spacy:passive:{lemma}")

                # -- Active + dobj: "Y founded X" (with coordination) ------ #
                if nsubj_toks and dobj_toks:
                    for nsubj in nsubj_toks:
                        for dobj in dobj_toks:
                            for subj_tok in _conj_chain(nsubj):
                                for obj_tok in _conj_chain(dobj):
                                    ns = _span_text(subj_tok)
                                    do = _span_text(obj_tok)
                                    if invert_active:
                                        emit(do, active_rel, ns, f"spacy:active:{lemma}")
                                    else:
                                        emit(ns, active_rel, do, f"spacy:active:{lemma}")

        # ------------------------------------------------------------------ #
        # Copula patterns (root "be" → leader_of or is_a)
        # ------------------------------------------------------------------ #
        if token.lemma_.lower() == "be" and token.dep_ == "ROOT":
            nsubj_toks = [c for c in token.children if c.dep_ == "nsubj"]
            attr_toks  = [c for c in token.children if c.dep_ == "attr"]

            for nsubj in nsubj_toks:
                subj_text = _span_text(nsubj)

                for attr in attr_toks:
                    attr_low = attr.lemma_.lower()

                    # "X is the CEO of Y" → leader_of
                    if attr_low in _LEADERSHIP_TITLES:
                        for prep_ch in attr.children:
                            if prep_ch.dep_ == "prep" and prep_ch.text.lower() == "of":
                                for pobj in prep_ch.children:
                                    if pobj.dep_ == "pobj":
                                        for obj_tok in _conj_chain(pobj):
                                            emit(subj_text, "leader_of",
                                                 _span_text(obj_tok),
                                                 "spacy:copula:leader_of")

                    # "X is an aerospace company" → is_a
                    elif attr_low in _ORG_TYPE_NOUNS:
                        emit(subj_text, "is_a", _span_text(attr),
                             "spacy:copula:is_a")

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_candidates_from_doc_spacy(
    doc: "ReadySnapshotDoc",
    index: "EntitySurfaceIndex",
    *,
    lead_only: bool = False,
) -> tuple[list[ExtractedRelationCandidate], int]:
    """Extract relation candidates from one snapshot doc using spaCy.

    Returns (candidates, sentences_scanned).
    """
    from worldpgt.relation_extraction_v2.sentence_splitter import (
        extract_full_body,
        split_paragraphs,
        split_sentences,
    )

    _LEAD_PARA_LIMIT = 5
    _MAX_SENTENCES_PER_DOC = 300
    _PIPE_BATCH_SIZE = 64

    text = Path(doc.normalized_doc_path).read_text(encoding="utf-8")
    body = extract_full_body(text)
    paragraphs = split_paragraphs(body)
    if lead_only:
        paragraphs = paragraphs[:_LEAD_PARA_LIMIT]

    # Phase 1: collect sentences with their per-sentence metadata up to the cap.
    sentence_records: list[tuple[str, dict]] = []
    for para_idx, para in enumerate(paragraphs):
        if len(sentence_records) >= _MAX_SENTENCES_PER_DOC:
            break
        for sent_idx, sent_text in enumerate(split_sentences(para)):
            if len(sentence_records) >= _MAX_SENTENCES_PER_DOC:
                break
            sentence_records.append((sent_text, {
                "title": doc.title,
                "url": doc.source_url,
                "retrieved_at": doc.retrieved_at,
                "sha": doc.raw_text_sha256,
                "sent_idx": sent_idx,
                "para_idx": para_idx,
            }))

    if not sentence_records:
        return [], 0

    # Phase 2: batch-parse with nlp.pipe — avoids per-call Python overhead.
    nlp = _get_nlp()
    candidates: list[ExtractedRelationCandidate] = []
    seen_triples: set[tuple[str, str, str]] = set()
    texts = [sent_text for sent_text, _ in sentence_records]

    for spacy_doc, (_, meta) in zip(
        nlp.pipe(texts, batch_size=_PIPE_BATCH_SIZE),
        sentence_records,
    ):
        for cand in _extract_from_spacy_doc(spacy_doc, index, meta):
            triple = (cand.subject.lower(), cand.relation, cand.object.lower())
            if triple in seen_triples:
                continue
            seen_triples.add(triple)
            candidates.append(cand)

    return candidates, len(sentence_records)
