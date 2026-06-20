"""Relation candidate extractor v2.

For each ready snapshot doc:
1. Parse body text into paragraphs and sentences (using sentence_splitter).
2. For each sentence, run hard safety screens (current/live/private/universal).
3. For each allowed pattern, match the sentence.
4. For each match, resolve subject/object to known entities via the surface index.
5. Emit a raw ExtractedRelationCandidate.

No ML, no embeddings, no network. Read-only over snapshot docs. Does not write
any overlay or memory.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterator

from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex
from worldpgt.relation_extraction_v2.relation_patterns import (
    CURRENT_LIVE_SCREEN,
    PATTERNS,
    PRIVATE_SCREEN,
    UNIVERSAL_SCREEN,
    PatternSpec,
)
from worldpgt.relation_extraction_v2.sentence_splitter import (
    extract_full_body,
    split_paragraphs,
    split_sentences,
)
from worldpgt.relation_extraction_v2.types import (
    ExtractedRelationCandidate,
    RelationExtractionEvidence,
)
from worldpgt.wiki_snapshot_ingestion.types import ReadySnapshotDoc

# Maximum object phrase word count (guards against runaway multi-word matches).
_MAX_OBJ_WORDS = 8

# Lead sentences to extract from per doc (intro is most authoritative).
_LEAD_PARA_LIMIT = 5

# Total sentences per doc (to limit runtime on large docs).
_MAX_SENTENCES_PER_DOC = 300

# Minimum entity recognition length for a captured span to count as a valid
# entity reference (avoids junk like "a", "the", "its").
_MIN_ENTITY_LEN = 3


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _clean_capture(s: str) -> str:
    """Strip trailing stopwords/punctuation from a captured entity span."""

    s = s.strip().rstrip("., ;:)")
    # Remove trailing relational words that bled into the capture.
    s = re.sub(
        r"\s+(?:and|or|with|for|in|on|at|by|of|to|the|a|an|its|their|which|that)$",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()
    return s


def _word_count(s: str) -> int:
    return len(s.split())


def _norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _is_too_generic(s: str) -> bool:
    low = s.lower().strip()
    if len(low) < _MIN_ENTITY_LEN:
        return True
    # Pure article or pronoun.
    if low in {"it", "he", "she", "they", "its", "the", "a", "an", "this", "that"}:
        return True
    return False


def _screen_sentence(sentence: str) -> list[str]:
    """Return list of quarantine reasons if sentence is unsafe; empty = safe."""

    reasons: list[str] = []
    if CURRENT_LIVE_SCREEN.search(sentence):
        reasons.append("current_or_live_claim")
    if PRIVATE_SCREEN.search(sentence):
        reasons.append("private_or_sensitive")
    if UNIVERSAL_SCREEN.search(sentence):
        reasons.append("unsupported_universal")
    return reasons


def _resolve_entity(
    surface: str,
    index: EntitySurfaceIndex,
) -> str:
    """Return canonical name from index, or the cleaned surface if not found.

    If the surface itself doesn't resolve, try to find the longest known entity
    surface that appears as a prefix or substring of the captured span. This
    handles cases where the regex captured a truncated name (e.g. "Elon" when
    "Elon Musk" is the full surface in the index).
    """

    resolved = index.resolve(surface)
    if resolved:
        return resolved

    # Try longest-match entity within the captured span.
    best_surface = ""
    best_canonical = ""
    for known in sorted(index.all_surfaces(), key=len, reverse=True):
        pattern = r"(?<!\w)" + re.escape(known) + r"(?!\w)"
        if re.search(pattern, surface, re.IGNORECASE):
            if len(known) > len(best_surface):
                best_surface = known
                best_canonical = index.resolve(known) or known
    if best_canonical:
        return best_canonical

    return surface


def _find_best_entity_in_span(
    text: str,
    span_start: int,
    span_end: int,
    index: EntitySurfaceIndex,
    sentence: str,
) -> str | None:
    """Find the best (longest) known entity surface that overlaps the regex span.

    Searches within an expanded window around [span_start, span_end] in the
    sentence to recover multi-word entity names that the non-greedy regex may
    have truncated. Returns canonical name or None if nothing useful found.
    """

    # Expand the window slightly to catch names that bled beyond the span.
    window_start = max(0, span_start - 2)
    window_end = min(len(sentence), span_end + 40)
    window = sentence[window_start:window_end]

    # Find all known surfaces in the window, return the longest that starts
    # close to span_start (within 5 chars).
    best_surface = ""
    best_canonical = ""
    for surface in sorted(index.all_surfaces(), key=len, reverse=True):
        pattern = r"(?<!\w)" + re.escape(surface) + r"(?!\w)"
        for m in re.finditer(pattern, window, re.IGNORECASE):
            # Only accept if the match starts near the original span start.
            abs_start = window_start + m.start()
            if abs(abs_start - span_start) <= 8:
                if len(surface) > len(best_surface):
                    best_surface = surface
                    best_canonical = index.resolve(surface) or surface
    return best_canonical if best_canonical else None


def extract_candidates_from_doc(
    doc: ReadySnapshotDoc,
    index: EntitySurfaceIndex,
    *,
    lead_only: bool = False,
) -> tuple[list[ExtractedRelationCandidate], int]:
    """Extract raw relation candidates from one snapshot doc.

    Returns (candidates, sentences_scanned).
    """

    text = Path(doc.normalized_doc_path).read_text(encoding="utf-8")
    body = extract_full_body(text)

    paragraphs = split_paragraphs(body)
    if lead_only:
        paragraphs = paragraphs[:_LEAD_PARA_LIMIT]

    candidates: list[ExtractedRelationCandidate] = []
    sentences_scanned = 0
    seen_triples: set[tuple[str, str, str]] = set()

    for para_idx, para in enumerate(paragraphs):
        sentences = split_sentences(para)
        for sent_idx, sentence in enumerate(sentences):
            if sentences_scanned >= _MAX_SENTENCES_PER_DOC:
                break
            sentences_scanned += 1

            screen_reasons = _screen_sentence(sentence)

            for pattern in PATTERNS:
                for m in pattern.regex.finditer(sentence):
                    raw_a = _clean_capture(m.group("A"))
                    raw_b = _clean_capture(m.group("B"))

                    if not raw_a or not raw_b:
                        continue
                    if _is_too_generic(raw_a) or _is_too_generic(raw_b):
                        continue
                    if _word_count(raw_a) > _MAX_OBJ_WORDS:
                        continue
                    if _word_count(raw_b) > _MAX_OBJ_WORDS:
                        continue

                    # Try surface-index-aware entity resolution. For each
                    # captured span, look for the longest known entity surface
                    # that overlaps that span position in the sentence.
                    a_span_start = m.start("A")
                    a_span_end = m.end("A")
                    b_span_start = m.start("B")
                    b_span_end = m.end("B")

                    # Try to find a better canonical for A.
                    best_a = _find_best_entity_in_span(
                        text, a_span_start, a_span_end, index, sentence
                    )
                    canon_a = best_a if best_a else _resolve_entity(raw_a, index)

                    # Try to find a better canonical for B.
                    best_b = _find_best_entity_in_span(
                        text, b_span_start, b_span_end, index, sentence
                    )
                    canon_b = best_b if best_b else _resolve_entity(raw_b, index)

                    if pattern.direction == "A_subj_B_obj":
                        subject, obj = canon_a, canon_b
                    else:
                        subject, obj = canon_b, canon_a

                    if pattern.relation in ("is_a", "type_of") and _norm_label(subject) == _norm_label(obj):
                        continue

                    # Deduplicate within this doc.
                    triple = (subject.lower(), pattern.relation, obj.lower())
                    if triple in seen_triples:
                        continue
                    seen_triples.add(triple)

                    cand_id = (
                        f"rv2:{doc.raw_text_sha256[:8]}:"
                        f"{_sha8(subject + pattern.relation + obj)}"
                    )

                    evidence = RelationExtractionEvidence(
                        sentence=sentence,
                        pattern_id=pattern.pattern_id,
                        pattern_description=pattern.description,
                        subject_surface=raw_a if pattern.direction == "A_subj_B_obj" else raw_b,
                        object_surface=raw_b if pattern.direction == "A_subj_B_obj" else raw_a,
                        sentence_index=sent_idx,
                        paragraph_index=para_idx,
                    )

                    # Confidence degradation: screened sentences stay in output
                    # but will be quarantined by the validator.
                    confidence = pattern.confidence
                    if screen_reasons:
                        confidence = "low"

                    candidates.append(
                        ExtractedRelationCandidate(
                            id=cand_id,
                            subject=subject,
                            relation=pattern.relation,
                            object=obj,
                            confidence=confidence,
                            stability=pattern.stability,
                            risk=pattern.risk,
                            source_title=doc.title,
                            source_url=doc.source_url,
                            retrieved_at=doc.retrieved_at,
                            raw_text_sha256=doc.raw_text_sha256,
                            evidence_sentence=sentence,
                            pattern_id=pattern.pattern_id,
                            directionality="forward",
                            requires_review=True,
                            safe_for_overlay_delta=False,
                            evidence=evidence,
                            notes="; ".join(screen_reasons) if screen_reasons else "",
                        )
                    )

    return candidates, sentences_scanned


def extract_all_candidates(
    docs: list[ReadySnapshotDoc],
    index: EntitySurfaceIndex,
) -> tuple[list[ExtractedRelationCandidate], int, list[str]]:
    """Extract from all docs. Returns (candidates, total_sentences, errors)."""

    all_candidates: list[ExtractedRelationCandidate] = []
    total_sentences = 0
    errors: list[str] = []

    for doc in docs:
        try:
            cands, sents = extract_candidates_from_doc(doc, index)
            all_candidates.extend(cands)
            total_sentences += sents
        except Exception as exc:
            errors.append(f"{doc.title}: {exc}")

    return all_candidates, total_sentences, errors
