"""Weightless semantic renderer v2.

Pipeline (per already-selected sense):

    frame -> candidate phrases -> connector-aware rendering
          -> semantic/surface validation -> deterministic ranker
          -> best safe candidate (or None -> caller audits)

The ranker is fully deterministic. Its primary signal is *relative novelty*:
among the candidate phrases for a connector, it prefers the one that repeats the
fewest content words from the prompt's recent context, breaking ties by the
phrase's original library order and then lexically. This keeps already-clean
continuations stable while steering away from awkward word/action repetition.

No ML, no neural weights, no sampling.
"""

from __future__ import annotations

import re
from typing import Optional

from worldpgt.continuation import phrase_library
from worldpgt.continuation.realization import (
    ENDING_NEUTRAL,
    _trim_connector_mismatch,
    _trim_repeated_subject,
)
from worldpgt.continuation.semantic_frame import FrameCandidate, SemanticFrame
from worldpgt.continuation.surface_validator import validate_surface_text

RENDERER_NAME = "semantic_renderer_v2"

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_RECENT_WINDOW = 6

# Words that do not count as repeated content when shared with the prompt.
_STOPWORDS = {
    "the", "a", "an", "to", "and", "as", "it", "its", "his", "her", "he", "she",
    "they", "them", "their", "was", "were", "is", "are", "be", "been", "of", "in",
    "on", "at", "with", "for", "by", "from", "into", "below", "above", "near",
    "over", "under", "down", "up", "out", "off", "this", "that", "then", "only",
    "so", "but", "or", "if", "after", "before", "until", "when", "while", "where",
    "there", "here", "him", "had", "has", "have", "did", "do", "does", "would",
    "could", "should", "will", "would", "about", "back", "place", "another",
    "more", "other", "same", "fresh", "second",
}

# A repeated content word is forgiven when preceded by one of these (e.g.
# "catch another fish" after "...chasing fish" is acceptable).
_SOFTENERS = {"another", "more", "other", "same", "fresh", "second"}

# Story-drift / dialogue markers that must never appear in an emitted candidate.
_DRIFT_WORDS = {
    "said", "asked", "told", "replied", "shouted", "whispered",
    "mother", "father", "boyfriend", "girlfriend", "pregnancy",
    "wife", "husband", "daughter", "son",
}
_DRIFT_CHARS = {'"', "'", "“", "”", "‘", "’", "\n"}


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _phrase_part(prompt: str, text: str) -> str:
    """The continuation phrase appended after the prompt (best effort)."""
    base = prompt.rstrip()
    if text.lower().startswith(base.lower()):
        return text[len(base):].strip()
    return text.strip()


def _recent_content(prompt: str, term: str) -> set[str]:
    recent = _tokens(prompt)[-_RECENT_WINDOW:]
    return {t for t in recent if t not in _STOPWORDS and t != term}


def _overlap_count(prompt: str, phrase_part: str, term: str) -> int:
    """Number of content words the phrase repeats from the prompt's recent context."""
    recent = _recent_content(prompt, term)
    phrase_tokens = _tokens(phrase_part)
    count = 0
    for idx, token in enumerate(phrase_tokens):
        if token in _STOPWORDS or token == term:
            continue
        if token in recent:
            prev = phrase_tokens[idx - 1] if idx > 0 else ""
            if prev in _SOFTENERS:
                continue
            count += 1
    return count


def _drift_markers(phrase_part: str) -> list[str]:
    markers = []
    if any(ch in phrase_part for ch in _DRIFT_CHARS):
        markers.append("dialogue_or_newline")
    for token in _tokens(phrase_part):
        if token in _DRIFT_WORDS:
            markers.append(token)
    return markers


def _render_text(prompt: str, frame: SemanticFrame, phrase: str) -> str:
    """Apply the same deterministic trims the legacy realizer uses, then join."""
    trimmed = _trim_repeated_subject(prompt, phrase)
    trimmed = _trim_connector_mismatch(frame.connector_type, trimmed)
    if not trimmed.strip():
        return ""
    return f"{prompt.rstrip()} {trimmed}".strip()


def make_frame_candidates(
    prompt: str,
    frame: SemanticFrame,
    phrases: list[str],
) -> list[FrameCandidate]:
    """Render up to 5 connector-aware candidates from an explicit phrase pool.

    Each candidate carries validation flags in ``reasons`` (``surface_invalid``,
    ``surface_pattern=...``, ``drift``, ``drift_marker=...``, ``empty``,
    ``connector_match``). Ranking/dropping happens in ``rank_frame_candidates``.
    """
    candidates: list[FrameCandidate] = []
    for phrase in phrases[:5]:
        text = _render_text(prompt, frame, phrase)
        reasons = [f"connector_match={frame.connector_type}"]
        if not text:
            reasons.append("empty")
            candidates.append(FrameCandidate(text="", frame=frame, renderer_name=RENDERER_NAME, score=0.0, reasons=reasons))
            continue

        validation = validate_surface_text(prompt, text)
        if not validation.ok:
            reasons.append("surface_invalid")
            reasons.extend(f"surface_pattern={p}" for p in validation.matched_patterns)

        phrase_part = _phrase_part(prompt, text)
        drift = _drift_markers(phrase_part)
        if drift:
            reasons.append("drift")
            reasons.extend(f"drift_marker={m}" for m in drift)

        candidates.append(
            FrameCandidate(
                text=text,
                frame=frame,
                renderer_name=RENDERER_NAME,
                score=0.0,
                reasons=reasons,
            )
        )
    return candidates


def generate_frame_candidates(prompt: str, frame: SemanticFrame) -> list[FrameCandidate]:
    """Generate candidates for a frame using the explicit phrase library."""
    phrases = phrase_library.get_phrases(frame.term, frame.sense_id, frame.connector_type)
    if not phrases:
        phrases = phrase_library.get_phrases(frame.term, frame.sense_id, ENDING_NEUTRAL)
    return make_frame_candidates(prompt, frame, phrases)


def _is_safe(candidate: FrameCandidate) -> bool:
    return not (
        "empty" in candidate.reasons
        or "surface_invalid" in candidate.reasons
        or "drift" in candidate.reasons
    )


def rank_frame_candidates(
    prompt: str,
    candidates: list[FrameCandidate],
) -> Optional[FrameCandidate]:
    """Return the best safe candidate, or None if none is safe.

    Ranking key (ascending, lower is better):
      1. overlap with prompt's recent content words (+1 if too short/generic)
      2. original candidate order (prefer earlier curated phrases on ties)
      3. text (lexical, for full determinism)
    """
    scored: list[tuple[tuple[float, int, str], FrameCandidate]] = []
    for index, candidate in enumerate(candidates):
        if not _is_safe(candidate):
            continue
        phrase_part = _phrase_part(prompt, candidate.text)
        overlap = _overlap_count(prompt, phrase_part, candidate.frame.term)
        too_short = len([t for t in _tokens(phrase_part) if t not in _STOPWORDS]) < 1
        penalty = overlap + (1 if too_short else 0)
        candidate.score = float(-penalty)
        candidate.reasons.append(f"overlap={overlap}")
        if too_short:
            candidate.reasons.append("too_short")
        scored.append(((float(penalty), index, candidate.text), candidate))

    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0])
    best = scored[0][1]
    best.reasons.append("selected_best_candidate")
    return best
