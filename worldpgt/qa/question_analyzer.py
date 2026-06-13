"""Deterministic question analyzer for AnswerPlanner v1.

Parses controlled QA questions into AnalyzedQuestion objects.
No ML.  No external model.  Rule-based pattern matching only.
"""

from __future__ import annotations

import re
from typing import Optional

from worldpgt.qa.types import AnalyzedQuestion, QAIntent

_KNOWN_TERMS = frozenset({"bank", "bat", "seal", "crane", "rock", "spring"})

_TERM_SENSES: dict[str, list[str]] = {
    "bank": ["financial_institution", "river_edge"],
    "bat": ["animal", "sports_equipment"],
    "seal": ["animal", "closure_stamp"],
    "crane": ["bird", "machine"],
    "rock": ["stone", "music"],
    "spring": ["season", "coil"],
}

# (qualifier_word, term) -> sense_id
_QUALIFIER_SENSE: dict[tuple[str, str], str] = {
    # bank
    ("financial", "bank"): "financial_institution",
    ("investment", "bank"): "financial_institution",
    ("savings", "bank"): "financial_institution",
    ("money", "bank"): "financial_institution",
    ("river", "bank"): "river_edge",
    ("stream", "bank"): "river_edge",
    ("waterway", "bank"): "river_edge",
    # bat
    ("animal", "bat"): "animal",
    ("flying", "bat"): "animal",
    ("mammal", "bat"): "animal",
    ("nocturnal", "bat"): "animal",
    ("baseball", "bat"): "sports_equipment",
    ("cricket", "bat"): "sports_equipment",
    ("sports", "bat"): "sports_equipment",
    ("wooden", "bat"): "sports_equipment",
    # seal
    ("animal", "seal"): "animal",
    ("marine", "seal"): "animal",
    ("wax", "seal"): "closure_stamp",
    ("stamp", "seal"): "closure_stamp",
    ("envelope", "seal"): "closure_stamp",
    ("closure", "seal"): "closure_stamp",
    # crane
    ("bird", "crane"): "bird",
    ("avian", "crane"): "bird",
    ("feathered", "crane"): "bird",
    ("construction", "crane"): "machine",
    ("machine", "crane"): "machine",
    ("lifting", "crane"): "machine",
    # rock
    ("music", "rock"): "music",
    ("band", "rock"): "music",
    ("genre", "rock"): "music",
    ("stone", "rock"): "stone",
    ("mineral", "rock"): "stone",
    ("geological", "rock"): "stone",
    ("boulder", "rock"): "stone",
    # spring
    ("season", "spring"): "season",
    ("seasonal", "spring"): "season",
    ("coil", "spring"): "coil",
    ("mechanical", "spring"): "coil",
    ("metal", "spring"): "coil",
    ("elastic", "spring"): "coil",
}

# Phrases that appear after "as a/an" in explain_cue questions → sense_id (per term)
_SENSE_LABEL_TO_ID: dict[str, dict[str, str]] = {
    "bank": {
        "financial institution": "financial_institution",
        "financial_institution": "financial_institution",
        "river edge": "river_edge",
        "river_edge": "river_edge",
        "river bank": "river_edge",
    },
    "bat": {
        "animal": "animal",
        "sports equipment": "sports_equipment",
        "sports_equipment": "sports_equipment",
        "baseball bat": "sports_equipment",
    },
    "seal": {
        "animal": "animal",
        "closure stamp": "closure_stamp",
        "closure_stamp": "closure_stamp",
        "wax seal": "closure_stamp",
        "stamp": "closure_stamp",
    },
    "crane": {
        "bird": "bird",
        "machine": "machine",
        "construction crane": "machine",
    },
    "rock": {
        "stone": "stone",
        "music": "music",
        "rock music": "music",
    },
    "spring": {
        "season": "season",
        "coil": "coil",
        "mechanical device": "coil",
        "spring season": "season",
        "spring coil": "coil",
    },
}

_INLINE_CONTEXT_RE = re.compile(
    r"""[Ii]n\s+['""](.+?)['""][,\s]""", re.DOTALL
)
_CUE_IN_QUOTES_RE = re.compile(r"""['"'"]([^'"'"]+)['"'"]""")
_WHAT_DOES_RE = re.compile(
    r"[Ww]hat\s+does\s+(\w+)\s+mean", re.IGNORECASE
)


def _find_term(text: str) -> Optional[str]:
    """Return the first known term found in the lowercased text."""
    lower = text.lower()
    # Prefer longer matches; check all and pick first by position.
    found: list[tuple[int, str]] = []
    for term in _KNOWN_TERMS:
        idx = lower.find(term)
        if idx >= 0:
            found.append((idx, term))
    if not found:
        return None
    found.sort()
    return found[0][1]


def _find_all_terms(text: str) -> list[str]:
    """Return all known terms found in the text, in order of appearance."""
    lower = text.lower()
    found: list[tuple[int, str]] = []
    for term in _KNOWN_TERMS:
        idx = lower.find(term)
        if idx >= 0:
            found.append((idx, term))
    found.sort()
    return [t for _, t in found]


def _sense_from_qualifiers(question_lower: str, term: str) -> Optional[str]:
    """Try to determine a single sense from qualifier words in the question."""
    hits: list[str] = []
    for (qual, t), sense_id in _QUALIFIER_SENSE.items():
        if t == term and qual in question_lower:
            hits.append(sense_id)
    unique = list(dict.fromkeys(hits))
    return unique[0] if len(unique) == 1 else None


def _sense_from_label(after_as: str, term: str) -> Optional[str]:
    """Look up a sense_id from the text that follows 'as a/an' in explain_cue questions."""
    text = after_as.strip().lower()
    labels = _SENSE_LABEL_TO_ID.get(term, {})
    # Try longest match first
    for label in sorted(labels, key=len, reverse=True):
        if label in text:
            return labels[label]
    return None


def _extract_cue(question: str) -> Optional[str]:
    """Extract the cue token from an explain_cue question (word in quotes, or after 'does')."""
    m = _CUE_IN_QUOTES_RE.search(question)
    if m:
        return m.group(1).strip().lower()
    # Fallback: "Why does WORD suggest"
    m2 = re.search(r"[Ww]hy\s+does\s+(\w+)\s+suggest", question)
    if m2:
        return m2.group(1).strip().lower()
    return None


def _detect_inline_context(question: str) -> Optional[str]:
    """Extract the embedded sentence from classify_context questions."""
    m = _INLINE_CONTEXT_RE.search(question)
    if m:
        return m.group(1).strip()
    # Try without trailing punctuation requirement
    m2 = re.search(r"""[Ii]n\s+['""](.+?)['""]""", question, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return None


def _is_distinguish_question(question_lower: str) -> bool:
    return any(p in question_lower for p in (
        "difference between", "different from", "how is", "compare",
        "distinguish", "versus", " vs ",
    ))


def _is_explain_cue_question(question_lower: str) -> bool:
    return any(p in question_lower for p in (
        "why does", "why do", "how does", "what makes",
    )) and any(p in question_lower for p in ("suggest", "indicate", "signal", "relate"))


def _is_classify_context_question(question: str) -> bool:
    return bool(_detect_inline_context(question))


def _extract_distinguish_senses(question: str) -> tuple[Optional[str], Optional[str]]:
    """Extract two senses from a distinguish_senses question."""
    lower = question.lower()
    terms = _find_all_terms(lower)
    if not terms:
        return None, None

    # Split on "and" / "from" / "versus"
    for sep in (" and ", " from ", " versus ", " vs ", " or "):
        if sep in lower:
            left, right = lower.split(sep, 1)
            sense_a: Optional[str] = None
            sense_b: Optional[str] = None
            for t in terms:
                if not sense_a:
                    sense_a = _sense_from_qualifiers(left, t)
                if not sense_b:
                    sense_b = _sense_from_qualifiers(right, t)
            if sense_a and sense_b and sense_a != sense_b:
                return sense_a, sense_b
            # Try each side independently
            if not sense_a:
                sense_a = _sense_from_qualifiers(left, terms[0]) if terms else None
            if not sense_b:
                sense_b = _sense_from_qualifiers(right, terms[-1]) if terms else None
            if sense_a and sense_b:
                return sense_a, sense_b

    return None, None


def analyze(question: str) -> AnalyzedQuestion:
    """Parse a question into an AnalyzedQuestion deterministically."""
    lower = question.lower()

    # ── classify_context (has embedded sentence in quotes) ──────────────
    inline_ctx = _detect_inline_context(question)
    if inline_ctx:
        term = _find_term(inline_ctx) or _find_term(question)
        cues_in_q: list[str] = []
        return AnalyzedQuestion(
            question=question,
            intent="classify_context",
            term=term,
            target_sense=None,
            second_sense=None,
            cues_in_question=cues_in_q,
            inline_context=inline_ctx,
            underconstrained=term is None,
        )

    # ── distinguish_senses ───────────────────────────────────────────────
    if _is_distinguish_question(lower):
        term = _find_term(lower)
        sense_a, sense_b = _extract_distinguish_senses(question)
        return AnalyzedQuestion(
            question=question,
            intent="distinguish_senses",
            term=term,
            target_sense=sense_a,
            second_sense=sense_b,
            cues_in_question=[],
            inline_context=None,
            underconstrained=(sense_a is None or sense_b is None),
        )

    # ── explain_cue ──────────────────────────────────────────────────────
    if _is_explain_cue_question(lower):
        term = _find_term(lower)
        cue = _extract_cue(question)
        sense_id: Optional[str] = None
        if term:
            # Try "as a/an [sense label]" or "as [sense label]"
            m = re.search(r"\bas\s+(?:a\s+|an\s+)?(.+?)(?:\?|$)", lower)
            if m:
                sense_id = _sense_from_label(m.group(1), term)
            if not sense_id:
                sense_id = _sense_from_qualifiers(lower, term)
        return AnalyzedQuestion(
            question=question,
            intent="explain_cue",
            term=term,
            target_sense=sense_id,
            second_sense=None,
            cues_in_question=[cue] if cue else [],
            inline_context=None,
            underconstrained=(term is None or cue is None or sense_id is None),
        )

    # ── unknown_or_ambiguous ("What does X mean?") ───────────────────────
    m_wd = _WHAT_DOES_RE.search(question)
    if m_wd:
        term = m_wd.group(1).lower()
        if term not in _KNOWN_TERMS:
            term = _find_term(lower)
        return AnalyzedQuestion(
            question=question,
            intent="unknown_or_ambiguous",
            term=term,
            target_sense=None,
            second_sense=None,
            cues_in_question=[],
            inline_context=None,
            underconstrained=True,
        )

    # ── define_sense ("What is a/an [qualifier] [term]?") ────────────────
    term = _find_term(lower)
    if term:
        sense_id = _sense_from_qualifiers(lower, term)
        if sense_id:
            cue = _extract_cue(question)
            return AnalyzedQuestion(
                question=question,
                intent="define_sense",
                term=term,
                target_sense=sense_id,
                second_sense=None,
                cues_in_question=[cue] if cue else [],
                inline_context=None,
                underconstrained=False,
            )

    # ── fallback: unknown_or_ambiguous ───────────────────────────────────
    return AnalyzedQuestion(
        question=question,
        intent="unknown_or_ambiguous",
        term=term,
        target_sense=None,
        second_sense=None,
        cues_in_question=[],
        inline_context=None,
        underconstrained=True,
    )
