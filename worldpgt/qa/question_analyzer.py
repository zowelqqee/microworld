"""Deterministic question analyzer for AnswerPlanner v1.

Parses controlled QA questions into AnalyzedQuestion objects.
No ML.  No external model.  Rule-based pattern matching only.

GeneralizedQuestionAnalyzer v1 additions (new phrasings only):
  - classify_context/define_sense: "SENTENCE. What does TERM mean?"
  - classify_context/define_sense: "SENTENCE. What kind of TERM is it?"
  - classify_context/define_sense: "Is SUBJECT [prep CUE] an A or B?"
  - explain_cue:                   "Why does/do CUE point to TERM as SENSE"
  - distinguish_senses:            "Compare X with Y."

For new context-bearing phrasings, the inline context is scored against
_CONTEXT_CUE_TO_SENSE (derived from sense_memory builtin cues + select
overlay tokens).  If exactly one sense scores, the question is routed as
define_sense so the renderer emits the rich description format.  If multiple
senses score (conflict), it falls back to classify_context for the planner
to handle via margin/conflict guards.
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

# (qualifier_word, term) -> sense_id  — used for define_sense and explain_cue detection
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

# Context cue → sense_id lookup, keyed by (token, term).
# Derived from ExplicitSenseMemory builtin cues plus select overlay tokens.
# Used ONLY in new context-bearing phrasings to determine the winning sense
# when the context is unambiguous.  Does NOT replace sense_memory scoring.
_CONTEXT_CUE_TO_SENSE: dict[tuple[str, str], str] = {
    # bank / financial_institution
    ("money", "bank"): "financial_institution",
    ("loan", "bank"): "financial_institution",
    ("account", "bank"): "financial_institution",
    ("teller", "bank"): "financial_institution",
    ("deposit", "bank"): "financial_institution",
    ("cash", "bank"): "financial_institution",
    ("card", "bank"): "financial_institution",
    ("mortgage", "bank"): "financial_institution",
    ("credit", "bank"): "financial_institution",
    ("customer", "bank"): "financial_institution",
    ("client", "bank"): "financial_institution",
    ("counter", "bank"): "financial_institution",
    ("manager", "bank"): "financial_institution",
    ("lobby", "bank"): "financial_institution",
    # bank / river_edge
    ("river", "bank"): "river_edge",
    ("fisherman", "bank"): "river_edge",
    ("water", "bank"): "river_edge",
    ("shore", "bank"): "river_edge",
    ("mud", "bank"): "river_edge",
    ("stream", "bank"): "river_edge",
    ("current", "bank"): "river_edge",
    ("boat", "bank"): "river_edge",
    ("bridge", "bank"): "river_edge",
    ("reeds", "bank"): "river_edge",
    ("path", "bank"): "river_edge",
    # bat / animal
    ("cave", "bat"): "animal",
    ("flew", "bat"): "animal",
    ("flying", "bat"): "animal",
    ("wings", "bat"): "animal",
    ("night", "bat"): "animal",
    ("animal", "bat"): "animal",
    ("hanging", "bat"): "animal",
    ("attic", "bat"): "animal",
    ("dusk", "bat"): "animal",
    ("eaves", "bat"): "animal",
    # bat / sports_equipment
    ("baseball", "bat"): "sports_equipment",
    ("player", "bat"): "sports_equipment",
    ("hit", "bat"): "sports_equipment",
    ("cracked", "bat"): "sports_equipment",
    ("swing", "bat"): "sports_equipment",
    ("game", "bat"): "sports_equipment",
    ("batter", "bat"): "sports_equipment",
    ("plate", "bat"): "sports_equipment",
    ("dugout", "bat"): "sports_equipment",
    ("swung", "bat"): "sports_equipment",
    ("lighter", "bat"): "sports_equipment",
    ("pitcher", "bat"): "sports_equipment",   # present in accepted overlay
    # seal / animal
    ("ocean", "seal"): "animal",
    ("fish", "seal"): "animal",
    ("zoo", "seal"): "animal",
    ("flippers", "seal"): "animal",
    ("animal", "seal"): "animal",
    ("water", "seal"): "animal",
    ("trainer", "seal"): "animal",
    ("treat", "seal"): "animal",
    ("slid", "seal"): "animal",
    ("pier", "seal"): "animal",
    ("tourists", "seal"): "animal",
    ("splash", "seal"): "animal",
    ("swimming", "seal"): "animal",           # present in accepted overlay
    ("coast", "seal"): "animal",              # near-synonym of "shore"
    # seal / closure_stamp
    ("envelope", "seal"): "closure_stamp",
    ("document", "seal"): "closure_stamp",
    ("stamp", "seal"): "closure_stamp",
    ("wax", "seal"): "closure_stamp",
    ("official", "seal"): "closure_stamp",
    ("package", "seal"): "closure_stamp",
    ("clerk", "seal"): "closure_stamp",
    ("closing", "seal"): "closure_stamp",
    ("parcel", "seal"): "closure_stamp",
    ("flap", "seal"): "closure_stamp",
    ("label", "seal"): "closure_stamp",
    # crane / bird
    ("bird", "crane"): "bird",
    ("wings", "crane"): "bird",
    ("wing", "crane"): "bird",                # singular; also present in overlay
    ("marsh", "crane"): "bird",
    ("flew", "crane"): "bird",
    ("nest", "crane"): "bird",
    ("lake", "crane"): "bird",
    ("dawn", "crane"): "bird",
    ("reeds", "crane"): "bird",
    ("neck", "crane"): "bird",
    ("photographer", "crane"): "bird",
    ("wetland", "crane"): "bird",             # present in accepted overlay
    # crane / machine
    ("construction", "crane"): "machine",
    ("building", "crane"): "machine",
    ("lifted", "crane"): "machine",
    ("steel", "crane"): "machine",
    ("operator", "crane"): "machine",
    ("site", "crane"): "machine",
    ("foreman", "crane"): "machine",
    ("crew", "crane"): "machine",
    ("hook", "crane"): "machine",
    ("load", "crane"): "machine",
    ("lift", "crane"): "machine",
    # rock / stone
    ("mountain", "rock"): "stone",
    ("stone", "rock"): "stone",
    ("heavy", "rock"): "stone",
    ("ground", "rock"): "stone",
    ("cliff", "rock"): "stone",
    ("trail", "rock"): "stone",
    ("boulder", "rock"): "stone",
    # rock / music
    ("band", "rock"): "music",
    ("guitar", "rock"): "music",
    ("concert", "rock"): "music",
    ("song", "rock"): "music",
    ("drummer", "rock"): "music",
    ("stage", "rock"): "music",
    ("crowd", "rock"): "music",
    ("louder", "rock"): "music",
    ("venue", "rock"): "music",
    # spring / season
    ("flowers", "spring"): "season",
    ("april", "spring"): "season",
    ("warm", "spring"): "season",
    ("weather", "spring"): "season",
    ("garden", "spring"): "season",
    ("rain", "spring"): "season",
    ("thaw", "spring"): "season",
    ("mornings", "spring"): "season",
    ("warmed", "spring"): "season",
    # spring / coil
    ("metal", "spring"): "coil",
    ("compressed", "spring"): "coil",
    ("mechanism", "spring"): "coil",
    ("tension", "spring"): "coil",
    ("bounce", "spring"): "coil",
    ("device", "spring"): "coil",
    ("latch", "spring"): "coil",
    ("handle", "spring"): "coil",
    ("snapped", "spring"): "coil",            # from gen context ("snapped back")
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

# GeneralizedQuestionAnalyzer v1 — new phrasing patterns
# "SENTENCE. What does TERM mean?" where SENTENCE contains contextual cues
_SENTENCE_WHAT_DOES_RE = re.compile(
    r"^(.+?)\.\s+[Ww]hat\s+does\s+\w+\s+mean\??\s*$",
    re.DOTALL,
)

# "SENTENCE. What kind of TERM is it?"
_SENTENCE_WHAT_KIND_RE = re.compile(
    r"^(.+?)\.\s+[Ww]hat\s+kind\s+of\s+\w+\s+is\s+it\??\s*$",
    re.DOTALL,
)

# "Why does/do [an?] CUE point to TERM ..."
_POINT_TO_CUE_RE = re.compile(
    r"[Ww]hy\s+do(?:es)?\s+(?:an?\s+)?(\w+)\s+point\s+to",
    re.IGNORECASE,
)

# "Why does/do [an?] CUE suggest/indicate/... TERM"
_SUGGEST_CUE_RE = re.compile(
    r"[Ww]hy\s+do(?:es)?\s+(?:an?\s+)?(\w+)\s+(?:suggest|indicate|signal|relate)",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize_lower(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text.lower()))


def _find_term(text: str) -> Optional[str]:
    """Return the first known term found in the lowercased text."""
    lower = text.lower()
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
    for label in sorted(labels, key=len, reverse=True):
        if label in text:
            return labels[label]
    return None


def _extract_cue(question: str) -> Optional[str]:
    """Extract the cue token from an explain_cue question."""
    m = _CUE_IN_QUOTES_RE.search(question)
    if m:
        return m.group(1).strip().lower()

    # "Why does/do [an?] WORD point to TERM ..."
    m_pt = _POINT_TO_CUE_RE.search(question)
    if m_pt:
        return m_pt.group(1).strip().lower()

    # "Why does/do [an?] WORD suggest/indicate/..."
    m_sg = _SUGGEST_CUE_RE.search(question)
    if m_sg:
        return m_sg.group(1).strip().lower()

    return None


def _detect_inline_context(question: str) -> Optional[str]:
    """Extract the embedded sentence from In '...' classify_context questions."""
    m = _INLINE_CONTEXT_RE.search(question)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"""[Ii]n\s+['""](.+?)['""]""", question, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return None


def _resolve_context_sense(inline_ctx: str, term: str) -> Optional[str]:
    """Score inline_ctx against _CONTEXT_CUE_TO_SENSE for a given term.

    Returns the unique matching sense_id if exactly one sense is supported,
    or None if zero or multiple senses are supported (ambiguous / conflict).
    This is used to route unambiguous new phrasings to define_sense so the
    renderer emits the rich description format expected by the benchmark.
    """
    tokens = _tokenize_lower(inline_ctx)
    sense_hits: dict[str, bool] = {}
    for (cue, t), sense_id in _CONTEXT_CUE_TO_SENSE.items():
        if t == term and cue in tokens:
            sense_hits[sense_id] = True

    hit_senses = list(sense_hits)
    return hit_senses[0] if len(hit_senses) == 1 else None


def _extract_sentence_before_what_does(question: str) -> Optional[str]:
    """Return the leading sentence from 'SENTENCE. What does TERM mean?' format."""
    m = _SENTENCE_WHAT_DOES_RE.match(question)
    if not m:
        return None
    sentence = m.group(1).strip()
    # Only proceed if a known term is present in the leading sentence
    if _find_term(sentence) is None:
        return None
    return sentence


def _extract_sentence_before_what_kind(question: str) -> Optional[str]:
    """Return the leading sentence from 'SENTENCE. What kind of TERM is it?' format."""
    m = _SENTENCE_WHAT_KIND_RE.match(question)
    if not m:
        return None
    sentence = m.group(1).strip()
    if _find_term(sentence) is None:
        return None
    return sentence


def _extract_is_a_or_b_subject(question: str) -> Optional[str]:
    """Extract the subject phrase from 'Is SUBJECT [prep CUE] an A or B?' questions.

    Returns only the part before 'an A or B' to avoid polluting the context
    with sense-label tokens (e.g. 'animal', 'sports equipment').
    """
    lower = question.lower()
    if not lower.startswith("is "):
        return None
    or_idx = lower.rfind(" or ")
    if or_idx < 0:
        return None
    before_or = lower[:or_idx]
    # Find the rightmost " an " or " a " which precedes the sense options
    an_idx = before_or.rfind(" an ")
    a_idx = before_or.rfind(" a ")
    boundary = max(an_idx, a_idx)
    if boundary < 0:
        return None
    subject = question[3:boundary].strip()  # skip leading "Is "
    if not subject or _find_term(subject) is None:
        return None
    return subject


def _analyze_new_context_pattern(question: str) -> Optional[tuple[str, Optional[str]]]:
    """Match new context-bearing phrasings and resolve the inline context.

    Returns (inline_ctx, sense_id_or_None) where:
      - sense_id is the unique winning sense when context is unambiguous
      - sense_id is None when context is ambiguous/conflicting

    Returns None when no new phrasing pattern matches.
    The original In '...' classify_context format is handled separately.
    """
    inline_ctx: Optional[str] = None

    inline_ctx = _extract_sentence_before_what_kind(question)
    if inline_ctx is None:
        inline_ctx = _extract_sentence_before_what_does(question)
    if inline_ctx is None:
        inline_ctx = _extract_is_a_or_b_subject(question)

    if inline_ctx is None:
        return None

    term = _find_term(inline_ctx) or _find_term(question)
    if term is None:
        return None

    sense_id = _resolve_context_sense(inline_ctx, term)
    return (inline_ctx, sense_id)


def _is_distinguish_question(question_lower: str) -> bool:
    return any(p in question_lower for p in (
        "difference between", "different from", "how is", "compare",
        "distinguish", "versus", " vs ",
    ))


def _is_explain_cue_question(question_lower: str) -> bool:
    has_trigger = any(p in question_lower for p in (
        "why does", "why do", "how does", "what makes",
    ))
    if not has_trigger:
        return False
    has_relation = any(p in question_lower for p in (
        "suggest", "indicate", "signal", "relate", "point to",
    ))
    return has_relation


def _extract_distinguish_senses(question: str) -> tuple[Optional[str], Optional[str]]:
    """Extract two senses from a distinguish_senses question."""
    lower = question.lower()
    terms = _find_all_terms(lower)
    if not terms:
        return None, None

    # " with " added for "Compare X with Y." phrasings
    for sep in (" and ", " with ", " from ", " versus ", " vs ", " or "):
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

    # ── classify_context: original In '...' format (main benchmark) ─────────
    inline_ctx = _detect_inline_context(question)
    if inline_ctx:
        term = _find_term(inline_ctx) or _find_term(question)
        return AnalyzedQuestion(
            question=question,
            intent="classify_context",
            term=term,
            target_sense=None,
            second_sense=None,
            cues_in_question=[],
            inline_context=inline_ctx,
            underconstrained=term is None,
        )

    # ── new context-bearing phrasings (generalized forms) ───────────────────
    new_ctx = _analyze_new_context_pattern(question)
    if new_ctx is not None:
        inline_ctx, sense_id = new_ctx
        term = _find_term(inline_ctx) or _find_term(question)
        if sense_id is not None:
            # Unambiguous context → define_sense for the rich description format
            return AnalyzedQuestion(
                question=question,
                intent="define_sense",
                term=term,
                target_sense=sense_id,
                second_sense=None,
                cues_in_question=[],
                inline_context=inline_ctx,
                underconstrained=term is None,
            )
        else:
            # Ambiguous/conflicting context → classify_context for planner scoring
            return AnalyzedQuestion(
                question=question,
                intent="classify_context",
                term=term,
                target_sense=None,
                second_sense=None,
                cues_in_question=[],
                inline_context=inline_ctx,
                underconstrained=term is None,
            )

    # ── distinguish_senses ───────────────────────────────────────────────────
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

    # ── explain_cue ──────────────────────────────────────────────────────────
    if _is_explain_cue_question(lower):
        term = _find_term(lower)
        cue = _extract_cue(question)
        sense_id_ec: Optional[str] = None
        if term:
            m = re.search(r"\bas\s+(?:a\s+|an\s+)?(.+?)(?:\?|$)", lower)
            if m:
                sense_id_ec = _sense_from_label(m.group(1), term)
            if not sense_id_ec:
                sense_id_ec = _sense_from_qualifiers(lower, term)
        return AnalyzedQuestion(
            question=question,
            intent="explain_cue",
            term=term,
            target_sense=sense_id_ec,
            second_sense=None,
            cues_in_question=[cue] if cue else [],
            inline_context=None,
            underconstrained=(term is None or cue is None or sense_id_ec is None),
        )

    # ── unknown_or_ambiguous ("What does X mean?") ───────────────────────────
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

    # ── define_sense ("What is a/an [qualifier] [term]?") ────────────────────
    term = _find_term(lower)
    if term:
        sense_id_ds = _sense_from_qualifiers(lower, term)
        if sense_id_ds:
            cue = _extract_cue(question)
            return AnalyzedQuestion(
                question=question,
                intent="define_sense",
                term=term,
                target_sense=sense_id_ds,
                second_sense=None,
                cues_in_question=[cue] if cue else [],
                inline_context=None,
                underconstrained=False,
            )

    # ── fallback: unknown_or_ambiguous ───────────────────────────────────────
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
