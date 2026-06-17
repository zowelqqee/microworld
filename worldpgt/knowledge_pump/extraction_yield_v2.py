"""Knowledge Pump Extraction Yield v2.

Targeted deterministic patterns for relations and definitions that the v1
extractor misses, primarily because the non-greedy _E pattern in
relation_patterns.py captures the first word of a descriptor phrase instead
of the actual entity name.

Root cause fixed:
  "was founded by serial startup entrepreneur Steve Blank" →
  v1 captures B="serial", quarantined for missing_explicit_evidence.
  v2 captures raw span after "by", then extracts proper-name sequences.

Pattern groups:
  1. founded_by_v2    — greedy raw capture + person-name extraction
  2. subsidiary_v2   — allows modifier words before "subsidiary of"
  3. service_of_v2   — same as subsidiary but for division/arm/unit/branch
  4. lead_definition_v2 — first sentence only, stable risk
  5. publishes_named_v2 — "publishes … including the [adj] NAMED_PUBLICATION"
  6. publishes_direct_v2 — "publishes [the] [adj] PUBLICATION_NAME"
  7. developed_by_v2  — same descriptor-stripping fix as founded_by

Output: list of overlay-format dicts compatible with merge_overlay_items().
Pass these as additional fresh_snapshot_overlay_items in the pump runner
(--enable-extraction-yield-v2 flag).

No network calls. No ML. Does not modify accepted memory or overlays.
"""

from __future__ import annotations

import re
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, TYPE_CHECKING

from worldpgt.relation_extraction_v2.sentence_splitter import (
    extract_full_body,
    split_paragraphs,
    split_sentences,
)

if TYPE_CHECKING:
    from worldpgt.wiki_snapshot_ingestion.types import ReadySnapshotDoc


# ---------------------------------------------------------------------------
# Safety screens — same policy as relation_patterns.py
# ---------------------------------------------------------------------------

_NEGATION_RE = re.compile(
    r"\b(?:not|never|no\s+longer|wasn't|weren't|isn't|aren't|hadn't"
    r"|cannot|can't|won't|wouldn't|shouldn't|couldn't|mustn't)\b",
    re.IGNORECASE,
)
_CURRENT_LIVE_RE = re.compile(
    r"\b(?:currently|today|right\s+now|as\s+of\s+today"
    r"|current\s+(?:ceo|president|chairman|chief|leader|head|director|owner)"
    r"|stock\s+price|share\s+price|market\s+cap(?:italization)?"
    r"|net\s+worth|latest\s+(?:revenue|ranking)"
    r"|quarterly\s+revenue|live\s+(?:market|price|data))\b",
    re.IGNORECASE,
)
_PRIVATE_RE = re.compile(
    r"\b(?:phone\s+number|home\s+address|private\s+email"
    r"|date\s+of\s+birth|social\s+security|passport\s+number)\b",
    re.IGNORECASE,
)
_UNIVERSAL_RE = re.compile(
    r"\b(?:all\s+\w+\s+are\b|is\s+every\b|are\s+all\b|every\s+\w+\s+is\b)\b",
    re.IGNORECASE,
)

# Volatile years: 2023-2029 — not stable facts
_VOLATILE_YEAR_RE = re.compile(r"\b(202[3-9])\b")


def screen_sentence(sentence: str) -> bool:
    """Return True if sentence passes all safety screens (safe to extract from)."""
    if _NEGATION_RE.search(sentence):
        return False
    if _CURRENT_LIVE_RE.search(sentence):
        return False
    if _PRIVATE_RE.search(sentence):
        return False
    if _UNIVERSAL_RE.search(sentence):
        return False
    if _VOLATILE_YEAR_RE.search(sentence):
        return False
    return True


# ---------------------------------------------------------------------------
# Fragment / generic subject rejection
# ---------------------------------------------------------------------------

_FRAGMENT_STARTERS = frozenset({
    "in", "on", "at", "from", "when", "what", "while", "with", "after",
    "before", "during", "following", "therefore", "although", "however",
    "since", "is", "are", "was", "were", "they", "this", "that",
    "he", "she", "it", "his", "her", "their", "its",
    # Participial / adverbial starters (section headers, apposition context)
    "headquartered", "incorporated", "established", "founded",
    "known", "located", "based", "owned", "operated", "history",
    "background", "overview", "today", "currently", "previously",
    "originally", "formerly", "initially", "eventually",
    # Temporal / discourse starters
    "by", "as", "under", "through", "between", "among", "within",
    "alongside", "together", "later", "soon", "also", "meanwhile",
    "however", "nonetheless", "instead", "further", "moreover",
    "if", "many",
})

# Subjects that are generic nouns, not proper entities.
_GENERIC_SUBJECTS = frozenset({
    "company", "corporation", "manufacturer", "provider", "service",
    "platform", "organization", "entity", "firm", "startup", "system",
    "program", "project", "network", "model", "product", "vehicle",
    "technology", "business", "enterprise", "group", "consortium",
    "association", "agency", "institute", "foundation",
})

# Subject prefixes that are always clause fragments (multi-word starters).
_FRAGMENT_PREFIX_RE = re.compile(
    r"^(?:the\s+(?:original|former|late|new|old|current|first|second)\b"
    r"|the\s+following\b"
    r"|history\s+of\b"
    r")",
    re.IGNORECASE,
)


def is_fragment_subject(s: str) -> bool:
    """Return True if subject string looks like a sentence fragment starter."""
    if not s:
        return True
    words = s.strip().lower().split()
    if not words:
        return True
    if words[0] in _FRAGMENT_STARTERS:
        return True
    if _FRAGMENT_PREFIX_RE.match(s):
        return True
    return False


def is_generic_subject(s: str) -> bool:
    """Return True if subject is a generic noun (not a specific entity name)."""
    norm = s.strip().lower()
    if norm in _GENERIC_SUBJECTS:
        return True
    # "The/A <generic>" forms (e.g. "The company", "A subsidiary")
    words = norm.split()
    if len(words) == 2 and words[0] in ("the", "a", "an") and words[1] in _GENERIC_SUBJECTS:
        return True
    return False


def _canonical_subject(s: str) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", (s or "").strip(), flags=re.IGNORECASE)


def _ambiguous_single_word_subject(subject: str, source_page: str) -> bool:
    subj = _canonical_subject(subject)
    page = _canonical_subject(source_page.replace("_", " "))
    return (
        len(subj.split()) == 1
        and len(page.split()) > 1
        and subj.lower() != page.lower()
        and subj.lower() in page.lower().split()
    )


# ---------------------------------------------------------------------------
# Person name extraction from a raw "after by/for/by" span
# ---------------------------------------------------------------------------

# Matches a proper name sequence: 1-4 capitalized words.
# Descriptor/title words are lowercase → automatically excluded by [A-Z] anchor.
_PERSON_SEQ_RE = re.compile(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}")
_ENTITY_SEQ_RE = re.compile(
    r"[A-Z][A-Za-z0-9&.'\-]*(?:\s+(?:&\s+)?[A-Z][A-Za-z0-9&.'\-]*){0,5}"
)


def extract_person_names(raw: str) -> list[str]:
    """Find proper person names (capitalized word sequences) in a raw span.

    Automatically strips leading descriptor/title words because they are
    lowercase and the pattern only starts at an uppercase letter.

    Examples::

        extract_person_names("serial startup entrepreneur Steve Blank")
            → ["Steve Blank"]
        extract_person_names("Michael Bloomberg and Matthew Winkler")
            → ["Michael Bloomberg", "Matthew Winkler"]
        extract_person_names("Elon Musk") → ["Elon Musk"]
    """
    clean = re.sub(r"\s+in\s+\d{4}\s*$", "", raw.strip()).rstrip(".,; ")
    names = _PERSON_SEQ_RE.findall(clean)
    multi_word = [n for n in names if len(n.split()) >= 2]
    if multi_word:
        return multi_word
    # Fall back: accept single capitalized word if nothing else found
    return [n for n in names if n]


def extract_named_entities(raw: str) -> list[str]:
    """Extract capitalized organization/person-like spans from an object clause."""
    clean = re.sub(r"\s+in\s+\d{4}\s*$", "", raw.strip()).rstrip(".,; ")
    clean = re.sub(r"^(?:the|a|an)\s+", "", clean, flags=re.IGNORECASE)
    names = []
    for name in _ENTITY_SEQ_RE.findall(clean):
        stripped = name.strip(" ,.;")
        if stripped and stripped.lower() not in {"the", "a", "an"}:
            names.append(stripped)
    return names


# ---------------------------------------------------------------------------
# Definition truncation helpers
# ---------------------------------------------------------------------------

_DEF_TRUNCATE_RE = re.compile(
    r"\s+(?:and\b|or\b|that\b|which\b|who\b|with\b|through\b"
    r"|founded\b|based\b|known\b|located\b|headquartered\b"
    r"|developed\b|owned\b|operated\b|manufactured\b"
    r"|serving\b|offering\b|also\b|originally\b|currently\b|formerly\b"
    r")",
    re.IGNORECASE,
)

_BROAD_DEFINITIONS = frozenset({
    "artificial intelligence",
    "american infrastructure",
    "good thing",
})

_BAD_DEFINITION_PHRASES = (
    "award winner",
    "entrepreneur of the year",
    "third son",
    "good thing",
    "proposal to rename",
    "winner in the",
)

_BAD_DEFINITION_HEADS = frozenset({
    "artificial", "american", "british", "canadian", "german", "french",
})

_GOOD_NATIONALITY_HEADS = frozenset({
    "company", "manufacturer", "studio", "organization", "agency", "engineer",
    "entrepreneur", "publication", "newspaper", "provider", "service",
})


def _truncate_definition(defn: str) -> str:
    """Truncate definition at clause-starter words, return stripped result."""
    m = _DEF_TRUNCATE_RE.search(defn)
    if m:
        return defn[: m.start()].rstrip(" ,")
    return defn.rstrip(" ,")


def _definition_ok(defn: str) -> bool:
    norm = " ".join((defn or "").lower().split())
    if not norm:
        return False
    if norm in _BROAD_DEFINITIONS:
        return False
    if any(phrase in norm for phrase in _BAD_DEFINITION_PHRASES):
        return False
    words = norm.split()
    if len(words) < 2:
        return False
    if words[0] in _BAD_DEFINITION_HEADS and not any(
        head in words[1:] for head in _GOOD_NATIONALITY_HEADS
    ):
        return False
    return True


# ---------------------------------------------------------------------------
# Overlay item factories
# ---------------------------------------------------------------------------

_CANDIDATE_SOURCE = "pump_extraction_yield_v2"


def _make_relation(
    subject: str,
    predicate: str,
    obj: str,
    source_page: str,
    evidence_text: str,
    pattern_id: str,
    risk: str = "medium",
    stability: str = "semi_stable",
) -> dict:
    return {
        "overlay_type": "overlay_relation",
        "subject": subject.strip(),
        "predicate": predicate,
        "object": obj.strip(),
        "source_page": source_page,
        "evidence_text": evidence_text,
        "trust": "overlay_candidate",
        "risk": risk,
        "stability": stability,
        "candidate_source": _CANDIDATE_SOURCE,
        "extraction_pattern": pattern_id,
        "v2_pattern_id": pattern_id,
        "confidence_label": "pattern_explicit",
        "evidence_span": evidence_text,
    }


def _make_definition(
    subject: str,
    definition: str,
    source_page: str,
    evidence_text: str,
    pattern_id: str,
) -> dict:
    return {
        "overlay_type": "overlay_definition",
        "subject": subject.strip(),
        "definition": definition.strip(),
        "predicate": "is_a",
        "source_page": source_page,
        "evidence_text": evidence_text,
        "trust": "overlay_candidate",
        "risk": "low",
        "stability": "stable",
        "candidate_source": _CANDIDATE_SOURCE,
        "extraction_pattern": pattern_id,
        "v2_pattern_id": pattern_id,
        "confidence_label": "pattern_explicit",
        "evidence_span": evidence_text,
    }


# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------

# Group 1 — founded_by_v2
# Captures everything after "by" up to "in YYYY" or a clause boundary.
# Post-processing extracts proper name sequences (handles descriptor stripping).
_FOUNDED_BY_V2 = re.compile(
    r"\b(?P<subj>[A-Z][A-Za-z0-9 ,.'&\-]{1,59}?)"
    r"\s+was\s+founded\s+(?:in\s+\d{4}\s+)?by\s+"
    r"(?P<raw_obj>[A-Za-z][^.!?\n]{0,120}?)"
    r"(?=\s+in\s+\d{4}|\s*[.,;]|$)",
    re.IGNORECASE,
)

# Group 2 — subsidiary_of_v2
# Allows arbitrary lowercase modifier words between "a" and "subsidiary of".
_SUBSIDIARY_V2 = re.compile(
    r"\b(?P<subj>[A-Z][A-Za-z0-9 ,.'&\-]{1,59}?)"
    r"\s+is\s+a\s+"
    r"(?:[a-z][A-Za-z0-9 ,]{0,80}?\s+)?subsidiary\s+of\s+"
    r"(?P<obj>[A-Z][A-Za-z0-9 ,.'&\-]{1,59}?)"
    r"(?=[,. ]|$)",
)

# Group 3 — service_of_v2
# Same modifier-tolerance pattern but for division / arm / unit / branch.
_SERVICE_OF_V2 = re.compile(
    r"\b(?P<subj>[A-Z][A-Za-z0-9 ,.'&\-]{1,59}?)"
    r"\s+is\s+a\s+"
    r"(?:[a-z][A-Za-z0-9 ,]{0,80}?\s+)?(?:division|service|arm|unit|branch)\s+of\s+"
    r"(?P<obj>[A-Z][A-Za-z0-9 ,.'&\-]{1,59}?)"
    r"(?=[,. ]|$)",
)

# Group 4 — lead_definition_v2  (applied to first sentence only)
# Matches "ENTITY [optional (abbr)] is/was/are/were a/an/the DEFINITION_PHRASE"
# and stops the definition at clause boundaries.
_LEAD_DEF_V2 = re.compile(
    r"(?P<subj>[A-Z][A-Za-z0-9 ,.'&\-]{1,60}?)"
    r"(?:\s*\([^)]{0,30}\))?"       # optional abbreviation: (XYZ)
    r"\s+(?:is|are|was|were)\s+(?:a|an|the)\s+"
    r"(?P<defn>[A-Za-z][A-Za-z0-9 \-']{2,80}?)"
    r"(?=[,.]"
    r"|\s+(?:and\b|that\b|which\b|who\b|founded\b|based\b|known\b"
    r"|located\b|headquartered\b|serving\b|formerly\b|also\b)"
    r"|$)",
)

# Group 5a — publishes_named_v2
# Targets "X publishes … including the [adj] NAMED_PUBLICATION".
_PUBLISHES_NAMED_V2 = re.compile(
    r"\b(?P<subj>[A-Z][A-Za-z0-9 ,.'&\-]{1,59}?)"
    r"\s+publishes?\s+.{0,120}?including\s+the\s+"
    r"(?:[a-z]+\s+)?(?P<obj>[A-Z][A-Za-z0-9 ]{1,59}?)"
    r"(?=[,.]|$)",
    re.IGNORECASE,
)

# Group 5b — publishes_direct_v2
# Targets "X publishes [the] [adj]* NAMED_WORK" (direct form, no "including").
_PUBLISHES_DIRECT_V2 = re.compile(
    r"\b(?P<subj>[A-Z][A-Za-z0-9 ,.'&\-]{1,59}?)"
    r"\s+publishes?\s+(?:the\s+)?(?:[a-z]+\s+){0,2}"
    r"(?P<obj>[A-Z][A-Za-z0-9 ]{1,59}?)"
    r"(?=[,.]|$)",
)

# Group 6 — developed_by_v2
# Same greedy raw-object approach as founded_by_v2 but for development verbs.
_DEVELOPED_BY_V2 = re.compile(
    r"\b(?P<subj>[A-Z][A-Za-z0-9 ,.'&\-]{1,59}?)"
    r"\s+(?:was|is|were)\s+(?:developed|created|built|designed)\s+by\s+"
    r"(?P<raw_obj>[A-Za-z][^.!?\n]{0,120}?)"
    r"(?=\s+in\s+\d{4}|\s*[.,;]|$)",
    re.IGNORECASE,
)

# Group 6b — passive_by_v2
_PASSIVE_BY_V2 = re.compile(
    r"\b(?P<subj>[A-Z][A-Za-z0-9 ,.'&\-]{1,59}?)"
    r"\s+(?:was|is|were)\s+"
    r"(?P<verb>manufactured|operated|owned|acquired)\s+by\s+"
    r"(?P<raw_obj>[A-Za-z][^.!?\n]{0,120}?)"
    r"(?=\s+in\s+\d{4}|\s*[.,;]|$)",
    re.IGNORECASE,
)

_ACQUIRED_ACTIVE_V2 = re.compile(
    r"\b(?P<subj>[A-Z][A-Za-z0-9 ,.'&\-]{1,59}?)"
    r"\s+acquired\s+"
    r"(?P<obj>[A-Z][A-Za-z0-9 ,.'&\-]{1,59}?)"
    r"(?=\s+in\s+\d{4}|\s*[.,;]|$)",
)

# Group 7 — headquartered_in_v2  (optional; low risk)
_HEADQUARTERED_V2 = re.compile(
    r"\b(?P<subj>[A-Z][A-Za-z0-9 ,.'&\-]{1,59}?)"
    r"\s+(?:is\s+)?headquartered\s+(?:in|at)\s+"
    r"(?P<obj>[A-Z][A-Za-z, ]{1,59}?)"
    r"(?=[,.]|$)",
)


# ---------------------------------------------------------------------------
# Per-sentence extraction
# ---------------------------------------------------------------------------

def extract_from_sentence(
    sentence: str,
    source_page: str,
    *,
    is_lead: bool = False,
    stats: Counter | None = None,
) -> list[dict]:
    """Apply all v2 patterns to one sentence and return overlay items.

    ``is_lead`` must be True for the first sentence of the first paragraph
    (the only context where lead_definition_v2 fires).
    ``stats`` is an optional Counter updated in place for diagnostics.
    """
    if not screen_sentence(sentence):
        if stats is not None:
            stats["screened_out"] += 1
        return []

    items: list[dict] = []

    def _bump(key: str) -> None:
        if stats is not None:
            stats[key] += 1

    # -- Group 1: founded_by_v2 -----------------------------------------------
    for m in _FOUNDED_BY_V2.finditer(sentence):
        subj = _canonical_subject(m.group("subj"))
        if not subj or not subj[0].isupper():
            continue
        if is_fragment_subject(subj) or is_generic_subject(subj):
            continue
        raw_obj = m.group("raw_obj").strip()
        names = extract_person_names(raw_obj)
        for name in names:
            if len(name.split()) < 2:
                continue  # require at least first + last name
            items.append(_make_relation(subj, "founded_by", name, source_page, sentence, "founded_by_v2"))
            _bump("founded_by_v2")

    # -- Group 2: subsidiary_of_v2 --------------------------------------------
    for m in _SUBSIDIARY_V2.finditer(sentence):
        subj = _canonical_subject(m.group("subj"))
        obj = m.group("obj").strip()
        if not subj or not subj[0].isupper():
            continue
        if is_fragment_subject(subj) or is_generic_subject(subj):
            continue
        if not obj or not obj[0].isupper():
            continue
        items.append(_make_relation(subj, "subsidiary_of", obj, source_page, sentence, "subsidiary_of_v2"))
        _bump("subsidiary_of_v2")

    # -- Group 3: service_of_v2 -----------------------------------------------
    for m in _SERVICE_OF_V2.finditer(sentence):
        subj = _canonical_subject(m.group("subj"))
        obj = m.group("obj").strip()
        if not subj or not subj[0].isupper():
            continue
        if is_fragment_subject(subj) or is_generic_subject(subj):
            continue
        if not obj or not obj[0].isupper():
            continue
        items.append(_make_relation(subj, "service_of", obj, source_page, sentence, "service_of_v2"))
        _bump("service_of_v2")

    # -- Group 4: lead_definition_v2  (first sentence of first para only) -----
    if is_lead:
        for m in _LEAD_DEF_V2.finditer(sentence):
            if m.start() > 10:
                # Subject must start near the beginning of the sentence.
                continue
            subj = _canonical_subject(m.group("subj"))
            if not subj or not subj[0].isupper():
                continue
            if is_fragment_subject(subj) or is_generic_subject(subj):
                continue
            if _ambiguous_single_word_subject(subj, source_page):
                continue
            defn_raw = m.group("defn").strip()
            defn = _truncate_definition(defn_raw)
            if not _definition_ok(defn):
                continue
            items.append(_make_definition(subj, defn, source_page, sentence, "lead_definition_v2"))
            _bump("lead_definition_v2")

    # -- Group 5a: publishes_named_v2 -----------------------------------------
    for m in _PUBLISHES_NAMED_V2.finditer(sentence):
        subj = _canonical_subject(m.group("subj"))
        obj = m.group("obj").strip()
        if not subj or not subj[0].isupper():
            continue
        if is_fragment_subject(subj) or is_generic_subject(subj):
            continue
        if not obj or not obj[0].isupper():
            continue
        items.append(_make_relation(subj, "publishes", obj, source_page, sentence, "publishes_named_v2"))
        _bump("publishes_named_v2")

    # -- Group 5b: publishes_direct_v2 ----------------------------------------
    for m in _PUBLISHES_DIRECT_V2.finditer(sentence):
        subj = _canonical_subject(m.group("subj"))
        obj = m.group("obj").strip()
        if not subj or not subj[0].isupper():
            continue
        if is_fragment_subject(subj) or is_generic_subject(subj):
            continue
        if not obj or not obj[0].isupper():
            continue
        if "range of" in obj.lower():
            continue  # vague object — precision firewall would also catch this
        items.append(_make_relation(subj, "publishes", obj, source_page, sentence, "publishes_direct_v2"))
        _bump("publishes_direct_v2")

    # -- Group 6: developed_by_v2 ---------------------------------------------
    for m in _DEVELOPED_BY_V2.finditer(sentence):
        subj = _canonical_subject(m.group("subj"))
        if not subj or not subj[0].isupper():
            continue
        if is_fragment_subject(subj) or is_generic_subject(subj):
            continue
        raw_obj = m.group("raw_obj").strip()
        names = extract_named_entities(raw_obj)
        for name in names:
            if not name:
                continue
            items.append(_make_relation(subj, "developed_by", name, source_page, sentence, "developed_by_v2"))
            items.append(_make_relation(name, "develops", subj, source_page, sentence, "develops_inverse_v2"))
            _bump("developed_by_v2")

    # -- Group 6b: passive manufacture/operation/ownership/acquisition --------
    passive_predicates = {
        "manufactured": ("produces", "manufactured_by_v2", True),
        "operated": ("operated_by", "operated_by_v2", False),
        "owned": ("owned_by", "owned_by_v2", False),
        "acquired": ("owned_by", "acquired_by_v2", False),
    }
    for m in _PASSIVE_BY_V2.finditer(sentence):
        subj = _canonical_subject(m.group("subj"))
        verb = m.group("verb").lower()
        if not subj or not subj[0].isupper():
            continue
        if is_fragment_subject(subj) or is_generic_subject(subj):
            continue
        raw_obj = m.group("raw_obj").strip()
        names = extract_named_entities(raw_obj)
        if not names:
            continue
        predicate, pattern_id, inverse = passive_predicates[verb]
        for name in names:
            if inverse:
                items.append(_make_relation(name, predicate, subj, source_page, sentence, pattern_id))
            else:
                items.append(_make_relation(subj, predicate, name, source_page, sentence, pattern_id))
            _bump(pattern_id)

    for m in _ACQUIRED_ACTIVE_V2.finditer(sentence):
        subj = _canonical_subject(m.group("subj"))
        obj = m.group("obj").strip()
        if not subj or not obj or not subj[0].isupper() or not obj[0].isupper():
            continue
        if is_fragment_subject(subj) or is_generic_subject(subj):
            continue
        if is_fragment_subject(obj) or is_generic_subject(obj):
            continue
        items.append(_make_relation(obj, "owned_by", subj, source_page, sentence, "acquired_active_v2"))
        _bump("acquired_active_v2")

    # -- Group 7: headquartered_in_v2 (optional, low risk) --------------------
    for m in _HEADQUARTERED_V2.finditer(sentence):
        subj = _canonical_subject(m.group("subj"))
        obj = m.group("obj").strip().rstrip(",")
        if not subj or not subj[0].isupper():
            continue
        if is_fragment_subject(subj) or is_generic_subject(subj):
            continue
        if not obj or not obj[0].isupper():
            continue
        items.append(
            _make_relation(
                subj, "headquartered_in", obj, source_page, sentence,
                "headquartered_in_v2", risk="low", stability="semi_stable",
            )
        )
        _bump("headquartered_in_v2")

    return items


# ---------------------------------------------------------------------------
# Doc-level extraction
# ---------------------------------------------------------------------------

def extract_from_doc(
    doc: "ReadySnapshotDoc",
    *,
    max_sentences: int = 50,
    stats: Counter | None = None,
) -> list[dict]:
    """Extract overlay items from one snapshot doc.

    Reads the normalized markdown file. Only the first ``max_sentences``
    sentences are scanned. The very first sentence triggers the
    lead_definition_v2 pattern; all others do not.
    """
    text = Path(doc.normalized_doc_path).read_text(encoding="utf-8")
    body = extract_full_body(text)
    paragraphs = split_paragraphs(body)

    items: list[dict] = []
    sentences_scanned = 0
    seen: set[tuple] = set()

    for para_idx, para in enumerate(paragraphs):
        if sentences_scanned >= max_sentences:
            break
        for sent_idx, sentence in enumerate(split_sentences(para)):
            if sentences_scanned >= max_sentences:
                break
            sentences_scanned += 1
            is_lead = para_idx == 0 and sent_idx == 0

            for item in extract_from_sentence(
                sentence, doc.title, is_lead=is_lead, stats=stats
            ):
                otype = item.get("overlay_type", "")
                if otype == "overlay_relation":
                    key = (
                        "r",
                        item["subject"].lower(),
                        item["predicate"],
                        item["object"].lower(),
                    )
                elif otype == "overlay_definition":
                    key = ("d", item["subject"].lower())
                else:
                    key = (otype, str(item))
                if key not in seen:
                    seen.add(key)
                    items.append(item)

    return items


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_yield_v2(
    docs: "list[ReadySnapshotDoc]",
    index=None,  # EntitySurfaceIndex — reserved for future quality checks
    *,
    max_sentences_per_doc: int = 50,
) -> tuple[list[dict], dict]:
    """Extract relation and definition overlay items from all batch snapshot docs.

    ``index`` is accepted but not currently used; it is reserved for future
    surface-index-aware filtering without requiring an interface change.

    Returns ``(overlay_items, stats)`` where ``overlay_items`` is ready to be
    appended to ``fresh_snapshot_overlay_items`` in :func:`merge_with_fresh_deltas`.
    """
    all_items: list[dict] = []
    pattern_stats: Counter = Counter()
    docs_processed = 0
    docs_errors: list[str] = []

    # Cross-doc deduplication: same (subj, pred, obj) should not appear twice.
    seen_global: set[tuple] = set()

    for doc in docs:
        try:
            items = extract_from_doc(
                doc, max_sentences=max_sentences_per_doc, stats=pattern_stats
            )
            for item in items:
                otype = item.get("overlay_type", "")
                if otype == "overlay_relation":
                    gkey = (
                        "r",
                        item["subject"].lower(),
                        item["predicate"],
                        item["object"].lower(),
                    )
                elif otype == "overlay_definition":
                    gkey = ("d", item["subject"].lower())
                else:
                    gkey = (otype, str(item))
                if gkey not in seen_global:
                    seen_global.add(gkey)
                    all_items.append(item)
            docs_processed += 1
        except Exception as exc:
            docs_errors.append(f"{doc.title}: {exc}")

    relations = [i for i in all_items if i.get("overlay_type") == "overlay_relation"]
    definitions = [i for i in all_items if i.get("overlay_type") == "overlay_definition"]

    stats = {
        "docs_processed": docs_processed,
        "docs_errors": len(docs_errors),
        "docs_error_details": docs_errors[:10],
        "total_candidates": len(all_items),
        "relation_candidates": len(relations),
        "definition_candidates": len(definitions),
        "pattern_counts": dict(sorted(pattern_stats.items())),
        "predicates": sorted({i.get("predicate", "") for i in relations}),
        "network_calls": False,
        "candidate_source": _CANDIDATE_SOURCE,
    }

    return all_items, stats


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------

_CANDIDATE_FIELDS = [
    "overlay_type",
    "subject",
    "predicate",
    "object",
    "definition",
    "source_page",
    "evidence_text",
    "evidence_span",
    "stability",
    "risk",
    "trust",
    "candidate_source",
    "extraction_pattern",
    "confidence_label",
]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _candidate_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **item,
            "object": item.get("object", ""),
            "definition": item.get("definition", ""),
            "extraction_pattern": item.get("extraction_pattern") or item.get("v2_pattern_id", ""),
        }
        for item in items
    ]


def _by_source_page_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, Counter] = {}
    for item in items:
        page = str(item.get("source_page") or "(unknown)")
        bucket = counts.setdefault(page, Counter())
        bucket["candidate_count"] += 1
        if item.get("overlay_type") == "overlay_relation":
            bucket["relation_count"] += 1
        elif item.get("overlay_type") == "overlay_definition":
            bucket["definition_count"] += 1
    return [
        {
            "source_page": page,
            "candidate_count": bucket["candidate_count"],
            "relation_count": bucket["relation_count"],
            "definition_count": bucket["definition_count"],
        }
        for page, bucket in sorted(
            counts.items(), key=lambda kv: (-kv[1]["candidate_count"], kv[0])
        )
    ]


def _pattern_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(item.get("extraction_pattern") or item.get("v2_pattern_id") or "unknown")
        for item in items
    )
    return dict(sorted(counts.items()))


def build_artifact_summary(
    candidates: list[dict[str, Any]],
    stats: dict[str, Any],
    *,
    rejected: list[dict[str, Any]] | None = None,
    quarantine: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rejected = rejected or []
    quarantine = quarantine or []
    relation_count = sum(1 for item in candidates if item.get("overlay_type") == "overlay_relation")
    definition_count = sum(1 for item in candidates if item.get("overlay_type") == "overlay_definition")
    return {
        "extraction_yield_v2_enabled": True,
        "candidate_source": _CANDIDATE_SOURCE,
        "extraction_yield_v2_candidate_count": len(candidates),
        "extraction_yield_v2_relation_candidate_count": relation_count,
        "extraction_yield_v2_definition_candidate_count": definition_count,
        "extraction_yield_v2_rejected_count": len(rejected),
        "extraction_yield_v2_quarantined_count": len(quarantine),
        "extraction_yield_v2_candidate_by_pattern": _pattern_counts(candidates),
        "docs_processed": stats.get("docs_processed", 0),
        "docs_errors": stats.get("docs_errors", 0),
        "screened_out_sentence_count": stats.get("screened_out", 0),
        "network_calls": False,
    }


def write_extraction_yield_v2_artifacts(
    out_dir: str | Path,
    candidates: list[dict[str, Any]],
    stats: dict[str, Any],
    *,
    rejected: list[dict[str, Any]] | None = None,
    quarantine: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write the public Extraction Yield v2 artifact set.

    Rejected/quarantine lists are for candidate-level filtering. Sentence-level
    safety skips remain summarized as ``screened_out_sentence_count`` because no
    candidate was emitted for them.
    """
    target = Path(out_dir)
    rejected = rejected or []
    quarantine = quarantine or []
    summary = build_artifact_summary(
        candidates, stats, rejected=rejected, quarantine=quarantine
    )
    rows = _candidate_rows(candidates)
    by_source = _by_source_page_rows(candidates)

    _write_json(target / "extraction_yield_v2_candidates.json", candidates)
    _write_csv(target / "extraction_yield_v2_candidates.csv", rows, _CANDIDATE_FIELDS)
    _write_json(target / "extraction_yield_v2_rejected.json", rejected)
    _write_json(target / "extraction_yield_v2_quarantine.json", quarantine)
    _write_json(target / "extraction_yield_v2_summary.json", summary)
    _write_json(target / "extraction_yield_v2_by_pattern.json", summary["extraction_yield_v2_candidate_by_pattern"])
    _write_csv(
        target / "extraction_yield_v2_by_source_page.csv",
        by_source,
        ["source_page", "candidate_count", "relation_count", "definition_count"],
    )
    _write_json(
        target / "extraction_yield_v2_report.json",
        {
            "summary": summary,
            "stats": stats,
            "candidate_examples": candidates[:20],
            "rejected_examples": rejected[:20],
            "quarantine_examples": quarantine[:20],
            "by_source_page": by_source[:50],
        },
    )
    return summary
