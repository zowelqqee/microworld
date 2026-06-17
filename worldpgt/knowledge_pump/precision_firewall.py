"""Answerable Delta Precision Firewall for Knowledge Pump v1.4.

This runs *after* the v1.3 Delta Quality Firewall. The v1.3 firewall already
separated weak ``mentioned_with`` context links from answerable knowledge and
caught the most blatant safety problems (self-relations, volatile evidence,
negated development, too-broad ``is_a`` objects). What it did *not* do is
guarantee precision: the answerable bucket still contained semantic-extraction
garbage – sentence fragments as subjects, adjective-only objects, cross-page
apposition errors, opinion/award/relative "definitions", and so on.

This module classifies each *already-answerable* item into one of:

* ``accept``              – clean, supported, high-precision knowledge
* ``reject``              – fragment/generic/unsupported – drop entirely
* ``quarantine``          – plausible but needs review (e.g. ``known_for``)
* ``property_candidate``  – not an answerable definition, but a clean property
                            relation (e.g. ``July | has_birthstone | Ruby``)

It is purely classifying: it never writes files, never calls the network, and
never mutates accepted/promoted/runtime memory. Precision is preferred over
coverage – it is fine if the answerable count drops sharply.
"""

from __future__ import annotations

import re
import string
from typing import Any

_PUNCT = string.punctuation

# Function words that may appear lowercase inside an otherwise-proper-noun name.
_FUNCTION_WORDS = frozenset({
    "of", "the", "a", "an", "and", "for", "in", "to", "on", "at", "by",
    "de", "von", "van", "del", "della", "&", "-",
})

# Tokens that, when they lead a "subject", mark it as a sentence fragment.
_FRAGMENT_LEAD = frozenset({
    "if", "many", "some", "therefore", "however", "although", "because",
    "while", "such", "what", "most", "several", "various",
})

# Verb-ish tokens that should not appear inside an entity-like subject; their
# presence means the "subject" is actually a clause fragment.
_CLAUSE_VERB = frozenset({
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "owns", "own", "owned",
    "uses", "use", "used", "operates", "operate", "operated",
    "makes", "make", "made", "intended", "includes", "include", "included",
    "provides", "provide", "develops", "develop", "developed",
    "produces", "produce", "produced", "said", "says", "announced",
    "acquired", "acquires", "brought", "consists",
})

# Adverb/discourse markers that signal a fragment when present anywhere.
_FRAGMENT_MARKERS = frozenset({"also", "usually", "therefore", "previously"})

# Subjects that are generic non-entities (fine as objects, not as subjects).
_GENERIC_SUBJECTS = frozenset({
    "economy of the united states", "satellite internet",
    "internet service provider", "stock exchange", "business journalism",
    "initial public offering", "subsidiary", "founder", "technology",
    "company", "corporation", "vehicle", "electric vehicle", "product",
    "business", "organization", "model", "satellite constellation",
    "the following exchanges", "private spaceflight",
})

# Objects that are adjective-only / generic / fragmentary.
_GENERIC_OBJECTS = frozenset({
    "aggressive", "control", "british", "american", "rocket", "rockets",
    "fuselage", "novels", "singer", "clergyman", "economic news",
    "social media", "private spaceflight",
})

# Object lead tokens that mark a clause/continuation fragment.
_OBJECT_FRAGMENT_LEAD = frozenset({
    "which", "but", "and", "as", "other", "that", "including", "or", "who",
    "where", "although", "began", "especially", "multiple", "his", "her",
})

# Substrings that mark an object as an uncontrolled clause.
_OBJECT_CLAUSE_MARKERS = (
    "as well as", " which ", " but ", "that can", "that meant",
    "including", " who ", " where ", "such as",
)

# is_a objects that are too broad to be answerable (mirrors v1.3 + extensions).
_BAD_IS_A_OBJECTS = frozenset({
    "artificial intelligence", "machine learning", "computer science",
    "information technology", "deep learning", "electric vehicle",
    "renewable energy", "social media", "e-commerce", "cloud computing",
})

_NATIONALITY_ADJECTIVES = frozenset({
    "american", "chinese", "european", "british", "german", "japanese",
    "korean", "indian", "canadian", "australian", "french", "italian",
    "spanish", "russian", "israeli",
})

# Evidence phrases that indicate the relation is negated or non-affirming.
_NEGATION_PHRASES = (
    "never developed", "not developed", "considered but never",
    "planned but not", "never built", "never produced", "not produced",
)

# Strong ownership/operation wording required for owned_by.
_OWNERSHIP_PHRASES = (
    "owned by", "operated by", "acquired by", "wholly owned subsidiary",
    "part of", "subsidiary of", "purchased by",
)

# Definition phrases that mark it as an award/relative/opinion/proposal/etc.
_BAD_DEFINITION_MARKERS = (
    "good thing", "best ", "greatest", "largest", "biggest",
    "proposal to", "rename", "award", "winner",
    "son of", "daughter of", "father of", "mother of",
    "brother of", "sister of", "wife of", "husband of",
    "physical place where", "co-inventor", " who ",
)

_GEMSTONES = frozenset({
    "ruby", "sapphire", "emerald", "diamond", "pearl", "opal", "topaz",
    "garnet", "amethyst", "aquamarine", "peridot", "citrine", "turquoise",
    "tanzanite", "alexandrite", "moonstone", "spinel", "zircon", "onyx",
})

_YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _strip(token: str) -> str:
    return token.lower().strip(_PUNCT)


def _in_evidence(value: str, evidence: str) -> bool:
    nv = _norm(value)
    return bool(nv) and nv in _norm(evidence)


def _subject_ok(subject: str) -> bool:
    """True if subject reads as a proper-noun entity, not a sentence fragment."""
    s = (subject or "").strip()
    if not s:
        return False
    norm = _norm(s)
    if norm in _GENERIC_SUBJECTS:
        return False
    words = s.split()
    if len(words) > 7:
        return False
    lowers = [_strip(w) for w in words]
    if lowers and lowers[0] in _FRAGMENT_LEAD:
        return False
    if any(w in _FRAGMENT_MARKERS for w in lowers):
        return False
    if any(w in _CLAUSE_VERB for w in lowers):
        return False
    # Casing: every content word must be capitalised, an acronym, or a number.
    for word in words:
        wl = _strip(word)
        if not wl or wl in _FUNCTION_WORDS:
            continue
        if word.isupper():
            continue
        if word[0].isupper() or word[0].isdigit():
            continue
        return False
    return True


def _object_entity_ok(obj: str) -> bool:
    """True if object reads as a proper-noun organisation/person/product."""
    o = (obj or "").strip()
    if not o:
        return False
    if _norm(o) in _GENERIC_OBJECTS:
        return False
    words = o.split()
    if len(words) > 6:
        return False
    lowers = [_strip(w) for w in words]
    if lowers and lowers[0] in _OBJECT_FRAGMENT_LEAD:
        return False
    if any(w in _CLAUSE_VERB for w in lowers):
        return False
    for word in words:
        wl = _strip(word)
        if not wl or wl in _FUNCTION_WORDS:
            continue
        if word.isupper():
            continue
        if word[0].isupper() or word[0].isdigit():
            continue
        return False
    return True


def _object_phrase_ok(obj: str) -> bool:
    """True if object is a clean named product/technology/publication phrase.

    Unlike :func:`_object_entity_ok` this allows lowercase common-noun phrases
    (e.g. ``tunnels``, ``brain-computer interfaces``) but still rejects
    adjective-only words, generic words, fragments, and long clauses.
    """
    o = (obj or "").strip()
    if not o:
        return False
    norm = _norm(o)
    if norm in _GENERIC_OBJECTS:
        return False
    if "(" in o or ")" in o:
        return False
    if _YEAR_RE.search(o):
        return False
    words = o.split()
    if len(words) > 6:
        return False
    lowers = [_strip(w) for w in words]
    if lowers and lowers[0] in _OBJECT_FRAGMENT_LEAD:
        return False
    for marker in _OBJECT_CLAUSE_MARKERS:
        if marker in f" {norm} ":
            return False
    # Reject a single adjective-only / bare generic word with no proper noun.
    return True


def _check_known_for(item: dict[str, Any]) -> tuple[str, str]:
    # Pump-generated known_for is unreliable; default to quarantine. Only a
    # clean entity + clean named object + explicit evidence can be accepted.
    subject = item.get("subject", "")
    obj = item.get("object", "")
    evidence = _norm(item.get("evidence_text", ""))
    if not _subject_ok(subject):
        return "reject", "known_for_bad_subject"
    explicit = any(p in evidence for p in ("known for", "best known", "famous for"))
    # Only a short, clean, named object backed by explicit wording may pass;
    # disambiguation/surname snippets (long apposition lists) stay quarantined.
    if explicit and _object_entity_ok(obj) and len(obj.split()) <= 3:
        return "accept", ""
    return "quarantine", "known_for_default_quarantine"


def _check_subsidiary_of(item: dict[str, Any]) -> tuple[str, str]:
    subject, obj = item.get("subject", ""), item.get("object", "")
    evidence = item.get("evidence_text", "")
    if not _subject_ok(subject):
        return "reject", "subsidiary_bad_subject"
    if not _object_entity_ok(obj):
        return "reject", "subsidiary_bad_object"
    # Both parties must appear in the cited sentence – otherwise this is a
    # cross-page apposition error (e.g. an Ampex sentence attached to Oracle).
    if not _in_evidence(subject, evidence):
        return "reject", "subsidiary_subject_not_in_evidence"
    if not _in_evidence(obj, evidence):
        return "reject", "subsidiary_object_not_in_evidence"
    return "accept", ""


def _check_owned_by(item: dict[str, Any]) -> tuple[str, str]:
    subject, obj = item.get("subject", ""), item.get("object", "")
    evidence = item.get("evidence_text", "")
    nev = _norm(evidence)
    if not _subject_ok(subject):
        return "reject", "owned_by_bad_subject"
    if not _object_entity_ok(obj):
        return "reject", "owned_by_bad_object"
    if not any(p in nev for p in _OWNERSHIP_PHRASES):
        return "reject", "owned_by_no_ownership_evidence"
    if not _in_evidence(obj, evidence):
        return "reject", "owned_by_object_not_in_evidence"
    # Anti-inversion: if the subject is itself the agent of a "by <subject>"
    # phrase, it is an operator, not the owned entity (e.g. "contracted by
    # NASA and operated by SpaceX" must not yield NASA owned_by SpaceX).
    if f"by {_norm(subject)}" in nev:
        return "reject", "owned_by_subject_is_agent"
    return "accept", ""


def _check_develops(item: dict[str, Any]) -> tuple[str, str]:
    subject, obj = item.get("subject", ""), item.get("object", "")
    predicate = _norm(item.get("predicate", ""))
    evidence = item.get("evidence_text", "")
    nev = _norm(evidence)
    for phrase in _NEGATION_PHRASES:
        if phrase in nev:
            return "reject", "develops_negated"
    if not _subject_ok(subject):
        return "reject", "develops_bad_subject"
    if predicate == "developed_by":
        if not _object_entity_ok(obj):
            return "reject", "developed_by_bad_object"
    elif not _object_phrase_ok(obj):
        return "reject", "develops_bad_object"
    if not _in_evidence(obj, evidence):
        return "reject", "develops_object_not_in_evidence"
    return "accept", ""


def _check_produces(item: dict[str, Any]) -> tuple[str, str]:
    subject, obj = item.get("subject", ""), item.get("object", "")
    evidence = item.get("evidence_text", "")
    if not _subject_ok(subject):
        return "reject", "produces_bad_subject"
    if not _object_phrase_ok(obj):
        return "reject", "produces_bad_object"
    if not _in_evidence(obj, evidence):
        return "reject", "produces_object_not_in_evidence"
    return "accept", ""


def _check_publishes(item: dict[str, Any]) -> tuple[str, str]:
    subject, obj = item.get("subject", ""), item.get("object", "")
    evidence = item.get("evidence_text", "")
    if not _subject_ok(subject):
        return "reject", "publishes_bad_subject"
    if not _object_phrase_ok(obj):
        return "reject", "publishes_bad_object"
    if "range of" in _norm(obj):
        return "quarantine", "publishes_vague_object"
    # A real publication is a named work: it must contain a capitalised token.
    if not _has_capitalised_word(obj):
        return "quarantine", "publishes_unnamed_publication"
    if not _in_evidence(obj, evidence):
        return "reject", "publishes_object_not_in_evidence"
    return "accept", ""


def _check_founded(item: dict[str, Any]) -> tuple[str, str]:
    subject, obj = item.get("subject", ""), item.get("object", "")
    evidence = item.get("evidence_text", "")
    if not _subject_ok(subject):
        return "reject", "founded_bad_subject"
    if not _object_entity_ok(obj):
        return "reject", "founded_bad_object"
    if not _in_evidence(obj, evidence):
        return "reject", "founded_object_not_in_evidence"
    return "accept", ""


def _check_is_a(item: dict[str, Any]) -> tuple[str, str]:
    subject, obj = item.get("subject", ""), item.get("object", "")
    nobj = _norm(obj)
    if not _subject_ok(subject):
        return "reject", "is_a_bad_subject"
    if nobj in _BAD_IS_A_OBJECTS:
        return "reject", "is_a_object_too_broad"
    words = nobj.split()
    if len(words) >= 2 and words[0] in _NATIONALITY_ADJECTIVES:
        return "reject", "is_a_object_truncated_nationality"
    if not _object_phrase_ok(obj):
        # A truncated/clause class may have a better full phrase in evidence.
        return "quarantine", "truncated_is_a_object"
    return "accept", ""


def _check_generic_relation(item: dict[str, Any]) -> tuple[str, str]:
    subject, obj = item.get("subject", ""), item.get("object", "")
    evidence = item.get("evidence_text", "")
    if not _subject_ok(subject):
        return "reject", "generic_relation_bad_subject"
    if not (_object_entity_ok(obj) or _object_phrase_ok(obj)):
        return "reject", "generic_relation_bad_object"
    if not _in_evidence(obj, evidence):
        return "reject", "generic_relation_object_not_in_evidence"
    return "accept", ""


_RELATION_HANDLERS = {
    "known_for": _check_known_for,
    "subsidiary_of": _check_subsidiary_of,
    "owned_by": _check_owned_by,
    "develops": _check_develops,
    "developed_by": _check_develops,
    "produces": _check_produces,
    "publishes": _check_publishes,
    "published_by": _check_generic_relation,
    "founded": _check_founded,
    "founded_by": _check_founded,
    "is_a": _check_is_a,
}


def _check_relation(item: dict[str, Any]) -> tuple[str, str]:
    predicate = _norm(item.get("predicate", ""))
    handler = _RELATION_HANDLERS.get(predicate, _check_generic_relation)
    return handler(item)


def _has_capitalised_word(text: str) -> bool:
    for word in text.split():
        if _strip(word) in _FUNCTION_WORDS:
            continue
        if word[:1].isalpha() and word[0].isupper() and not word.isupper():
            return True
    return False


def _check_definition(item: dict[str, Any]) -> tuple[str, str, dict[str, Any] | None]:
    """Return (verdict, reason, property_candidate)."""
    subject = item.get("subject", "")
    definition = item.get("definition", "")
    evidence = item.get("evidence_text", "")
    nsubj = _norm(subject)
    ndef = _norm(definition)
    nev = _norm(evidence)

    # Birthstone special case: a month "= <gem>" is not a definition, but may
    # cleanly support a has_birthstone property relation.
    if "birthstone" in nev:
        if ndef in _GEMSTONES and ndef in nev:
            prop = {
                "overlay_type": "overlay_relation",
                "subject": subject,
                "predicate": "has_birthstone",
                "object": definition.strip().title(),
                "source_page": item.get("source_page", ""),
                "evidence_text": evidence,
                "trust": "overlay_candidate",
                "risk": "low",
                "stability": "stable",
                "pump_precision_origin": "definition_to_property",
            }
            return "property_candidate", "definition_birthstone", prop
        return "quarantine", "property_candidate_not_answerable_definition", None

    if nsubj in _GENERIC_SUBJECTS:
        return "reject", "generic_definition_subject", None

    for marker in _BAD_DEFINITION_MARKERS:
        if marker in ndef:
            return "reject", "definition_not_lead_style", None

    # Truncated: unbalanced parentheses.
    if definition.count("(") != definition.count(")"):
        return "reject", "truncated_definition", None

    # Require clean lead-style support: the subject must be followed by a copula
    # and an article, *and* the definition text must actually be what follows
    # (otherwise "July is the seventh month ... ruby" would falsely back
    # "July = ruby"). Anchor on the first few words of the definition.
    def_lead = " ".join(ndef.split()[:3])
    if not def_lead:
        return "reject", "definition_not_lead_style", None
    pattern = re.compile(
        r"\b" + re.escape(nsubj)
        + r"(?:\s*\([^)]*\))?\s+(?:is|was|are|were)\s+(?:a\s+|an\s+|the\s+)?"
        + re.escape(def_lead)
    )
    if not pattern.search(nev):
        return "reject", "definition_not_lead_style", None

    return "accept", "", None


def apply_precision_firewall(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify already-answerable items into precision buckets.

    ``items`` is the v1.3 answerable delta (entities + definitions + relations).
    Entities pass through unchanged (they are page-existence cards). Definitions
    and relations are gated for precision.
    """
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    property_candidates: list[dict[str, Any]] = []
    rejection_by_reason: dict[str, int] = {}
    quarantine_by_reason: dict[str, int] = {}
    relation_before: dict[str, int] = {}
    relation_after: dict[str, int] = {}
    definition_before = 0
    definition_after = 0

    def _bump(table: dict[str, int], key: str) -> None:
        table[key] = table.get(key, 0) + 1

    for item in items:
        otype = item.get("overlay_type")

        if otype == "overlay_relation":
            predicate = _norm(item.get("predicate", "")) or "(none)"
            _bump(relation_before, predicate)
            verdict, reason = _check_relation(item)
            if verdict == "accept":
                accepted.append(item)
                _bump(relation_after, predicate)
            elif verdict == "quarantine":
                quarantine.append({"item": item, "reason": reason})
                _bump(quarantine_by_reason, reason)
            else:
                rejected.append({"item": item, "reason": reason})
                _bump(rejection_by_reason, reason)
            continue

        if otype == "overlay_definition":
            definition_before += 1
            verdict, reason, prop = _check_definition(item)
            if verdict == "accept":
                accepted.append(item)
                definition_after += 1
            elif verdict == "property_candidate":
                property_candidates.append(prop)
                _bump(quarantine_by_reason, reason)
            elif verdict == "quarantine":
                quarantine.append({"item": item, "reason": reason})
                _bump(quarantine_by_reason, reason)
            else:
                rejected.append({"item": item, "reason": reason})
                _bump(rejection_by_reason, reason)
            continue

        # Entities and any other already-cleared types pass through.
        accepted.append(item)

    return {
        "accepted": accepted,
        "rejected": rejected,
        "quarantine": quarantine,
        "property_candidates": property_candidates,
        "rejection_by_reason": dict(sorted(rejection_by_reason.items())),
        "quarantine_by_reason": dict(sorted(quarantine_by_reason.items())),
        "relation_before_by_predicate": dict(sorted(relation_before.items())),
        "relation_after_by_predicate": dict(sorted(relation_after.items())),
        "definition_before_count": definition_before,
        "definition_after_count": definition_after,
    }
