"""Knowledge Pump Precision Firewall v2 (v1.7 pass).

Runs *after* the v1.4 Answerable Delta Precision Firewall. The v1.4 pass
removed sentence-fragment subjects and adjective-only objects. What it did
*not* do is catch extraction-yield-v2-specific noise:

* Dangling-comma subjects (``Amazon Leo,``, ``CNBC Asia, and CNBC Europe,``)
* Appositive-alias subjects (``France, officially the French Republic,``)
* List-fragment subjects (``Founding Elevation Partners``)
* Geopolitical locations as corporate-predicate subjects
* ``headquartered_in`` with a list/clause subject
* ``owned_by`` with a country/region object
* ``develops``/``developed_by`` with a location as developer or a
  design-artifact as object
* ``founded_by`` with a truncated-name object (``Roger Mc``)
* ``publishes`` with a phrase/section title as subject
* ``leader_of`` and other volatile political predicates
* Truncated definitions ending mid-sentence
* Appositive-alias subjects for definitions

This module is purely classifying: it never writes files, never calls the
network, and never mutates accepted/promoted/runtime memory. Precision is
preferred over coverage.
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Relation subject sanity checks
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


# Subjects that are geopolitical regions / generic locations, not orgs.
# These should never be the *subject* of a corporate predicate.
_GEOPOLITICAL_SUBJECTS = frozenset({
    "united states", "us", "usa", "u.s.", "u.s.a.",
    "uk", "united kingdom", "eu", "european union",
    "china", "russia", "germany", "france", "japan",
    "india", "canada", "australia", "brazil",
    # Cities/states that are not orgs
    "washington", "huntington beach", "auburn hills", "opelika", "london",
    "new york", "san jose", "seattle", "palo alto", "mountain view",
    "menlo park", "redwood city", "chicago", "houston", "detroit",
    "california", "texas", "nevada",
    # Generic non-entity nouns
    "stock exchange", "the stock exchange",
    "government", "the government",
    "congress", "senate", "parliament",
})

# Corporate predicates that should not have geopolitical subjects.
_CORPORATE_PREDICATES = frozenset({
    "headquartered_in",
    "owned_by",
    "subsidiary_of",
    "developed_by",
    "develops",
    "founded_by",
    "acquired_by",
    "operated_by",
    "manufactured_by",
})

# Predicates where the *object* must not be a bare geopolitical name.
_PREDICATES_NEED_ORG_OBJECT = frozenset({
    "owned_by",
    "acquired_by",
})

_GEOPOLITICAL_OBJECTS = frozenset({
    "us", "usa", "u.s.", "u.s.a.",
    "uk", "united kingdom", "eu",
    "china", "russia", "germany", "france", "japan",
    "india", "canada", "australia", "brazil",
    "washington",
})

# Location words that should not appear as the developer/manufacturer.
_LOCATION_WORDS = frozenset({
    "huntington beach", "auburn hills", "opelika", "london", "washington",
    "new york", "seattle", "chicago", "houston", "detroit",
    "cape canaveral", "palo alto", "mountain view", "menlo park",
    "redwood city",
})

# Design/document artifacts that should not be the object of develops.
_DESIGN_ARTIFACT_RE = re.compile(
    r"\b(?:insignia|livery|logo|mission\s+patch|badge|emblem|seal|crest"
    r"|coat\s+of\s+arms|flag|icon|symbol)\b",
    re.IGNORECASE,
)

# Predicates that are volatile/current and should be quarantined.
_VOLATILE_PREDICATES = frozenset({
    "leader_of",
    "leads",
    "chairs",
    "heads",
    "current_ceo",
    "current_president",
    "current_chairman",
})

# Subject prefixes that mark a noun-phrase fragment, not a clean entity.
_FRAGMENT_SUBJECT_PREFIXES = (
    "founding ",
    "the founding of ",
    "history of ",
    "overview of ",
    "background of ",
)

# Minimum token length for a "founded_by" object to be a real person/org.
# "Roger Mc" has 2 tokens but "Mc" looks like a truncated last name.
_TRUNCATED_NAME_RE = re.compile(
    r"^[A-Z][a-z]+ [A-Z][a-z]{0,2}\.?\s*$"  # "Roger Mc", "John St.", etc.
)

# Objects that are generic structures/concepts rather than entities.
_GENERIC_ENTITY_OBJECTS = frozenset({
    "stock exchange",
    "the stock exchange",
    "government",
    "congress",
    "senate",
    "parliament",
    "the internet",
    "internet",
})

# Objects with "All ..." or "Every ..." are list/universal claims.
_UNIVERSAL_OBJECT_RE = re.compile(r"^\s*(?:all|every|each|most|many|several|various)\s+", re.IGNORECASE)

# "publishes" subject tokens that suggest a phrase/section, not an org.
# A real publisher rarely has a trailing subordinate clause before the org name.
_PUBLISHES_PHRASE_RE = re.compile(
    r"(?:the\s+society\s*$|the\s+association\s*$|the\s+institute\s*$"
    r"|the\s+foundation\s*$|\bsociety\s*$|\bassociation\s*$)",
    re.IGNORECASE,
)

# Objects with long descriptive tails after the real entity name
# (e.g. "Grumman E-2 Hawkeye airborne early warning").
# We allow up to 4 capitalised-lead tokens; beyond that it's likely a tail.
_OBJECT_WITH_DESCRIPTIVE_TAIL_RE = re.compile(
    r"^(?:[A-Z][A-Za-z0-9/\-'.&]*)(?:\s+[A-Za-z0-9/\-'.&]+){4,}$"
)


# ---------------------------------------------------------------------------
# Subject-level checks
# ---------------------------------------------------------------------------

def _subject_has_dangling_comma(subject: str) -> bool:
    """True if subject ends with a comma or contains ', and '."""
    s = (subject or "").strip()
    return s.endswith(",") or ", and " in s


def _subject_is_appositive_alias(subject: str) -> bool:
    """True if subject has an embedded appositive alias clause.

    Matches patterns like:
      ``France, officially the French Republic,``
      ``Azure, sometimes stylized Azure, and formerly Windows Azure,``
      ``Hyperloop One, known as Virgin Hyperloop until November 2022,``
      ``Las Vegas, colloquially shortened to Vegas,``
    """
    s = (subject or "").strip()
    if "," not in s:
        return False
    return bool(re.search(
        r",\s+(?:officially|also\s+known\s+as|known\s+as|colloquially|"
        r"formerly|sometimes\s+stylized|stylized|also\s+called|"
        r"shortened\s+to|abbreviated\s+to)",
        s, re.IGNORECASE,
    ))


def _subject_is_fragment_prefix(subject: str) -> bool:
    """True if subject starts with a known fragment-marking prefix."""
    s = (subject or "").strip().lower()
    return any(s.startswith(p) for p in _FRAGMENT_SUBJECT_PREFIXES)


def _subject_is_geopolitical(subject: str) -> bool:
    return _norm(subject) in _GEOPOLITICAL_SUBJECTS


# ---------------------------------------------------------------------------
# Relation checks
# ---------------------------------------------------------------------------

def _check_relation_v2(item: dict[str, Any]) -> tuple[str, str]:
    """Return (verdict, reason) for a single relation item.

    ``accept``    — passes all v2 checks
    ``reject``    — definitively bad
    ``quarantine`` — plausible but unstable/volatile
    """
    subject = (item.get("subject") or "").strip()
    obj = (item.get("object") or "").strip()
    predicate = _norm(item.get("predicate") or "")
    nsubj = _norm(subject)
    nobj = _norm(obj)

    # --- Volatile predicates ---
    if predicate in _VOLATILE_PREDICATES:
        return "quarantine", "volatile_predicate"

    # --- Dangling-comma / list-fragment subjects ---
    if _subject_has_dangling_comma(subject):
        return "reject", "dangling_comma_subject"

    # --- Appositive-alias subjects ---
    if _subject_is_appositive_alias(subject):
        return "reject", "appositive_alias_subject"

    # --- Fragment-prefix subjects ---
    if _subject_is_fragment_prefix(subject):
        return "reject", "fragment_prefix_subject"

    # --- Geopolitical subjects for corporate predicates ---
    if predicate in _CORPORATE_PREDICATES and _subject_is_geopolitical(subject):
        return "reject", "geopolitical_subject_corporate_predicate"

    # --- Predicate-specific checks ---

    if predicate == "headquartered_in":
        # Must not have a list/clause subject (already caught by dangling comma).
        # Geopolitical subject already caught above.
        # Require evidence contains "headquartered in" or "headquarters in".
        evidence = _norm(item.get("evidence_text") or "")
        if "headquartered in" not in evidence and "headquarters in" not in evidence:
            return "quarantine", "headquarters_subject_or_clause_uncertain"
        return "accept", ""

    if predicate in ("owned_by", "acquired_by"):
        if nobj in _GEOPOLITICAL_OBJECTS:
            return "reject", "owned_by_geopolitical_object"
        return "accept", ""

    if predicate in ("develops", "developed_by"):
        # Reject location-as-developer (subject for develops, object for developed_by).
        if predicate == "developed_by" and nobj in _LOCATION_WORDS:
            return "reject", "developed_by_location_object"
        if predicate == "develops" and nsubj in _LOCATION_WORDS:
            return "reject", "develops_location_subject"
        # Reject design/document-artifact objects.
        if _DESIGN_ARTIFACT_RE.search(obj):
            return "reject", "develops_design_artifact_object"
        return "accept", ""

    if predicate == "founded_by":
        # Reject truncated-name objects.
        if _TRUNCATED_NAME_RE.match(obj):
            return "reject", "founded_by_truncated_object"
        if nobj in _GENERIC_ENTITY_OBJECTS:
            return "reject", "founded_by_generic_object"
        return "accept", ""

    if predicate == "publishes":
        if _PUBLISHES_PHRASE_RE.search(subject):
            return "reject", "publishes_phrase_subject"
        return "accept", ""

    if predicate == "produces":
        if _UNIVERSAL_OBJECT_RE.match(obj):
            return "reject", "produces_universal_object"
        return "accept", ""

    # Object descriptive tail (applies to develops and other predicates).
    if _OBJECT_WITH_DESCRIPTIVE_TAIL_RE.match(obj) and len(obj.split()) > 5:
        return "quarantine", "object_has_descriptive_tail"

    return "accept", ""


# ---------------------------------------------------------------------------
# Definition checks
# ---------------------------------------------------------------------------

# Endings that indicate a truncated definition sentence.
_TRUNCATED_DEF_ENDINGS = (
    " the u",
    " the u.",
    " to",
    ", to",
    " of",  # very short defs ending in "of" only
    " in",
    "primary purpose is to establish",
    "primary purpose is to",
    " concerned",
    " associated",
    "primarily",
    " in consumer",
    " in the u",
)

# Short one-word endings that signal truncation only when the def is short.
_SHORT_TRUNCATED_ENDS = frozenset({"to", "of", "in", "at", "by", "for", "and", "or"})


def _definition_is_truncated(definition: str) -> bool:
    """True if the definition looks cut off mid-sentence."""
    d = (definition or "").strip()
    dn = d.lower()

    # Explicit truncation patterns.
    for ending in _TRUNCATED_DEF_ENDINGS:
        if dn.endswith(ending):
            return True

    # Check for "the U" or "the U." at end (matches "in the U", "of the U" etc.)
    if re.search(r"\bthe\s+[Uu]\.?\s*$", d):
        return True

    # Short definitions (< 6 words) that end with a preposition/conjunction.
    words = d.split()
    if len(words) <= 6 and words and words[-1].lower() in _SHORT_TRUNCATED_ENDS:
        return True

    return False


def _check_definition_v2(item: dict[str, Any]) -> tuple[str, str]:
    """Return (verdict, reason) for a single definition item."""
    subject = (item.get("subject") or "").strip()
    definition = (item.get("definition") or "").strip()

    # --- Appositive-alias subjects ---
    if _subject_is_appositive_alias(subject):
        return "reject", "appositive_alias_subject"

    # --- Dangling-comma subjects ---
    if _subject_has_dangling_comma(subject):
        return "reject", "dangling_comma_subject"

    # --- Truncated definitions ---
    if _definition_is_truncated(definition):
        return "reject", "truncated_definition"

    # --- Definitions that are too vague (single nationality adjective + noun) ---
    ndef = _norm(definition)
    if re.match(r"^(?:american|british|french|german|chinese|canadian|"
                r"australian|japanese|korean|indian|spanish|italian|"
                r"russian)\s+\w{1,12}\s*$", ndef):
        return "reject", "definition_nationality_plus_single_noun"

    return "accept", ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_precision_firewall_v2(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the v1.7 precision pass to already-v1.4-accepted items.

    ``items`` is the output of ``apply_precision_firewall`` (the v1.4 pass).
    Entities and other non-relation/non-definition types pass through unchanged.

    Returns a dict with:
      accepted, rejected, quarantine,
      rejection_by_reason, quarantine_by_reason,
      relation_before_count, relation_after_count,
      definition_before_count, definition_after_count,
    """
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    rejection_by_reason: dict[str, int] = {}
    quarantine_by_reason: dict[str, int] = {}
    relation_before = 0
    relation_after = 0
    definition_before = 0
    definition_after = 0

    def _bump(table: dict[str, int], key: str) -> None:
        table[key] = table.get(key, 0) + 1

    for item in items:
        otype = item.get("overlay_type")

        if otype == "overlay_relation":
            relation_before += 1
            verdict, reason = _check_relation_v2(item)
            if verdict == "accept":
                accepted.append(item)
                relation_after += 1
            elif verdict == "quarantine":
                quarantine.append({"item": item, "reason": reason})
                _bump(quarantine_by_reason, reason)
            else:
                rejected.append({"item": item, "reason": reason})
                _bump(rejection_by_reason, reason)
            continue

        if otype == "overlay_definition":
            definition_before += 1
            verdict, reason = _check_definition_v2(item)
            if verdict == "accept":
                accepted.append(item)
                definition_after += 1
            elif verdict == "quarantine":
                quarantine.append({"item": item, "reason": reason})
                _bump(quarantine_by_reason, reason)
            else:
                rejected.append({"item": item, "reason": reason})
                _bump(rejection_by_reason, reason)
            continue

        # Entities and other cleared types pass through.
        accepted.append(item)

    return {
        "accepted": accepted,
        "rejected": rejected,
        "quarantine": quarantine,
        "rejection_by_reason": dict(sorted(rejection_by_reason.items())),
        "quarantine_by_reason": dict(sorted(quarantine_by_reason.items())),
        "relation_before_count": relation_before,
        "relation_after_count": relation_after,
        "definition_before_count": definition_before,
        "definition_after_count": definition_after,
    }


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------

import json
from pathlib import Path


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_precision_v2_report(
    result: dict[str, Any],
    *,
    answerable_before: int,
    answerable_after: int,
) -> dict[str, Any]:
    accepted = result["accepted"]
    rejected = result["rejected"]
    quarantine = result["quarantine"]

    relation_examples_rejected = [
        {"subject": e["item"].get("subject"), "predicate": e["item"].get("predicate"),
         "object": e["item"].get("object"), "reason": e["reason"]}
        for e in rejected
        if e["item"].get("overlay_type") == "overlay_relation"
    ][:15]
    definition_examples_rejected = [
        {"subject": e["item"].get("subject"), "definition": e["item"].get("definition"),
         "reason": e["reason"]}
        for e in rejected
        if e["item"].get("overlay_type") == "overlay_definition"
    ][:15]
    relation_examples_preserved = [
        {"subject": i.get("subject"), "predicate": i.get("predicate"), "object": i.get("object")}
        for i in accepted
        if i.get("overlay_type") == "overlay_relation"
    ][:15]
    definition_examples_preserved = [
        {"subject": i.get("subject"), "definition": i.get("definition")}
        for i in accepted
        if i.get("overlay_type") == "overlay_definition"
    ][:15]

    return {
        "precision_v2_enabled": True,
        "answerable_before_v2": answerable_before,
        "answerable_after_v2": answerable_after,
        "relation_before_count": result["relation_before_count"],
        "relation_after_count": result["relation_after_count"],
        "definition_before_count": result["definition_before_count"],
        "definition_after_count": result["definition_after_count"],
        "rejected_count": len(rejected),
        "quarantined_count": len(quarantine),
        "rejection_by_reason": result["rejection_by_reason"],
        "quarantine_by_reason": result["quarantine_by_reason"],
        "relation_examples_rejected": relation_examples_rejected,
        "definition_examples_rejected": definition_examples_rejected,
        "relation_examples_preserved": relation_examples_preserved,
        "definition_examples_preserved": definition_examples_preserved,
        "quarantine_examples": [
            {"subject": e["item"].get("subject"), "predicate": e["item"].get("predicate"),
             "object": e["item"].get("object"), "reason": e["reason"]}
            for e in quarantine
        ][:15],
    }


def write_precision_v2_artifacts(
    out_dir: str | Path,
    result: dict[str, Any],
    *,
    answerable_before: int,
    answerable_after: int,
) -> dict[str, Any]:
    """Write the v1.7 precision firewall artifact set.

    Never modifies accepted memory, accepted/promoted overlays, or the snapshot
    dry-run overlay.
    """
    target = Path(out_dir)
    report = build_precision_v2_report(
        result, answerable_before=answerable_before, answerable_after=answerable_after
    )

    _write_json(target / "precision_v2_answerable_delta.json", result["accepted"])
    _write_json(target / "precision_v2_rejected_delta.json", result["rejected"])
    _write_json(target / "precision_v2_quarantine.json", result["quarantine"])
    _write_json(target / "precision_v2_report.json", report)
    _write_json(target / "precision_v2_by_reason.json", {
        "rejection_by_reason": result["rejection_by_reason"],
        "quarantine_by_reason": result["quarantine_by_reason"],
    })

    summary = {
        "precision_v2_enabled": True,
        "precision_v2_answerable_before_count": answerable_before,
        "precision_v2_answerable_after_count": answerable_after,
        "precision_v2_relation_before_count": result["relation_before_count"],
        "precision_v2_relation_after_count": result["relation_after_count"],
        "precision_v2_definition_before_count": result["definition_before_count"],
        "precision_v2_definition_after_count": result["definition_after_count"],
        "precision_v2_rejected_count": len(result["rejected"]),
        "precision_v2_quarantined_count": len(result["quarantine"]),
        "precision_v2_rejection_by_reason": result["rejection_by_reason"],
        "precision_v2_quarantine_by_reason": result["quarantine_by_reason"],
    }
    _write_json(target / "precision_v2_summary.json", summary)

    return summary
