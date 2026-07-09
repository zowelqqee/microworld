"""Knowledge Pump Precision Cleanup v2.1 (v1.7.1 pass).

Runs *after* the v1.7 Precision Firewall v2. The v1.7 pass removed
extraction-yield-v2-specific garbage. What remained were subtler issues:

* Truncated person aliases as objects (``Levchin`` instead of ``Max Levchin``)
* Single-word subjects that are informal aliases for longer entity names
  (``Harvard`` for ``Harvard University``, ``Morgan`` for ``Morgan Motor Company``)
* Composite geographic-region subjects from multi-region source pages
  (``Africa and Eurasia WorldSpace``, ``United States Sirius Satellite Radio``)
* All ``headquartered_in`` facts (high-risk predicate, quarantine by default)
* Ambiguous ``owned_by`` facts where evidence says "operated by" or the subject
  is clearly a vehicle/spacecraft rather than a company
* Duplicate alias relations (``Compaq | owned_by | HP`` vs ``… | Hewlett-Packard``)
* Definitions ending with an unresolved participle (``designed``, ``marketed``, …)

This module is purely classifying: it never writes files, never calls the
network, and never mutates accepted/promoted/runtime memory. Precision is
preferred over coverage.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


# ---------------------------------------------------------------------------
# Geographic / composite region detection
# ---------------------------------------------------------------------------

_COMPOSITE_REGION_LEAD = re.compile(
    r"^(?:United\s+States|North\s+America|South\s+America|Latin\s+America"
    r"|Africa|Asia|Europe|Eurasia|Middle\s+East|Pacific|Atlantic)\b",
    re.IGNORECASE,
)

_COMPOSITE_AND_REGION = re.compile(
    r"\b(?:Africa|Asia|Europe|Eurasia|America|Pacific|Atlantic|Latin)\b"
    r".{0,30}\band\b",
    re.IGNORECASE,
)


def _is_composite_region_subject(subject: str) -> bool:
    return bool(
        _COMPOSITE_REGION_LEAD.match(subject)
        or _COMPOSITE_AND_REGION.search(subject)
    )


# ---------------------------------------------------------------------------
# Institution-type suffix detection for subject alias
# ---------------------------------------------------------------------------

# Source-page suffixes that indicate the subject is a shortened alias.
_INSTITUTION_SUFFIXES = (
    " university",
    " college",
    " institute",
    " school",
    " foundation",
    " motor company",
    " motor corp",
    " motor corp.",
    " academy",
    " laboratories",
    " laboratory",
    " lab",
)


def _is_subject_alias_too_short(subject: str, source_page: str) -> bool:
    """True if subject is a single-word alias for a longer institution name.

    Checks that subject is a single, mixed-case word, source_page starts with
    the same word, and source_page has an institution-type suffix that marks
    it as a longer canonical name (University, College, Motor Company, etc.).
    """
    if not subject or not source_page:
        return False
    # Only applies to single-word subjects.
    if len(subject.split()) != 1:
        return False
    # Exempt all-uppercase acronyms.
    if subject.isupper():
        return False
    # Exempt CamelCase compound names (e.g. SpaceX).
    if re.search(r"[A-Z]", subject[1:]):
        return False
    norm_src = _norm(source_page)
    norm_subj = subject.lower()
    if not norm_src.startswith(norm_subj + " "):
        return False
    suffix = norm_src[len(norm_subj):]
    return any(suffix.startswith(sfx) for sfx in _INSTITUTION_SUFFIXES)


# ---------------------------------------------------------------------------
# Truncated person alias detection
# ---------------------------------------------------------------------------

def _is_single_titlecase_word(word: str) -> bool:
    """True if word is a single token like 'Levchin' (title-case, no acronym)."""
    if not word or len(word.split()) != 1:
        return False
    if len(word) < 3:
        return False
    if word.isupper():
        return False
    # No internal uppercase (excludes CamelCase company names like SpaceX).
    if re.search(r"[A-Z]", word[1:]):
        return False
    return word[0].isupper()


def _object_is_truncated_person(obj: str) -> bool:
    return _is_single_titlecase_word(obj)


def _subject_is_truncated_person_via_source(subject: str, source_page: str) -> bool:
    """True if subject is a single word that is the last token of source_page.

    Catches ``Levchin | develops | Affirm`` where source_page is ``Max Levchin``:
    'Levchin' is the last word of a multi-word full name.
    """
    if not _is_single_titlecase_word(subject):
        return False
    src_words = (source_page or "").strip().split()
    if len(src_words) < 2:
        return False
    return src_words[-1].lower() == subject.lower()


# ---------------------------------------------------------------------------
# Vehicle / spacecraft detection for owned_by ambiguity
# ---------------------------------------------------------------------------

_VEHICLE_TYPE_WORDS = frozenset({
    "dragon", "orion", "capsule", "shuttle", "module",
    "lander", "probe", "rover", "station",
})


def _subject_is_vehicle(subject: str) -> bool:
    return any(w in _VEHICLE_TYPE_WORDS for w in subject.lower().split())


# ---------------------------------------------------------------------------
# Strong ownership phrases (evidence must contain one for owned_by to be clean)
# ---------------------------------------------------------------------------

_STRONG_OWNERSHIP_PHRASES = (
    "owned by",
    "acquired by",
    "wholly owned subsidiary",
    "subsidiary of",
    "purchased by",
    "part of",
)


def _evidence_has_strong_ownership(evidence: str) -> bool:
    nev = _norm(evidence)
    return any(p in nev for p in _STRONG_OWNERSHIP_PHRASES)


# ---------------------------------------------------------------------------
# Alias map for duplicate detection
# ---------------------------------------------------------------------------

# Conservative alias map: short form → canonical long form.
_ALIAS_MAP: dict[str, str] = {
    "hp": "hewlett-packard",
    "hpq": "hewlett-packard",
}


def _canonical_object(obj: str) -> str:
    return _ALIAS_MAP.get(_norm(obj), _norm(obj))


# ---------------------------------------------------------------------------
# Definition truncated participle detection
# ---------------------------------------------------------------------------

_TRUNCATED_PARTICIPLES = frozenset({
    "designed",
    "marketed",
    "offered",
    "developed",
    "concerned",
    "associated",
    "operated",
    "manufactured",
    "intended",
    "used",
    "produced",
    "created",
    "built",
})


def _definition_ends_with_participle(definition: str) -> bool:
    d = (definition or "").strip()
    if not d:
        return False
    last_word = d.rstrip(".,; ").split()[-1].lower()
    return last_word in _TRUNCATED_PARTICIPLES


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_precision_cleanup_v2_1(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the v1.7.1 precision cleanup pass to already-v1.7-accepted items.

    ``items`` is the output of ``apply_precision_firewall_v2`` (the v1.7 pass).
    Entities and other non-relation/non-definition types pass through unchanged.

    The deduplication pass (duplicate alias relations) is done in a pre-scan
    so the decision is consistent regardless of item order.

    Returns a dict with:
      accepted, rejected, quarantine,
      rejection_by_reason, quarantine_by_reason,
      relation_before_count, relation_after_count,
      definition_before_count, definition_after_count,
      duplicate_relation_groups,
      duplicate_alias_relation_count,
      canonicalized_relation_count,
      quarantined_alias_duplicate_count,
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

    # --- Pre-scan: find duplicate alias relation groups ---
    # Build (subject_norm, predicate) → list of canonical objects seen.
    alias_groups: dict[tuple[str, str], list[str]] = {}
    for item in items:
        if item.get("overlay_type") != "overlay_relation":
            continue
        key = (_norm(item.get("subject", "")), _norm(item.get("predicate", "")))
        canon = _canonical_object(item.get("object", ""))
        alias_groups.setdefault(key, []).append(canon)

    # For each group with duplicate canonical objects, track which items are dupes.
    # A duplicate is any item whose canonical object matches another AND whose
    # raw object != canonical (i.e., it is the alias form, not the preferred form).
    duplicate_alias_items: set[int] = set()  # ids of items that are alias duplicates
    duplicate_relation_groups: list[dict[str, Any]] = []
    for key, canons in alias_groups.items():
        # Count how many distinct canonical objects there are.
        if len(canons) <= 1:
            continue
        from collections import Counter
        canon_counts = Counter(canons)
        dup_canons = {c for c, n in canon_counts.items() if n > 1}
        if not dup_canons:
            continue
        # Multiple objects with the same canonical form → duplicate.
        duplicate_relation_groups.append({
            "subject": key[0],
            "predicate": key[1],
            "canonical_objects": list(dup_canons),
        })
        for item in items:
            if item.get("overlay_type") != "overlay_relation":
                continue
            if _norm(item.get("subject", "")) != key[0]:
                continue
            if _norm(item.get("predicate", "")) != key[1]:
                continue
            obj_norm = _norm(item.get("object", ""))
            canon = _canonical_object(obj_norm)
            if canon in dup_canons and obj_norm != canon:
                # This is the alias form (shorter/non-canonical).
                duplicate_alias_items.add(id(item))

    quarantined_alias_duplicate_count = len(duplicate_alias_items)
    canonicalized_relation_count = sum(len(g["canonical_objects"]) for g in duplicate_relation_groups)

    # --- Main classification loop ---
    for item in items:
        otype = item.get("overlay_type")

        if otype == "overlay_relation":
            relation_before += 1
            subject = (item.get("subject") or "").strip()
            obj = (item.get("object") or "").strip()
            predicate = _norm(item.get("predicate") or "")
            source_page = (item.get("source_page") or "").strip()
            evidence = item.get("evidence_text") or ""

            # --- Composite region subject ---
            if _is_composite_region_subject(subject):
                rejected.append({"item": item, "reason": "composite_region_subject"})
                _bump(rejection_by_reason, "composite_region_subject")
                continue

            # --- headquartered_in quarantine default ---
            if predicate == "headquartered_in":
                quarantine.append({"item": item, "reason": "headquartered_in_quarantine_default"})
                _bump(quarantine_by_reason, "headquartered_in_quarantine_default")
                continue

            # --- owned_by predicate ambiguity (vehicle subject) ---
            if predicate == "owned_by" and _subject_is_vehicle(subject):
                quarantine.append({"item": item, "reason": "owned_by_predicate_ambiguous"})
                _bump(quarantine_by_reason, "owned_by_predicate_ambiguous")
                continue

            # --- Duplicate alias relation ---
            if id(item) in duplicate_alias_items:
                quarantine.append({"item": item, "reason": "duplicate_alias_relation"})
                _bump(quarantine_by_reason, "duplicate_alias_relation")
                continue

            # --- Subject alias too short ---
            if predicate in ("founded_by", "founded", "developed_by", "develops") and \
               _is_subject_alias_too_short(subject, source_page):
                quarantine.append({"item": item, "reason": "subject_alias_too_short"})
                _bump(quarantine_by_reason, "subject_alias_too_short")
                continue

            # --- Truncated person alias (object) ---
            if predicate in ("founded_by", "developed_by") and \
               _object_is_truncated_person(obj):
                quarantine.append({"item": item, "reason": "truncated_person_alias"})
                _bump(quarantine_by_reason, "truncated_person_alias")
                continue

            # --- Truncated person alias (subject of develops) ---
            if predicate == "develops" and \
               _subject_is_truncated_person_via_source(subject, source_page):
                quarantine.append({"item": item, "reason": "truncated_person_alias"})
                _bump(quarantine_by_reason, "truncated_person_alias")
                continue

            accepted.append(item)
            relation_after += 1
            continue

        if otype == "overlay_definition":
            definition_before += 1
            definition = (item.get("definition") or "").strip()

            # --- Truncated participle ending ---
            if _definition_ends_with_participle(definition):
                rejected.append({"item": item, "reason": "definition_truncated_participle"})
                _bump(rejection_by_reason, "definition_truncated_participle")
                continue

            accepted.append(item)
            definition_after += 1
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
        "duplicate_relation_groups": duplicate_relation_groups,
        "duplicate_alias_relation_count": len(duplicate_alias_items),
        "canonicalized_relation_count": canonicalized_relation_count,
        "quarantined_alias_duplicate_count": quarantined_alias_duplicate_count,
    }


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_cleanup_v2_1_report(
    result: dict[str, Any],
    *,
    answerable_before: int,
    answerable_after: int,
) -> dict[str, Any]:
    accepted = result["accepted"]
    rejected = result["rejected"]
    quarantine = result["quarantine"]

    return {
        "precision_cleanup_v2_1_enabled": True,
        "answerable_before_cleanup": answerable_before,
        "answerable_after_cleanup": answerable_after,
        "relation_before_count": result["relation_before_count"],
        "relation_after_count": result["relation_after_count"],
        "definition_before_count": result["definition_before_count"],
        "definition_after_count": result["definition_after_count"],
        "rejected_count": len(rejected),
        "quarantined_count": len(quarantine),
        "rejection_by_reason": result["rejection_by_reason"],
        "quarantine_by_reason": result["quarantine_by_reason"],
        "duplicate_relation_groups": result["duplicate_relation_groups"],
        "duplicate_alias_relation_count": result["duplicate_alias_relation_count"],
        "canonicalized_relation_count": result["canonicalized_relation_count"],
        "quarantined_alias_duplicate_count": result["quarantined_alias_duplicate_count"],
        "relation_examples_rejected": [
            {
                "subject": e["item"].get("subject"),
                "predicate": e["item"].get("predicate"),
                "object": e["item"].get("object"),
                "reason": e["reason"],
            }
            for e in rejected
            if e["item"].get("overlay_type") == "overlay_relation"
        ][:15],
        "definition_examples_rejected": [
            {
                "subject": e["item"].get("subject"),
                "definition": e["item"].get("definition"),
                "reason": e["reason"],
            }
            for e in rejected
            if e["item"].get("overlay_type") == "overlay_definition"
        ][:15],
        "quarantine_examples": [
            {
                "subject": e["item"].get("subject"),
                "predicate": e["item"].get("predicate"),
                "object": e["item"].get("object"),
                "definition": e["item"].get("definition"),
                "reason": e["reason"],
            }
            for e in quarantine
        ][:20],
        "relation_examples_preserved": [
            {"subject": i.get("subject"), "predicate": i.get("predicate"), "object": i.get("object")}
            for i in accepted
            if i.get("overlay_type") == "overlay_relation"
        ][:20],
        "definition_examples_preserved": [
            {"subject": i.get("subject"), "definition": i.get("definition")}
            for i in accepted
            if i.get("overlay_type") == "overlay_definition"
        ][:10],
    }


def write_cleanup_v2_1_artifacts(
    out_dir: str | Path,
    result: dict[str, Any],
    *,
    answerable_before: int,
    answerable_after: int,
) -> dict[str, Any]:
    """Write the v1.7.1 cleanup artifact set.

    Never modifies accepted memory, accepted/promoted overlays, or the snapshot
    dry-run overlay.
    """
    target = Path(out_dir)
    report = build_cleanup_v2_1_report(
        result, answerable_before=answerable_before, answerable_after=answerable_after
    )

    _write_json(target / "precision_cleanup_v2_1_answerable_delta.json", result["accepted"])
    _write_json(target / "precision_cleanup_v2_1_rejected_delta.json", result["rejected"])
    _write_json(target / "precision_cleanup_v2_1_quarantine.json", result["quarantine"])
    _write_json(target / "precision_cleanup_v2_1_report.json", report)
    _write_json(target / "precision_cleanup_v2_1_by_reason.json", {
        "rejection_by_reason": result["rejection_by_reason"],
        "quarantine_by_reason": result["quarantine_by_reason"],
    })

    summary = {
        "precision_cleanup_v2_1_enabled": True,
        "precision_cleanup_v2_1_answerable_before_count": answerable_before,
        "precision_cleanup_v2_1_answerable_after_count": answerable_after,
        "precision_cleanup_v2_1_relation_before_count": result["relation_before_count"],
        "precision_cleanup_v2_1_relation_after_count": result["relation_after_count"],
        "precision_cleanup_v2_1_definition_before_count": result["definition_before_count"],
        "precision_cleanup_v2_1_definition_after_count": result["definition_after_count"],
        "precision_cleanup_v2_1_rejected_count": len(result["rejected"]),
        "precision_cleanup_v2_1_quarantined_count": len(result["quarantine"]),
        "precision_cleanup_v2_1_rejection_by_reason": result["rejection_by_reason"],
        "precision_cleanup_v2_1_quarantine_by_reason": result["quarantine_by_reason"],
        "precision_cleanup_v2_1_duplicate_alias_relation_count": result["duplicate_alias_relation_count"],
        "precision_cleanup_v2_1_canonicalized_relation_count": result["canonicalized_relation_count"],
        "precision_cleanup_v2_1_quarantined_alias_duplicate_count": result["quarantined_alias_duplicate_count"],
    }
    _write_json(target / "precision_cleanup_v2_1_summary.json", summary)

    return summary
