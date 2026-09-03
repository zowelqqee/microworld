"""Relation candidate validator for Relation Extraction v2.

Takes raw extracted candidates and applies:
1. Safety screens (current/live, private, universal re-checked against evidence).
2. Directionality checks (person cannot found person, product cannot own company…).
3. Entity type conflict checks.
4. Minimum evidence quality (both subject and object must be resolvable to a
   non-trivial entity, or appear in the surface index).
5. Duplicate check against existing overlay relations.
6. Conflict check against existing overlay relations.
7. Generic entity match quarantine (if both subject and object are not in any
   known overlay and the pattern match is low confidence).

Outputs: (safe_candidates, quarantine_items, duplicate_ids, conflict_ids)

No ML, no network, no overlay writes.
"""

from __future__ import annotations

import re
from typing import Optional

from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex
from worldpgt.relation_extraction_v2.relation_patterns import (
    CURRENT_LIVE_SCREEN,
    PRIVATE_SCREEN,
    UNIVERSAL_SCREEN,
)
from worldpgt.relation_extraction_v2.types import (
    ALLOWED_RELATIONS,
    ExtractedRelationCandidate,
    RelationExtractionQuarantineItem,
)

# These relations must have an organization or product as subject, not a person.
_ORG_SUBJ_RELATIONS = frozenset({
    "manufactures", "produces", "develops", "operates",
    "parent_company_of", "subsidiary_of",
    "headquartered_in", "industry",
})

# These relations must have a non-person as the subject (the thing being founded/owned).
_NON_PERSON_SUBJ_RELATIONS = frozenset({
    "founded_by", "developed_by", "owned_by", "subsidiary_of", "service_of",
    "platform_of", "product_of", "created_by", "published_by", "part_of",
})

# Relations where the object should not be a person.
_NON_PERSON_OBJ_RELATIONS = frozenset({
    "founded", "develops", "manufactures", "produces", "operates",
    "product_of", "service_of", "platform_of", "part_of",
    "headquartered_in", "industry",
    # Ad-hoc pilot: object is a dataset / method / prior work, not a person.
    # ("about" is intentionally excluded — "The paper is about Newton" is a
    # legitimate topic relation with a person object.)
    "trained_on", "based_on", "extends",
})

# Maximum subject/object phrase word count — longer captures are usually noise.
_MAX_PHRASE_WORDS = 7

# Minimum subject/object length (chars).
_MIN_PHRASE_LEN = 2

# A structured (conditional) edge keeps its predicate short: the rule's
# conditions and exceptions live in their own fields, so a long predicate means
# the rule was welded back into the predicate string — the exact failure the
# conditional-edge schema exists to remove. Length alone is the detector,
# deliberately, instead of a list of "suspicious" connective words: every
# welded predicate observed across both legal pilots ran 8-51 words, so the
# threshold below catches the real failure shape without hand-picking which
# specific words signal it (a list like that would need updating for every new
# phrasing, exactly the hardcoding this module avoids elsewhere).
_MAX_STRUCTURED_PREDICATE_WORDS = 6


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _is_literal_span(span: str, evidence: str) -> bool:
    """True if ``span`` occurs verbatim (case/space-insensitive) in ``evidence``."""
    span_n = _norm(span)
    return bool(span_n) and span_n in _norm(evidence)


def _structured_reasons(cand: ExtractedRelationCandidate) -> list[str]:
    """Structural verification for a conditional / class-subject edge.

    A structured edge is admitted by *shape*, not by predicate vocabulary: a
    short un-welded predicate, a verbatim-grounded object, and every condition,
    exception, and object-alternative anchored to a literal span of the
    evidence.  This keeps the same literal-support discipline the rest of the
    graph uses, without an allowlist of legal predicates (which would be the
    hardcoding this design avoids).  It returns reasons; an empty list means the
    edge is a valid proposal (still never auto-admitted — see the caller).
    """

    reasons: list[str] = []
    evidence = cand.evidence_sentence or ""
    predicate_surface = str(cand.relation or "").replace("_", " ")

    if _word_count(predicate_surface) > _MAX_STRUCTURED_PREDICATE_WORDS:
        reasons.append("welded_predicate")
    if cand.polarity not in ("affirm", "negate"):
        reasons.append("invalid_polarity")

    if len(cand.object.strip()) < _MIN_PHRASE_LEN:
        reasons.append("missing_explicit_evidence")
    elif not _is_literal_span(cand.object, evidence):
        reasons.append("object_span_not_literal")

    if cand.subject_kind == "class_subject":
        # A class subject earns its place only by being grounded in the text.
        if not _is_literal_span(cand.subject, evidence):
            reasons.append("class_subject_span_not_literal")
    elif len(cand.subject.strip()) < _MIN_PHRASE_LEN:
        reasons.append("missing_explicit_evidence")

    for clause in (*cand.conditions, *cand.exceptions, *cand.object_alternatives):
        if not _is_literal_span(getattr(clause, "evidence_span", ""), evidence):
            reasons.append("unverified_clause_span")
            break

    return reasons


def _word_count(s: str) -> int:
    return len(s.split())


def _build_existing_relations(overlay_items: list[dict]) -> set[tuple[str, str, str]]:
    """Build a set of normalized (subject, predicate, object) triples."""

    triples: set[tuple[str, str, str]] = set()
    for item in overlay_items:
        if item.get("overlay_type") not in ("overlay_relation", "overlay_definition"):
            continue
        subj = _norm(str(item.get("subject") or ""))
        pred = _norm(str(item.get("predicate") or ""))
        obj = _norm(str(item.get("object") or item.get("definition") or ""))
        if subj and pred and obj:
            triples.add((subj, pred, obj))
    return triples


def _conflicts_existing(
    candidate: ExtractedRelationCandidate,
    existing_triples: set[tuple[str, str, str]],
    inverse_map: dict[tuple[str, str, str], str],
) -> Optional[str]:
    """Return conflict description if the candidate contradicts an existing relation."""

    subj_n = _norm(candidate.subject)
    obj_n = _norm(candidate.object)
    rel_n = _norm(candidate.relation)

    # Only true contradictory inverses are conflicts. Compatible inverse
    # predicates such as "X founded_by Y" and "Y founded X" are duplicates,
    # handled in _is_duplicate().
    inverse_rel = {
        "owned_by": "owns",
        "subsidiary_of": "parent_company_of",
        "parent_company_of": "subsidiary_of",
    }
    inv = inverse_rel.get(candidate.relation)
    if inv:
        if (_norm(candidate.object), _norm(inv), _norm(candidate.subject)) in existing_triples:
            return f"inverse_of_existing:{inv}"
    return None


def _is_duplicate(
    candidate: ExtractedRelationCandidate,
    existing_triples: set[tuple[str, str, str]],
) -> bool:
    subj_n = _norm(candidate.subject)
    obj_n = _norm(candidate.object)
    rel_n = _norm(candidate.relation)
    # Also check compatible predicate aliases.
    aliases: dict[str, list[str]] = {
        "founded_by": ["founded_by"],
        "founded": ["founded"],
        "develops": ["develops"],
        "developed_by": ["developed_by"],
        "manufactures": ["manufactures", "produces"],
        "produces": ["produces", "manufactures"],
        "owned_by": ["owned_by"],
        "is_a": ["is_a", "type_of"],
        "type_of": ["type_of", "is_a"],
    }
    for pred in aliases.get(candidate.relation, [candidate.relation]):
        if (subj_n, _norm(pred), obj_n) in existing_triples:
            return True
    if candidate.relation == "founded_by":
        if (obj_n, _norm("founded"), subj_n) in existing_triples:
            return True
    if candidate.relation == "founded":
        if (obj_n, _norm("founded_by"), subj_n) in existing_triples:
            return True
    return False


def validate_candidates(
    raw_candidates: list[ExtractedRelationCandidate],
    index: EntitySurfaceIndex,
    all_overlay_items: list[dict],
) -> tuple[
    list[ExtractedRelationCandidate],
    list[RelationExtractionQuarantineItem],
    list[str],   # duplicate ids
    list[str],   # conflict ids
]:
    existing_triples = _build_existing_relations(all_overlay_items)
    inverse_map: dict[tuple[str, str, str], str] = {}  # not used currently

    safe: list[ExtractedRelationCandidate] = []
    quarantine: list[RelationExtractionQuarantineItem] = []
    duplicate_ids: list[str] = []
    conflict_ids: list[str] = []

    known_surfaces = index.known_surfaces_set()

    for cand in raw_candidates:
        reasons: list[str] = []

        # ---- 1. Re-screen evidence sentence for live/private/universal ----
        if cand.notes:
            for note in cand.notes.split("; "):
                if note:
                    reasons.append(note)

        # Also screen the evidence sentence directly in case notes are missing.
        if CURRENT_LIVE_SCREEN.search(cand.evidence_sentence):
            reasons.append("current_or_live_claim")
        if PRIVATE_SCREEN.search(cand.evidence_sentence):
            reasons.append("private_or_sensitive")
        if UNIVERSAL_SCREEN.search(cand.evidence_sentence):
            reasons.append("unsupported_universal")

        # ---- Structured (conditional / class-subject) edge path ----------
        # An edge that carries conditions, exceptions, disjunctive objects, a
        # non-affirm polarity, or a class subject is validated by structure
        # rather than the entity/predicate-vocabulary checks below.  It is
        # *never* auto-admitted: a passing structured edge is a review-only
        # proposal (safe_for_overlay_delta stays False), preserving the gate's
        # conservatism while giving the rule a lossless home.  Simple edges skip
        # this branch entirely and follow the unchanged path below.
        if not cand.is_simple():
            reasons.extend(_structured_reasons(cand))
            if reasons:
                quarantine.append(
                    RelationExtractionQuarantineItem(
                        id=cand.id, subject=cand.subject, relation=cand.relation,
                        object=cand.object, reason="; ".join(sorted(set(reasons))),
                        source_title=cand.source_title,
                        evidence_sentence=cand.evidence_sentence,
                        pattern_id=cand.pattern_id,
                        risk="high" if any(
                            r in reasons for r in (
                                "current_or_live_claim", "private_or_sensitive",
                            )
                        ) else "medium",
                        notes="; ".join(reasons),
                    )
                )
            else:
                cand.requires_review = True
                cand.safe_for_overlay_delta = False
                safe.append(cand)
            continue

        # ---- 2. Relation allowed? ----------------------------------------
        if cand.relation not in ALLOWED_RELATIONS:
            reasons.append("ambiguous_relation")

        # ---- 3. Phrase quality checks ------------------------------------
        if len(cand.subject.strip()) < _MIN_PHRASE_LEN:
            reasons.append("missing_explicit_evidence")
        if len(cand.object.strip()) < _MIN_PHRASE_LEN:
            reasons.append("missing_explicit_evidence")
        if _word_count(cand.subject) > _MAX_PHRASE_WORDS:
            reasons.append("missing_explicit_evidence")
        if _word_count(cand.object) > _MAX_PHRASE_WORDS:
            reasons.append("missing_explicit_evidence")

        # ---- 3b. Entity quality: subject and object must look like entities ----
        # The subject must either resolve to a known entity, or be a clean
        # proper-noun phrase (starts capital, no digits, ≤ 5 words, no commas).
        def _is_clean_entity_phrase(s: str) -> bool:
            """Return True if s looks like a valid entity phrase."""
            s = s.strip()
            if not s:
                return False
            # Known in index: always valid.
            if index.resolve(s) is not None:
                return True
            # Check that it looks like a proper noun phrase.
            if not s[0].isupper():
                return False
            if any(c.isdigit() for c in s):
                return False
            words = s.split()
            if len(words) > 5:
                return False
            if "," in s:
                return False
            return True

        if not _is_clean_entity_phrase(cand.subject):
            reasons.append("missing_explicit_evidence")
        if not _is_clean_entity_phrase(cand.object):
            # Objects can also be location phrases with no index entry.
            # Allow them if they start with a capital and are clean.
            if not (cand.object.strip() and cand.object.strip()[0].isupper()
                    and not any(c.isdigit() for c in cand.object)
                    and len(cand.object.split()) <= 6
                    and "," not in cand.object):
                reasons.append("missing_explicit_evidence")

        # ---- 3c. Subject fragment detection ---------------------------------
        # Subjects that begin with sentence-fragment preamble patterns are noise:
        # "In total, Tesla", "As of 2020, Neuralink was", "History SolarCity", etc.
        _FRAGMENT_PREFIXES = re.compile(
            r"^(In\s+\w+|As\s+of|Following|During|By\s+\w+|On\s+\w+|"
            r"History|Founding|Background|Meanwhile|Currently|According|"
            r"After|Before|Since|While|When|Later|Soon|Also|However|"
            r"From|Until|Between|Among|Several|Three|Two|One|For|Although"
            r")",
            re.IGNORECASE,
        )
        if _FRAGMENT_PREFIXES.match(cand.subject.strip()):
            reasons.append("missing_explicit_evidence")

        # ---- 3c. Lone first-name object check ------------------------------
        # A single-word object that is a capitalized name not in the surface
        # index is almost certainly a truncated entity (e.g. "Elon", "Michael").
        def _is_truncated_name(s: str) -> bool:
            words = s.strip().split()
            if len(words) != 1:
                return False
            w = words[0]
            # Starts with capital, all-alpha, not in surface index.
            if w[0].isupper() and w.isalpha() and index.resolve(w) is None:
                return True
            return False

        if _is_truncated_name(cand.object):
            reasons.append("missing_explicit_evidence")
        if _is_truncated_name(cand.subject):
            reasons.append("missing_explicit_evidence")

        # Require that at least one of subject/object is a known surface or
        # looks like a proper noun (starts with capital). Single-word lowercase
        # objects (e.g. "and", "more", "torque", "devices") are junk captures.
        _COMMON_WORDS = frozenset({
            "and", "or", "the", "a", "an", "its", "it", "more", "many",
            "some", "most", "several", "nearly", "about", "seven", "five",
            "multiple", "various", "advanced", "new", "large", "small",
            "global", "american", "international", "modern", "early",
            "artificial", "further", "additional", "key", "major", "certain",
            "fixed", "both", "other", "different", "significant", "numerous",
            "unique", "all", "each", "any", "specific", "former", "various",
            "primary", "main", "full", "including", "excellent", "good",
            "devices", "torque", "information", "data", "services", "products",
            "components", "features", "materials", "technologies", "operations",
            "missions", "flights", "systems", "capabilities", "models",
            "approximately", "extensive", "additional", "commercial",
        })
        def _is_junk_phrase(s: str) -> bool:
            words = s.strip().split()
            if not words:
                return True
            if len(words) == 1:
                w = words[0].lower().rstrip(".,;:)")
                if w in _COMMON_WORDS:
                    return True
                # Single-word object must either be in the surface index or
                # start with a capital letter (proper noun heuristic).
                if not words[0][0].isupper() and index.resolve(words[0]) is None:
                    return True
            return False

        if _is_junk_phrase(cand.subject):
            reasons.append("missing_explicit_evidence")
        if _is_junk_phrase(cand.object):
            reasons.append("missing_explicit_evidence")

        # ---- 4. Entity type conflict / directionality -------------------
        subj_type = index.entity_type(cand.subject) or ""
        obj_type = index.entity_type(cand.object) or ""

        if cand.relation in _ORG_SUBJ_RELATIONS and subj_type == "person":
            reasons.append("directionality_conflict")
        if cand.relation in _NON_PERSON_SUBJ_RELATIONS and subj_type == "person":
            # Allowed: a person "founded_by" another person? Actually person
            # founded an org, so person being the subject of founded_by is wrong.
            # Exception: created_by / developed_by can have person as object (obj is the creator).
            # Here subj is the thing, obj is the creator — subj should not be person for most.
            if cand.relation not in ("created_by", "developed_by", "published_by"):
                reasons.append("entity_type_conflict")
        if cand.relation in _NON_PERSON_OBJ_RELATIONS and obj_type == "person":
            reasons.append("entity_type_conflict")

        # ---- 5. Generic entity match (both unknown, low confidence) -----
        subj_known = cand.subject in known_surfaces or index.resolve(cand.subject) is not None
        obj_known = cand.object in known_surfaces or index.resolve(cand.object) is not None
        if not subj_known and not obj_known and cand.confidence == "low":
            reasons.append("generic_entity_match")

        # ---- 6. Low confidence -------------------------------------------
        if cand.confidence == "low" and not reasons:
            reasons.append("low_confidence")

        # ---- 7. Duplicate / conflict check --------------------------------
        is_dup = _is_duplicate(cand, existing_triples)
        conflict_reason = _conflicts_existing(cand, existing_triples, inverse_map)

        if is_dup:
            duplicate_ids.append(cand.id)
            # Duplicates are accepted extraction evidence but are not safe for
            # overlay delta because the relation already exists/equivalently
            # exists in an overlay.
            cand.safe_for_overlay_delta = False
            safe.append(cand)
            continue

        if conflict_reason:
            reasons.append("conflict_existing_overlay")
            conflict_ids.append(cand.id)

        # ---- Decision ---------------------------------------------------
        if reasons:
            quarantine.append(
                RelationExtractionQuarantineItem(
                    id=cand.id,
                    subject=cand.subject,
                    relation=cand.relation,
                    object=cand.object,
                    reason="; ".join(sorted(set(reasons))),
                    source_title=cand.source_title,
                    evidence_sentence=cand.evidence_sentence,
                    pattern_id=cand.pattern_id,
                    risk="high" if any(
                        r in reasons for r in (
                            "current_or_live_claim", "private_or_sensitive",
                            "directionality_conflict", "entity_type_conflict",
                            "conflict_existing_overlay",
                        )
                    ) else "medium",
                    notes="; ".join(reasons),
                )
            )
        else:
            cand.safe_for_overlay_delta = True
            safe.append(cand)

    return safe, quarantine, duplicate_ids, conflict_ids
