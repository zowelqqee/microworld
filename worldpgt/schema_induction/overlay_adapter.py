"""Convert schema_induction artifacts to overlay_relation / overlay_definition dicts.

The overlay format is what the existing precision gate, safe_delta_merger, and
QA benchmark understand. This adapter is a one-way projection:

    RelationFamily + ArgumentFrame + RawClaim
        -> overlay_relation / overlay_definition dicts

Mapping rules
-------------
* promoted family, role=requirement   -> predicate="requires"
* promoted family, role=permission    -> predicate="allows"
* promoted family, role=prohibition   -> predicate="prohibits"
* promoted family, role=destination   -> predicate="located_in"
* promoted family, role=agent         -> predicate="operated_by"
* promoted family, role=attribute     -> overlay_definition (is_a)
* generated family (not promoted)     -> stability="semi_stable",
                                         trust="overlay_candidate_generated"
* source trace doc:sentence           -> evidence_text field

All produced items carry:
  pump_source_kind = "schema_induced"
  candidate_source = "pump_schema_induction"
  schema_family_id, schema_promotion_status  (for later audit)

Generated (non-promoted) items additionally carry:
  schema_source_doc_count  (the source diversity gate value)
so the precision gate can apply a stricter threshold when needed.
"""

from __future__ import annotations

from worldpgt.schema_induction.types import (
    ArgumentFrame,
    RawClaim,
    RelationFamily,
)

_CANDIDATE_SOURCE = "pump_schema_induction"

# Generic role -> overlay predicate (promoted families only).
# These are conservative: only roles that map cleanly to existing known
# predicates become full overlay_relation items. Unknown roles produce
# a lower-trust item with the role name as predicate (for inspection).
_ROLE_TO_PREDICATE: dict[str, str] = {
    "requirement": "requires",
    "permission": "allows",
    "prohibition": "prohibits",
    "destination": "located_in",
    "agent": "operated_by",
    "patient": "feeds_on",
}

# Roles that should become overlay_definition (is_a) rather than
# overlay_relation.
_DEFINITION_ROLES = frozenset({"attribute"})

# Family canonical labels that are too high-frequency / generic to be useful
# as overlay items in a Wikipedia-scale corpus.  The copula "is/are/was/were"
# fires on nearly every sentence and produces noisy subjects.  It remains
# useful for small targeted corpora (e.g. demo docs) but must be suppressed
# when converting bulk Wikipedia docs to the existing overlay format.
_SUPPRESSED_FAMILIES = frozenset({"is", "was", "are", "were"})

# Maximum character length for subject / object strings.  Wikipedia bodies
# sometimes produce multi-sentence fragments as subjects when the trigger
# appears late in a long clause.  Hard-limit keeps overlay items clean.
_MAX_SPAN_LEN = 200

# Stability/trust by promotion status.
_PROMOTED_STABILITY = "semi_stable"
_PROMOTED_TRUST = "overlay_candidate"
_GENERATED_STABILITY = "semi_stable"
_GENERATED_TRUST = "overlay_candidate_generated"


def _norm(s: str) -> str:
    return (s or "").strip()


def _is_clean_span(text: str) -> bool:
    """Return False for spans that are clearly malformed for overlay use."""
    t = _norm(text)
    if not t:
        return False
    if len(t) > _MAX_SPAN_LEN:
        return False
    # Reject spans that look like markdown document headers / URL lines.
    if t.startswith("#") or t.startswith("Source:") or t.startswith("Retrieved"):
        return False
    # Reject pronoun-only subjects.
    if t.lower() in {"it", "they", "this", "that", "he", "she", "we", "who", "which"}:
        return False
    # Reject sentence fragments: must start with an alphabetic character.
    if not t[0].isalpha():
        return False
    # Reject spans with semicolons (list items, committee rosters, etc.).
    if ";" in t:
        return False
    # Reject spans containing wiki-style data placeholders.
    if "[data" in t.lower() or "[missing" in t.lower():
        return False
    # Reject spans that are clearly clausal (contain a verb phrase).
    # Heuristic: if there are 8+ words, it's likely a sentence fragment.
    if len(t.split()) > 10:
        return False
    return True


def _evidence_text(
    frame: ArgumentFrame,
    claims: dict[str, RawClaim],
) -> str:
    """Return first available sentence from the frame's claims."""
    for cid in frame.claim_ids:
        claim = claims.get(cid)
        if claim and claim.sentence:
            return claim.sentence
    return ""


def _source_page(
    frame: ArgumentFrame,
    claims: dict[str, RawClaim],
) -> str:
    for cid in frame.claim_ids:
        claim = claims.get(cid)
        if claim:
            return claim.source_doc_id
    return ""


def _make_relation(
    subject: str,
    predicate: str,
    obj: str,
    source_page: str,
    evidence_text: str,
    *,
    stability: str,
    trust: str,
    family: RelationFamily,
) -> dict:
    return {
        "overlay_type": "overlay_relation",
        "subject": _norm(subject),
        "predicate": predicate,
        "object": _norm(obj),
        "source_page": source_page,
        "evidence_text": evidence_text,
        "trust": trust,
        "risk": "medium",
        "stability": stability,
        "candidate_source": _CANDIDATE_SOURCE,
        "extraction_pattern": f"schema_induction_{family.canonical_label}",
        "v2_pattern_id": f"schema_induction_{family.canonical_label}",
        "confidence_label": "schema_induced",
        "evidence_span": evidence_text,
        # Schema-induction specific metadata.
        "pump_source_kind": "schema_induced",
        "schema_family_id": family.family_id,
        "schema_canonical_label": family.canonical_label,
        "schema_promotion_status": family.promotion_status,
        "schema_source_doc_count": family.source_doc_count,
        "schema_evidence_count": family.evidence_count,
    }


def _make_definition(
    subject: str,
    definition: str,
    source_page: str,
    evidence_text: str,
    *,
    family: RelationFamily,
) -> dict:
    return {
        "overlay_type": "overlay_definition",
        "subject": _norm(subject),
        "definition": _norm(definition),
        "predicate": "is_a",
        "source_page": source_page,
        "evidence_text": evidence_text,
        "trust": "overlay_candidate",
        "risk": "low",
        "stability": "stable",
        "candidate_source": _CANDIDATE_SOURCE,
        "extraction_pattern": "schema_induction_attribute",
        "v2_pattern_id": "schema_induction_attribute",
        "confidence_label": "schema_induced",
        "evidence_span": evidence_text,
        "pump_source_kind": "schema_induced",
        "schema_family_id": family.family_id,
        "schema_canonical_label": family.canonical_label,
        "schema_promotion_status": family.promotion_status,
        "schema_source_doc_count": family.source_doc_count,
    }


def family_to_overlay_items(
    family: RelationFamily,
    frames: list[ArgumentFrame],
    claims_by_id: dict[str, RawClaim],
) -> list[dict]:
    """Convert one relation family's frames into overlay dicts.

    ``frames`` should be the frames that belong to this family
    (i.e. their frame_id is in family.frame_ids).
    """
    items: list[dict] = []

    # Suppress high-frequency generic copula families — they fire on almost
    # every Wikipedia sentence and produce noisy overlay items at scale.
    if family.canonical_label in _SUPPRESSED_FAMILIES:
        return items

    is_promoted = family.promotion_status == "promoted"
    stability = _PROMOTED_STABILITY if is_promoted else _GENERATED_STABILITY
    trust = _PROMOTED_TRUST if is_promoted else _GENERATED_TRUST

    family_frames = [f for f in frames if f.frame_id in set(family.frame_ids)]

    for frame in family_frames:
        subject = frame.roles.get("subject", "")
        if not _is_clean_span(subject):
            continue

        evidence = _evidence_text(frame, claims_by_id)
        src_page = _source_page(frame, claims_by_id)

        # Try known role -> predicate mappings first.
        emitted = False
        for role, predicate in _ROLE_TO_PREDICATE.items():
            val = frame.roles.get(role)
            if not val:
                continue
            if not _is_clean_span(val):
                continue
            items.append(
                _make_relation(
                    subject, predicate, val, src_page, evidence,
                    stability=stability,
                    trust=trust,
                    family=family,
                )
            )
            emitted = True

        # Attribute roles become definitions — only when subject is a proper
        # noun-like token (starts with capital) to avoid sentence fragments.
        for role in _DEFINITION_ROLES:
            val = frame.roles.get(role)
            if val and _is_clean_span(val) and subject and subject[0].isupper():
                items.append(
                    _make_definition(subject, val, src_page, evidence, family=family)
                )
                emitted = True

        # For unknown non-subject roles in promoted families, emit a generic
        # overlay_relation with role name as predicate (for inspection/audit).
        if is_promoted and not emitted:
            for role, val in frame.roles.items():
                if role == "subject" or not val:
                    continue
                if role not in _ROLE_TO_PREDICATE and role not in _DEFINITION_ROLES:
                    items.append(
                        _make_relation(
                            subject, f"schema_{role}", val, src_page, evidence,
                            stability="semi_stable",
                            trust="overlay_candidate_generated",
                            family=family,
                        )
                    )

    return items


def schema_result_to_overlay_items(
    families: list[RelationFamily],
    frames: list[ArgumentFrame],
    claims: list[RawClaim],
    *,
    include_generated: bool = True,
) -> list[dict]:
    """Convert a full schema induction result into a flat list of overlay dicts.

    Args:
        include_generated: if False, only promoted families are converted.
            Defaults True so callers can decide; the precision gate applies
            stricter thresholds for generated items anyway.
    """
    claims_by_id: dict[str, RawClaim] = {c.claim_id: c for c in claims}
    all_items: list[dict] = []

    for family in families:
        if not include_generated and family.promotion_status != "promoted":
            continue
        items = family_to_overlay_items(family, list(frames), claims_by_id)
        all_items.extend(items)

    return all_items
