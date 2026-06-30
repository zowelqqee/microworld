"""Build a minimal QA-ready overlay for an arbitrary domain (Pass 3).

Combines the bootstrapped entities (Pass 1, entity_bootstrapper) with the
induced relations (Pass 2, schema_induction) into the SAME overlay format the
pump produces — so the existing QA stack (AnswerOrchestrator, EntityAnswerPlanner,
synthesis engine) reads it without knowing it came from a cold-start domain.

Output item types:
  * overlay_entity     — one per bootstrapped entity (label, aliases, type)
  * overlay_definition — from copula ("is/are") frames whose subject resolves
                         to a bootstrapped entity   -> predicate is_a
  * overlay_relation   — from requires/allows/prohibits/destination/agent
                         families, subject+object resolved to bootstrapped
                         entities where possible

Subjects/objects are resolved against the bootstrapped entity list (longest
surface match) instead of the overlay-derived EntitySurfaceIndex — that is the
whole point: no prior entity list is required.
"""

from __future__ import annotations

import re

from worldpgt.schema_induction.entity_bootstrapper import (
    BootstrappedEntity,
    bootstrap_entities,
)
from worldpgt.schema_induction.promotion_gates import GateConfig
from worldpgt.schema_induction.run_schema_induction import run_induction
from worldpgt.schema_induction.types import ArgumentFrame, RawClaim, SchemaInductionResult

_WS = re.compile(r"\s+")

# Generic family label -> overlay predicate. Universal relation verbs, NOT a
# domain ontology. Copula ("is") becomes a definition, handled separately.
_FAMILY_TO_PREDICATE: dict[str, str] = {
    "requires": "requires",
    "allows": "allows",
    "prohibits": "prohibits",
    "founded by": "founded_by",
    "operated by": "operated_by",
    "move/migrate": "located_in",
    "depends on": "depends_on",
}

# Role within a family that holds the object value.
_FAMILY_OBJECT_ROLE: dict[str, str] = {
    "requires": "requirement",
    "allows": "permission",
    "prohibits": "prohibition",
    "founded by": "agent",
    "operated by": "agent",
    "move/migrate": "destination",
    "depends on": "cause",
}

_MAX_DEF_LEN = 240
_MAX_OBJ_LEN = 160


_ABBREV_ARTIFACT = re.compile(r"(\S+?)\1\x00")


def _norm(text: str) -> str:
    # The shared sentence splitter mangles abbreviations into a doubled form
    # with a trailing NUL (e.g. "U.S." -> "U.SU.S\x00", "Inc." -> "IncInc\x00").
    # Collapse that artifact (keyed on the \x00 marker, which never appears in
    # normal text) before stripping any remaining control chars.
    text = text or ""
    text = _ABBREV_ARTIFACT.sub(r"\1", text)
    text = text.replace("\x00", "")
    return _WS.sub(" ", text.strip())


# ---------------------------------------------------------------------------
# Entity resolver over bootstrapped entities (no prior list needed).
# ---------------------------------------------------------------------------

class _BootstrapResolver:
    def __init__(self, entities: list[BootstrappedEntity]) -> None:
        # surface(lower) -> canonical_label
        self._surface_to_canonical: dict[str, str] = {}
        for e in entities:
            for surface in (e.canonical_label, *e.aliases):
                s = _norm(surface).lower()
                if s:
                    self._surface_to_canonical.setdefault(s, e.canonical_label)
        # Longest first for greedy containment.
        self._surfaces = sorted(
            self._surface_to_canonical, key=len, reverse=True
        )

    def resolve(self, raw: str, *, min_coverage: float = 0.0) -> str | None:
        """Resolve a raw span to a canonical bootstrapped label.

        ``min_coverage`` requires the matched entity surface to cover at least
        that fraction of the raw span's length. Use a high value (e.g. 0.6) for
        objects so a descriptive phrase ("self-petition without a U.S. agent")
        is NOT collapsed to a short entity it merely contains. Subjects use 0.0
        so "The O-1A visa" still resolves to "O-1A".
        """
        low = _norm(raw).lower()
        if not low:
            return None
        # Exact.
        if low in self._surface_to_canonical:
            return self._surface_to_canonical[low]
        # Longest known surface contained in the raw span (word boundary).
        for surface in self._surfaces:
            if re.search(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", low):
                if min_coverage and len(surface) < min_coverage * len(low):
                    continue
                return self._surface_to_canonical[surface]
        return None


# ---------------------------------------------------------------------------
# Overlay item builders
# ---------------------------------------------------------------------------

def _entity_item(e: BootstrappedEntity) -> dict:
    return {
        "overlay_type": "overlay_entity",
        "entity_id": e.entity_id,
        "label": e.canonical_label,
        "aliases": list(e.aliases),
        "entity_type": e.entity_type,
        "source_page": e.source_doc_ids[0] if e.source_doc_ids else "",
        "source_candidate_type": "domain_bootstrap",
        "trust": "overlay_candidate",
        "risk": "low",
        "bootstrap_source": "schema_induction_domain_bootstrap",
        "bootstrap_occurrences": e.occurrences,
        "bootstrap_spacy_labels": list(e.spacy_labels),
    }


def _definition_item(subject: str, definition: str, evidence: str, source: str) -> dict:
    return {
        "overlay_type": "overlay_definition",
        "subject": subject,
        "definition": definition,
        "predicate": "is_a",
        "source_page": source,
        "evidence_text": evidence,
        "trust": "overlay_candidate",
        "risk": "low",
        "stability": "stable",
        "bootstrap_source": "schema_induction_domain_bootstrap",
    }


def _relation_item(subject: str, predicate: str, obj: str, evidence: str, source: str) -> dict:
    return {
        "overlay_type": "overlay_relation",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "source_page": source,
        "evidence_text": evidence,
        "trust": "overlay_candidate",
        "risk": "medium",
        "stability": "semi_stable",
        "bootstrap_source": "schema_induction_domain_bootstrap",
    }


def _claim_for_frame(frame: ArgumentFrame, claims: dict[str, RawClaim]) -> RawClaim | None:
    for cid in frame.claim_ids:
        if cid in claims:
            return claims[cid]
    return None


def build_overlay_from_result(
    result: SchemaInductionResult,
    entities: list[BootstrappedEntity],
) -> tuple[list[dict], dict]:
    """Build overlay items from a schema induction result + bootstrapped entities."""
    resolver = _BootstrapResolver(entities)
    claims_by_id = {c.claim_id: c for c in result.claims}
    frames_by_family: dict[str, list[ArgumentFrame]] = {}
    frame_by_id = {f.frame_id: f for f in result.frames}
    for fam in result.families:
        frames_by_family[fam.family_id] = [
            frame_by_id[fid] for fid in fam.frame_ids if fid in frame_by_id
        ]

    overlay: list[dict] = []
    seen_def: set[str] = set()
    seen_rel: set[tuple] = set()

    # 1. Entities.
    for e in entities:
        overlay.append(_entity_item(e))

    # 2. Definitions (from copula families) + 3. Relations.
    n_defs = 0
    n_rels = 0
    for fam in result.families:
        is_copula = fam.canonical_label in {"is", "are", "was", "were"}
        predicate = _FAMILY_TO_PREDICATE.get(fam.canonical_label)
        obj_role = _FAMILY_OBJECT_ROLE.get(fam.canonical_label)

        for frame in frames_by_family.get(fam.family_id, []):
            raw_subject = frame.roles.get("subject", "")
            subject = resolver.resolve(raw_subject)
            if not subject:
                continue
            claim = _claim_for_frame(frame, claims_by_id)
            evidence = _norm(claim.sentence) if claim else ""
            source = claim.source_doc_id if claim else ""

            if is_copula:
                attr = _norm(frame.roles.get("attribute", ""))
                if not attr or len(attr) > _MAX_DEF_LEN:
                    continue
                key = subject.lower()
                if key in seen_def:
                    continue
                seen_def.add(key)
                overlay.append(_definition_item(subject, attr, evidence, source))
                n_defs += 1
                continue

            if predicate and obj_role:
                raw_obj = _norm(frame.roles.get(obj_role, ""))
                if not raw_obj or len(raw_obj) > _MAX_OBJ_LEN:
                    continue
                # Resolve object to a canonical entity only when it essentially
                # IS that entity (high coverage); else keep the descriptive raw
                # phrase verbatim.
                obj = resolver.resolve(raw_obj, min_coverage=0.6) or raw_obj
                if subject.lower() == obj.lower():
                    continue
                key = (subject.lower(), predicate, obj.lower())
                if key in seen_rel:
                    continue
                seen_rel.add(key)
                overlay.append(_relation_item(subject, predicate, obj, evidence, source))
                n_rels += 1

    stats = {
        "entities": len(entities),
        "definitions": n_defs,
        "relations": n_rels,
        "overlay_items": len(overlay),
    }
    return overlay, stats


def build_domain_overlay(
    docs: list[dict],
    *,
    domain: str = "domain",
    min_evidence: int = 1,
    min_sources: int = 1,
) -> dict:
    """Full three-pass bootstrap: NER -> schema induction -> overlay.

    ``docs`` is a list of {"doc_id","title","url","text"} dicts. Returns a dict
    with the overlay item list and diagnostic stats.
    """
    pairs = [(str(d.get("doc_id") or f"d{i}"), str(d.get("text") or ""))
             for i, d in enumerate(docs)]

    # Pass 1 — entity discovery from raw text (no prior list).
    entities = bootstrap_entities(pairs)

    # Pass 2 — relation extraction (schema induction over the same docs).
    result = run_induction(docs, GateConfig(min_evidence=min_evidence, min_sources=min_sources))

    # Pass 3 — overlay construction.
    overlay, stats = build_overlay_from_result(result, entities)

    return {
        "domain": domain,
        "overlay": overlay,
        "entities": entities,
        "schema_result": result,
        "stats": stats,
    }
