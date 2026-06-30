"""Relation family induction groups repeated relation structure; local types
are induced without a global enum."""

from __future__ import annotations

from worldpgt.schema_induction.entity_discovery import discover_entities
from worldpgt.schema_induction.frame_builder import build_frames
from worldpgt.schema_induction.local_type_inducer import induce_local_types
from worldpgt.schema_induction.raw_claim_extractor import extract_claims
from worldpgt.schema_induction.relation_family_builder import build_relation_families
from worldpgt.schema_induction.types import DocumentRecord


def _pipeline(docs):
    sentences, claims = extract_claims(docs)
    entities = discover_entities(claims, sentences)
    frames = build_frames(claims, entities)
    claim_doc = {c.claim_id: c.source_doc_id for c in claims}
    families = build_relation_families(frames, claim_doc)
    return claims, entities, frames, families


def test_requires_family_groups_synonyms():
    docs = [
        DocumentRecord("d1", "t", "", "Portugal D7 visa requires proof of passive income."),
        DocumentRecord("d1b", "t", "", "Portugal D7 visa needs accommodation."),
        DocumentRecord("d2", "t", "", "Spain non-lucrative visa requires financial means."),
    ]
    _, _, _, families = _pipeline(docs)
    requires = [f for f in families if f.canonical_label == "requires"]
    assert len(requires) == 1
    fam = requires[0]
    # require + need collapse into one generated family.
    assert {"requires", "needs"} <= set(fam.surface_forms)
    assert "requirement" in fam.roles
    assert fam.evidence_count >= 3
    assert fam.promotion_status == "generated"


def test_local_type_induced_without_global_enum():
    docs = [
        DocumentRecord("d1", "t", "", "Portugal D7 visa requires proof of passive income."),
        DocumentRecord("d2", "t", "", "Spain non-lucrative visa prohibits work."),
        DocumentRecord("d3", "t", "", "Digital nomad visa allows remote work under conditions."),
    ]
    _, entities, frames, _ = _pipeline(docs)
    local_types = induce_local_types(entities, frames)
    visa_types = [lt for lt in local_types if lt.label == "visa"]
    assert visa_types, "expected an induced 'visa' local type"
    members = visa_types[0].members
    assert len(members) >= 2
    assert any("Portugal D7 visa" in m for m in members)
    assert any("Spain non-lucrative visa" in m for m in members)


def test_movement_families_keep_distinct_role_structure():
    docs = [
        DocumentRecord("d4", "t", "", "Giraffes move seasonally in search of food and water."),
        DocumentRecord("d5", "t", "", "Wildebeest migrate toward areas with fresh grass."),
    ]
    _, _, _, families = _pipeline(docs)
    moves = [f for f in families if f.canonical_label == "move/migrate"]
    # Same canonical label, but cause-bearing and destination-bearing frames
    # form separate families (role structure differs).
    role_sets = {tuple(f.roles) for f in moves}
    assert any("cause" in rs for rs in role_sets)
    assert any("destination" in rs for rs in role_sets)
