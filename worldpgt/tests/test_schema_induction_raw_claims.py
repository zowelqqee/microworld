"""Raw claim extraction preserves SURFACE relations (no domain predicates)."""

from __future__ import annotations

from worldpgt.schema_induction.raw_claim_extractor import (
    extract_claims,
    extract_claims_from_sentence,
)
from worldpgt.schema_induction.types import DocumentRecord, SentenceRecord


def _claim(text: str):
    sent = SentenceRecord(sentence_id="d1:s0", doc_id="d1", index=0, text=text)
    claims = extract_claims_from_sentence(sent)
    assert claims, f"no claim extracted from: {text}"
    return claims[0]


def test_requires_surface_preserved():
    claim = _claim("Portugal D7 visa requires proof of passive income.")
    assert "Portugal D7 visa" in claim.subject
    # Surface relation, NOT a canonical predicate like requires_document.
    assert claim.relation_surface == "requires"
    assert claim.object is not None
    assert "proof of passive income" in claim.object
    # No manual predicate field exists on the claim.
    assert not hasattr(claim, "predicate")
    assert claim.source_doc_id == "d1"
    assert claim.source_sentence_id == "d1:s0"


def test_directional_movement_exposes_destination():
    claim = _claim("Wildebeest migrate toward areas with fresh grass.")
    assert claim.relation_surface == "migrate toward"
    assert claim.modifiers.get("destination") == "areas with fresh grass"


def test_cause_modifier_captured():
    claim = _claim("Giraffes move seasonally in search of food and water.")
    assert claim.relation_surface == "move"
    assert "food and water" == claim.modifiers.get("cause")
    assert claim.modifiers.get("time") == "seasonally"


def test_passive_agentive_surface():
    claim = _claim("Starlink is operated by SpaceX.")
    assert claim.relation_surface == "is operated by"
    assert claim.object == "SpaceX"


def test_no_canonical_mapping_for_founded():
    claim = _claim("SpaceX was founded by Elon Musk.")
    # Surface kept verbatim; not mapped to "founded_by".
    assert claim.relation_surface == "was founded by"
    assert claim.relation_surface != "founded_by"


def test_extract_claims_over_documents():
    docs = [
        DocumentRecord(
            doc_id="d1", title="t", url="",
            text="Spain non-lucrative visa prohibits work. "
                 "Spain non-lucrative visa requires proof of financial means.",
        )
    ]
    sentences, claims = extract_claims(docs)
    assert len(sentences) == 2
    surfaces = {c.relation_surface for c in claims}
    assert "prohibits" in surfaces
    assert "requires" in surfaces


def test_every_claim_has_source_trace():
    docs = [DocumentRecord(doc_id="dX", title="t", url="",
                           text="Digital nomad visa allows remote work under conditions.")]
    _, claims = extract_claims(docs)
    assert claims
    for c in claims:
        assert c.source_doc_id == "dX"
        assert c.source_sentence_id.startswith("dX:s")
        assert c.sentence
