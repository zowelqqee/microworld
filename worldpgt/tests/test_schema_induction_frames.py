"""Frame builder assigns GENERIC semantic roles."""

from __future__ import annotations

from worldpgt.schema_induction.entity_discovery import discover_entities
from worldpgt.schema_induction.frame_builder import build_frame, build_frames
from worldpgt.schema_induction.raw_claim_extractor import extract_claims_from_sentence
from worldpgt.schema_induction.types import SentenceRecord


def _one_claim(text: str):
    sent = SentenceRecord(sentence_id="d1:s0", doc_id="d1", index=0, text=text)
    return extract_claims_from_sentence(sent)[0]


def test_requires_maps_to_requirement_role():
    claim = _one_claim("Portugal D7 visa requires proof of passive income.")
    frame = build_frame(claim)
    assert frame.trigger == "requires"
    assert frame.roles["subject"] == "Portugal D7 visa"
    assert "requirement" in frame.roles
    assert "proof of passive income" in frame.roles["requirement"]


def test_prohibits_maps_to_prohibition_role():
    claim = _one_claim("Spain non-lucrative visa prohibits work.")
    frame = build_frame(claim)
    assert frame.roles.get("prohibition") == "work"


def test_allows_maps_to_permission_role():
    claim = _one_claim("Digital nomad visa allows remote work under conditions.")
    frame = build_frame(claim)
    assert frame.roles.get("permission") == "remote work"
    assert frame.roles.get("condition") == "conditions"


def test_movement_maps_to_destination_role():
    claim = _one_claim("Wildebeest migrate toward areas with fresh grass.")
    frame = build_frame(claim)
    assert frame.roles.get("destination") == "areas with fresh grass"


def test_cause_role_from_modifier():
    claim = _one_claim("Giraffes move seasonally in search of food and water.")
    frame = build_frame(claim)
    assert frame.roles.get("cause") == "food and water"
    assert frame.roles.get("time") == "seasonally"


def test_role_types_use_local_hints():
    claim = _one_claim("Portugal D7 visa requires proof of passive income.")
    entities = discover_entities([claim])
    frames = build_frames([claim], entities)
    assert frames[0].role_types.get("subject") == "visa"
