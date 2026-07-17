from worldpgt.knowledge_pump.wikidata_property_gate import validate_wikidata_property_proposals
from worldpgt.knowledge_pump.wikidata_relation_layer import build_content_property_proposals


def _claim(qid: str) -> dict:
    return {"mainsnak": {"snaktype": "value", "datatype": "wikibase-item", "datavalue": {"value": {"id": qid}}}}


def test_wikidata_seed_maps_existing_properties_and_quarantines_one_off_schema_gaps():
    subjects = [
        {"subject": "Alpha", "canonical_qid": "Q1"},
        {"subject": "Beta", "canonical_qid": "Q2"},
        {"subject": "Gamma", "canonical_qid": "Q3"},
    ]
    entities = {
        "Q1": {"claims": {"P178": [_claim("Q10")], "P999": [_claim("Q11")]}},
        "Q2": {"claims": {"P999": [_claim("Q12")]}},
        "Q3": {"claims": {"P999": [_claim("Q13")]}},
    }
    candidates, quarantine, counts = build_content_property_proposals(
        subjects, entities=entities,
        labels={"Q10": "Developer", "Q11": "One", "Q12": "Two", "Q13": "Three"},
        property_labels={"P178": "developer", "P999": "example property"},
    )
    assert counts == {"P178": 1, "P999": 3}
    assert {(row["wikidata_property"], row["predicate"]) for row in candidates} == {
        ("P178", "developed_by"), ("P999", "wikidata_p999_example_property"),
    }
    assert quarantine == []


def test_wikidata_seed_gate_is_proposal_only_and_uses_source_specific_provenance():
    row = {
        "overlay_type": "overlay_relation", "subject": "Alpha", "canonical_qid": "Q1",
        "predicate": "developed_by", "object": "Developer", "source_kind": "wikidata_api",
        "open_web_extraction": "wikidata_api_structured_property_v1",
        "evidence_text": "Wikidata statement P178 for Alpha identifies Developer.",
    }
    report = validate_wikidata_property_proposals([row])
    assert report["proposal_only"] is True
    assert report["accepted_memory_modified"] is False
    assert report["passed_source_gate"] == 1
