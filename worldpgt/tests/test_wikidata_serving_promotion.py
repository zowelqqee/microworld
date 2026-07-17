from worldpgt.knowledge_pump.wikidata_serving_promotion import build_wikidata_serving_overlay


def test_wikidata_serving_promotion_is_queryable_but_not_accepted_memory():
    rows, summary = build_wikidata_serving_overlay([{
        "overlay_type": "overlay_relation", "subject": "Alpha", "predicate": "uses", "object": "Beta",
        "canonical_qid": "Q1", "source_kind": "wikidata_api", "open_web_extraction": "wikidata_api_structured_property_v1",
        "evidence_text": "Wikidata statement P2283 for Alpha identifies Beta.",
    }])
    assert len(rows) == 1
    assert rows[0]["serving_status"] == "user_authorized_experimental"
    assert summary["accepted_memory_modified"] is False
    assert summary["serving_relation_count"] == 1
