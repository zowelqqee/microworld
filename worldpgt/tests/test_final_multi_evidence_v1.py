from worldpgt.experiments.build_final_multi_evidence_v1 import _phrase


def test_final_mixed_explicit_question_names_predicates_without_templates():
    assert _phrase("created_by") == "created by"
    assert _phrase("wikidata_p527_has_part_s") == "has part"
