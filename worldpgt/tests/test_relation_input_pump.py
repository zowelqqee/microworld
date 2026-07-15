from worldpgt.knowledge_pump.relation_input_pump import build_relation_input_proposal, write_relation_input_proposal


def test_relation_input_pump_emits_only_supported_unambiguous_proposal_edges(tmp_path):
    examples = [
        {"id": "a", "question": "What does Alpha make possible?", "expected_subject": "Alpha", "expected_predicate": ["enables"]},
        {"id": "b", "question": "What does Beta make possible?", "expected_subject": "Beta", "expected_predicate": ["enables"]},
        {"id": "c", "question": "What does Gamma make possible?", "expected_subject": "Gamma", "expected_predicate": ["supports"]},
        {"id": "d", "question": "What mechanism does Delta use?", "expected_subject": "Delta", "expected_predicate": ["works_by"]},
    ]
    proposal = build_relation_input_proposal(examples, min_support=2)
    assert proposal["proposal_only"] is True
    assert proposal["accepted_memory_modified"] is False
    assert proposal["runtime_graph_modified"] is False
    assert proposal["edges"] == []
    assert {item["reason"] for item in proposal["rejected"]} == {"conflicting_predicate_labels", "insufficient_support"}

    stable = [item for item in examples if item["id"] != "c"]
    written = write_relation_input_proposal(stable, tmp_path, min_support=2)
    assert written["edges"] == [{
        "source": "phrase:make_possible", "predicate": "denotes", "target": "predicate:enables",
        "support": 2, "evidence_ids": ["a", "b"],
    }]
    assert (tmp_path / "relation_input_graph_proposal.json").exists()
    assert (tmp_path / "relation_input_graph_training_report.json").exists()
