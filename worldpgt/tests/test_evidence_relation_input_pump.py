from worldpgt.knowledge_pump.evidence_relation_input_pump import (
    build_evidence_relation_input_proposal,
    evidence_examples,
)


def test_evidence_pump_uses_local_edge_evidence_and_never_modifies_runtime():
    items = [
        {"overlay_type": "overlay_relation", "predicate": "enables", "evidence_text": "Alpha makes Beta possible."},
        {"overlay_type": "overlay_relation", "predicate": "enables", "evidence_text": "Gamma makes Delta possible."},
        {"overlay_type": "overlay_relation", "predicate": "supports", "evidence_text": "Epsilon supports Zeta."},
    ]
    phrases = lambda text: ["make possible"] if "makes" in text else ["support"]
    always_valid = lambda _phrase, _predicate: True
    examples = evidence_examples(items, extract_phrases=phrases, validate_phrase=always_valid)
    result = build_evidence_relation_input_proposal(
        items, min_support=2, extract_phrases=phrases, validate_phrase=always_valid,
    )
    assert len(examples) == 3
    assert result["training_lane"] == "local_graph_evidence"
    assert result["accepted_memory_modified"] is False
    assert result["runtime_graph_modified"] is False
    assert result["edges"] == [{
        "source": "phrase:make_possible", "predicate": "denotes", "target": "predicate:enables",
        "support": 2,
        "evidence_ids": ["evidence:0:make possible", "evidence:1:make possible"],
    }]


def test_evidence_pump_rejects_unverified_parser_noise():
    items = [{"overlay_type": "overlay_relation", "predicate": "located_in", "evidence_text": "Daimler is in Stuttgart."}]
    assert evidence_examples(items, extract_phrases=lambda _text: ["daimler"]) == []
