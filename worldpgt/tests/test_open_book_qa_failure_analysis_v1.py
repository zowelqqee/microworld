from worldpgt.benchmarks.open_book_qa.failure_analysis import classify, collapse_results, evaluator_flags


def _case():
    return {"id": "a", "category": "multi_evidence", "expected_objects": ["one", "two"], "expected_decision": "answer", "relation_ids": ["e1", "e2"]}


def _result(edges=(), answer="one"):
    return {"id": "a", "answer": answer, "decision": "answer", "trace": {"answer_plan": {"blocks": [{"step": {"edge": {"evidence_id": edge, "object": edge}}} for edge in edges]}}}


def test_result_collapse_requires_one_representative_and_counts_repeats():
    collapsed, counts = collapse_results([_result(), _result()])
    assert list(collapsed) == ["a"] and counts == {"a": 2}


def test_partial_multi_evidence_is_classified_before_wrong_plan():
    stage, _, _ = classify(_case(), _result(("e1",)))
    assert stage == "partial_plan"


def test_earliest_resolution_failure_precedes_plan_outcome():
    debug = {"resolved_targets": [], "parsed_predicate": "uses", "graph_has_target": True, "expected_edge_available": True, "planner_invoked": False}
    assert classify(_case(), _result(("e1",)), debug)[0] == "entity_resolution_failed"


def test_evaluator_detects_full_surface_coverage():
    flags = evaluator_flags(_case(), _result(("e1", "e2"), answer="one and two"))
    assert flags["correct"] and flags["object_recall"] == 1.0
