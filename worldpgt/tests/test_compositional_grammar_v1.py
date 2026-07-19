import pytest

from worldpgt.reasoning.compositional_grammar_v1 import AndQuery, ChainQuery, CompositionalGrammar, RelationRequest, parse_candidate


def row(subject, predicate, obj, **extra):
    return {"subject": subject, "predicate": predicate, "object": obj, "evidence_id": f"e:{subject}:{predicate}:{obj}", **extra}


@pytest.mark.parametrize("left,right", [
    ("uses", "developed_by"), ("created_by", "published_by"), ("country_of_origin", "language"),
    ("runs_on", "used_for"), ("founded_by", "headquartered_in"), ("owned_by", "produces"),
    ("parent_company_of", "product_of"), ("enables", "supports"), ("works_by", "uses"),
    ("named_after", "opposite_of"), ("has_part", "writing_system"), ("created_by", "uses"),
])
def test_and_is_predicate_data_driven_not_a_pair_enum(left, right):
    plan = CompositionalGrammar([row("X", left, "A"), row("X", right, "B")]).execute(
        AndQuery("X", (RelationRequest(left), RelationRequest(right)))
    )
    assert plan.decision == "answer"
    assert [ref.edge.predicate for ref in plan.evidence] == [left, right]
    assert len({ref.evidence_id for ref in plan.evidence}) == 2


def test_and_preserves_fanout_and_audits_missing_support():
    grammar = CompositionalGrammar([row("X", "created_by", "A"), row("X", "created_by", "B"), row("X", "published_by", "C")])
    plan = grammar.execute(AndQuery("X", (RelationRequest("created_by"), RelationRequest("published_by"))))
    assert plan.decision == "answer" and [len(c) for c in plan.components] == [2, 1]
    assert grammar.execute(AndQuery("X", (RelationRequest("created_by"), RelationRequest("missing")))).audit_reason == "missing_predicate_support:missing"


def test_chain_joins_on_object_subject_and_preserves_both_provenances():
    plan = CompositionalGrammar([row("X", "uses", "Y"), row("Y", "developed_by", "Z")]).execute(ChainQuery("X", "uses", "developed_by"))
    assert plan.decision == "answer"
    assert [(r.edge.subject, r.edge.predicate, r.edge.object) for r in plan.evidence] == [("X", "uses", "Y"), ("Y", "developed_by", "Z")]


def test_chain_refuses_missing_or_unsafe_component():
    assert CompositionalGrammar([row("X", "uses", "Y")]).execute(ChainQuery("X", "uses", "developed_by")).audit_reason == "missing_second_hop_support"
    unsafe = CompositionalGrammar([row("X", "uses", "Y", stability="volatile"), row("Y", "developed_by", "Z")]).execute(ChainQuery("X", "uses", "developed_by"))
    assert unsafe.decision == "audit" and unsafe.audit_reason.startswith("unsafe_component:")


def test_parser_finds_dynamic_explicit_and_and_bounded_implicit_marker():
    relations = [row("Adobe", "country_of_origin", "US"), row("Adobe", "writing_system", "Latin")]
    parsed = parse_candidate("For Adobe, what are its country of origin and writing system relations?", relations)
    assert isinstance(parsed, AndQuery) and [x.predicate for x in parsed.relations] == ["country_of_origin", "writing_system"]
    assert isinstance(parse_candidate("Tell me two key relations about Adobe.", relations), AndQuery)
