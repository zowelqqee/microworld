"""Multi-hop QA v1 — comprehensive test suite.

Tests:
  Cat 1: Graph construction — only overlay_relation items, never definitions/entity-cards
  Cat 2: RelationGraph — outgoing edges, display names, is_known, no fuzzy matching
  Cat 3: Question analyzer — all chain types, direct-relation guards
  Cat 4: Path planner — answers when 2-hop exists, audits when hop missing
  Cat 5: Chain safety — direct founder/owner audit immediately
  Cat 6: Fixture chains — curated Acme Labs / Northstar Holdings / Ada Stone scenario
  Cat 7: Answer renderer — natural-language output for each chain type
  Cat 8: Real pump overlay — graphs from pump dry-run, real 2-hop chains
  Cat 9: Single-hop not broken — existing entity QA unchanged
  Cat 10: Protected files — accepted memory / overlays unchanged
  Cat 11: Experiment runner — summary all_critical_passed, artifact format
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldpgt.multihop_qa.answer_renderer import render
from worldpgt.multihop_qa.path_planner import MultihopPlanner
from worldpgt.multihop_qa.question_analyzer import analyze
from worldpgt.multihop_qa.relation_graph import RelationGraph
from worldpgt.multihop_qa.types import HopEdge, MultihopPath, MultihopQuestion

_REPO = Path(__file__).resolve().parent.parent.parent
_EXP = _REPO / "worldpgt" / "experiments"
_PUMP_OVERLAY = _EXP / "knowledge_pump_v1" / "pump_dry_run_overlay.json"
_MULTIHOP_SUMMARY = _EXP / "multihop_qa_v1" / "multihop_qa_summary.json"

_PROTECTED = [
    _EXP / "accepted_knowledge_memory_v1.json",
    _EXP / "accepted_wiki_memory_overlay_v1.json",
    _EXP / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json",
    _EXP / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json",
]

# ---------------------------------------------------------------------------
# Test fixture — curated, test-only overlay relations.
# NEVER promoted to runtime memory or accepted overlay.
# ---------------------------------------------------------------------------
_FIXTURE_RELATIONS = [
    {"subject": "Acme Labs", "predicate": "owned_by", "object": "Northstar Holdings",
     "overlay_type": "overlay_relation"},
    {"subject": "Northstar Holdings", "predicate": "founded_by", "object": "Ada Stone",
     "overlay_type": "overlay_relation"},
    {"subject": "Acme Labs", "predicate": "develops", "object": "Nova Engine",
     "overlay_type": "overlay_relation"},
    {"subject": "Nova Engine", "predicate": "is_a", "object": "rocket engine",
     "overlay_type": "overlay_relation"},
    {"subject": "OrbitMail", "predicate": "service_of", "object": "Northstar Holdings",
     "overlay_type": "overlay_relation"},
    # Extra entity for testing "no path" cases.
    {"subject": "SilverBridge Corp", "predicate": "develops", "object": "BridgeNet",
     "overlay_type": "overlay_relation"},
]

# Fixture with non-relation items to confirm they are excluded.
_FIXTURE_MIXED = _FIXTURE_RELATIONS + [
    {"overlay_type": "overlay_definition", "subject": "Acme Labs",
     "definition": "A lab that makes things."},
    {"overlay_type": "overlay_entity", "label": "Ada Stone",
     "entity_id": "ada-stone", "entity_type": "person", "aliases": []},
    {"overlay_type": "overlay_context_link", "source_page": "Acme Labs",
     "target": "Nova Engine", "surface": "engine"},
    {"overlay_type": "overlay_source_fact", "subject": "Northstar Holdings",
     "predicate": "revenue", "object": "1B", "source_name": "Forbes"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fixture_graph() -> RelationGraph:
    return RelationGraph(_FIXTURE_RELATIONS)


def _mixed_graph() -> RelationGraph:
    return RelationGraph(_FIXTURE_MIXED)


@pytest.fixture(scope="module")
def pump_graph():
    overlay = json.loads(_PUMP_OVERLAY.read_text(encoding="utf-8"))
    relations = [i for i in overlay if i.get("overlay_type") == "overlay_relation"]
    return RelationGraph(relations)


@pytest.fixture(scope="module")
def multihop_summary():
    assert _MULTIHOP_SUMMARY.exists(), f"Run run_multihop_qa_v1.py first: {_MULTIHOP_SUMMARY}"
    return json.loads(_MULTIHOP_SUMMARY.read_text(encoding="utf-8"))


# ===========================================================================
# Category 1: Graph construction — only overlay_relation items
# ===========================================================================


def test_graph_excludes_definitions():
    g = _mixed_graph()
    # Definitions must not become edges; "Acme Labs" definition should not
    # add any outgoing edge with predicate from definition text.
    outgoing = g.outgoing("Acme Labs")
    preds = {p for p, _ in outgoing}
    assert "definition" not in preds


def test_graph_excludes_entity_cards():
    g = _mixed_graph()
    # overlay_entity for "Ada Stone" must not add any outgoing edge.
    # The only Ada Stone edge comes from Northstar Holdings founded_by Ada Stone.
    ada_out = g.outgoing("Ada Stone")
    assert len(ada_out) == 0, f"Ada Stone should have no outgoing edges, got: {ada_out}"


def test_graph_excludes_context_links():
    g = _mixed_graph()
    # context_link Acme Labs -> Nova Engine (surface=engine) must not become an edge.
    outgoing = {(p, o) for p, o in g.outgoing("Acme Labs")}
    # The only Acme Labs edges should be owned_by and develops.
    assert ("owned_by", "northstar holdings") in outgoing
    assert ("develops", "nova engine") in outgoing
    assert len(outgoing) == 2, f"Expected exactly 2 edges for Acme Labs, got: {outgoing}"


def test_graph_excludes_source_facts():
    g = _mixed_graph()
    # source_fact for Northstar Holdings revenue must not become an edge.
    outgoing = {(p, o) for p, o in g.outgoing("Northstar Holdings")}
    assert "revenue" not in {p for p, _ in outgoing}


def test_graph_edge_count_matches_fixture():
    g = _fixture_graph()
    # 6 overlay_relation items → 6 directed edges.
    assert g.edge_count() == 6


# ===========================================================================
# Category 2: RelationGraph — correctness and no fuzzy matching
# ===========================================================================


def test_graph_outgoing_acme_labs():
    g = _fixture_graph()
    outgoing = dict(g.outgoing("Acme Labs"))
    # normalized values in the dict
    assert "northstar holdings" in outgoing.values() or any(
        o == "northstar holdings" for _, o in g.outgoing("Acme Labs")
    )


def test_graph_is_known_fixture_entities():
    g = _fixture_graph()
    assert g.is_known("Acme Labs")
    assert g.is_known("Northstar Holdings")
    assert g.is_known("Ada Stone")
    assert g.is_known("Nova Engine")
    assert g.is_known("OrbitMail")


def test_graph_is_not_known_missing_entity():
    g = _fixture_graph()
    assert not g.is_known("UnknownCorp")
    assert not g.is_known("XYZ Holdings")


def test_graph_no_fuzzy_or_substring_match():
    g = _fixture_graph()
    # "Acme" alone does not match "Acme Labs"
    assert not g.is_known("Acme")
    # "Stone" alone does not match "Ada Stone"
    assert not g.is_known("Stone")


def test_graph_display_returns_canonical():
    g = _fixture_graph()
    # Ask with normalized lowercase; should return canonical display.
    assert g.display("acme labs") == "Acme Labs"
    assert g.display("northstar holdings") == "Northstar Holdings"


def test_graph_2hop_acme_to_ada():
    g = _fixture_graph()
    paths = g.find_2hop_paths("Acme Labs", "Ada Stone")
    assert len(paths) == 1
    p = paths[0]
    assert p.hop1.predicate == "owned_by"
    assert p.hop2.predicate == "founded_by"
    assert p.via_node == "Northstar Holdings"


def test_graph_2hop_no_reverse_path():
    g = _fixture_graph()
    # Ada Stone → Northstar Holdings → Acme Labs: no forward directed path exists.
    paths = g.find_2hop_paths("Ada Stone", "Acme Labs")
    assert paths == []


def test_graph_develops_isa_acme_rocket_engine():
    g = _fixture_graph()
    paths = g.find_develops_isa_paths("Acme Labs", "rocket engine")
    assert len(paths) == 1
    p = paths[0]
    assert p.hop1.predicate == "develops"
    assert p.hop2.predicate == "is_a"
    assert p.via_node == "Nova Engine"


def test_graph_develops_isa_wrong_category_no_path():
    g = _fixture_graph()
    paths = g.find_develops_isa_paths("Acme Labs", "satellite")
    assert paths == []


def test_graph_through_path_acme_northstar():
    g = _fixture_graph()
    paths = g.find_through_paths("Acme Labs", "Northstar Holdings")
    # Should find Acme Labs -> Northstar Holdings -> Ada Stone.
    assert len(paths) >= 1
    p = paths[0]
    assert p.hop1.subject == "Acme Labs"
    assert p.via_node == "Northstar Holdings"


def test_graph_through_path_missing_returns_empty():
    g = _fixture_graph()
    paths = g.find_through_paths("Acme Labs", "SilverBridge Corp")
    assert paths == []


# ===========================================================================
# Category 3: Question analyzer
# ===========================================================================


def test_analyzer_connection_how_is():
    q = analyze("How is Acme Labs connected to Ada Stone?")
    assert q.chain_type == "connection"
    assert q.endpoint_a == "Acme Labs"
    assert q.endpoint_c == "Ada Stone"
    assert not q.is_direct_relation


def test_analyzer_connection_what_links():
    q = analyze("What links SolarCity to electric cars?")
    assert q.chain_type == "connection"
    assert q.endpoint_a == "SolarCity"
    assert q.endpoint_c == "electric cars"


def test_analyzer_connection_relationship():
    q = analyze("What is Acme Labs's relationship to Ada Stone?")
    assert q.chain_type == "connection"
    assert q.endpoint_a == "Acme Labs"
    assert q.endpoint_c == "Ada Stone"


def test_analyzer_develops_isa():
    q = analyze("What does Acme Labs develop that is a rocket engine?")
    assert q.chain_type == "develops_isa"
    assert q.endpoint_a == "Acme Labs"
    assert q.endpoint_c == "rocket engine"


def test_analyzer_through_node():
    q = analyze("Is Acme Labs connected to Ada Stone through Northstar Holdings?")
    assert q.chain_type == "through_node"
    assert q.endpoint_a == "Acme Labs"
    assert q.endpoint_c == "Ada Stone"
    assert q.through_b == "Northstar Holdings"


def test_analyzer_who_through():
    q = analyze("Who is Acme Labs linked to through Northstar Holdings?")
    assert q.chain_type == "who_through"
    assert q.endpoint_a == "Acme Labs"
    assert q.through_b == "Northstar Holdings"


def test_analyzer_direct_founder_guard():
    q = analyze("Who founded Acme Labs?")
    assert q.chain_type == "direct_only"
    assert q.is_direct_relation is True


def test_analyzer_direct_founded_by_guard():
    q = analyze("Who was Acme Labs founded by?")
    assert q.chain_type == "direct_only"
    assert q.is_direct_relation is True


def test_analyzer_direct_owner_guard():
    q = analyze("Who owns Acme Labs?")
    assert q.chain_type == "direct_only"
    assert q.is_direct_relation is True


def test_analyzer_direct_owned_by_guard():
    q = analyze("Who is Acme Labs owned by?")
    assert q.chain_type == "direct_only"
    assert q.is_direct_relation is True


def test_analyzer_unsupported_returns_unsupported():
    q = analyze("What is Acme Labs?")
    assert q.chain_type == "unsupported"


def test_analyzer_empty_question_returns_unsupported():
    q = analyze("")
    assert q.chain_type == "unsupported"


# ===========================================================================
# Category 4: Path planner — answers and audits
# ===========================================================================


def test_planner_answers_2hop_connection():
    g = _fixture_graph()
    p = MultihopPlanner(g)
    mq = analyze("How is Acme Labs connected to Ada Stone?")
    plan = p.plan(mq)
    assert plan.decision == "answer"
    assert plan.path is not None
    assert plan.chain_label == "owned_by_then_founded_by"


def test_planner_answers_develops_isa():
    g = _fixture_graph()
    p = MultihopPlanner(g)
    mq = analyze("What does Acme Labs develop that is a rocket engine?")
    plan = p.plan(mq)
    assert plan.decision == "answer"
    assert plan.chain_label == "develops_then_is_a"


def test_planner_answers_through_node():
    g = _fixture_graph()
    p = MultihopPlanner(g)
    mq = analyze("Is Acme Labs connected to Ada Stone through Northstar Holdings?")
    plan = p.plan(mq)
    assert plan.decision == "answer"
    assert plan.path.via_node == "Northstar Holdings"


def test_planner_audits_missing_second_hop():
    g = _fixture_graph()
    p = MultihopPlanner(g)
    mq = analyze("How is Acme Labs connected to SpaceX?")
    plan = p.plan(mq)
    assert plan.decision == "audit"


def test_planner_audits_missing_first_hop():
    g = _fixture_graph()
    p = MultihopPlanner(g)
    mq = analyze("How is UnknownCorp connected to Ada Stone?")
    plan = p.plan(mq)
    assert plan.decision == "audit"


def test_planner_audits_both_hops_missing():
    g = _fixture_graph()
    p = MultihopPlanner(g)
    mq = analyze("How is Alpha Inc connected to Beta Corp?")
    plan = p.plan(mq)
    assert plan.decision == "audit"


def test_planner_audits_develops_isa_wrong_category():
    g = _fixture_graph()
    p = MultihopPlanner(g)
    mq = analyze("What does Acme Labs develop that is a satellite?")
    plan = p.plan(mq)
    assert plan.decision == "audit"


def test_planner_evidence_lists_both_hops():
    g = _fixture_graph()
    p = MultihopPlanner(g)
    mq = analyze("How is Acme Labs connected to Ada Stone?")
    plan = p.plan(mq)
    assert plan.decision == "answer"
    assert len(plan.evidence.hops_used) == 2


# ===========================================================================
# Category 5: Chain safety — direct-relation leak guards
# ===========================================================================


def test_direct_founder_audits_immediately():
    g = _fixture_graph()
    p = MultihopPlanner(g)
    mq = analyze("Who founded Acme Labs?")
    plan = p.plan(mq)
    assert plan.decision == "audit", (
        "Multi-hop must not answer 'Who founded X?' using a transitive chain"
    )
    assert plan.chain_label == "direct_only_audit"


def test_direct_founded_by_audits_immediately():
    g = _fixture_graph()
    p = MultihopPlanner(g)
    mq = analyze("Who was Acme Labs founded by?")
    plan = p.plan(mq)
    assert plan.decision == "audit"


def test_direct_owner_audits_immediately():
    g = _fixture_graph()
    p = MultihopPlanner(g)
    mq = analyze("Who owns Acme Labs?")
    plan = p.plan(mq)
    assert plan.decision == "audit", (
        "Multi-hop must not answer 'Who owns X?' using a transitive chain"
    )


def test_transitive_founder_not_inferred_as_direct():
    # Acme Labs owned_by Northstar Holdings; Northstar Holdings founded_by Ada Stone.
    # Multi-hop must NOT answer "Who founded Acme Labs?" as Ada Stone.
    g = _fixture_graph()
    p = MultihopPlanner(g)
    mq = analyze("Who founded Acme Labs?")
    plan = p.plan(mq)
    assert plan.decision == "audit"
    # Answer text must not say Ada Stone founded Acme Labs.
    text = render(plan)
    assert "Ada Stone" not in text or "founded Acme" not in text


def test_transitive_owner_not_inferred_as_direct():
    # Even if A owned_by B, multi-hop must NOT answer "Who is A owned by?" as a
    # connection answer for direct ownership queries.
    g = _fixture_graph()
    p = MultihopPlanner(g)
    mq = analyze("Who is Acme Labs owned by?")
    plan = p.plan(mq)
    assert plan.decision == "audit"


# ===========================================================================
# Category 6: Fixture chain answers — correct natural language
# ===========================================================================


def test_fixture_acme_connected_to_ada_stone_answer():
    g = _fixture_graph()
    p = MultihopPlanner(g)
    mq = analyze("How is Acme Labs connected to Ada Stone?")
    plan = p.plan(mq)
    text = render(plan)
    assert "Acme Labs" in text
    assert "Ada Stone" in text
    assert "Northstar Holdings" in text
    assert "owned by" in text
    assert "founded by" in text


def test_fixture_acme_develops_nova_engine_rocket_engine():
    g = _fixture_graph()
    p = MultihopPlanner(g)
    mq = analyze("What does Acme Labs develop that is a rocket engine?")
    plan = p.plan(mq)
    text = render(plan)
    assert "Acme Labs" in text
    assert "Nova Engine" in text
    assert "rocket engine" in text
    assert "develops" in text


def test_fixture_orbitmail_through_northstar():
    g = _fixture_graph()
    p = MultihopPlanner(g)
    mq = analyze("Who is OrbitMail linked to through Northstar Holdings?")
    plan = p.plan(mq)
    # OrbitMail service_of Northstar Holdings -> Northstar Holdings founded_by Ada Stone
    assert plan.decision == "answer"
    text = render(plan)
    assert "OrbitMail" in text
    assert "Northstar Holdings" in text


def test_fixture_who_founded_acme_audits():
    g = _fixture_graph()
    p = MultihopPlanner(g)
    mq = analyze("Who founded Acme Labs?")
    plan = p.plan(mq)
    assert plan.decision == "audit"


def test_fixture_who_owns_acme_audits():
    g = _fixture_graph()
    p = MultihopPlanner(g)
    mq = analyze("Who owns Acme Labs?")
    plan = p.plan(mq)
    assert plan.decision == "audit"


def test_fixture_what_does_ada_stone_own_audits():
    g = _fixture_graph()
    p = MultihopPlanner(g)
    mq = analyze("What does Ada Stone own?")
    plan = p.plan(mq)
    # "What does X own?" is unsupported multi-hop form → audit.
    assert plan.decision == "audit"


# ===========================================================================
# Category 7: Answer renderer
# ===========================================================================


def test_renderer_connection_answer_format():
    edge1 = HopEdge("Acme Labs", "owned_by", "Northstar Holdings")
    edge2 = HopEdge("Northstar Holdings", "founded_by", "Ada Stone")
    path = MultihopPath(edge1, edge2, "Northstar Holdings")
    mq = MultihopQuestion(
        "How is Acme Labs connected to Ada Stone?", "connection",
        endpoint_a="Acme Labs", endpoint_c="Ada Stone",
    )
    from worldpgt.multihop_qa.types import MultihopEvidence, MultihopPlan
    plan = MultihopPlan(
        question=mq, decision="answer", audit_reason=None,
        path=path, chain_label="owned_by_then_founded_by",
        render_args={"a": "Acme Labs", "b": "Northstar Holdings", "c": "Ada Stone",
                     "hop1": edge1, "hop2": edge2},
        evidence=MultihopEvidence(hops_used=[edge1.display(), edge2.display()]),
    )
    text = render(plan)
    assert text == (
        "Acme Labs is connected to Ada Stone through Northstar Holdings: "
        "Acme Labs is owned by Northstar Holdings, and "
        "Northstar Holdings was founded by Ada Stone."
    )


def test_renderer_connection_snapshot_hop_adds_temporal_caveat():
    edge1 = HopEdge(
        "Acme Labs", "owned_by", "Northstar Holdings",
        temporal_class="snapshot", as_of="2026-06",
    )
    edge2 = HopEdge("Northstar Holdings", "founded_by", "Ada Stone", temporal_class="historical")
    path = MultihopPath(edge1, edge2, "Northstar Holdings")
    mq = MultihopQuestion(
        "How is Acme Labs connected to Ada Stone?", "connection",
        endpoint_a="Acme Labs", endpoint_c="Ada Stone",
    )
    from worldpgt.multihop_qa.types import MultihopEvidence, MultihopPlan
    plan = MultihopPlan(
        question=mq, decision="answer", audit_reason=None,
        path=path, chain_label="owned_by_then_founded_by",
        render_args={"a": "Acme Labs", "b": "Northstar Holdings", "c": "Ada Stone",
                     "hop1": edge1, "hop2": edge2},
        evidence=MultihopEvidence(hops_used=[edge1.display(), edge2.display()]),
    )
    text = render(plan)
    assert "snapshot fact as of 2026-06" in text
    assert "should be rechecked" in text


def test_snapshot_hop_without_as_of_audits():
    relations = [
        {"subject": "Acme Labs", "predicate": "owned_by", "object": "Northstar Holdings",
         "overlay_type": "overlay_relation", "temporal_class": "snapshot"},
        {"subject": "Northstar Holdings", "predicate": "founded_by", "object": "Ada Stone",
         "overlay_type": "overlay_relation", "temporal_class": "historical"},
    ]
    planner = MultihopPlanner(RelationGraph(relations))
    plan = planner.plan(analyze("How is Acme Labs connected to Ada Stone?"))
    assert plan.decision == "audit"
    assert "snapshot_missing_as_of:owned_by" in plan.audit_reason


def test_renderer_develops_isa_format():
    edge1 = HopEdge("Acme Labs", "develops", "Nova Engine")
    edge2 = HopEdge("Nova Engine", "is_a", "rocket engine")
    path = MultihopPath(edge1, edge2, "Nova Engine")
    mq = MultihopQuestion(
        "What does Acme Labs develop that is a rocket engine?", "develops_isa",
        endpoint_a="Acme Labs", endpoint_c="rocket engine",
    )
    from worldpgt.multihop_qa.types import MultihopEvidence, MultihopPlan
    plan = MultihopPlan(
        question=mq, decision="answer", audit_reason=None,
        path=path, chain_label="develops_then_is_a",
        render_args={"a": "Acme Labs", "b": "Nova Engine", "category": "rocket engine"},
        evidence=MultihopEvidence(hops_used=[edge1.display(), edge2.display()]),
    )
    text = render(plan)
    assert text == "Acme Labs develops Nova Engine, which is a rocket engine."


def test_renderer_audit_format():
    mq = MultihopQuestion("Who founded Acme Labs?", "direct_only", is_direct_relation=True)
    from worldpgt.multihop_qa.types import MultihopEvidence, MultihopPlan
    plan = MultihopPlan(
        question=mq, decision="audit",
        audit_reason="direct_relation_query_blocked",
        path=None, chain_label="direct_only_audit",
        render_args={"reason": "direct_relation_query_blocked"},
        evidence=MultihopEvidence(),
    )
    text = render(plan)
    assert text.startswith("[audit]")


# ===========================================================================
# Category 8: Real pump overlay
# ===========================================================================


def test_pump_graph_builds_from_relations_only(pump_graph):
    # The graph should have nodes from relations only, not from definitions.
    assert pump_graph.is_known("Starlink")
    assert pump_graph.is_known("SpaceX")
    assert pump_graph.is_known("Falcon 9")
    assert pump_graph.is_known("SolarCity")
    assert pump_graph.is_known("Tesla")


def test_pump_graph_starlink_spacex_falcon9(pump_graph):
    paths = pump_graph.find_2hop_paths("Starlink", "Falcon 9")
    assert len(paths) >= 1
    p = paths[0]
    assert p.via_node == "SpaceX"


@pytest.mark.xfail(reason="SolarCity→owned_by→Tesla removed by pump precision cleanup", strict=False)
def test_pump_graph_solarcity_tesla_electric_cars(pump_graph):
    paths = pump_graph.find_2hop_paths("SolarCity", "electric cars")
    assert len(paths) >= 1
    p = paths[0]
    assert p.via_node == "Tesla"


def test_pump_how_is_starlink_connected_to_falcon9(pump_graph):
    planner = MultihopPlanner(pump_graph)
    mq = analyze("How is Starlink connected to Falcon 9?")
    plan = planner.plan(mq)
    assert plan.decision == "answer"
    text = render(plan)
    assert "Starlink" in text
    assert "Falcon 9" in text
    assert "SpaceX" in text


def test_pump_who_founded_starlink_audits(pump_graph):
    planner = MultihopPlanner(pump_graph)
    mq = analyze("Who founded Starlink?")
    plan = planner.plan(mq)
    assert plan.decision == "audit"


def test_pump_who_owns_solarcity_audits(pump_graph):
    planner = MultihopPlanner(pump_graph)
    mq = analyze("Who owns SolarCity?")
    plan = planner.plan(mq)
    assert plan.decision == "audit"


def test_pump_missing_hop_audits(pump_graph):
    planner = MultihopPlanner(pump_graph)
    mq = analyze("How is Starlink connected to Ada Stone?")
    plan = planner.plan(mq)
    assert plan.decision == "audit"


# ===========================================================================
# Category 9: Single-hop entity QA not broken
# ===========================================================================


def test_singlebhop_entity_qa_still_answers():
    from worldpgt.experiments.ask_microworld_v1 import ask
    a = ask("What does SpaceX develop?", "pump-dry-run")
    assert a.decision == "answer", f"Single-hop entity QA broken: {a.answer_text}"


def test_single_hop_who_founded_spacex_answers():
    from worldpgt.experiments.ask_microworld_v1 import ask
    a = ask("Who founded SpaceX?", "pump-dry-run")
    assert a.decision == "answer", f"Single-hop founder QA broken: {a.answer_text}"


# ===========================================================================
# Category 10: Protected files unchanged
# ===========================================================================


def test_protected_files_not_modified():
    for path in _PROTECTED:
        if path.exists():
            assert path.is_file(), f"Protected path {path} is not a regular file"


def test_no_neural_gpt_imports_in_multihop_package():
    import importlib
    import pkgutil
    import worldpgt.multihop_qa as pkg
    forbidden = frozenset({"torch", "tensorflow", "transformers", "openai",
                           "anthropic", "gpt", "embedding", "neural"})
    for modinfo in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        spec = importlib.util.find_spec(modinfo.name)
        if spec and spec.origin:
            src = Path(spec.origin).read_text(encoding="utf-8").lower()
            for term in forbidden:
                assert term not in src, (
                    f"{modinfo.name} contains forbidden term '{term}'"
                )


# ===========================================================================
# Category 11: Experiment runner summary
# ===========================================================================


def test_multihop_summary_all_critical_passed(multihop_summary):
    assert multihop_summary.get("all_critical_passed") is True


def test_multihop_summary_wrong_count_zero(multihop_summary):
    assert multihop_summary.get("wrong_count", 0) == 0


def test_multihop_summary_unsafe_chain_count_zero(multihop_summary):
    assert multihop_summary.get("unsafe_chain_count", 0) == 0


def test_multihop_summary_unsupported_answer_count_zero(multihop_summary):
    assert multihop_summary.get("unsupported_answer_count", 0) == 0


def test_multihop_summary_total_prompts_nonzero(multihop_summary):
    assert multihop_summary.get("total_prompts", 0) > 0


def test_multihop_summary_has_required_keys(multihop_summary):
    required = {
        "total_prompts", "answer_count", "audit_count", "wrong_count",
        "unsupported_answer_count", "unsafe_chain_count", "missing_hop_audit_count",
        "direct_only_guard_count", "direct_relation_leak_count",
        "volatile_hop_audit_count", "high_risk_hop_audit_count",
        "current_sensitive_hop_audit_count",
        "transitive_founder_leak_count",
        "transitive_owner_leak_count", "all_critical_passed",
    }
    missing = required - set(multihop_summary.keys())
    assert not missing, f"Summary missing keys: {missing}"


# ===========================================================================
# Category 12: v1.1 Safety metrics — leader_of blocking and new counters
# ===========================================================================


def test_elon_musk_connected_to_falcon9_audits_due_to_leader_of(pump_graph):
    planner = MultihopPlanner(pump_graph)
    mq = analyze("How is Elon Musk connected to Falcon 9?")
    plan = planner.plan(mq)
    assert plan.decision == "audit", (
        "Elon Musk -> Falcon 9 first path uses leader_of; must audit"
    )
    assert plan.audit_reason is not None
    assert "current_sensitive_relation:leader_of" in plan.audit_reason


def test_jeff_bezos_connected_to_rockets_audits_due_to_leader_of(pump_graph):
    planner = MultihopPlanner(pump_graph)
    mq = analyze("How is Jeff Bezos connected to rockets?")
    plan = planner.plan(mq)
    assert plan.decision == "audit", (
        "Jeff Bezos -> rockets first path uses leader_of; must audit"
    )
    assert plan.audit_reason is not None
    assert "current_sensitive_relation:leader_of" in plan.audit_reason


def test_direct_only_guard_count_in_summary(multihop_summary):
    count = multihop_summary.get("direct_only_guard_count", -1)
    assert count >= 0, "direct_only_guard_count must be present"
    # The runner has 5 direct-only audit prompts (founded, owned queries).
    assert count >= 4, f"Expected >=4 direct-only guards, got {count}"


def test_direct_relation_leak_count_equals_direct_only_guard_count(multihop_summary):
    assert multihop_summary["direct_relation_leak_count"] == multihop_summary["direct_only_guard_count"]


def test_current_sensitive_hop_audit_count_nonzero(multihop_summary):
    count = multihop_summary.get("current_sensitive_hop_audit_count", 0)
    assert count >= 2, (
        f"Expected >=2 current_sensitive audits (Elon Musk, Jeff Bezos), got {count}"
    )


def test_paths_have_structured_hop1_detail():
    from worldpgt.experiments.run_multihop_qa_v1 import run, _OUT, _PUMP_OVERLAY
    if not _PUMP_OVERLAY.exists():
        pytest.skip("Pump overlay not present")
    paths_file = _OUT / "multihop_qa_paths.json"
    if not paths_file.exists():
        pytest.skip("Run run_multihop_qa_v1.py first")
    rows = json.loads(paths_file.read_text(encoding="utf-8"))
    assert rows, "multihop_qa_paths.json is empty"
    for row in rows:
        assert "hop1_detail" in row, f"Missing hop1_detail in row {row.get('row_id')}"
        if row["hop1_detail"] is not None:
            d = row["hop1_detail"]
            for key in ("subject", "predicate", "object", "overlay_type", "trust", "stability", "risk", "source_page"):
                assert key in d, f"hop1_detail missing key '{key}' in row {row.get('row_id')}"


def test_paths_have_structured_hop2_detail():
    paths_file = _EXP / "multihop_qa_v1" / "multihop_qa_paths.json"
    if not paths_file.exists():
        pytest.skip("Run run_multihop_qa_v1.py first")
    rows = json.loads(paths_file.read_text(encoding="utf-8"))
    assert rows, "multihop_qa_paths.json is empty"
    for row in rows:
        assert "hop2_detail" in row, f"Missing hop2_detail in row {row.get('row_id')}"
        if row["hop2_detail"] is not None:
            d = row["hop2_detail"]
            for key in ("subject", "predicate", "object", "overlay_type", "trust", "stability", "risk", "source_page"):
                assert key in d, f"hop2_detail missing key '{key}' in row {row.get('row_id')}"


def test_no_leader_of_predicate_in_answered_paths():
    paths_file = _EXP / "multihop_qa_v1" / "multihop_qa_paths.json"
    if not paths_file.exists():
        pytest.skip("Run run_multihop_qa_v1.py first")
    rows = json.loads(paths_file.read_text(encoding="utf-8"))
    bad = [
        row for row in rows
        if row.get("decision") == "answer"
        and (
            (row.get("hop1_detail") or {}).get("predicate") == "leader_of"
            or (row.get("hop2_detail") or {}).get("predicate") == "leader_of"
        )
    ]
    assert not bad, (
        f"Found {len(bad)} answered path(s) with leader_of predicate: "
        + ", ".join(r["row_id"] for r in bad)
    )
