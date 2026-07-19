"""Tests for the isolated reflective-reasoning experiment.

Behaviour is pinned to the two gate pilots
(``artifacts/reflective_reasoning_core_v1/pilot_report.md`` and
``pilot_abduction_report.md``): the rules fire only with the validated structural
filters and decline (audit) otherwise.
"""

from __future__ import annotations

import pytest

from worldpgt.reasoning.reflective_reasoning_v1 import (
    abduction_explanation,
    counterfactual_removal,
    load_edges,
    reflect,
    render_reflective_plan,
)


def _rel(subject, predicate, obj, eid=None):
    return {
        "overlay_type": "overlay_relation",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "evidence_id": eid or f"edge:{subject}|{predicate}|{obj}".lower(),
    }


# A small graph mirroring the real capability_overlay subgraph used in the pilots.
OVERLAY = [
    _rel("Elon Musk", "founded", "SpaceX"),
    _rel("Elon Musk", "leader_of", "SpaceX"),
    _rel("Elon Musk", "known_for", "SpaceX"),
    _rel("Elon Musk", "leader_of", "Tesla"),
    _rel("Elon Musk", "founded", "Neuralink"),
    _rel("Elon Musk", "estimated_net_worth", "US$1.1 trillion"),
    _rel("SpaceX", "develops", "rockets"),
    _rel("SpaceX", "develops", "spacecraft"),
    _rel("SpaceX", "located_in", "Starbase, Texas"),
    _rel("Tesla", "produces", "electric cars"),
    _rel("Gwynne Shotwell", "leader_of", "SpaceX"),
    _rel("Jeff Bezos", "founded", "Blue Origin"),
    _rel("Blue Origin", "develops", "rockets"),
    _rel("LVMH", "located_in", "Paris"),
]


@pytest.fixture
def edges():
    return load_edges(OVERLAY)


# --- counterfactual removal ------------------------------------------------- #

def test_counterfactual_fires_on_founding_of_entity(edges):
    plan = counterfactual_removal(edges, "Elon Musk", "founded", "SpaceX")
    assert plan.decision == "speculative"
    assert plan.step is not None
    assert plan.step.premises[0].evidence_id == "edge:elon musk|founded|spacex"
    # Every conclusion fact references the SpaceX node.
    for fact in plan.step.conclusion_facts:
        assert "spacex" in (fact.s, fact.o) or fact.s == "spacex" or fact.o == "spacex"
    # Includes SpaceX's downstream facts.
    concl = {(f.subject, f.predicate, f.object) for f in plan.step.conclusion_facts}
    assert ("SpaceX", "develops", "rockets") in concl


def test_counterfactual_declines_incidental_predicate(edges):
    # leader_of is not existence-conferring -> audit.
    plan = counterfactual_removal(edges, "Elon Musk", "leader_of", "Tesla")
    assert plan.decision == "audit"
    assert "existence-conferring" in (plan.audit_reason or "")


def test_counterfactual_declines_generic_object(edges):
    # 'rockets' is not itself a graph entity (never a subject) -> audit even though
    # 'develops' feels dependency-ish. Matches the pilot's refined filter.
    plan = counterfactual_removal(edges, "SpaceX", "develops", "rockets")
    assert plan.decision == "audit"


def test_counterfactual_declines_unknown_fact(edges):
    plan = counterfactual_removal(edges, "Nobody", "founded", "Nothing")
    assert plan.decision == "audit"
    assert "not in the evidence slice" in (plan.audit_reason or "")


def test_counterfactual_never_includes_unrelated_cosubject(edges):
    plan = counterfactual_removal(edges, "Elon Musk", "founded", "SpaceX")
    concl = {(f.subject, f.predicate, f.object) for f in plan.step.conclusion_facts}
    # Musk's OTHER ventures must not appear (non-sequiturs the naive rule emitted).
    assert ("Elon Musk", "founded", "Neuralink") not in concl
    assert ("Elon Musk", "estimated_net_worth", "US$1.1 trillion") not in concl


# --- abduction -------------------------------------------------------------- #

def test_abduction_fires_on_two_hop_bridge(edges):
    plan = abduction_explanation(edges, "Elon Musk", "rockets")
    assert plan.decision == "speculative"
    assert plan.step.bridge_node == "SpaceX"
    assert plan.step.premises[1].object.lower() == "rockets"


def test_abduction_prefers_strong_first_hop(edges):
    # founded/leader_of should be chosen over known_for as the explanatory hop.
    plan = abduction_explanation(edges, "Elon Musk", "rockets")
    assert plan.step.premises[0].predicate in {"founded", "leader_of"}


def test_abduction_defers_on_direct_edge(edges):
    plan = abduction_explanation(edges, "Blue Origin", "rockets")
    assert plan.decision == "grounded_deferral"


def test_abduction_declines_no_bridge(edges):
    plan = abduction_explanation(edges, "Elon Musk", "Paris")
    assert plan.decision == "audit"


def test_abduction_declines_spurious_three_hop(edges):
    # Tesla ~ rockets only bridges through Musk (3 hops) -> decline.
    plan = abduction_explanation(edges, "Tesla", "rockets")
    assert plan.decision == "audit"


# --- routing + rendering ---------------------------------------------------- #

def test_reflect_routes_what_if(edges):
    plan = reflect("What if Elon Musk had not founded SpaceX?", OVERLAY)
    assert plan is not None and plan.rule == "counterfactual_removal"
    assert plan.decision == "speculative"


def test_reflect_routes_why_might(edges):
    plan = reflect("Why might Elon Musk be associated with rockets?", OVERLAY)
    assert plan is not None and plan.rule == "abduction_path_explanation"
    assert plan.decision == "speculative"


def test_reflect_rejects_unsupported_pattern():
    assert reflect("Who founded SpaceX?", OVERLAY) is None


def test_render_marks_speculation_explicitly():
    plan = reflect("Why might Elon Musk be associated with rockets?", OVERLAY)
    text = render_reflective_plan(plan)
    assert "speculative inference" in text.lower()
    assert "stored fact" in text.lower()


def test_render_audit_is_honest():
    plan = reflect("What if Elon Musk had not leader_of Tesla?", OVERLAY)
    text = render_reflective_plan(plan)
    assert "can't responsibly speculate" in text.lower()


def test_plan_to_dict_carries_support_kind():
    plan = reflect("What if Elon Musk had not founded SpaceX?", OVERLAY)
    d = plan.to_dict()
    assert d["support_kind"] == "speculative_inference"
    assert d["step"]["premise_evidence_ids"] == ["edge:elon musk|founded|spacex"]
