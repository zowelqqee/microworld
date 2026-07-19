"""Tests for the reflective-reasoning EXTENSION (speculative_extended, Pattern A).

Pins the pilot-validated behaviour: co-attribution fires only via a KINSHIP
predicate (distribution/location shared objects are excluded), carries the
lower-confidence support_kind, and renders the structure-driven caution. Also
guards that this module does not disturb the proven v1 rules.
"""

from __future__ import annotations

from worldpgt.reasoning.reflective_reasoning_extended_v2 import (
    SUPPORT_KIND,
    co_attribution_for_pair,
    discover_co_attributions,
    render_extended,
)
from worldpgt.reasoning.reflective_reasoning_v1 import load_edges


def _rel(s, p, o):
    return {"overlay_type": "overlay_relation", "subject": s, "predicate": p,
            "object": o, "evidence_id": f"e:{s}|{p}|{o}".lower()}


OVERLAY = [
    _rel("SpaceX", "develops", "rockets"),
    _rel("Blue Origin", "develops", "rockets"),
    _rel("NASA", "develops", "spacecraft"),
    _rel("SpaceX", "develops", "spacecraft"),
    _rel("Martin Eberhard", "founded", "Tesla"),
    _rel("Marc Tarpenning", "founded", "Tesla"),
    # distribution links that must NOT create co-attributions:
    _rel("Book A", "published_by", "Oxford University Press"),
    _rel("Book B", "published_by", "Oxford University Press"),
    _rel("SpaceX", "located_in", "Texas"),
    _rel("Tesla", "located_in", "Texas"),
]


def edges():
    return load_edges(OVERLAY)


def test_co_attribution_fires_on_shared_capability():
    plan = co_attribution_for_pair(edges(), "SpaceX", "Blue Origin")
    assert plan.decision == "speculative_extended"
    assert any(s.shared_object == "rockets" for s in plan.steps)
    assert plan.to_dict()["support_kind"] == SUPPORT_KIND


def test_co_attribution_fires_on_co_founders():
    plan = co_attribution_for_pair(edges(), "Martin Eberhard", "Marc Tarpenning")
    assert plan.decision == "speculative_extended"
    assert plan.steps[0].shared_object == "Tesla"
    assert plan.steps[0].predicate == "founded"


def test_distribution_shared_object_is_excluded():
    # Two books sharing only a PUBLISHER must not be linked.
    plan = co_attribution_for_pair(edges(), "Book A", "Book B")
    assert plan.decision == "audit"


def test_shared_location_is_excluded():
    plan = co_attribution_for_pair(edges(), "SpaceX", "Tesla")
    assert plan.decision == "audit"  # only share located_in Texas -> excluded


def test_same_entity_audits():
    plan = co_attribution_for_pair(edges(), "SpaceX", "SpaceX")
    assert plan.decision == "audit"


def test_discover_returns_only_kinship_pairs():
    pairs = discover_co_attributions(OVERLAY)
    objs = {p.shared_object.lower() for p in pairs}
    assert "oxford university press" not in objs
    assert "texas" not in objs
    assert any(p.shared_object == "rockets" for p in pairs)
    # co-founders discovered
    assert any({p.x.lower(), p.y.lower()} == {"martin eberhard", "marc tarpenning"} for p in pairs)


def test_render_marks_lower_confidence():
    plan = co_attribution_for_pair(edges(), "SpaceX", "Blue Origin")
    text = render_extended(plan.steps[0])
    assert "not directly linked" in text
    assert "less-tested inference" in text


def test_does_not_disturb_proven_rules():
    # The proven v1 rules still behave as before (smoke check of isolation).
    from worldpgt.reasoning.reflective_reasoning_v1 import reflect
    plan = reflect("Why might Elon Musk be associated with rockets?", [
        _rel("Elon Musk", "founded", "SpaceX"), _rel("SpaceX", "develops", "rockets"),
    ])
    assert plan is not None and plan.decision == "speculative"
    assert plan.to_dict()["support_kind"] == "speculative_inference"
