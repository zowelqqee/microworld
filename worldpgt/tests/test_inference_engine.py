"""Tests for the inference engine over overlay fact graphs.

The engine is now data-driven: ``run_inference`` delegates to the rule
interpreter, loading rules from ``knowledge/base_inference_rules.json``. These
tests verify the *end-to-end behaviour* of the base rule set. The unification
mechanics themselves are covered in ``test_rule_interpreter.py``.

Most tests use minimal synthetic overlays (fast, no dependency on the live
promoted overlay). The final class runs against the real promoted overlay.
"""

from __future__ import annotations

import pytest

from worldpgt.cognition.inference_engine import (
    InferenceWorkspace,
    InferredFact,
    run_inference,
)


# Base rule ids (data-defined in knowledge/base_inference_rules.json)
CAPABILITY = "capability_inheritance_v1"
TRANSITIVITY = "ownership_transitivity_v1"
FOUNDER = "founder_shared_v1"
LEADER = "leader_shared_v1"


# ---------------------------------------------------------------------------
# Helpers — minimal overlay builders
# ---------------------------------------------------------------------------

def _def(subject: str, definition: str, stability: str = "stable") -> dict:
    return {
        "overlay_type": "overlay_definition",
        "subject": subject,
        "definition": definition,
        "stability": stability,
    }


def _rel(subject: str, predicate: str, obj: str, stability: str = "semi_stable") -> dict:
    return {
        "overlay_type": "overlay_relation",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "stability": stability,
    }


def _ent(label: str, entity_type: str = "organization") -> dict:
    return {
        "overlay_type": "overlay_entity",
        "label": label,
        "entity_type": entity_type,
    }


# ---------------------------------------------------------------------------
# Rule 1 — Capability inheritance
# ---------------------------------------------------------------------------

class TestCapabilityInheritance:
    def test_starlink_inherits_spacex_develops_rockets(self):
        overlay = [
            _ent("SpaceX", "organization"),
            _ent("Starlink", "service"),
            _def("Starlink", "satellite internet constellation operated by SpaceX"),
            _rel("SpaceX", "develops", "rockets"),
        ]
        ws = run_inference(overlay)
        facts = ws.for_subject("Starlink")
        assert any(f.predicate == "develops" and f.object == "rockets" for f in facts)

    def test_inherited_fact_is_marked_inferred_with_correct_rule(self):
        overlay = [
            _ent("SpaceX", "organization"),
            _ent("Starlink", "service"),
            _def("Starlink", "satellite internet constellation operated by SpaceX"),
            _rel("SpaceX", "produces", "broadband services"),
        ]
        ws = run_inference(overlay)
        facts = ws.for_subject("Starlink")
        assert len(facts) == 1
        f = facts[0]
        assert f.source == "inferred"
        assert f.rule == CAPABILITY
        assert f.predicate == "produces"
        assert f.object == "broadband services"

    def test_chain_contains_explicit_proof_steps(self):
        overlay = [
            _ent("SpaceX", "organization"),
            _ent("Starlink", "service"),
            _def("Starlink", "satellite internet constellation operated by SpaceX"),
            _rel("SpaceX", "develops", "spacecraft"),
        ]
        ws = run_inference(overlay)
        f = next(f for f in ws.for_subject("Starlink") if f.object == "spacecraft")
        assert ("Starlink", "owned_by", "SpaceX") in f.chain
        assert ("SpaceX", "develops", "spacecraft") in f.chain
        assert len(f.chain) == 2

    def test_confidence_is_min_of_edge_and_relation_stability(self):
        overlay = [
            _ent("SpaceX", "organization"),
            _ent("Starlink", "service"),
            _def("Starlink", "satellite internet constellation operated by SpaceX",
                 stability="stable"),
            _rel("SpaceX", "develops", "rockets", stability="semi_stable"),
        ]
        ws = run_inference(overlay)
        f = next(f for f in ws.for_subject("Starlink") if f.predicate == "develops")
        assert f.confidence == pytest.approx(0.7)  # min(1.0, 0.7)

    def test_both_stable_gives_confidence_one(self):
        overlay = [
            _ent("SpaceX", "organization"),
            _ent("Starlink", "service"),
            _def("Starlink", "satellite internet constellation operated by SpaceX", stability="stable"),
            _rel("SpaceX", "develops", "rockets", stability="stable"),
        ]
        ws = run_inference(overlay)
        f = next(f for f in ws.for_subject("Starlink") if f.predicate == "develops")
        assert f.confidence == pytest.approx(1.0)

    def test_service_owner_qualifier_stripped(self):
        overlay = [
            _ent("SpaceX", "organization"),
            _ent("Starlink Business", "service"),
            _def("Starlink Business", "internet service operated by SpaceX for business customers"),
            _rel("SpaceX", "develops", "spacecraft"),
        ]
        ws = run_inference(overlay)
        facts = ws.for_subject("Starlink Business")
        assert any(f.predicate == "develops" for f in facts)
        f = next(f for f in facts if f.predicate == "develops")
        assert ("Starlink Business", "owned_by", "SpaceX") in f.chain

    def test_division_of_pattern_matched(self):
        overlay = [
            _ent("Tesla", "organization"),
            _ent("Tesla Energy", "organization"),
            _def("Tesla Energy", "clean energy division of Tesla"),
            _rel("Tesla", "produces", "electric cars"),
        ]
        ws = run_inference(overlay)
        facts = ws.for_subject("Tesla Energy")
        assert any(f.predicate == "produces" and f.object == "electric cars" for f in facts)

    def test_class_only_definition_without_creator_produces_no_facts(self):
        overlay = [
            _ent("SpaceX", "organization"),
            _def("SpaceX", "aerospace manufacturer and space transportation company"),
        ]
        ws = run_inference(overlay)
        assert ws.for_subject("SpaceX") == ()

    def test_type_guard_blocks_inheritance_when_parent_is_not_organization(self):
        """capability requires ?B (the owner) to be an organization. A person owner blocks it."""
        overlay = [
            _ent("Elon Musk", "person"),
            _ent("Some Product", "product"),
            # 'operated by Elon Musk' → owned_by Elon Musk, but Elon Musk is a person
            _def("Some Product", "gadget operated by Elon Musk"),
            _rel("Elon Musk", "develops", "ideas"),
        ]
        ws = run_inference(overlay)
        assert ws.by_rule(CAPABILITY) == ()

    def test_self_reference_excluded(self):
        overlay = [
            _ent("Tesla", "organization"),
            _def("Tesla", "American company called Tesla"),
            _rel("Tesla", "produces", "cars"),
        ]
        ws = run_inference(overlay)
        assert not any(
            f.subject == "Tesla" and f.rule == CAPABILITY for f in ws.facts
        )

    def test_multiple_relations_all_inherited(self):
        overlay = [
            _ent("SpaceX", "organization"),
            _ent("Starlink", "service"),
            _def("Starlink", "satellite internet constellation operated by SpaceX"),
            _rel("SpaceX", "develops", "rockets"),
            _rel("SpaceX", "produces", "Dragon spacecraft"),
        ]
        ws = run_inference(overlay)
        predicates = {f.predicate for f in ws.for_subject("Starlink")}
        assert "develops" in predicates
        assert "produces" in predicates

    def test_unknown_entity_reference_in_definition_produces_no_fact(self):
        overlay = [
            _ent("Satellite X", "service"),
            _def("Satellite X", "satellite operated by UnknownCorp"),
            _rel("UnknownCorp", "develops", "satellites"),
        ]
        ws = run_inference(overlay)
        assert ws.for_subject("Satellite X") == ()

    def test_product_type_guard_blocks_capability_inheritance(self):
        overlay = [
            _ent("SpaceX", "organization"),
            _ent("Falcon 9", "product"),
            _def("Falcon 9", "reusable launch vehicle developed by SpaceX"),
            _rel("SpaceX", "develops", "rockets"),
        ]
        ws = run_inference(overlay)
        assert not any(
            f.predicate == "develops" and f.object == "rockets"
            for f in ws.for_subject("Falcon 9")
        )


# ---------------------------------------------------------------------------
# Rule 2 — Ownership transitivity
# ---------------------------------------------------------------------------

class TestOwnershipTransitivity:
    @staticmethod
    def _chain_overlay(powerwall_stability: str = "stable") -> list[dict]:
        return [
            _ent("Tesla", "organization"),
            _ent("Tesla Energy", "organization"),
            _ent("Powerwall", "product"),
            _def("Tesla Energy", "clean energy division of Tesla", stability="stable"),
            _def("Powerwall", "home battery product produced by Tesla Energy",
                 stability=powerwall_stability),
        ]

    def test_powerwall_owned_by_tesla_through_tesla_energy(self):
        ws = run_inference(self._chain_overlay())
        facts = [f for f in ws.for_subject("Powerwall") if f.rule == TRANSITIVITY]
        assert len(facts) == 1
        f = facts[0]
        assert f.predicate == "owned_by"
        assert f.object == "Tesla"
        assert f.source == "inferred"

    def test_transitivity_chain_is_two_hops(self):
        ws = run_inference(self._chain_overlay())
        f = next(f for f in ws.for_subject("Powerwall") if f.rule == TRANSITIVITY)
        assert ("Powerwall", "owned_by", "Tesla Energy") in f.chain
        assert ("Tesla Energy", "owned_by", "Tesla") in f.chain
        assert len(f.chain) == 2

    def test_transitivity_confidence_is_min_of_both_hops(self):
        ws = run_inference(self._chain_overlay(powerwall_stability="semi_stable"))
        f = next(f for f in ws.for_subject("Powerwall") if f.rule == TRANSITIVITY)
        assert f.confidence == pytest.approx(0.7)  # min(stable=1.0, semi_stable=0.7)

    def test_transitivity_full_stable_chain_confidence_one(self):
        ws = run_inference(self._chain_overlay(powerwall_stability="stable"))
        f = next(f for f in ws.for_subject("Powerwall") if f.rule == TRANSITIVITY)
        assert f.confidence == pytest.approx(1.0)

    def test_direct_owned_by_does_not_produce_transitivity_fact(self):
        overlay = [
            _ent("SpaceX", "organization"),
            _ent("Falcon 9", "product"),
            _def("Falcon 9", "reusable launch vehicle developed by SpaceX"),
        ]
        ws = run_inference(overlay)
        assert ws.by_rule(TRANSITIVITY) == ()

    def test_cycle_prevention_no_reflexive_owned_by(self):
        overlay = [
            _ent("Alpha", "organization"),
            _ent("Beta", "organization"),
            _def("Alpha", "division of Beta", stability="stable"),
            _def("Beta", "division of Alpha", stability="stable"),
        ]
        ws = run_inference(overlay)
        transitive = ws.by_rule(TRANSITIVITY)
        assert not any(f.subject == f.object for f in transitive)


# ---------------------------------------------------------------------------
# Rule 3 — Founder / leader shared
# ---------------------------------------------------------------------------

class TestFounderShared:
    def test_spacex_and_neuralink_share_founder_elon_musk(self):
        overlay = [
            _ent("Elon Musk", "person"),
            _rel("Elon Musk", "founded", "SpaceX"),
            _rel("Elon Musk", "founded", "Neuralink"),
        ]
        ws = run_inference(overlay)
        assert any(
            f.predicate == "share_founder" and f.object == "Neuralink"
            for f in ws.for_subject("SpaceX")
        )

    def test_founder_inheritance_is_symmetric(self):
        overlay = [
            _ent("Elon Musk", "person"),
            _rel("Elon Musk", "founded", "SpaceX"),
            _rel("Elon Musk", "founded", "Neuralink"),
        ]
        ws = run_inference(overlay)
        assert any(
            f.subject == "Neuralink" and f.predicate == "share_founder"
            and f.object == "SpaceX"
            for f in ws.facts
        )

    def test_founder_chain_runs_in_founded_by_direction(self):
        overlay = [
            _ent("Elon Musk", "person"),
            _rel("Elon Musk", "founded", "SpaceX"),
            _rel("Elon Musk", "founded", "The Boring Company"),
        ]
        ws = run_inference(overlay)
        f = next(
            f for f in ws.facts
            if f.subject == "SpaceX" and f.predicate == "share_founder"
        )
        assert ("SpaceX", "founded_by", "Elon Musk") in f.chain
        assert ("The Boring Company", "founded_by", "Elon Musk") in f.chain
        assert f.rule == FOUNDER
        assert f.source == "inferred"

    def test_single_company_founder_produces_no_share_founder(self):
        overlay = [
            _ent("Jeff Bezos", "person"),
            _rel("Jeff Bezos", "founded", "Blue Origin"),
        ]
        ws = run_inference(overlay)
        assert not any(f.predicate == "share_founder" for f in ws.facts)

    def test_type_guard_blocks_share_founder_when_founder_not_a_person(self):
        """If the founder is not a known person, the person type_guard blocks the rule."""
        overlay = [
            # No overlay_entity for the founder → entity_type unknown → guard fails
            _rel("MysteryFounder", "founded", "SpaceX"),
            _rel("MysteryFounder", "founded", "Neuralink"),
        ]
        ws = run_inference(overlay)
        assert ws.by_rule(FOUNDER) == ()

    def test_leader_of_variant_produces_share_leader(self):
        overlay = [
            _ent("Elon Musk", "person"),
            _rel("Elon Musk", "leader_of", "SpaceX"),
            _rel("Elon Musk", "leader_of", "Tesla"),
        ]
        ws = run_inference(overlay)
        assert any(
            f.subject == "SpaceX" and f.predicate == "share_leader"
            and f.object == "Tesla" and f.rule == LEADER
            for f in ws.facts
        )
        assert any(
            f.subject == "Tesla" and f.predicate == "share_leader"
            and f.object == "SpaceX"
            for f in ws.facts
        )

    def test_founder_confidence_is_min_of_both_relations(self):
        overlay = [
            _ent("Elon Musk", "person"),
            _rel("Elon Musk", "founded", "SpaceX", stability="semi_stable"),
            _rel("Elon Musk", "founded", "Neuralink", stability="semi_stable"),
        ]
        ws = run_inference(overlay)
        f = next(
            f for f in ws.facts
            if f.subject == "SpaceX" and f.predicate == "share_founder"
        )
        assert f.confidence == pytest.approx(0.7)

    def test_four_companies_same_founder_all_pairs_produced(self):
        companies = ["SpaceX", "Neuralink", "The Boring Company", "xAI"]
        overlay = [_ent("Elon Musk", "person")]
        overlay += [_rel("Elon Musk", "founded", c) for c in companies]
        ws = run_inference(overlay)
        founder_facts = [f for f in ws.facts if f.predicate == "share_founder"]
        assert len(founder_facts) == 12  # C(4,2) = 6 pairs × 2 symmetric

    def test_founded_and_leader_of_are_independent_rules(self):
        overlay = [
            _ent("Elon Musk", "person"),
            _rel("Elon Musk", "founded", "SpaceX"),
            _rel("Elon Musk", "founded", "Neuralink"),
            _rel("Elon Musk", "leader_of", "SpaceX"),
            _rel("Elon Musk", "leader_of", "Tesla"),
        ]
        ws = run_inference(overlay)
        share_founder = [f for f in ws.facts if f.predicate == "share_founder"]
        share_leader = [f for f in ws.facts if f.predicate == "share_leader"]
        assert len(share_founder) == 2
        assert len(share_leader) == 2


# ---------------------------------------------------------------------------
# InferenceWorkspace API
# ---------------------------------------------------------------------------

class TestInferenceWorkspace:
    def test_for_subject_case_insensitive(self):
        overlay = [
            _ent("SpaceX", "organization"),
            _ent("Starlink", "service"),
            _def("Starlink", "satellite internet constellation operated by SpaceX"),
            _rel("SpaceX", "develops", "rockets"),
        ]
        ws = run_inference(overlay)
        assert ws.for_subject("Starlink") == ws.for_subject("starlink")
        assert ws.for_subject("STARLINK") == ws.for_subject("Starlink")

    def test_involving_finds_by_object(self):
        overlay = [
            _ent("Elon Musk", "person"),
            _rel("Elon Musk", "founded", "SpaceX"),
            _rel("Elon Musk", "founded", "Neuralink"),
        ]
        ws = run_inference(overlay)
        found = ws.involving("Neuralink")
        names = {f.subject for f in found} | {f.object for f in found}
        assert "Neuralink" in names

    def test_by_rule_filters_by_rule_id(self):
        overlay = [
            _ent("Tesla", "organization"),
            _ent("Tesla Energy", "organization"),
            _ent("Powerwall", "product"),
            _def("Tesla Energy", "clean energy division of Tesla"),
            _def("Powerwall", "home battery product produced by Tesla Energy"),
            _rel("Tesla", "produces", "electric cars"),
        ]
        ws = run_inference(overlay)
        assert all(f.rule == TRANSITIVITY for f in ws.by_rule(TRANSITIVITY))
        assert all(f.rule == CAPABILITY for f in ws.by_rule(CAPABILITY))

    def test_for_subject_returns_empty_for_unknown(self):
        ws = InferenceWorkspace(facts=())
        assert ws.for_subject("Unknown Corp") == ()

    def test_involving_returns_empty_for_unknown(self):
        ws = InferenceWorkspace(facts=())
        assert ws.involving("Unknown Corp") == ()


# ---------------------------------------------------------------------------
# InferredFact serialisation
# ---------------------------------------------------------------------------

class TestInferredFactAsDict:
    def test_as_dict_contains_required_keys(self):
        fact = InferredFact(
            subject="Starlink",
            predicate="develops",
            object="rockets",
            rule=CAPABILITY,
            chain=(("Starlink", "owned_by", "SpaceX"), ("SpaceX", "develops", "rockets")),
            confidence=0.7,
        )
        d = fact.as_dict()
        assert d["subject"] == "Starlink"
        assert d["predicate"] == "develops"
        assert d["object"] == "rockets"
        assert d["source"] == "inferred"
        assert d["rule"] == CAPABILITY
        assert d["confidence"] == pytest.approx(0.7)
        assert d["chain"] == [
            ["Starlink", "owned_by", "SpaceX"],
            ["SpaceX", "develops", "rockets"],
        ]

    def test_as_dict_chain_is_list_of_lists(self):
        fact = InferredFact(
            subject="A", predicate="owned_by", object="C",
            rule=TRANSITIVITY,
            chain=(("A", "owned_by", "B"), ("B", "owned_by", "C")),
        )
        d = fact.as_dict()
        assert isinstance(d["chain"], list)
        assert all(isinstance(step, list) for step in d["chain"])


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_run_inference_deduplicates_same_rule_same_fact(self):
        overlay = [
            _ent("SpaceX", "organization"),
            _ent("Starlink", "service"),
            _def("Starlink", "satellite internet constellation operated by SpaceX"),
            _rel("SpaceX", "develops", "rockets"),
            _rel("SpaceX", "develops", "rockets"),  # duplicate
        ]
        ws = run_inference(overlay)
        starlink_develops = [
            f for f in ws.for_subject("Starlink")
            if f.predicate == "develops" and f.object == "rockets"
        ]
        assert len(starlink_develops) == 1


# ---------------------------------------------------------------------------
# Rule combinations (integration)
# ---------------------------------------------------------------------------

class TestRuleCombinations:
    def test_all_rules_fire_on_realistic_subgraph(self):
        overlay = [
            _ent("SpaceX", "organization"),
            _ent("Tesla", "organization"),
            _ent("Tesla Energy", "organization"),
            _ent("Starlink", "service"),
            _ent("Falcon 9", "product"),
            _ent("Powerwall", "product"),
            _ent("Elon Musk", "person"),
            _def("Starlink", "satellite internet constellation operated by SpaceX"),
            _def("Falcon 9", "reusable launch vehicle developed by SpaceX"),
            _def("Tesla Energy", "clean energy division of Tesla"),
            _def("Powerwall", "home battery product produced by Tesla Energy"),
            _rel("SpaceX", "develops", "rockets"),
            _rel("SpaceX", "develops", "spacecraft"),
            _rel("Tesla", "produces", "electric cars"),
            _rel("Elon Musk", "founded", "SpaceX"),
            _rel("Elon Musk", "founded", "The Boring Company"),
            _rel("Elon Musk", "leader_of", "SpaceX"),
            _rel("Elon Musk", "leader_of", "Tesla"),
        ]
        ws = run_inference(overlay)

        # Rule 1: capability inheritance (Starlink service, Tesla Energy organization)
        assert any(
            f.subject == "Starlink" and f.predicate == "develops"
            and f.object == "rockets" for f in ws.facts
        )
        assert not any(
            f.subject == "Falcon 9" and f.predicate == "develops"
            and f.object == "spacecraft" for f in ws.facts
        )
        assert any(
            f.subject == "Tesla Energy" and f.predicate == "produces"
            and f.object == "electric cars" for f in ws.facts
        )
        # Rule 2: ownership transitivity
        assert any(
            f.subject == "Powerwall" and f.rule == TRANSITIVITY
            and f.object == "Tesla" for f in ws.facts
        )
        # Rule 3: shared founder
        assert any(
            f.subject == "SpaceX" and f.predicate == "share_founder"
            and f.object == "The Boring Company" for f in ws.facts
        )
        # Rule 4: shared leader
        assert any(
            f.subject == "SpaceX" and f.predicate == "share_leader"
            and f.object == "Tesla" for f in ws.facts
        )


# ---------------------------------------------------------------------------
# Integration against real promoted overlay
# ---------------------------------------------------------------------------

class TestIntegrationRealOverlay:
    @pytest.fixture(scope="class")
    def ws(self):
        from worldpgt.assistant_surface.context_selector import resolve_overlay
        import json, pathlib
        path, _ = resolve_overlay("promoted")
        items = json.loads(pathlib.Path(path).read_text())
        return run_inference(items)

    def test_starlink_inherits_spacex_develops_rockets(self, ws):
        facts = ws.for_subject("Starlink")
        assert any(f.predicate == "develops" and f.object == "rockets" for f in facts), (
            f"Got: {[f.as_dict() for f in facts]}"
        )

    def test_powerwall_transitively_owned_by_tesla(self, ws):
        facts = ws.by_rule("ownership_transitivity_v1")
        assert any(
            f.subject == "Powerwall" and f.object == "Tesla" for f in facts
        ), f"Got: {[f.as_dict() for f in facts]}"

    def test_spacex_and_neuralink_share_founder_elon_musk(self, ws):
        assert any(
            f.predicate == "share_founder" and f.object == "Neuralink"
            for f in ws.for_subject("SpaceX")
        )

    def test_spacex_and_tesla_share_leader_elon_musk(self, ws):
        assert any(
            f.predicate == "share_leader" and f.object == "Tesla"
            for f in ws.for_subject("SpaceX")
        )

    def test_all_inferred_facts_are_marked_inferred(self, ws):
        assert all(f.source == "inferred" for f in ws.facts)

    def test_all_facts_have_non_empty_chain(self, ws):
        assert all(len(f.chain) >= 1 for f in ws.facts)

    def test_all_facts_have_valid_confidence(self, ws):
        assert all(0.0 < f.confidence <= 1.0 for f in ws.facts)

    def test_all_facts_carry_a_known_base_rule_id(self, ws):
        known = {CAPABILITY, TRANSITIVITY, FOUNDER, LEADER,
                 "classification_chain_v1", "activity_inheritance_v1"}
        assert all(f.rule in known for f in ws.facts)
