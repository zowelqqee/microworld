"""Tests for the data-driven rule interpreter (unification engine).

Covers the mechanics the base-rule end-to-end tests assume:
  - unification with one and several variables
  - predicate_class matching
  - type_guard blocking
  - confidence propagation (min/max/constant)
  - inverse-relation and definition normalization
  - adding a brand-new rule purely as data, with no code change (the key
    flexibility test)
"""

from __future__ import annotations

import json

import pytest

from worldpgt.cognition.inference_engine import run_inference
from worldpgt.cognition.rule_interpreter import (
    Pattern,
    apply_rule,
    apply_rules,
    load_base_rules,
    normalize_overlay,
    parse_rule,
)


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
    return {"overlay_type": "overlay_entity", "label": label, "entity_type": entity_type}


# ---------------------------------------------------------------------------
# Rule parsing
# ---------------------------------------------------------------------------

class TestRuleParsing:
    def test_parse_rule_reads_all_fields(self):
        rule = parse_rule({
            "rule_id": "r1",
            "name": "Rule One",
            "description": "desc",
            "if_pattern": [
                {"subject": "?A", "predicate": "owned_by", "object": "?B"},
                {"subject": "?B", "predicate": "?P", "object": "?C",
                 "predicate_class": "capability"},
            ],
            "then_pattern": {"subject": "?A", "predicate": "?P", "object": "?C"},
            "confidence_rule": "min_stability",
            "type_guards": {"?A": ["product"], "?B": ["organization"]},
            "enabled": True,
        })
        assert rule.rule_id == "r1"
        assert len(rule.if_pattern) == 2
        assert rule.if_pattern[1].predicate_class == "capability"
        assert rule.then_pattern.predicate == "?P"
        assert rule.type_guards == {"?A": ("product",), "?B": ("organization",)}
        assert rule.enabled is True

    def test_load_base_rules_returns_enabled_rules(self):
        rules = load_base_rules()
        ids = {r.rule_id for r in rules}
        assert "capability_inheritance_v1" in ids
        assert "ownership_transitivity_v1" in ids
        assert "founder_shared_v1" in ids
        assert "causal_chain_v1" in ids
        assert "expertise_inheritance_v1" in ids
        assert "competitor_detection_v1" in ids

    def test_base_rules_json_is_valid_overlay_inference_rule_items(self):
        from worldpgt.cognition import rule_interpreter
        path = rule_interpreter._BASE_RULES_PATH
        data = json.loads(path.read_text())
        assert all(item["overlay_type"] == "overlay_inference_rule" for item in data)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_relation_becomes_fact(self):
        norm = normalize_overlay([_rel("SpaceX", "develops", "rockets")])
        assert any(
            f.subject == "SpaceX" and f.predicate == "develops" and f.object == "rockets"
            for f in norm.facts
        )

    def test_founded_gets_inverse_founded_by(self):
        norm = normalize_overlay([_rel("Elon Musk", "founded", "SpaceX")])
        assert any(
            f.subject == "SpaceX" and f.predicate == "founded_by"
            and f.object == "Elon Musk"
            for f in norm.facts
        )

    def test_leader_of_gets_inverse_led_by(self):
        norm = normalize_overlay([_rel("Elon Musk", "leader_of", "Tesla")])
        assert any(
            f.subject == "Tesla" and f.predicate == "led_by" and f.object == "Elon Musk"
            for f in norm.facts
        )

    def test_definition_becomes_is_a_fact(self):
        norm = normalize_overlay([_def("SpaceX", "aerospace manufacturer")])
        assert any(
            f.subject == "SpaceX" and f.predicate == "is_a"
            and f.object == "aerospace manufacturer"
            for f in norm.facts
        )

    def test_definition_creator_phrase_becomes_owned_by(self):
        norm = normalize_overlay([
            _ent("SpaceX", "organization"),
            _def("Starlink", "satellite constellation operated by SpaceX"),
        ])
        assert any(
            f.subject == "Starlink" and f.predicate == "owned_by" and f.object == "SpaceX"
            for f in norm.facts
        )

    def test_entity_type_map_populated(self):
        norm = normalize_overlay([_ent("SpaceX", "organization")])
        assert norm.entity_types["spacex"] == "organization"


# ---------------------------------------------------------------------------
# Unification
# ---------------------------------------------------------------------------

class TestUnification:
    def test_single_variable_binding(self):
        """One variable in if_pattern binds to every matching fact."""
        rule = parse_rule({
            "rule_id": "single_var",
            "if_pattern": [{"subject": "?A", "predicate": "develops", "object": "rockets"}],
            "then_pattern": {"subject": "?A", "predicate": "builds", "object": "rockets"},
            "type_guards": {},
        })
        norm = normalize_overlay([
            _rel("SpaceX", "develops", "rockets"),
            _rel("Boeing", "develops", "rockets"),
        ])
        facts = apply_rule(rule, norm)
        subjects = {f.subject for f in facts}
        assert subjects == {"SpaceX", "Boeing"}

    def test_multiple_variables_join_on_shared_binding(self):
        """?B must bind to the same value across both patterns."""
        rule = parse_rule({
            "rule_id": "join",
            "if_pattern": [
                {"subject": "?A", "predicate": "owned_by", "object": "?B"},
                {"subject": "?B", "predicate": "develops", "object": "?C"},
            ],
            "then_pattern": {"subject": "?A", "predicate": "develops", "object": "?C"},
            "type_guards": {},
        })
        norm = normalize_overlay([
            _rel("Starlink", "owned_by", "SpaceX"),
            _rel("SpaceX", "develops", "rockets"),
            _rel("OtherCo", "develops", "cars"),  # must NOT leak in (?B != OtherCo)
        ])
        facts = apply_rule(rule, norm)
        assert len(facts) == 1
        assert facts[0].subject == "Starlink"
        assert facts[0].object == "rockets"

    def test_literal_in_pattern_must_match_exactly(self):
        rule = parse_rule({
            "rule_id": "literal",
            "if_pattern": [{"subject": "?A", "predicate": "founded_by", "object": "Elon Musk"}],
            "then_pattern": {"subject": "?A", "predicate": "tagged", "object": "musk_company"},
            "type_guards": {},
        })
        norm = normalize_overlay([
            _rel("Elon Musk", "founded", "SpaceX"),
            _rel("Jeff Bezos", "founded", "Blue Origin"),
        ])
        facts = apply_rule(rule, norm)
        assert {f.subject for f in facts} == {"SpaceX"}

    def test_predicate_variable_binds_concrete_predicate(self):
        """?P binds to the concrete predicate and flows into the conclusion."""
        rule = parse_rule({
            "rule_id": "pred_var",
            "if_pattern": [
                {"subject": "?A", "predicate": "owned_by", "object": "?B"},
                {"subject": "?B", "predicate": "?P", "object": "?C",
                 "predicate_class": "capability"},
            ],
            "then_pattern": {"subject": "?A", "predicate": "?P", "object": "?C"},
            "type_guards": {},
        })
        norm = normalize_overlay([
            _rel("Starlink", "owned_by", "SpaceX"),
            _rel("SpaceX", "produces", "Dragon"),
        ])
        facts = apply_rule(rule, norm)
        assert len(facts) == 1
        assert facts[0].predicate == "produces"  # ?P bound to 'produces'


# ---------------------------------------------------------------------------
# Predicate class matching
# ---------------------------------------------------------------------------

class TestPredicateClassMatching:
    def _capability_rule(self) -> dict:
        return {
            "rule_id": "cap",
            "if_pattern": [
                {"subject": "?A", "predicate": "owned_by", "object": "?B"},
                {"subject": "?B", "predicate": "?P", "object": "?C",
                 "predicate_class": "capability"},
            ],
            "then_pattern": {"subject": "?A", "predicate": "?P", "object": "?C"},
            "type_guards": {},
        }

    def test_predicate_class_matches_member_predicate(self):
        rule = parse_rule(self._capability_rule())
        norm = normalize_overlay([
            _rel("Starlink", "owned_by", "SpaceX"),
            _rel("SpaceX", "manufactures", "satellites"),  # 'manufactures' ∈ capability
        ])
        facts = apply_rule(rule, norm)
        assert any(f.predicate == "manufactures" for f in facts)

    def test_predicate_class_rejects_non_member_predicate(self):
        rule = parse_rule(self._capability_rule())
        norm = normalize_overlay([
            _rel("Starlink", "owned_by", "SpaceX"),
            _rel("SpaceX", "located_in", "USA"),  # 'located_in' ∉ capability
        ])
        facts = apply_rule(rule, norm)
        assert facts == []


# ---------------------------------------------------------------------------
# Type guards
# ---------------------------------------------------------------------------

class TestTypeGuards:
    def _rule(self) -> dict:
        return {
            "rule_id": "guarded",
            "if_pattern": [{"subject": "?A", "predicate": "owned_by", "object": "?B"}],
            "then_pattern": {"subject": "?A", "predicate": "controlled_by", "object": "?B"},
            "type_guards": {"?B": ["organization"]},
        }

    def test_type_guard_allows_matching_type(self):
        rule = parse_rule(self._rule())
        norm = normalize_overlay([
            _ent("SpaceX", "organization"),
            _rel("Starlink", "owned_by", "SpaceX"),
        ])
        facts = apply_rule(rule, norm)
        assert len(facts) == 1

    def test_type_guard_blocks_wrong_type(self):
        rule = parse_rule(self._rule())
        norm = normalize_overlay([
            _ent("Elon Musk", "person"),
            _rel("Starlink", "owned_by", "Elon Musk"),
        ])
        facts = apply_rule(rule, norm)
        assert facts == []

    def test_type_guard_blocks_unknown_entity(self):
        """An entity with no known type cannot satisfy a type guard."""
        rule = parse_rule(self._rule())
        norm = normalize_overlay([
            _rel("Starlink", "owned_by", "MysteryOrg"),  # no overlay_entity → unknown type
        ])
        facts = apply_rule(rule, norm)
        assert facts == []


# ---------------------------------------------------------------------------
# Confidence propagation
# ---------------------------------------------------------------------------

class TestConfidencePropagation:
    def _two_hop_rule(self, confidence_rule: str) -> dict:
        return {
            "rule_id": "conf",
            "if_pattern": [
                {"subject": "?A", "predicate": "owned_by", "object": "?B"},
                {"subject": "?B", "predicate": "owned_by", "object": "?C"},
            ],
            "then_pattern": {"subject": "?A", "predicate": "owned_by", "object": "?C"},
            "confidence_rule": confidence_rule,
            "type_guards": {},
        }

    def test_min_stability_takes_lowest(self):
        rule = parse_rule(self._two_hop_rule("min_stability"))
        norm = normalize_overlay([
            _rel("A", "owned_by", "B", stability="stable"),       # 1.0
            _rel("B", "owned_by", "C", stability="semi_stable"),  # 0.7
        ])
        facts = apply_rule(rule, norm)
        assert facts[0].confidence == pytest.approx(0.7)

    def test_max_stability_takes_highest(self):
        rule = parse_rule(self._two_hop_rule("max_stability"))
        norm = normalize_overlay([
            _rel("A", "owned_by", "B", stability="volatile"),     # 0.3
            _rel("B", "owned_by", "C", stability="stable"),       # 1.0
        ])
        facts = apply_rule(rule, norm)
        assert facts[0].confidence == pytest.approx(1.0)

    def test_constant_confidence(self):
        data = self._two_hop_rule("constant")
        data["constant_confidence"] = 0.42
        rule = parse_rule(data)
        norm = normalize_overlay([
            _rel("A", "owned_by", "B", stability="stable"),
            _rel("B", "owned_by", "C", stability="volatile"),
        ])
        facts = apply_rule(rule, norm)
        assert facts[0].confidence == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Disabled rules
# ---------------------------------------------------------------------------

class TestDisabledRules:
    def test_disabled_rule_produces_nothing(self):
        rule = parse_rule({
            "rule_id": "off",
            "if_pattern": [{"subject": "?A", "predicate": "develops", "object": "?C"}],
            "then_pattern": {"subject": "?A", "predicate": "builds", "object": "?C"},
            "type_guards": {},
            "enabled": False,
        })
        facts = apply_rules([_rel("SpaceX", "develops", "rockets")], [rule])
        assert facts == []


# ---------------------------------------------------------------------------
# THE FLEXIBILITY TEST — new rule added as data only, no code change
# ---------------------------------------------------------------------------

class TestNewRuleAsDataOnly:
    def test_expertise_inheritance_added_purely_as_json(self):
        """A rule that did not exist in the codebase, defined entirely as data,
        produces a new inferred fact — no Python was added for it."""
        expertise_rule = parse_rule({
            "overlay_type": "overlay_inference_rule",
            "rule_id": "expertise_inheritance",
            "name": "Expertise Inheritance",
            "if_pattern": [
                {"subject": "?A", "predicate": "founded_by", "object": "?B"},
                {"subject": "?B", "predicate": "known_for", "object": "?C"},
            ],
            "then_pattern": {
                "subject": "?A", "predicate": "associated_with_expertise", "object": "?C",
            },
            "confidence_rule": "min_stability",
            "type_guards": {"?B": ["person"]},
            "enabled": True,
        })

        overlay = [
            _ent("Elon Musk", "person"),
            _rel("Elon Musk", "founded", "SpaceX"),
            _rel("Elon Musk", "known_for", "Tesla"),
        ]
        ws = run_inference(overlay, rules=[expertise_rule])

        assert any(
            f.subject == "SpaceX"
            and f.predicate == "associated_with_expertise"
            and f.object == "Tesla"
            for f in ws.facts
        )
        f = next(f for f in ws.facts if f.predicate == "associated_with_expertise")
        assert f.rule == "expertise_inheritance"
        assert ("SpaceX", "founded_by", "Elon Musk") in f.chain
        assert ("Elon Musk", "known_for", "Tesla") in f.chain

    def test_new_rule_respects_its_own_type_guard(self):
        """The data-defined rule's type_guard is enforced by the generic engine."""
        expertise_rule = parse_rule({
            "rule_id": "expertise_inheritance",
            "if_pattern": [
                {"subject": "?A", "predicate": "founded_by", "object": "?B"},
                {"subject": "?B", "predicate": "known_for", "object": "?C"},
            ],
            "then_pattern": {
                "subject": "?A", "predicate": "associated_with_expertise", "object": "?C",
            },
            "type_guards": {"?B": ["person"]},
        })
        # Founder is an organization, not a person → guard blocks the rule
        overlay = [
            _ent("HoldingCo", "organization"),
            _rel("HoldingCo", "founded", "SubCo"),
            _rel("HoldingCo", "known_for", "investing"),
        ]
        ws = run_inference(overlay, rules=[expertise_rule])
        assert ws.facts == ()
