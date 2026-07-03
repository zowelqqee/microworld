"""Tests for Part 3 of the reasoning layer: counterfactual traces.

A counterfactual is never answered as a fact. The analyzer computes what in
the world model *structurally rests* on the hypothetically removed fact:
lost inferences (with the removed link highlighted in each proof chain),
patterns losing evidence, and the affected entities.
"""

from __future__ import annotations

import pytest

from worldpgt.reasoning import analyze_counterfactual, discover_patterns, try_answer_reasoning
from worldpgt.reasoning.counterfactual import find_target_items, render


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


def _musk_overlay() -> list[dict]:
    """Musk founded two companies → founder_shared inference depends on each
    founding fact; removing one breaks the shared-founder derivations."""
    return [
        _ent("SpaceX"),
        _ent("Neuralink"),
        _ent("Elon Musk", "person"),
        _rel("Elon Musk", "founded", "SpaceX", "stable"),
        _rel("Elon Musk", "founded", "Neuralink", "stable"),
        _rel("Elon Musk", "is_a", "entrepreneur", "stable"),
        _rel("SpaceX", "is_a", "aerospace company", "stable"),
        _rel("Neuralink", "is_a", "neurotechnology company", "stable"),
        _rel("SpaceX", "develops", "Falcon 9"),
    ]


# ---------------------------------------------------------------------------
# Target fact location
# ---------------------------------------------------------------------------

class TestTargetLocation:
    def test_finds_fact_in_stated_direction(self):
        targets = find_target_items(_musk_overlay(), "Elon Musk", "founded", "SpaceX")
        assert len(targets) == 1

    def test_finds_fact_via_inverse_direction(self):
        # "What if SpaceX had not been founded by Musk?" — stored as
        # Musk founded SpaceX; the inverse pair must still find it.
        targets = find_target_items(_musk_overlay(), "SpaceX", "founded_by", "Elon Musk")
        assert len(targets) == 1
        assert targets[0]["subject"] == "Elon Musk"

    def test_entity_wide_hypothesis_collects_all_touching_relations(self):
        targets = find_target_items(_musk_overlay(), "Elon Musk")
        assert len(targets) == 3  # two foundings + is_a entrepreneur

    def test_absent_fact_finds_nothing(self):
        assert find_target_items(_musk_overlay(), "Elon Musk", "founded", "Nokia") == []


# ---------------------------------------------------------------------------
# Inference dependencies
# ---------------------------------------------------------------------------

class TestLostInferences:
    def test_shared_founder_inference_is_lost(self):
        trace = analyze_counterfactual(_musk_overlay(), "Elon Musk", "founded", "SpaceX")
        assert trace.decision == "analysis"
        lost = {(f.subject, f.predicate, f.object) for f in trace.lost_inferences}
        assert ("SpaceX", "share_founder", "Neuralink") in lost
        assert ("Neuralink", "share_founder", "SpaceX") in lost

    def test_lost_inference_highlights_removed_link(self):
        trace = analyze_counterfactual(_musk_overlay(), "Elon Musk", "founded", "SpaceX")
        for f in trace.lost_inferences:
            assert f.removed_links, (
                f"lost inference {f.display()} must cite the removed link"
            )
            assert all(link in f.chain for link in f.removed_links)

    def test_unrelated_inferences_survive(self):
        overlay = _musk_overlay() + [
            _ent("Tesla"),
            _rel("Elon Musk", "founded", "Tesla", "stable"),
        ]
        trace = analyze_counterfactual(overlay, "Elon Musk", "founded", "SpaceX")
        lost = {(f.subject, f.predicate, f.object) for f in trace.lost_inferences}
        # Tesla/Neuralink still share a founder — that inference must survive.
        assert ("Tesla", "share_founder", "Neuralink") not in lost

    def test_dependent_entities_listed(self):
        trace = analyze_counterfactual(_musk_overlay(), "Elon Musk", "founded", "SpaceX")
        assert "SpaceX" in trace.dependent_entities
        assert "Neuralink" in trace.dependent_entities

    def test_removal_is_hypothetical_overlay_untouched(self):
        overlay = _musk_overlay()
        before = [dict(item) for item in overlay]
        analyze_counterfactual(overlay, "Elon Musk", "founded", "SpaceX")
        assert overlay == before


# ---------------------------------------------------------------------------
# Pattern dependencies
# ---------------------------------------------------------------------------

class TestAffectedPatterns:
    def _overlay(self) -> list[dict]:
        return [
            _ent("SpaceX"),
            _ent("Blue Origin"),
            _ent("Rocket Lab"),
            _rel("SpaceX", "is_a", "launch vehicle company", "stable"),
            _rel("Blue Origin", "is_a", "launch vehicle company", "stable"),
            _rel("Rocket Lab", "is_a", "launch vehicle company", "stable"),
            _rel("SpaceX", "develops", "Falcon 9"),
            _rel("Blue Origin", "develops", "New Glenn"),
            _rel("Rocket Lab", "develops", "Electron"),
        ]

    def test_pattern_confidence_recomputed_after_removal(self):
        overlay = self._overlay()
        patterns = discover_patterns(overlay)
        trace = analyze_counterfactual(
            overlay, "SpaceX", "develops", "Falcon 9", patterns=patterns
        )
        affected = {p.pattern_id: p for p in trace.affected_patterns}
        p = affected.get("class_implication:launch_vehicle_company=>develops")
        assert p is not None
        assert p.old_confidence == 1.0
        assert p.new_confidence == pytest.approx(2 / 3)
        assert p.old_support == 3 and p.new_support == 2

    def test_affected_pattern_cites_removed_evidence(self):
        overlay = self._overlay()
        trace = analyze_counterfactual(overlay, "SpaceX", "develops", "Falcon 9")
        for p in trace.affected_patterns:
            assert p.removed_evidence

    def test_untouched_patterns_not_reported(self):
        overlay = self._overlay() + [
            _rel("Alpha", "is_a", "bakery", "stable"),
            _rel("Beta", "is_a", "bakery", "stable"),
            _rel("Alpha", "produces", "bread"),
            _rel("Beta", "produces", "bread"),
        ]
        trace = analyze_counterfactual(overlay, "SpaceX", "develops", "Falcon 9")
        ids = {p.pattern_id for p in trace.affected_patterns}
        assert not any("bakery" in pid for pid in ids)


# ---------------------------------------------------------------------------
# Honest audit for absent facts
# ---------------------------------------------------------------------------

class TestAbsentFactAudit:
    def test_absent_fact_is_audit(self):
        trace = analyze_counterfactual(_musk_overlay(), "Elon Musk", "founded", "Nokia")
        assert trace.decision == "audit"
        assert trace.removed_facts == []
        assert trace.lost_inferences == []

    def test_audit_render_is_explicit(self):
        trace = analyze_counterfactual(_musk_overlay(), "Elon Musk", "founded", "Nokia")
        text = render(trace)
        assert text.startswith("[audit]")


# ---------------------------------------------------------------------------
# Rendering and adapter
# ---------------------------------------------------------------------------

class TestRenderAndAdapter:
    def test_render_never_speculates(self):
        trace = analyze_counterfactual(_musk_overlay(), "Elon Musk", "founded", "SpaceX")
        text = render(trace)
        assert "can't answer a counterfactual as a fact" in text
        assert "Decision: analysis." in text

    def test_what_if_question_routes_to_counterfactual(self):
        result = try_answer_reasoning(
            "What if Elon Musk had not founded SpaceX?", _musk_overlay()
        )
        assert result.kind == "counterfactual"
        assert result.decision == "analysis"
        assert "share_founder" in result.answer_text

    def test_passive_form_parses(self):
        result = try_answer_reasoning(
            "What if SpaceX had never been founded by Elon Musk?", _musk_overlay()
        )
        assert result.kind == "counterfactual"
        assert result.decision == "analysis"

    def test_entity_nonexistence_form(self):
        result = try_answer_reasoning(
            "What would change if Elon Musk did not exist?", _musk_overlay()
        )
        assert result.kind == "counterfactual"
        assert result.decision == "analysis"
        assert result.detail["removed_facts"]

    def test_what_depends_on_form(self):
        result = try_answer_reasoning(
            "What depends on the fact that Elon Musk founded SpaceX?", _musk_overlay()
        )
        assert result.kind == "counterfactual"
        assert result.decision == "analysis"

    def test_deterministic_trace(self):
        a = analyze_counterfactual(_musk_overlay(), "Elon Musk", "founded", "SpaceX")
        b = analyze_counterfactual(_musk_overlay(), "Elon Musk", "founded", "SpaceX")
        assert a.to_dict() == b.to_dict()


# ---------------------------------------------------------------------------
# Universality — second domain
# ---------------------------------------------------------------------------

class TestSecondDomain:
    def test_counterfactual_in_pharma_domain(self):
        overlay = [
            _ent("Novo Nordisk"),
            _ent("Eli Lilly"),
            _ent("Lars Sørensen", "person"),
            _rel("Lars Sørensen", "founded", "Novo Nordisk", "stable"),
            _rel("Lars Sørensen", "founded", "Eli Lilly", "stable"),
            _rel("Novo Nordisk", "is_a", "pharmaceutical company", "stable"),
            _rel("Eli Lilly", "is_a", "pharmaceutical company", "stable"),
        ]
        trace = analyze_counterfactual(overlay, "Lars Sørensen", "founded", "Novo Nordisk")
        assert trace.decision == "analysis"
        lost = {(f.subject, f.predicate, f.object) for f in trace.lost_inferences}
        assert ("Novo Nordisk", "share_founder", "Eli Lilly") in lost


# ---------------------------------------------------------------------------
# Integration against the real promoted overlay
# ---------------------------------------------------------------------------

class TestIntegrationRealOverlay:
    @pytest.fixture(scope="class")
    def overlay_items(self):
        from worldpgt.assistant_surface.context_selector import resolve_overlay
        import json, pathlib
        path, _ = resolve_overlay("promoted")
        return json.loads(pathlib.Path(path).read_text())

    def test_musk_spacex_founding_has_dependents(self, overlay_items):
        trace = analyze_counterfactual(
            overlay_items, "Elon Musk", "founded", "SpaceX"
        )
        assert trace.decision == "analysis"
        assert trace.removed_facts
        assert trace.lost_inferences, (
            "shared-founder / expertise inferences should rest on this fact"
        )

    def test_absent_fact_audits_on_real_graph(self, overlay_items):
        trace = analyze_counterfactual(
            overlay_items, "Elon Musk", "founded", "Absolutely Nonexistent Corp"
        )
        assert trace.decision == "audit"
