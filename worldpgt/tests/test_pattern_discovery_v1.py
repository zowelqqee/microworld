"""Tests for Part 2 of the reasoning layer: graph pattern discovery.

Patterns are observations about the graph's own structure — never new facts.
Each carries supporting evidence (verified triples), confidence (share of
condition-matching entities satisfying the consequent), and counter-examples.
"""

from __future__ import annotations

import pytest

from worldpgt.reasoning import (
    discover_patterns,
    load_patterns,
    relevant_patterns,
    render_pattern_note,
    save_patterns,
)


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


def _founder_overlay() -> list[dict]:
    """Three aerospace companies founded by entrepreneurs; two develop launch
    vehicles, one does not — the reference cooccurrence pattern with one
    counter-example."""
    return [
        _ent("SpaceX"),
        _ent("Blue Origin"),
        _ent("Astra"),
        _ent("Elon Musk", "person"),
        _ent("Jeff Bezos", "person"),
        _ent("Chris Kemp", "person"),
        _rel("Elon Musk", "is_a", "entrepreneur", "stable"),
        _rel("Jeff Bezos", "is_a", "entrepreneur", "stable"),
        _rel("Chris Kemp", "is_a", "entrepreneur", "stable"),
        _rel("Elon Musk", "founded", "SpaceX", "stable"),
        _rel("Jeff Bezos", "founded", "Blue Origin", "stable"),
        _rel("Chris Kemp", "founded", "Astra", "stable"),
        _rel("SpaceX", "develops", "Falcon 9"),
        _rel("Blue Origin", "develops", "New Glenn"),
        # Astra has a different capability — the counter-example.
        _rel("Astra", "provides", "launch services"),
        _rel("Falcon 9", "is_a", "launch vehicle", "stable"),
        _rel("New Glenn", "is_a", "launch vehicle", "stable"),
    ]


def _find(patterns, pattern_id: str):
    return next((p for p in patterns if p.pattern_id == pattern_id), None)


# ---------------------------------------------------------------------------
# Cooccurrence family — the reference "founder is entrepreneur" pattern
# ---------------------------------------------------------------------------

class TestCooccurrencePatterns:
    def test_founder_entrepreneur_develops_pattern_found(self):
        patterns = discover_patterns(_founder_overlay(), min_confidence=0.5)
        p = _find(patterns, "cooccurrence:founded_by@entrepreneur=>develops")
        assert p is not None

    def test_confidence_is_share_of_matching_entities(self):
        patterns = discover_patterns(_founder_overlay(), min_confidence=0.5)
        p = _find(patterns, "cooccurrence:founded_by@entrepreneur=>develops")
        assert p.population == 3
        assert p.support == 2
        assert p.confidence == pytest.approx(2 / 3)

    def test_counter_example_recorded_with_note(self):
        patterns = discover_patterns(_founder_overlay(), min_confidence=0.5)
        p = _find(patterns, "cooccurrence:founded_by@entrepreneur=>develops")
        assert len(p.counter_examples) == 1
        assert p.counter_examples[0].entity == "Astra"
        assert "develops" in p.counter_examples[0].note

    def test_supporting_evidence_is_concrete_verified_triples(self):
        patterns = discover_patterns(_founder_overlay(), min_confidence=0.5)
        p = _find(patterns, "cooccurrence:founded_by@entrepreneur=>develops")
        assert ["SpaceX", "develops", "Falcon 9"] in p.supporting_evidence
        assert ["Blue Origin", "develops", "New Glenn"] in p.supporting_evidence
        # Condition facts are evidence too (inverse twin of founded).
        assert ["SpaceX", "founded_by", "Elon Musk"] in p.supporting_evidence

    def test_matched_entities_listed(self):
        patterns = discover_patterns(_founder_overlay(), min_confidence=0.5)
        p = _find(patterns, "cooccurrence:founded_by@entrepreneur=>develops")
        assert set(p.matched_entities) == {"SpaceX", "Blue Origin"}


# ---------------------------------------------------------------------------
# Class implication family
# ---------------------------------------------------------------------------

class TestClassImplicationPatterns:
    def _overlay(self) -> list[dict]:
        return [
            _rel("SpaceX", "is_a", "launch vehicle company", "stable"),
            _rel("Blue Origin", "is_a", "launch vehicle company", "stable"),
            _rel("Rocket Lab", "is_a", "launch vehicle company", "stable"),
            _rel("SpaceX", "develops", "Falcon 9"),
            _rel("Blue Origin", "develops", "New Glenn"),
            _rel("Rocket Lab", "develops", "Electron"),
            _rel("Falcon 9", "is_a", "launch vehicle", "stable"),
            _rel("New Glenn", "is_a", "launch vehicle", "stable"),
            _rel("Electron", "is_a", "launch vehicle", "stable"),
        ]

    def test_class_predicate_pattern_found(self):
        patterns = discover_patterns(self._overlay())
        p = _find(patterns, "class_implication:launch_vehicle_company=>develops")
        assert p is not None
        assert p.confidence == 1.0
        assert p.support == 3

    def test_shared_object_class_generalized(self):
        patterns = discover_patterns(self._overlay())
        p = _find(patterns, "class_implication:launch_vehicle_company=>develops")
        assert p.consequent.get("object_class") == "launch vehicle"
        assert "launch vehicle" in p.description

    def test_min_support_filters_small_classes(self):
        overlay = [
            _rel("Solo", "is_a", "unicorn company", "stable"),
            _rel("Solo", "develops", "magic"),
        ]
        patterns = discover_patterns(overlay, min_support=2)
        assert patterns == []

    def test_min_confidence_filters_weak_patterns(self):
        overlay = self._overlay() + [
            _rel("SpaceX", "operates", "Starbase"),
        ]
        patterns = discover_patterns(overlay, min_confidence=0.5)
        # Only 1 of 3 launch vehicle companies operates something → filtered.
        assert _find(patterns, "class_implication:launch_vehicle_company=>operates") is None


# ---------------------------------------------------------------------------
# General guarantees
# ---------------------------------------------------------------------------

class TestDiscoveryGuarantees:
    def test_empty_overlay_yields_no_patterns(self):
        assert discover_patterns([]) == []

    def test_deterministic_across_runs(self):
        a = [p.to_dict() for p in discover_patterns(_founder_overlay(), min_confidence=0.5)]
        b = [p.to_dict() for p in discover_patterns(_founder_overlay(), min_confidence=0.5)]
        assert a == b

    def test_deterministic_under_item_reordering(self):
        forward = discover_patterns(_founder_overlay(), min_confidence=0.5)
        reordered = discover_patterns(
            list(reversed(_founder_overlay())), min_confidence=0.5
        )
        assert [p.to_dict() for p in forward] == [p.to_dict() for p in reordered]

    def test_patterns_sorted_best_first(self):
        patterns = discover_patterns(_founder_overlay(), min_confidence=0.3)
        keys = [(-p.confidence, -p.support, p.pattern_id) for p in patterns]
        assert keys == sorted(keys)

    def test_max_patterns_cap_respected(self):
        patterns = discover_patterns(_founder_overlay(), min_confidence=0.3, max_patterns=2)
        assert len(patterns) <= 2

    def test_all_confidences_valid(self):
        for p in discover_patterns(_founder_overlay(), min_confidence=0.3):
            assert 0.0 < p.confidence <= 1.0
            assert p.support <= p.population


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPatternStore:
    def test_save_load_round_trip(self, tmp_path):
        patterns = discover_patterns(_founder_overlay(), min_confidence=0.5)
        target = tmp_path / "patterns.json"
        save_patterns(patterns, path=target, metadata={"run": "test"})
        loaded = load_patterns(target)
        assert [p.to_dict() for p in loaded] == [p.to_dict() for p in patterns]

    def test_missing_file_loads_empty(self, tmp_path):
        assert load_patterns(tmp_path / "absent.json") == []


# ---------------------------------------------------------------------------
# Patterns as reasoning context
# ---------------------------------------------------------------------------

class TestPatternsInReasoning:
    def test_relevant_patterns_by_entity_and_predicate(self):
        patterns = discover_patterns(_founder_overlay(), min_confidence=0.5)
        hits = relevant_patterns(patterns, "SpaceX", "develops")
        assert hits
        assert all(p.consequent.get("predicate") == "develops" for p in hits)

    def test_counter_example_entity_is_also_relevant(self):
        patterns = discover_patterns(_founder_overlay(), min_confidence=0.5)
        hits = relevant_patterns(patterns, "Astra", "develops")
        assert hits, "patterns an entity violates are relevant context for it"

    def test_irrelevant_entity_gets_no_patterns(self):
        patterns = discover_patterns(_founder_overlay(), min_confidence=0.5)
        assert relevant_patterns(patterns, "Nokia", "develops") == []

    def test_note_mentions_share_and_counter_examples(self):
        patterns = discover_patterns(_founder_overlay(), min_confidence=0.5)
        p = _find(patterns, "cooccurrence:founded_by@entrepreneur=>develops")
        note = render_pattern_note(p)
        assert "67%" in note
        assert "counter-example" in note


# ---------------------------------------------------------------------------
# Nightly runner
# ---------------------------------------------------------------------------

class TestNightlyRunner:
    def test_runner_writes_artifact(self, tmp_path):
        import json
        from worldpgt.reasoning.run_pattern_discovery import main

        overlay_path = tmp_path / "overlay.json"
        overlay_path.write_text(json.dumps(_founder_overlay()))
        out = tmp_path / "patterns.json"
        code = main([
            "--overlay-path", str(overlay_path),
            "--output", str(out),
            "--min-confidence", "0.5",
        ])
        assert code == 0
        loaded = load_patterns(out)
        assert loaded
        assert all(p.as_of for p in loaded)


# ---------------------------------------------------------------------------
# Integration against the real promoted overlay
# ---------------------------------------------------------------------------

class TestIntegrationRealOverlay:
    @pytest.fixture(scope="class")
    def patterns(self):
        from worldpgt.assistant_surface.context_selector import resolve_overlay
        import json, pathlib
        path, _ = resolve_overlay("promoted")
        items = json.loads(pathlib.Path(path).read_text())
        return discover_patterns(items)

    def test_real_graph_yields_patterns(self, patterns):
        assert patterns, "the promoted overlay should contain regularities"

    def test_every_pattern_has_evidence(self, patterns):
        assert all(p.supporting_evidence for p in patterns)

    def test_every_pattern_meets_thresholds(self, patterns):
        assert all(p.support >= 2 and p.confidence >= 0.5 for p in patterns)

    def test_pattern_ids_unique(self, patterns):
        ids = [p.pattern_id for p in patterns]
        assert len(ids) == len(set(ids))
