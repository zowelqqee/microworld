"""Tests for Part 1 of the reasoning layer: explanatory chains.

The builder answers *why* a fact exists, not just *what* is true. Every link
must be a verified fact, an explicit inference rule, or a discovered pattern
(marked as an observation). A chain that does not close is reported honestly
as partial, with the frontier it reached. A fact not in the graph is an audit.

Synthetic overlays keep the unit tests fast and domain-independent; the final
class runs against the real promoted overlay.
"""

from __future__ import annotations

import pytest

from worldpgt.reasoning import discover_patterns, explain_fact, try_answer_reasoning
from worldpgt.reasoning.explanation_renderer import render
from worldpgt.reasoning.reasoning_adapter import parse_reasoning_question


# ---------------------------------------------------------------------------
# Helpers — minimal overlay builders (same shapes as test_inference_engine)
# ---------------------------------------------------------------------------

def _rel(subject: str, predicate: str, obj: str, stability: str = "semi_stable") -> dict:
    return {
        "overlay_type": "overlay_relation",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "stability": stability,
    }


def _def(subject: str, definition: str, stability: str = "stable") -> dict:
    return {
        "overlay_type": "overlay_definition",
        "subject": subject,
        "definition": definition,
        "stability": stability,
    }


def _ent(label: str, entity_type: str = "organization") -> dict:
    return {
        "overlay_type": "overlay_entity",
        "label": label,
        "entity_type": entity_type,
    }


def _aerospace_overlay() -> list[dict]:
    """The reference example: SpaceX develops Falcon 9, and the graph holds
    the full explanatory context (classes, causal chain, mission link)."""
    return [
        _ent("SpaceX"),
        _ent("Blue Origin"),
        _ent("Falcon 9", "product"),
        _ent("New Glenn", "product"),
        _ent("Elon Musk", "person"),
        _ent("Jeff Bezos", "person"),
        _rel("SpaceX", "is_a", "launch vehicle company", "stable"),
        _rel("Blue Origin", "is_a", "launch vehicle company", "stable"),
        _rel("SpaceX", "develops", "Falcon 9"),
        _rel("Blue Origin", "develops", "New Glenn"),
        _rel("Falcon 9", "is_a", "reusable launch vehicle", "stable"),
        _rel("New Glenn", "is_a", "reusable launch vehicle", "stable"),
        _rel("reusable launch vehicle", "reduces", "launch cost", "stable"),
        _rel("launch cost", "enables", "commercial spaceflight", "stable"),
        _rel("SpaceX", "known_for", "commercial spaceflight"),
        _rel("Elon Musk", "founded", "SpaceX", "stable"),
        _rel("Jeff Bezos", "founded", "Blue Origin", "stable"),
        _rel("Elon Musk", "is_a", "entrepreneur", "stable"),
        _rel("Jeff Bezos", "is_a", "entrepreneur", "stable"),
    ]


def _pharma_overlay() -> list[dict]:
    """Second domain proving the architecture is not aerospace-specific."""
    return [
        _ent("Novo Nordisk"),
        _ent("Ozempic", "product"),
        _rel("Novo Nordisk", "is_a", "pharmaceutical company", "stable"),
        _rel("Novo Nordisk", "produces", "Ozempic"),
        _rel("Ozempic", "is_a", "GLP-1 medication", "stable"),
        _rel("GLP-1 medication", "reduces", "blood sugar", "stable"),
        _rel("blood sugar", "enables", "diabetes management", "stable"),
        _rel("Novo Nordisk", "known_for", "diabetes management"),
    ]


# ---------------------------------------------------------------------------
# Closed chains
# ---------------------------------------------------------------------------

class TestClosedChains:
    def test_reference_chain_closes(self):
        chain = explain_fact(_aerospace_overlay(), "SpaceX", "develops", "Falcon 9")
        assert chain.decision == "answer"
        assert chain.fact_status == "direct"

    def test_chain_starts_with_the_target_fact(self):
        chain = explain_fact(_aerospace_overlay(), "SpaceX", "develops", "Falcon 9")
        first = chain.steps[0]
        assert (first.subject, first.predicate, first.object) == (
            "SpaceX", "develops", "Falcon 9",
        )

    def test_chain_includes_subject_classification(self):
        chain = explain_fact(_aerospace_overlay(), "SpaceX", "develops", "Falcon 9")
        assert any(
            s.predicate == "is_a" and s.object == "launch vehicle company"
            for s in chain.steps
        )

    def test_chain_walks_causal_context_to_mission(self):
        chain = explain_fact(_aerospace_overlay(), "SpaceX", "develops", "Falcon 9")
        triples = [(s.subject, s.predicate, s.object) for s in chain.steps]
        assert ("Falcon 9", "is_a", "reusable launch vehicle") in triples
        assert ("reusable launch vehicle", "reduces", "launch cost") in triples
        assert ("launch cost", "enables", "commercial spaceflight") in triples
        # Closure: the chain loops back to the subject's own mission fact.
        assert ("SpaceX", "known_for", "commercial spaceflight") in triples

    def test_every_step_is_verified_kind(self):
        chain = explain_fact(_aerospace_overlay(), "SpaceX", "develops", "Falcon 9")
        assert all(s.kind in {"fact", "rule", "pattern", "note"} for s in chain.steps)

    def test_pattern_step_included_when_patterns_supplied(self):
        overlay = _aerospace_overlay()
        patterns = discover_patterns(overlay)
        chain = explain_fact(overlay, "SpaceX", "develops", "Falcon 9", patterns=patterns)
        pattern_steps = [s for s in chain.steps if s.kind == "pattern"]
        assert pattern_steps and pattern_steps[0].pattern_id

    def test_works_in_second_domain(self):
        chain = explain_fact(_pharma_overlay(), "Novo Nordisk", "produces", "Ozempic")
        assert chain.decision == "answer"
        triples = [(s.subject, s.predicate, s.object) for s in chain.steps]
        assert ("Ozempic", "is_a", "GLP-1 medication") in triples
        assert ("Novo Nordisk", "known_for", "diabetes management") in triples

    def test_deterministic_output(self):
        a = explain_fact(_aerospace_overlay(), "SpaceX", "develops", "Falcon 9")
        b = explain_fact(_aerospace_overlay(), "SpaceX", "develops", "Falcon 9")
        assert a.to_dict() == b.to_dict()


# ---------------------------------------------------------------------------
# Inferred facts — proof chain becomes the explanation
# ---------------------------------------------------------------------------

class TestInferredFactExplanation:
    def _overlay(self) -> list[dict]:
        return [
            _ent("SpaceX", "organization"),
            _ent("Starlink", "service"),
            _def("Starlink", "satellite internet constellation operated by SpaceX"),
            _rel("SpaceX", "develops", "rockets"),
        ]

    def test_inferred_fact_explained_by_proof_chain(self):
        chain = explain_fact(self._overlay(), "Starlink", "develops", "rockets")
        assert chain.fact_status == "inferred"
        assert chain.decision == "answer"

    def test_rule_step_present_with_rule_id(self):
        chain = explain_fact(self._overlay(), "Starlink", "develops", "rockets")
        rule_steps = [s for s in chain.steps if s.kind == "rule"]
        assert len(rule_steps) == 1
        assert rule_steps[0].rule == "capability_inheritance_v1"

    def test_proof_chain_facts_are_included(self):
        chain = explain_fact(self._overlay(), "Starlink", "develops", "rockets")
        fact_steps = [s for s in chain.steps if s.kind == "fact"]
        assert any(s.predicate == "develops" and s.subject == "SpaceX" for s in fact_steps)


# ---------------------------------------------------------------------------
# Honest partial chains
# ---------------------------------------------------------------------------

class TestPartialChains:
    def test_unclosable_chain_is_partial_not_answer(self):
        overlay = [
            _ent("Acme"),
            _ent("Widget", "product"),
            _rel("Acme", "develops", "Widget"),
            _rel("Widget", "is_a", "gadget", "stable"),
            _rel("gadget", "reduces", "effort", "stable"),
            # No link from Acme back to anything on the walk — cannot close.
        ]
        chain = explain_fact(overlay, "Acme", "develops", "Widget")
        assert chain.decision == "partial"
        assert chain.fact_status == "direct"

    def test_partial_chain_reports_frontier(self):
        overlay = [
            _ent("Acme"),
            _ent("Widget", "product"),
            _rel("Acme", "develops", "Widget"),
            _rel("Widget", "is_a", "gadget", "stable"),
            _rel("gadget", "reduces", "effort", "stable"),
        ]
        chain = explain_fact(overlay, "Acme", "develops", "Widget")
        assert chain.frontier, "partial chain must show how far exploration got"

    def test_partial_chain_has_honest_note_step(self):
        overlay = [
            _ent("Acme"),
            _rel("Acme", "develops", "Widget"),
        ]
        chain = explain_fact(overlay, "Acme", "develops", "Widget")
        assert chain.decision == "partial"
        assert any(s.kind == "note" for s in chain.steps)

    def test_partial_render_says_partially(self):
        overlay = [_ent("Acme"), _rel("Acme", "develops", "Widget")]
        chain = explain_fact(overlay, "Acme", "develops", "Widget")
        text = render(chain)
        assert "partially" in text.lower()
        assert "Decision: partial." in text


# ---------------------------------------------------------------------------
# Audit: absent facts are never explained
# ---------------------------------------------------------------------------

class TestAbsentFactAudit:
    def test_absent_fact_is_audit(self):
        chain = explain_fact(_aerospace_overlay(), "SpaceX", "develops", "New Glenn")
        assert chain.decision == "audit"
        assert chain.fact_status == "absent"
        assert chain.steps == []

    def test_unknown_entities_are_audit(self):
        chain = explain_fact(_aerospace_overlay(), "Atlantis", "develops", "warp drive")
        assert chain.decision == "audit"

    def test_audit_render_is_explicit(self):
        chain = explain_fact(_aerospace_overlay(), "SpaceX", "develops", "New Glenn")
        text = render(chain)
        assert text.startswith("[audit]")
        assert "Decision: audit." in text


# ---------------------------------------------------------------------------
# Question adapter
# ---------------------------------------------------------------------------

class TestReasoningAdapterExplanations:
    def test_why_does_question_answers(self):
        result = try_answer_reasoning(
            "Why does SpaceX develop Falcon 9?", _aerospace_overlay()
        )
        assert result.kind == "explanation"
        assert result.decision == "answer"
        assert "Falcon 9" in result.answer_text

    def test_why_is_a_question(self):
        result = try_answer_reasoning(
            "Why is Falcon 9 a reusable launch vehicle?", _aerospace_overlay()
        )
        assert result.kind == "explanation"
        # The fact exists; chain may be partial, but never audit.
        assert result.decision in {"answer", "partial"}

    def test_why_did_found_question(self):
        result = try_answer_reasoning(
            "Why did Elon Musk found SpaceX?", _aerospace_overlay()
        )
        assert result.kind == "explanation"
        assert result.decision in {"answer", "partial"}

    def test_unrecognized_question_is_unsupported(self):
        result = try_answer_reasoning("Who founded SpaceX?", _aerospace_overlay())
        assert result.kind == "unsupported"
        assert result.decision == "audit"

    def test_pattern_notes_attached_when_patterns_given(self):
        overlay = _aerospace_overlay()
        patterns = discover_patterns(overlay)
        result = try_answer_reasoning(
            "Why does SpaceX develop Falcon 9?", overlay, patterns=patterns
        )
        assert result.pattern_notes
        assert all("By the way" in note for note in result.pattern_notes)

    def test_parser_extracts_predicate_from_verb(self):
        parsed = parse_reasoning_question("Why does Novo Nordisk produce Ozempic?")
        assert parsed == {
            "kind": "explanation",
            "subject": "Novo Nordisk",
            "predicate": "produces",
            "object": "Ozempic",
        }

    def test_detail_carries_full_chain_dict(self):
        result = try_answer_reasoning(
            "Why does SpaceX develop Falcon 9?", _aerospace_overlay()
        )
        assert result.detail["decision"] == "answer"
        assert result.detail["steps"]


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

    def test_inferred_fact_gets_closed_explanation(self, overlay_items):
        chain = explain_fact(overlay_items, "Starlink", "develops", "rockets")
        assert chain.fact_status in {"inferred", "direct"}
        assert chain.decision in {"answer", "partial"}
        assert chain.steps

    def test_absent_fact_still_audits_on_real_graph(self, overlay_items):
        chain = explain_fact(overlay_items, "SpaceX", "develops", "submarines")
        assert chain.decision == "audit"

    def test_real_graph_explanation_is_deterministic(self, overlay_items):
        a = explain_fact(overlay_items, "Starlink", "develops", "rockets")
        b = explain_fact(overlay_items, "Starlink", "develops", "rockets")
        assert a.to_dict() == b.to_dict()
