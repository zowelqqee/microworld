"""Tests for the natural-language surface over reasoning artifacts.

``reasoning_verbalizer`` turns the structured ``ExplanationChain`` /
``CounterfactualTrace`` into flowing English — same principle as
``cognition.verbalization_engine``: every clause is assembled from a field
already present on the artifact, nothing is invented, and the same input
always verbalizes identically.
"""

from __future__ import annotations

from worldpgt.reasoning.explanation_builder import explain_fact
from worldpgt.reasoning.counterfactual import analyze_counterfactual
from worldpgt.reasoning.pattern_discovery import discover_patterns
from worldpgt.reasoning.reasoning_verbalizer import (
    verbalize_counterfactual,
    verbalize_explanation,
)


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
    return {"overlay_type": "overlay_entity", "label": label, "entity_type": entity_type}


def _aerospace_overlay() -> list[dict]:
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


def _musk_overlay() -> list[dict]:
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
# Explanation verbalization
# ---------------------------------------------------------------------------

class TestVerbalizeExplanation:
    def test_closed_chain_reads_as_prose_not_a_list(self):
        overlay = _aerospace_overlay()
        chain = explain_fact(overlay, "SpaceX", "develops", "Falcon 9")
        text = verbalize_explanation(chain)
        assert "[fact]" not in text
        assert "1." not in text
        assert text.startswith("SpaceX develops Falcon 9.")

    def test_closed_chain_mentions_every_step_fact(self):
        overlay = _aerospace_overlay()
        chain = explain_fact(overlay, "SpaceX", "develops", "Falcon 9")
        text = verbalize_explanation(chain)
        assert "launch vehicle company" in text
        assert "reusable launch vehicle" in text
        assert "commercial spaceflight" in text

    def test_pattern_step_rendered_as_observation_sentence(self):
        overlay = _aerospace_overlay()
        patterns = discover_patterns(overlay)
        chain = explain_fact(overlay, "SpaceX", "develops", "Falcon 9", patterns=patterns)
        text = verbalize_explanation(chain)
        assert "pattern I've noticed in my graph" in text

    def test_inferred_fact_uses_because_framing(self):
        overlay = [
            _ent("SpaceX", "organization"),
            _ent("Starlink", "service"),
            _def("Starlink", "satellite internet constellation operated by SpaceX"),
            _rel("SpaceX", "develops", "rockets"),
        ]
        chain = explain_fact(overlay, "Starlink", "develops", "rockets")
        text = verbalize_explanation(chain)
        assert "I infer this because" in text
        assert "confidence" in text

    def test_partial_chain_names_the_frontier(self):
        overlay = [
            _ent("Acme"),
            _ent("Widget", "product"),
            _rel("Acme", "develops", "Widget"),
            _rel("Widget", "is_a", "gadget", "stable"),
            _rel("gadget", "reduces", "effort", "stable"),
        ]
        chain = explain_fact(overlay, "Acme", "develops", "Widget")
        text = verbalize_explanation(chain)
        assert "trail runs out at" in text
        assert chain.frontier[0] in text

    def test_partial_chain_with_no_context_uses_audit_reason(self):
        overlay = [_ent("Acme"), _rel("Acme", "develops", "Widget")]
        chain = explain_fact(overlay, "Acme", "develops", "Widget")
        text = verbalize_explanation(chain)
        assert chain.audit_reason
        assert chain.audit_reason in text

    def test_audit_reads_as_a_single_sentence(self):
        chain = explain_fact(_aerospace_overlay(), "SpaceX", "develops", "New Glenn")
        text = verbalize_explanation(chain)
        assert text.startswith("I can't explain why")
        assert "[audit]" not in text

    def test_deterministic_across_calls(self):
        overlay = _aerospace_overlay()
        chain = explain_fact(overlay, "SpaceX", "develops", "Falcon 9")
        assert verbalize_explanation(chain) == verbalize_explanation(chain)

    def test_is_a_uses_correct_article_for_vowel_initial_class(self):
        overlay = [
            _ent("SpaceX"),
            _rel("SpaceX", "is_a", "aerospace manufacturer", "stable"),
        ]
        chain = explain_fact(overlay, "SpaceX", "is_a", "aerospace manufacturer")
        text = verbalize_explanation(chain)
        assert "is an aerospace manufacturer" in text
        assert "is a aerospace" not in text

    def test_never_fabricates_beyond_step_content(self):
        """Every entity mentioned in the prose traces to a step in the chain."""
        overlay = _aerospace_overlay()
        chain = explain_fact(overlay, "SpaceX", "develops", "Falcon 9")
        text = verbalize_explanation(chain)
        step_objects = {s.object for s in chain.steps if s.object}
        step_subjects = {s.subject for s in chain.steps if s.subject}
        for entity in ("Falcon 9", "commercial spaceflight", "launch vehicle company"):
            assert entity in step_objects | step_subjects
            assert entity in text


# ---------------------------------------------------------------------------
# Counterfactual verbalization
# ---------------------------------------------------------------------------

class TestVerbalizeCounterfactual:
    def test_reads_as_prose_not_structured_lists(self):
        trace = analyze_counterfactual(_musk_overlay(), "Elon Musk", "founded", "SpaceX")
        text = verbalize_counterfactual(trace)
        assert "Facts removed (" not in text
        assert text.startswith("If Elon Musk had not founded SpaceX")

    def test_lost_inferences_named_in_prose(self):
        trace = analyze_counterfactual(_musk_overlay(), "Elon Musk", "founded", "SpaceX")
        text = verbalize_counterfactual(trace)
        assert "Neuralink" in text
        assert "no longer be able to say" in text

    def test_affected_patterns_described_with_confidence_change(self):
        overlay = [
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
        trace = analyze_counterfactual(overlay, "SpaceX", "develops", "Falcon 9")
        text = verbalize_counterfactual(trace)
        assert "less confident" in text
        assert "%" in text

    def test_no_lost_inferences_says_so_plainly(self):
        overlay = [
            _ent("Acme"),
            _ent("Founder", "person"),
            _rel("Founder", "founded", "Acme", "stable"),
        ]
        trace = analyze_counterfactual(overlay, "Founder", "founded", "Acme")
        text = verbalize_counterfactual(trace)
        assert "No inferred facts in my graph depend on it." in text

    def test_entity_nonexistence_hypothesis_phrasing(self):
        trace = analyze_counterfactual(_musk_overlay(), "Elon Musk")
        text = verbalize_counterfactual(trace)
        assert text.startswith("If Elon Musk did not exist")

    def test_audit_reads_as_a_single_sentence(self):
        trace = analyze_counterfactual(_musk_overlay(), "Elon Musk", "founded", "Nokia")
        text = verbalize_counterfactual(trace)
        assert text.startswith("I can't analyze that counterfactual")
        assert "[audit]" not in text

    def test_deterministic_across_calls(self):
        trace = analyze_counterfactual(_musk_overlay(), "Elon Musk", "founded", "SpaceX")
        assert verbalize_counterfactual(trace) == verbalize_counterfactual(trace)

    def test_caps_very_long_inference_lists_with_a_remainder_note(self):
        overlay = [_ent("Founder", "person"), _rel("Founder", "founded", "F0", "stable")]
        for i in range(1, 10):
            overlay.append(_ent(f"F{i}"))
            overlay.append(_rel("Founder", "founded", f"F{i}", "stable"))
        overlay.append(_ent("F0"))
        trace = analyze_counterfactual(overlay, "Founder", "founded", "F0")
        text = verbalize_counterfactual(trace)
        assert "other conclusions that would stop holding too" in text


# ---------------------------------------------------------------------------
# Adapter now returns natural-language answer_text
# ---------------------------------------------------------------------------

class TestAdapterUsesVerbalizedText:
    def test_explanation_answer_text_is_natural_language(self):
        from worldpgt.reasoning.reasoning_adapter import try_answer_reasoning

        result = try_answer_reasoning(
            "Why does SpaceX develop Falcon 9?", _aerospace_overlay()
        )
        assert "[fact]" not in result.answer_text
        assert result.answer_text == verbalize_explanation(
            explain_fact(_aerospace_overlay(), "SpaceX", "develops", "Falcon 9")
        )

    def test_counterfactual_answer_text_is_natural_language(self):
        from worldpgt.reasoning.reasoning_adapter import try_answer_reasoning

        result = try_answer_reasoning(
            "What if Elon Musk had not founded SpaceX?", _musk_overlay()
        )
        assert "Facts removed (" not in result.answer_text
        assert "Neuralink" in result.answer_text
