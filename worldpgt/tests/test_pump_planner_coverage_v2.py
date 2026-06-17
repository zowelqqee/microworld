"""Knowledge Pump Planner Coverage v2 — deterministic unit tests.

17 test categories covering Parts A-E of v1.8:
  A. Define X. / define X  →  entity_definition  (router + analyzer)
  B. What kind/type/sort of thing is X?  →  entity_definition  (router + analyzer)
  C. What is/was developed by X?  →  entity_relation / relation_lookup  (router + analyzer)
  D. 2-char acronym "XM" indexed  (entity matcher _MIN_SURFACE_LEN = 2)
  E. Safety regressions — current/live still fires; adversarial still audited

No network. No overlay writes. No memory modifications.
"""

from __future__ import annotations

import pytest

from worldpgt.assistant_surface.question_router import route
from worldpgt.context_pack.entity_matcher import build_surface_index
from worldpgt.entity_qa.entity_answer_renderer import render
from worldpgt.entity_qa.entity_question_analyzer import analyze
from worldpgt.entity_qa.types import AnalyzedEntityQuestion, EntityQAEvidence, EntityQAPlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockOverlay:
    """Minimal overlay stub for entity matcher tests."""

    def __init__(
        self,
        entities=None,
        definitions=None,
        relations=None,
        source_facts=None,
    ):
        self._entities = entities or []
        self._definitions = definitions or []
        self._relations = relations or []
        self._source_facts = source_facts or []

    def entities(self):
        return self._entities

    def definitions(self):
        return self._definitions

    def relations(self):
        return self._relations

    def source_facts(self):
        return self._source_facts


def _analyzed(intent, subject, predicate=None):
    return AnalyzedEntityQuestion(
        question="",
        intent=intent,
        subject=subject,
        predicate_hint=predicate,
        secondary_entity=None,
        source_hint=None,
        is_current_query=False,
        is_unsupported=False,
    )


def _plan_with_definition(subject, definition_text):
    return EntityQAPlan(
        analyzed=_analyzed("define_entity", subject),
        decision="answer",
        audit_reason=None,
        evidence=EntityQAEvidence(),
        render_template="define_entity",
        render_args={
            "entity": None,
            "definition": {"subject": subject, "definition": definition_text},
            "relations": [],
            "subject": subject,
        },
        confidence=0.95,
    )


# ===========================================================================
# Category 1: Router — Define X. → entity_definition
# ===========================================================================


def test_router_define_period_routes_entity_definition():
    r = route("Define SpaceX.")
    assert r.intent == "entity_definition"


def test_router_define_period_extracts_subject():
    r = route("Define SpaceX.")
    assert r.subject == "SpaceX"


# ===========================================================================
# Category 2: Router — define X (no period, lowercase) → entity_definition
# ===========================================================================


def test_router_define_no_period_routes_entity_definition():
    r = route("define Starlink")
    assert r.intent == "entity_definition"


def test_router_define_no_period_extracts_subject():
    r = route("define Starlink")
    assert r.subject == "Starlink"


# ===========================================================================
# Category 3: Router — What kind of thing is X? → entity_definition
# ===========================================================================


def test_router_what_kind_of_thing_routes_entity_definition():
    r = route("What kind of thing is Starlink?")
    assert r.intent == "entity_definition"


def test_router_what_kind_of_thing_extracts_subject():
    r = route("What kind of thing is Starlink?")
    assert r.subject == "Starlink"


# ===========================================================================
# Category 4: Router — What type of thing is X? → entity_definition
# ===========================================================================


def test_router_what_type_of_thing_routes_entity_definition():
    r = route("What type of thing is Exos Aerospace?")
    assert r.intent == "entity_definition"


def test_router_what_type_of_thing_extracts_subject():
    r = route("What type of thing is Exos Aerospace?")
    assert r.subject == "Exos Aerospace"


# ===========================================================================
# Category 5: Router — What sort of thing is X? → entity_definition
# ===========================================================================


def test_router_what_sort_of_thing_routes_entity_definition():
    r = route("What sort of thing is Crew Dragon?")
    assert r.intent == "entity_definition"


# ===========================================================================
# Category 6: Router — What is developed by X? → entity_relation
# ===========================================================================


def test_router_what_is_developed_by_routes_entity_relation():
    r = route("What is developed by SpaceX?")
    assert r.intent == "entity_relation"


def test_router_what_is_developed_by_extracts_subject():
    r = route("What is developed by SpaceX?")
    assert r.subject == "SpaceX"


# ===========================================================================
# Category 7: Router — What was developed by X? → entity_relation
# ===========================================================================


def test_router_what_was_developed_by_routes_entity_relation():
    r = route("What was developed by SpaceX?")
    assert r.intent == "entity_relation"


def test_router_what_was_developed_by_extracts_subject():
    r = route("What was developed by SpaceX?")
    assert r.subject == "SpaceX"


# ===========================================================================
# Category 8: Router — passive develops does NOT route as entity_definition
# ===========================================================================


def test_router_passive_developed_by_not_entity_definition():
    r = route("What is developed by SpaceX?")
    assert r.intent != "entity_definition"


def test_router_passive_produced_by_routes_entity_relation():
    r = route("What is produced by Tesla?")
    assert r.intent == "entity_relation"


# ===========================================================================
# Category 9: Router — current/live safety regression still fires
# ===========================================================================


def test_router_current_stock_price_still_fires():
    r = route("What is Tesla's current stock price?")
    assert r.intent == "current_live_request"
    assert r.is_hard_safety is True


def test_router_define_current_not_hijacked():
    r = route("What is the current CEO of SpaceX?")
    assert r.intent == "current_live_request"


# ===========================================================================
# Category 10: Analyzer — Define X. → define_entity, correct subject
# ===========================================================================


def test_analyzer_define_period_intent():
    a = analyze("Define SpaceX.")
    assert a.intent == "define_entity"


def test_analyzer_define_period_subject():
    a = analyze("Define SpaceX.")
    assert a.subject == "SpaceX"


def test_analyzer_define_lowercase_intent():
    a = analyze("define Starlink")
    assert a.intent == "define_entity"


def test_analyzer_define_lowercase_subject():
    a = analyze("define Starlink")
    assert a.subject == "Starlink"


# ===========================================================================
# Category 11: Analyzer — What kind of thing is X? → define_entity, subject
# ===========================================================================


def test_analyzer_what_kind_of_thing_intent():
    a = analyze("What kind of thing is Starlink?")
    assert a.intent == "define_entity"


def test_analyzer_what_kind_of_thing_subject():
    a = analyze("What kind of thing is Starlink?")
    assert a.subject == "Starlink"


def test_analyzer_what_type_of_thing_intent():
    a = analyze("What type of thing is Exos Aerospace?")
    assert a.intent == "define_entity"


def test_analyzer_what_type_of_thing_subject():
    a = analyze("What type of thing is Exos Aerospace?")
    assert a.subject == "Exos Aerospace"


# ===========================================================================
# Category 12: Analyzer — What is developed by X? → relation_lookup, develops
# ===========================================================================


def test_analyzer_what_is_developed_by_intent():
    a = analyze("What is developed by SpaceX?")
    assert a.intent == "relation_lookup"


def test_analyzer_what_is_developed_by_predicate():
    a = analyze("What is developed by SpaceX?")
    assert a.predicate_hint == "develops"


def test_analyzer_what_is_developed_by_subject():
    a = analyze("What is developed by SpaceX?")
    assert a.subject == "SpaceX"


# ===========================================================================
# Category 13: Analyzer — What was developed by X? → relation_lookup, develops
# ===========================================================================


def test_analyzer_what_was_developed_by_intent():
    a = analyze("What was developed by SpaceX?")
    assert a.intent == "relation_lookup"


def test_analyzer_what_was_developed_by_predicate():
    a = analyze("What was developed by SpaceX?")
    assert a.predicate_hint == "develops"


def test_analyzer_what_was_developed_by_subject():
    a = analyze("What was developed by SpaceX?")
    assert a.subject == "SpaceX"


def test_analyzer_passive_produced_by_predicate():
    a = analyze("What is produced by Tesla?")
    assert a.intent == "relation_lookup"
    assert a.predicate_hint == "produces"


# ===========================================================================
# Category 14: Renderer — ordinal definition uses "the"
# ===========================================================================


def test_renderer_ordinal_sixth_uses_the():
    plan = _plan_with_definition("June", "sixth month of the year")
    answer = render(plan)
    assert "June is the sixth month" in answer


def test_renderer_ordinal_first_uses_the():
    plan = _plan_with_definition("January", "first month of the Gregorian calendar")
    answer = render(plan)
    assert "January is the first month" in answer


def test_renderer_ordinal_numeric_uses_the():
    plan = _plan_with_definition("July", "7th month of the year")
    answer = render(plan)
    assert "July is the 7th month" in answer


# ===========================================================================
# Category 15: Renderer — superlative definition uses "the"
# ===========================================================================


def test_renderer_superlative_largest_uses_the():
    plan = _plan_with_definition("Amazon", "largest online retailer by revenue")
    answer = render(plan)
    assert "Amazon is the largest" in answer


def test_renderer_superlative_most_uses_the():
    plan = _plan_with_definition("Falcon 9", "most frequently flown orbital rocket")
    answer = render(plan)
    assert "Falcon 9 is the most frequently" in answer


# ===========================================================================
# Category 15b: Renderer — vowel definition uses "an" (regression)
# ===========================================================================


def test_renderer_vowel_definition_uses_an():
    plan = _plan_with_definition("SpaceX", "aerospace manufacturer")
    answer = render(plan)
    assert "SpaceX is an aerospace manufacturer" in answer


def test_renderer_consonant_definition_uses_a():
    plan = _plan_with_definition("Tesla", "battery-electric vehicle manufacturer")
    answer = render(plan)
    assert "Tesla is a battery-electric" in answer


# ===========================================================================
# Category 16: Entity matcher — 2-char acronym "XM" is indexed
# ===========================================================================


def test_entity_matcher_two_char_acronym_indexed():
    overlay = _MockOverlay(
        relations=[
            {"subject": "XM", "predicate": "founded_by", "object": "Lon Levin"}
        ]
    )
    index = build_surface_index(overlay)
    surfaces = {s for s, _, _ in index}
    assert "XM" in surfaces


def test_entity_matcher_two_char_entity_label_indexed():
    overlay = _MockOverlay(
        entities=[
            {
                "label": "XM",
                "entity_id": "xm-radio",
                "entity_type": "organization",
                "aliases": [],
            }
        ]
    )
    index = build_surface_index(overlay)
    surfaces = {s for s, _, _ in index}
    assert "XM" in surfaces


# ===========================================================================
# Category 17: Entity matcher — generic 2-char stopwords NOT indexed
# ===========================================================================


@pytest.mark.parametrize("word", ["is", "of", "or", "in", "on", "by", "to", "an"])
def test_entity_matcher_generic_stopword_not_indexed(word):
    overlay = _MockOverlay(
        relations=[{"subject": word, "predicate": "is_a", "object": "thing"}]
    )
    index = build_surface_index(overlay)
    surfaces = {s.lower() for s, _, _ in index}
    assert word.lower() not in surfaces
