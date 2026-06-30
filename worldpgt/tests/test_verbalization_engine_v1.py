"""Tests for the verbal-reasoning layer (cognition.verbalization_engine).

Covers each artifact verbalizer in isolation plus the end-to-end think-aloud
surface (CLI/API) for the three demo questions in the task spec.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from worldpgt.cognition.inference_engine import InferredFact
from worldpgt.cognition.types import (
    ActionPlan,
    AllowedConclusion,
    EvidenceWorkspace,
    MissingEvidence,
    ReasoningProgram,
    ReasoningTrace,
    TaskFrame,
    WorkingMemory,
)
from worldpgt.cognition.verbalization_engine import (
    verbalize_audit,
    verbalize_inferred_fact,
    verbalize_multihop,
    verbalize_reasoning,
    verbalize_synthesis,
)
from worldpgt.entity_qa.types import SynthesisAnswer, SynthesisFactGroup


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def _trace(subject: str, conclusion: AllowedConclusion, *, action: str = "answer",
           missing=()):
    return ReasoningTrace(
        task=TaskFrame(intent="entity_profile", subject=subject, question_style="q"),
        workspace=EvidenceWorkspace(subject=subject),
        working_memory=WorkingMemory(current_subject=subject),
        program=ReasoningProgram(operations=()),
        steps=(),
        allowed_conclusions=(conclusion,),
        action=ActionPlan(next_action=action, surface_goal="state facts"),
        missing_evidence=tuple(missing),
    )


# --------------------------------------------------------------------------- #
# 1. Direct fact
# --------------------------------------------------------------------------- #

def test_verbalize_reasoning_direct_fact():
    trace = _trace(
        "SpaceX",
        AllowedConclusion(kind="direct_answer", text="SpaceX was founded by Elon Musk"),
    )
    out = verbalize_reasoning(trace)
    assert out == (
        "I found that SpaceX was founded by Elon Musk. "
        "This is a verified historical fact from my knowledge base."
    )


def test_verbalize_reasoning_direct_fact_strips_trailing_period():
    trace = _trace(
        "SpaceX",
        AllowedConclusion(kind="direct_answer", text="SpaceX was founded by Elon Musk."),
    )
    assert "founded by Elon Musk. This is a verified" in verbalize_reasoning(trace)


def test_verbalize_reasoning_gap_uses_missing_evidence():
    trace = _trace(
        "Starlink",
        AllowedConclusion(kind="mechanism_gap", text="I know Starlink is a satellite service"),
        action="answer_with_gap",
        missing=[
            MissingEvidence(
                role="mechanism",
                reason="how it works mechanically",
                next_questions=("what Starlink uses or operates via",),
            )
        ],
    )
    out = verbalize_reasoning(trace)
    assert "I know Starlink is a satellite service" in out
    assert "I don't have verified information about how it works mechanically" in out
    assert "what Starlink uses or operates via" in out


# --------------------------------------------------------------------------- #
# 2. Multi-hop chain
# --------------------------------------------------------------------------- #

def test_verbalize_multihop_chain():
    result = SimpleNamespace(
        hop1_detail={"subject": "Starlink", "predicate": "owned_by", "object": "SpaceX"},
        hop2_detail={"subject": "SpaceX", "predicate": "develops", "object": "Falcon 9"},
        hop1=None,
        hop2=None,
        answer_text="(fallback)",
    )
    out = verbalize_multihop(result)
    assert out == (
        "I reasoned through two steps: Starlink is owned by SpaceX, "
        "and SpaceX develops Falcon 9. "
        "Therefore Starlink is connected to Falcon 9 through SpaceX."
    )


def test_verbalize_multihop_accepts_display_strings():
    result = SimpleNamespace(
        hop1_detail=None,
        hop2_detail=None,
        hop1="Starlink | owned_by | SpaceX",
        hop2="SpaceX | develops | Falcon 9",
        answer_text="(fallback)",
    )
    out = verbalize_multihop(result)
    assert "Starlink is owned by SpaceX" in out
    assert "through SpaceX" in out


# --------------------------------------------------------------------------- #
# 3. Inferred fact
# --------------------------------------------------------------------------- #

def test_verbalize_inferred_fact():
    fact = InferredFact(
        subject="Starlink",
        predicate="develops",
        object="rockets",
        rule="capability_inheritance_v1",
        chain=(("Starlink", "owned_by", "SpaceX"), ("SpaceX", "develops", "rockets")),
        confidence=0.7,
    )
    out = verbalize_inferred_fact(fact)
    assert out == (
        "I inferred that Starlink develops rockets — not as a direct fact, "
        "but because Starlink is owned by SpaceX and SpaceX develops rockets. "
        "This conclusion carries medium confidence."
    )


def test_verbalize_inferred_fact_high_confidence():
    fact = InferredFact(
        subject="Powerwall", predicate="owned_by", object="Tesla",
        rule="ownership_transitivity_v1",
        chain=(("Powerwall", "owned_by", "Tesla Energy"), ("Tesla Energy", "owned_by", "Tesla")),
        confidence=1.0,
    )
    assert "high confidence" in verbalize_inferred_fact(fact)


# --------------------------------------------------------------------------- #
# 4. Audit / gap explanation
# --------------------------------------------------------------------------- #

def test_verbalize_audit_current_data():
    out = verbalize_audit(
        "Who is the current CEO of SpaceX?",
        reason="this requires current data which I don't have a verified snapshot for",
    )
    assert out.startswith("I looked for information about the current CEO of SpaceX")
    assert "requires current data" in out
    assert "I can't answer this reliably." in out


def test_verbalize_audit_with_known_and_needs():
    out = verbalize_audit(
        "How does Starlink work?",
        known="I know that Starlink is a satellite internet constellation operated by SpaceX",
        needs=["what Starlink uses or operates via"],
    )
    assert "I know that Starlink is a satellite internet constellation" in out
    assert "i don't have verified information about" in out.lower()
    assert "I would need facts about: what Starlink uses or operates via." in out


# --------------------------------------------------------------------------- #
# 5. Synthesis (three tiers)
# --------------------------------------------------------------------------- #

def _spacex_synthesis() -> SynthesisAnswer:
    return SynthesisAnswer(
        subject="SpaceX",
        matched=True,
        match_kind="exact",
        definition="an aerospace manufacturer and space transportation company",
        entity_type="organization",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation", predicate="develops",
                objects=["rockets", "spacecraft", "Falcon 9"], tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="inverse_relation", predicate="founded",
                objects=["Elon Musk"], tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="snapshot", predicate="net_worth",
                objects=["$1.1 trillion"], tier="SNAPSHOT",
                source_name="Forbes", as_of="June 2026",
            ),
        ],
    )


def test_verbalize_synthesis_three_tiers():
    inferred = [
        InferredFact(
            subject="SpaceX", predicate="operates", object="Starlink",
            rule="capability_inheritance_v1",
            chain=(("Starlink", "owned_by", "SpaceX"),), confidence=0.7,
        )
    ]
    out = verbalize_synthesis(_spacex_synthesis(), inferred)

    assert out.startswith("Here's what I know about SpaceX:")
    # VERIFIED tier
    assert "- It is an aerospace manufacturer and space transportation company. [verified]" in out
    assert "- It develops rockets, spacecraft, and Falcon 9. [verified]" in out
    assert "- It was founded by Elon Musk. [verified]" in out
    # SNAPSHOT tier
    assert (
        "- Its net worth was $1.1 trillion as of June 2026, according to Forbes. "
        "[snapshot — may be outdated]"
    ) in out
    # INFERRED tier
    assert "- It operates Starlink. [inferred — medium confidence]" in out


def test_verbalize_synthesis_uses_person_reference_for_definition():
    person = SynthesisAnswer(
        subject="Warren Buffett",
        matched=True,
        match_kind="exact",
        definition="American investor",
        entity_type="person",
        groups=[],
    )
    organization = SynthesisAnswer(
        subject="Standard Oil",
        matched=True,
        match_kind="exact",
        definition="corporate trust",
        entity_type="organization",
        groups=[],
    )

    person_out = verbalize_synthesis(person)
    org_out = verbalize_synthesis(organization)

    assert "- They are an American investor. [verified]" in person_out
    assert "- It is a corporate trust. [verified]" in org_out
    assert "- It is American investor. [verified]" not in person_out


def test_verbalize_synthesis_unmatched():
    ans = SynthesisAnswer(
        subject="Nonexistent", matched=False, match_kind="none",
        definition=None, entity_type=None,
    )
    assert "don't have anything verified about Nonexistent" in verbalize_synthesis(ans)


# --------------------------------------------------------------------------- #
# End-to-end think-aloud surface for the three demo questions
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def api():
    from worldpgt.api import server
    server._startup("pump-dry-run")
    return server


def _ask(api, question, **kw):
    from worldpgt.api.server import AskRequest
    return api.ask(AskRequest(question=question, think_aloud=True, enable_multihop=True, **kw))


def test_think_aloud_who_founded_spacex(api):
    r = _ask(api, "Who founded SpaceX?")
    assert r.thinking is not None
    assert "SpaceX" in r.thinking
    assert "Conclusion: direct answer available." in r.thinking
    assert r.answer == "SpaceX was founded by Elon Musk."


def test_think_aloud_shows_only_question_relevant_facts(api):
    # Fix 1: a founding question's thinking shows the founding fact, not the
    # entity's unrelated develop/produce relations.
    r = _ask(api, "Who founded SpaceX?")
    assert "founded by Elon Musk" in r.thinking
    assert "develops rockets" not in r.thinking
    assert "produces Falcon" not in r.thinking


def test_think_aloud_definition_question_shows_only_definition(api):
    r = _ask(api, "How does Starlink work?")
    assert "I found a definition:" in r.thinking
    # The definition line, not a dump of every Starlink relation.
    assert "located_in" not in r.thinking
    assert "owned_by SpaceX" not in r.thinking


def test_think_aloud_starlink_connected_to_falcon9(api):
    r = _ask(api, "How is Starlink connected to Falcon 9?")
    assert "Hop 1: Starlink is owned by SpaceX." in r.thinking
    assert "Hop 2: SpaceX develops Falcon 9." in r.thinking
    assert "I reasoned through two steps" in r.answer
    assert "through SpaceX" in r.answer


def test_think_aloud_how_does_starlink_work(api):
    r = _ask(api, "How does Starlink work?")
    assert r.thinking is not None
    # Honest: identifies Starlink but flags the missing operating mechanism.
    assert "Starlink" in r.answer
    assert "mechanism" in r.answer.lower() or "how it works" in r.answer.lower()


def test_think_aloud_off_has_no_thinking(api):
    from worldpgt.api.server import AskRequest
    r = api.ask(AskRequest(question="Who founded SpaceX?", think_aloud=False))
    assert r.thinking is None
    assert r.answer == "SpaceX was founded by Elon Musk."


# --------------------------------------------------------------------------- #
# Fix 2: capability inheritance is vetoed for divisions/subsidiaries
# --------------------------------------------------------------------------- #

def _build_overlay():
    return [
        {"overlay_type": "overlay_entity", "label": "Tesla", "entity_type": "organization"},
        {"overlay_type": "overlay_entity", "label": "Tesla Energy", "entity_type": "organization"},
        {"overlay_type": "overlay_entity", "label": "Starlink", "entity_type": "service"},
        {"overlay_type": "overlay_entity", "label": "SpaceX", "entity_type": "organization"},
        {"overlay_type": "overlay_definition", "subject": "Tesla Energy",
         "definition": "the clean energy division of Tesla", "stability": "stable"},
        {"overlay_type": "overlay_relation", "subject": "Tesla", "predicate": "produces",
         "object": "electric cars", "stability": "semi_stable"},
        {"overlay_type": "overlay_relation", "subject": "Starlink", "predicate": "owned_by",
         "object": "SpaceX", "stability": "semi_stable"},
        {"overlay_type": "overlay_relation", "subject": "SpaceX", "predicate": "develops",
         "object": "rockets", "stability": "semi_stable"},
    ]


def test_division_does_not_inherit_capability():
    from worldpgt.cognition.inference_engine import run_inference
    ws = run_inference(_build_overlay())
    # Tesla Energy is a *division* — it must NOT inherit "produces electric cars".
    assert all(f.predicate != "produces" for f in ws.for_subject("Tesla Energy"))


def test_owned_by_relation_still_inherits_capability():
    from worldpgt.cognition.inference_engine import run_inference
    ws = run_inference(_build_overlay())
    # Starlink is owned via a real relation (not a division) — it still inherits.
    inferred = {(f.predicate, f.object) for f in ws.for_subject("Starlink")}
    assert ("develops", "rockets") in inferred


def test_unless_pattern_is_optional():
    # Rules without unless_pattern parse and run unchanged.
    from worldpgt.cognition.rule_interpreter import load_base_rules
    rules = load_base_rules()
    assert any(r.rule_id == "ownership_transitivity_v1" and not r.unless_pattern for r in rules)
    assert any(r.rule_id == "capability_inheritance_v1" and r.unless_pattern for r in rules)
