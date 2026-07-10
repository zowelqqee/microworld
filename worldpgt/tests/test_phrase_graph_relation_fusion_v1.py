"""Tests for fusing consecutive facts that share a *learned grammatical frame*
into one shared-subject sentence in ``cognition.phrase_graph.generate``.

Before this change, ``generate`` only fused consecutive *capability* facts
(develops/produces/...) into one sentence, via a hardcoded predicate list.
The poetry_lab experiment taught the opposite discipline: a fact's combinable
role must be recovered from the surface form the corpus produced, not from a
curated vocabulary. So fusion is now decided by ``_fusion_class`` reading each
fact's LEARNED fragment ("develops X" -> active, "was founded by X" ->
past-passive, "is owned by X" -> copular). Facts sharing a frame coordinate
under one subject; a frame change starts a new sentence, so tense/voice never
collide inside one clause list -- and a brand-new relation type fuses (or not)
purely by whatever phrasing the graph learned for it, with no edit here.

All tests run against a hand-trained, isolated ``PhraseGraph`` -- no dependency
on the committed overlay/artifact data.
"""

from __future__ import annotations

from worldpgt.cognition.phrase_graph import PhraseGraph, generate
from worldpgt.entity_qa.types import SynthesisAnswer, SynthesisEnrichment, SynthesisFactGroup


def _graph() -> PhraseGraph:
    graph = PhraseGraph()
    graph.add_fragment("organization", "is_a", "is {definition}")
    graph.add_fragment("organization", "founded_by", "was founded by {object_list}")
    graph.add_fragment("organization", "headquartered_in", "is headquartered in {object_list}")
    graph.add_fragment("organization", "owned_by", "is owned by {object_list}")
    graph.add_fragment("organization", "develops", "develops {object_list}")
    graph.add_fragment("organization", "produces", "produces {object_list}")
    return graph


def _widgetco(*, groups: list[SynthesisFactGroup], enrichment=None) -> SynthesisAnswer:
    return SynthesisAnswer(
        subject="Widgetco",
        matched=True,
        match_kind="exact",
        definition="a small manufacturing company",
        entity_type="organization",
        groups=groups,
        enrichment=enrichment,
    )


def test_same_frame_relation_facts_fuse_into_one_sentence() -> None:
    # Both learned fragments start with the copular "is ..." frame, so they
    # coordinate under one subject -- decided by the fragment surface, not by
    # any "these predicates are fusible" list.
    result = _widgetco(groups=[
        SynthesisFactGroup(kind="forward_relation", predicate="headquartered_in", objects=["Springfield"], tier="VERIFIED"),
        SynthesisFactGroup(kind="forward_relation", predicate="owned_by", objects=["MegaCorp"], tier="VERIFIED"),
    ])

    answer = generate(result, graph=_graph())

    assert answer == (
        "Widgetco is a small manufacturing company. "
        "It is owned by MegaCorp, and is headquartered in Springfield."
    )
    assert answer.count("It ") == 1


def test_different_frames_do_not_fuse_even_though_both_are_relations() -> None:
    """The key demonstration that grouping is derived, not hardcoded: a
    past-passive fact ("was founded by") and a copular fact ("is headquartered
    in") are both 'relations', yet they render as two sentences because their
    learned frames differ -- so tense/voice never collide in one clause list."""

    result = _widgetco(groups=[
        SynthesisFactGroup(kind="forward_relation", predicate="founded_by", objects=["Ada Stone"], tier="VERIFIED"),
        SynthesisFactGroup(kind="forward_relation", predicate="headquartered_in", objects=["Springfield"], tier="VERIFIED"),
    ])

    answer = generate(result, graph=_graph())

    assert answer == (
        "Widgetco is a small manufacturing company. "
        "It was founded by Ada Stone. It is headquartered in Springfield."
    )


def test_frame_fusion_preserves_founding_enrichment() -> None:
    """A founding fact rendered through the run path (even as a run of one, in
    its own past-passive frame) must still carry the enrichment appositive that
    a solo founded_by fact gets. Regression guard: the run renderer originally
    had no enrichment-weaving path at all."""

    result = _widgetco(
        groups=[
            SynthesisFactGroup(kind="forward_relation", predicate="founded_by", objects=["Ada Stone"], tier="VERIFIED"),
            SynthesisFactGroup(kind="forward_relation", predicate="headquartered_in", objects=["Springfield"], tier="VERIFIED"),
        ],
        enrichment=SynthesisEnrichment(object="Ada Stone", note="an engineer", relation="founded_by"),
    )

    answer = generate(result, graph=_graph())

    assert "was founded by Ada Stone, an engineer" in answer
    assert "is headquartered in Springfield" in answer


def test_active_capability_facts_still_fuse() -> None:
    """Consecutive active-frame facts coordinate exactly as capability facts
    did before -- no regression for the common 'develops X, produces Y' case."""

    result = _widgetco(groups=[
        SynthesisFactGroup(kind="forward_relation", predicate="develops", objects=["widgets"], tier="VERIFIED"),
        SynthesisFactGroup(kind="forward_relation", predicate="produces", objects=["gadgets"], tier="VERIFIED"),
    ])

    answer = generate(result, graph=_graph())

    assert answer == (
        "Widgetco is a small manufacturing company. "
        "It develops widgets, and produces gadgets."
    )


def test_solo_relation_fact_still_renders_as_before() -> None:
    """A run of exactly one fact behaves like the old flat per-fact rendering
    -- no regression for the common single-fact case."""

    result = _widgetco(groups=[
        SynthesisFactGroup(kind="forward_relation", predicate="headquartered_in", objects=["Springfield"], tier="VERIFIED"),
    ])

    answer = generate(result, graph=_graph())

    assert answer == "Widgetco is a small manufacturing company. It is headquartered in Springfield."
