"""Tests for the subject-locative fact bundle (poetry_lab reasoning transfer).

poetry_lab's ``_plan_description_scene`` bundles a primary fact with a
compatible modifier and a prepositional *link* about the same subject into one
enriched clause ("Особая комната в трактире казалась мрачной."), deciding
compatibility in the reasoning layer and only positioning it in the speech
layer. This is the worldpgt analogue: the reasoning layer (``synthesize``)
chooses one locative relation to fold into the subject noun phrase, and the
speech layer (``phrase_graph``) renders it as a participial post-modifier whose
surface is derived from the learned fragment -- "a robotics company
headquartered in Boston" instead of a separate "It is headquartered in Boston."

The reasoning tests call the selection directly (no network, no LLM), mirroring
poetry_lab's ``test_description_reasoning_*`` forms.
"""

from __future__ import annotations

from worldpgt.cognition.phrase_graph import PhraseGraph, generate
from worldpgt.entity_qa.synthesis_engine import _select_subject_locative
from worldpgt.entity_qa.types import SynthesisAnswer, SynthesisFactGroup


def _fwd(predicate: str, *objects: str) -> SynthesisFactGroup:
    return SynthesisFactGroup(
        kind="forward_relation", predicate=predicate, objects=list(objects), tier="VERIFIED"
    )


def _graph() -> PhraseGraph:
    graph = PhraseGraph()
    graph.add_fragment("organization", "is_a", "is {definition}")
    graph.add_fragment("organization", "develops", "develops {object_list}")
    graph.add_fragment("organization", "founded_by", "was founded by {object_list}")
    graph.add_fragment("organization", "headquartered_in", "is headquartered in {object_list}")
    graph.add_fragment("organization", "located_in", "is located in {object_list}")
    graph.learn_subordinator("entity", "that")
    return graph


def _acme(groups, *, locative=None, definition="a robotics company") -> SynthesisAnswer:
    return SynthesisAnswer(
        subject="Acme", matched=True, match_kind="exact",
        definition=definition, entity_type="organization",
        groups=groups, subject_locative=locative,
    )


# --- reasoning layer: which fact bundles ------------------------------------


def test_reasoning_selects_a_locative_link_for_the_bundle() -> None:
    groups = [_fwd("develops", "robots"), _fwd("headquartered_in", "Boston")]

    chosen = _select_subject_locative(groups, "a robotics company")

    assert chosen is not None
    assert (chosen.predicate, chosen.objects) == ("headquartered_in", ["Boston"])


def test_reasoning_requires_a_definition_to_attach_the_link_to() -> None:
    # No noun phrase to post-modify without a definition, mirroring poetry_lab's
    # rule that a link needs a subject phrase to attach to.
    groups = [_fwd("headquartered_in", "Boston")]

    assert _select_subject_locative(groups, None) is None


def test_reasoning_never_promotes_a_non_locative_relation_into_the_bundle() -> None:
    groups = [_fwd("develops", "robots"), _fwd("founded_by", "Ada Stone")]

    assert _select_subject_locative(groups, "a robotics company") is None


def test_reasoning_folds_at_most_one_locative_deterministically() -> None:
    # headquartered_in outranks located_in in the fold order, so the choice is
    # stable when an entity carries more than one place relation.
    groups = [_fwd("located_in", "Massachusetts"), _fwd("headquartered_in", "Boston")]

    chosen = _select_subject_locative(groups, "a robotics company")

    assert chosen.predicate == "headquartered_in"


# --- speech layer: how the bundle renders -----------------------------------


def test_speech_folds_the_locative_into_the_subject_noun_phrase() -> None:
    locative = _fwd("headquartered_in", "Boston")
    result = _acme([_fwd("develops", "robots"), locative], locative=locative)

    answer = generate(result, graph=_graph())

    assert answer == "Acme is a robotics company headquartered in Boston that develops robots."
    # The folded fact is not also emitted as its own flat sentence.
    assert "It is headquartered in Boston" not in answer


def test_speech_without_a_chosen_bundle_keeps_the_old_flat_rendering() -> None:
    # subject_locative=None => reasoning layer chose no bundle => unchanged.
    locative = _fwd("headquartered_in", "Boston")
    result = _acme([_fwd("develops", "robots"), locative], locative=None)

    answer = generate(result, graph=_graph())

    assert answer == (
        "Acme is a robotics company that develops robots. It is headquartered in Boston."
    )


def test_speech_falls_back_to_a_flat_fact_when_no_phrasing_was_learned() -> None:
    """If the graph learned no copular phrasing for the locative, the fold
    cannot be surfaced -- the fact must render normally, never be dropped."""

    graph = PhraseGraph()
    graph.add_fragment("organization", "is_a", "is {definition}")
    graph.add_fragment("organization", "develops", "develops {object_list}")
    # headquartered_in taught only as a bare active verb, not a copular phrase.
    graph.add_fragment("organization", "headquartered_in", "headquarters {object_list}")
    graph.learn_subordinator("entity", "that")

    locative = _fwd("headquartered_in", "Boston")
    result = _acme([_fwd("develops", "robots"), locative], locative=locative)

    answer = generate(result, graph=graph)

    assert answer is not None
    assert "Boston" in answer  # never silently dropped
    assert "headquartered in Boston" not in answer  # not folded as a participle
