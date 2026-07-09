"""Tests for the generic active-voice verb fallback in phrase_graph.py
(2026-07-07).

Motivation: ``_PHRASE_MARKERS`` is a fixed, hand-enumerated list of ~26
surface phrases. "leader_of" has NO entry there at all, even though it's a
supported predicate everywhere else in the system (entity_qa,
relation_extraction_v2's own ``_VERB_RELATIONS``) -- so no amount of Reddit
training text could ever teach the phrase graph how to phrase a leader_of
fact. ``_generic_verb_fragment`` closes that gap by reusing
``relation_extraction_v2.spacy_extractor``'s own verb table instead of a
second, parallel list.
"""

from __future__ import annotations

from worldpgt.cognition.phrase_graph import (
    PhraseGraph,
    _fragment_from_sentence,
    _nodes_from_sentence,
    _train_from_sentences,
)


def test_leader_of_has_no_fixed_marker_but_is_learned_via_generic_fallback():
    graph = PhraseGraph()
    _train_from_sentences(
        graph, ["She leads the startup with a small team."], "person", learn_subordinators=False
    )
    assert graph.best_fragment(("person", "leader_of")) == "leads {object_list}"


def test_nodes_from_sentence_includes_the_generic_verb_node():
    nodes = _nodes_from_sentence("He leads the company from a small office.", "person")
    assert ("person", "leader_of") in nodes


def test_fragment_from_sentence_matches_generic_predicate():
    fragment = _fragment_from_sentence("He leads the company.", "leader_of")
    assert fragment == "leads {object_list}"


def test_fixed_marker_still_wins_and_generic_fallback_does_not_double_up():
    """A sentence that already matches a fixed _PHRASE_MARKERS entry must not
    also register a duplicate/competing generic node for the same predicate."""
    nodes = _nodes_from_sentence("Acme develops robots for warehouses.", "organization")
    develops_nodes = [n for n in nodes if n[1] == "develops"]
    assert len(develops_nodes) == 1


def test_arbitrary_content_verb_is_learned_by_surface_lemma():
    """SPEECH side: any content verb outside both _PHRASE_MARKERS and
    _VERB_RELATIONS ("win", "make", "have", "include") is a legitimate piece
    of language to learn a phrasing for -- keyed by its own surface lemma,
    since it maps to no canonical fact predicate. (Contrast the facts side,
    where "have"/"make" WOULD be noise.)"""
    nodes = _nodes_from_sentence("She won the championship last year.", "person")
    assert ("person", "win") in nodes


def test_canonical_verb_still_keys_to_its_canonical_predicate():
    """A verb that IS in _VERB_RELATIONS maps to the canonical predicate
    (lead -> leader_of), so learned fragments line up with overlay facts."""
    nodes = _nodes_from_sentence("He leads the company.", "person")
    assert ("person", "leader_of") in nodes
    assert ("person", "lead") not in nodes


def test_copular_be_is_not_learned_as_a_relation_phrasing():
    """"is/was" is the definition (is_a) path's job -- it must never be
    captured here as an ordinary relation verb."""
    nodes = _nodes_from_sentence("Tesla is a car company.", "organization")
    assert ("organization", "be") not in nodes


def test_person_and_org_have_are_distinct_typed_buckets():
    """The typed (entity_type, verb) key separates different senses of the
    same verb -- a person's "have" and a company's "have" don't collide."""
    graph = PhraseGraph()
    _train_from_sentences(graph, ["She has a doctorate."], "person", learn_subordinators=False)
    _train_from_sentences(graph, ["The company has offices."], "organization", learn_subordinators=False)
    assert ("person", "have") in graph.fragments
    assert ("organization", "have") in graph.fragments
