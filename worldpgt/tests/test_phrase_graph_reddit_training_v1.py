"""Tests for training the phrase graph's fluency model from Reddit text.

The phrase graph is a graph-based, non-backprop "tiny local language model":
it learns *how* things are phrased (fragment templates, connector phrases,
relative-clause pronouns), never *what* is true. These tests prove that
Reddit-sourced community-context items can genuinely teach new phrasing —
and that this can never leak Reddit content into the facts a rendered
answer states.
"""

from __future__ import annotations

from pathlib import Path

from worldpgt.cognition.phrase_graph import (
    PhraseGraph,
    _train_from_community_context,
    _train_from_community_context_file,
    build_phrase_graph,
    facts_from_synthesis,
    generate,
)
from worldpgt.entity_qa.types import SynthesisAnswer, SynthesisFactGroup


def _widgetco_result() -> SynthesisAnswer:
    return SynthesisAnswer(
        subject="Widgetco",
        matched=True,
        match_kind="exact",
        definition="a small manufacturing company",
        entity_type="organization",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="develops",
                objects=["widgets"],
                tier="VERIFIED",
            ),
        ],
    )


def test_untrained_graph_cannot_cover_a_fresh_predicate() -> None:
    result = _widgetco_result()
    facts = facts_from_synthesis(result)

    assert generate(result, graph=PhraseGraph()) is None
    assert not PhraseGraph().covers(facts)


def test_reddit_sentences_teach_new_fragments_and_enable_generation() -> None:
    result = _widgetco_result()
    reddit_items = [
        {
            "trust": "community_context_only",
            "title": "",
            "text": (
                "This place is a scrappy little shop. "
                "It develops some genuinely clever gadgets for hobbyists."
            ),
        }
    ]

    graph = PhraseGraph()
    _train_from_community_context(graph, reddit_items)

    assert graph.best_fragment(("entity", "is_a")) == "is {definition}"
    assert graph.best_fragment(("entity", "develops")) == "develops {object_list}"

    answer = generate(result, graph=graph)
    assert answer == "Widgetco is a small manufacturing company. It develops widgets."


def test_low_trust_items_without_the_trust_marker_are_ignored() -> None:
    graph = PhraseGraph()
    _train_from_community_context(
        graph,
        [{"trust": "unrelated_source", "title": "", "text": "It develops rockets."}],
    )
    assert not graph.fragments


def test_reddit_training_only_learns_phrasing_never_facts() -> None:
    """The graph can only ever store predicate -> fragment *templates*.

    A Reddit sentence naming a specific company/founder must not leave any
    trace of that subject in the graph — only the abstract verb template.
    """

    graph = PhraseGraph()
    _train_from_community_context(
        graph,
        [
            {
                "trust": "community_context_only",
                "title": "",
                "text": "AcmeCorp was founded by Jane Doe, a total legend in the space.",
            }
        ],
    )

    fragment = graph.best_fragment(("entity", "founded_by"))
    assert fragment == "was founded by {object_list}"
    assert "AcmeCorp" not in fragment
    assert "Jane Doe" not in fragment
    # No node in the graph is keyed by the Reddit subject/object at all.
    assert all("acmecorp" not in str(node).lower() for node in graph.fragments)


def test_reddit_training_does_not_bend_the_person_relative_pronoun() -> None:
    """Casual Reddit "that" phrasing must not out-vote the curated "who".

    The subordinator vote for people is intentionally left untouched by this
    low-trust source so grammar for defining a *person* stays correct even as
    verb phrasing gets richer from real conversational text.
    """

    graph = PhraseGraph()
    graph.learn_subordinator("person", "who")
    graph.learn_subordinator("person", "who")

    # Repeat a single distinctive "that" sentence many times to simulate a
    # flood of casual community phrasing.
    reddit_items = [
        {
            "trust": "community_context_only",
            "title": "",
            "text": "This person is an engineer that founded a startup.",
        }
        for _ in range(20)
    ]

    _train_from_community_context(graph, reddit_items)

    assert graph.best_subordinator("person") == "who"


def test_build_phrase_graph_loads_committed_reddit_community_context() -> None:
    graph = build_phrase_graph(overlay_paths=[], artifact_paths=[], snapshot_dir=Path("/missing"))
    assert graph.fragments, "the committed reddit_community_context.json should teach some phrasing"


def test_build_phrase_graph_can_fully_disable_reddit_training() -> None:
    without_reddit = build_phrase_graph(
        overlay_paths=[], artifact_paths=[], snapshot_dir=Path("/missing"), community_context_paths=[]
    )
    assert not without_reddit.fragments
    assert not without_reddit.subordinators


def test_missing_community_context_file_is_a_safe_no_op() -> None:
    graph = PhraseGraph()
    _train_from_community_context_file(graph, Path("/does/not/exist.json"))
    assert not graph.fragments
