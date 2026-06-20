"""Tests for explicit ontology ``is_a`` traversal."""

from __future__ import annotations

from worldpgt.knowledge.ontology_traversal import find_is_a_path


def _rel(subject: str, obj: str, *, stability: str = "stable", risk: str = "low") -> dict:
    return {
        "overlay_type": "overlay_relation",
        "subject": subject,
        "predicate": "is_a",
        "object": obj,
        "stability": stability,
        "risk": risk,
        "trust": "overlay_candidate",
    }


def _definition(subject: str, definition: str) -> dict:
    return {
        "overlay_type": "overlay_definition",
        "subject": subject,
        "predicate": "is_a",
        "definition": definition,
        "stability": "stable",
        "risk": "low",
        "trust": "overlay_candidate",
    }


def test_direct_is_a_path_one_hop():
    path = find_is_a_path("SpaceX", "aerospace manufacturer", [_rel("SpaceX", "aerospace manufacturer")])

    assert path is not None
    assert [edge.display() for edge in path] == ["SpaceX | is_a | aerospace manufacturer"]


def test_optional_ontology_layer_extends_is_a_path():
    overlay = [_rel("Elon Musk", "businessman")]
    ontology_layer = [
        _rel("businessman", "worker"),
        _rel("worker", "person with an activity"),
    ]

    path = find_is_a_path(
        "Elon Musk",
        "person with an activity",
        overlay,
        ontology_layer_items=ontology_layer,
    )

    assert path is not None
    assert [edge.display() for edge in path] == [
        "Elon Musk | is_a | businessman",
        "businessman | is_a | worker",
        "worker | is_a | person with an activity",
    ]


def test_without_ontology_layer_keeps_old_missing_path_behavior():
    assert find_is_a_path(
        "Elon Musk",
        "worker",
        [_rel("Elon Musk", "businessman")],
    ) is None


def test_definition_is_a_edge_is_explicit_path():
    path = find_is_a_path("Robert Zubrin", "American aerospace engineer", [
        _definition("Robert Zubrin", "American aerospace engineer"),
    ])

    assert path is not None
    assert [edge.overlay_type for edge in path] == ["overlay_definition"]


def test_transitive_is_a_path_two_hops():
    path = find_is_a_path("SpaceX", "organization", [
        _rel("SpaceX", "aerospace manufacturer"),
        _rel("aerospace manufacturer", "manufacturer"),
        _rel("manufacturer", "organization"),
    ])

    assert path is not None
    assert [edge.object for edge in path] == [
        "aerospace manufacturer",
        "manufacturer",
        "organization",
    ]


def test_missing_path_returns_none():
    assert find_is_a_path("SpaceX", "person", [_rel("SpaceX", "organization")]) is None


def test_reverse_inference_is_not_allowed():
    assert find_is_a_path("organization", "SpaceX", [_rel("SpaceX", "organization")]) is None


def test_volatile_is_a_edge_is_not_traversed():
    assert find_is_a_path("SpaceX", "organization", [
        _rel("SpaceX", "aerospace manufacturer"),
        _rel("aerospace manufacturer", "organization", stability="volatile"),
    ]) is None
