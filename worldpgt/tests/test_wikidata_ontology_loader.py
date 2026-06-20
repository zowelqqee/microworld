"""Tests for the read-only Wikidata P279 ontology loader."""

from __future__ import annotations

from worldpgt.knowledge.ontology_traversal import find_is_a_path
from worldpgt.knowledge.wikidata_ontology_loader import (
    WikidataP279Edge,
    WikidataSearchHit,
    build_wikidata_p279_ontology_layer,
    empty_is_a_object_labels,
    validate_ontology_layer,
)


def _rel(subject: str, obj: str, *, stability: str = "stable", risk: str = "low") -> dict:
    return {
        "overlay_type": "overlay_relation",
        "subject": subject,
        "predicate": "is_a",
        "object": obj,
        "evidence_text": f"{subject} is a {obj}.",
        "stability": stability,
        "risk": risk,
        "trust": "overlay_candidate",
    }


class FakeWikidataClient:
    def __init__(self) -> None:
        self.searches: list[str] = []
        self.edges = {
            "Q100": [WikidataP279Edge("Q100", "aerospace manufacturer", "Q200", "manufacturer")],
            "Q200": [WikidataP279Edge("Q200", "manufacturer", "Q300", "organization")],
            "Q300": [WikidataP279Edge("Q300", "organization", "Q400", "social group")],
            "Q500": [WikidataP279Edge("Q500", "businessperson", "Q600", "worker")],
        }

    def search_class(self, label: str):
        self.searches.append(label)
        if label == "aerospace manufacturer":
            return WikidataSearchHit("Q100", "aerospace manufacturer", "class of manufacturer")
        if label == "businessman":
            return WikidataSearchHit("Q500", "businessperson", "occupation")
        return None

    def p279_edges(self, qid: str, *, max_edges: int = 4):
        return list(self.edges.get(qid, []))[:max_edges]


def test_empty_is_a_object_labels_returns_leaf_classes_only() -> None:
    items = [
        _rel("SpaceX", "aerospace manufacturer"),
        _rel("aerospace manufacturer", "manufacturer"),
        _rel("Tesla", "company"),
    ]

    assert empty_is_a_object_labels(items) == ["company", "manufacturer"]


def test_builds_bounded_p279_layer_from_empty_class_labels() -> None:
    base = [_rel("SpaceX", "aerospace manufacturer")]
    client = FakeWikidataClient()

    layer, report = build_wikidata_p279_ontology_layer(base, client, max_depth=2)

    assert report.seed_empty_label_count == 1
    assert report.resolved_seed_count == 1
    assert report.raw_edge_count == 2
    assert report.accepted_edge_count == 2
    assert [(item["subject"], item["object"], item["wikidata_property"]) for item in layer] == [
        ("aerospace manufacturer", "manufacturer", "P279"),
        ("manufacturer", "organization", "P279"),
    ]


def test_layer_enables_ontology_traversal_chain() -> None:
    base = [_rel("SpaceX", "aerospace manufacturer")]
    layer, _report = build_wikidata_p279_ontology_layer(base, FakeWikidataClient(), max_depth=2)

    path = find_is_a_path("SpaceX", "organization", [*base, *layer])

    assert path is not None
    assert [(edge.subject, edge.object) for edge in path] == [
        ("SpaceX", "aerospace manufacturer"),
        ("aerospace manufacturer", "manufacturer"),
        ("manufacturer", "organization"),
    ]


def test_ontology_validation_rejects_self_loop() -> None:
    layer = [
        {
            "overlay_type": "overlay_relation",
            "subject": "organization",
            "predicate": "is_a",
            "object": "Organization",
            "evidence_text": "Wikidata P279 subclass of: organization -> Organization",
            "stability": "stable",
            "risk": "low",
        }
    ]

    accepted, rejected = validate_ontology_layer(layer, [])

    assert accepted == []
    assert rejected["self_loop"] == 1


def test_ontology_validation_rejects_volatile_edge() -> None:
    accepted, rejected = validate_ontology_layer([
        _rel("aerospace manufacturer", "manufacturer", stability="volatile"),
    ], [])

    assert accepted == []
    assert rejected["volatile_relation:is_a"] == 1


def test_first_wikidata_hop_preserves_local_seed_label() -> None:
    base = [_rel("Larry Ellison", "businessman")]

    layer, _report = build_wikidata_p279_ontology_layer(base, FakeWikidataClient(), max_depth=1)

    assert [(item["subject"], item["object"], item["wikidata_subject_label"]) for item in layer] == [
        ("businessman", "worker", "businessperson")
    ]
