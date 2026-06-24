"""Tests for entity_type_classifier.

Covers:
  - classify_entity_type() keyword rules for canonical types
  - Priority ordering (person > organization > publication > service > product > vehicle > program > place > concept > technology > other)
  - infer_entity_type() from overlay_entity, overlay_definition, overlay_relation items
  - Demo cases from the spec: Elon Musk, SpaceX, Falcon 9, Project Mercury,
    Tesla Semi, Robert Zubrin
"""

from __future__ import annotations

import pytest

from worldpgt.knowledge.entity_type_classifier import (
    classify_entity_type,
    infer_entity_type,
)


# ---------------------------------------------------------------------------
# classify_entity_type — direct keyword matching
# ---------------------------------------------------------------------------

class TestClassifyEntityType:
    # Person
    def test_elon_musk(self):
        assert classify_entity_type("American entrepreneur and businessman") == "person"

    def test_robert_zubrin(self):
        assert classify_entity_type("American aerospace engineer and author") == "person"

    def test_astronaut(self):
        assert classify_entity_type("Neil Armstrong was an astronaut and pilot") == "person"

    def test_ceo(self):
        assert classify_entity_type("CEO of Tesla, Inc.") == "person"

    # Organization
    def test_spacex(self):
        assert classify_entity_type(
            "American aerospace manufacturer and space transportation company"
        ) == "organization"

    def test_company(self):
        assert classify_entity_type("an American electric vehicle company") == "organization"

    def test_agency(self):
        assert classify_entity_type("the national space agency of the United States") == "organization"

    def test_institute(self):
        assert classify_entity_type("a research institute founded in 1999") == "organization"

    # Vehicle
    def test_falcon_9(self):
        assert classify_entity_type(
            "a medium-lift orbital launch vehicle developed by SpaceX"
        ) == "vehicle"

    def test_tesla_semi(self):
        assert classify_entity_type("an all-electric semi truck made by Tesla") == "vehicle"

    def test_spacecraft(self):
        assert classify_entity_type("a reusable spacecraft designed for deep space") == "vehicle"

    def test_satellite(self):
        assert classify_entity_type("a low Earth orbit satellite") == "vehicle"

    # Program
    def test_project_mercury(self):
        assert classify_entity_type(
            "the first United States human spaceflight program"
        ) == "program"

    def test_space_mission(self):
        assert classify_entity_type("a robotic space mission to Mars") == "program"

    def test_initiative(self):
        assert classify_entity_type("a government initiative launched in 2010") == "program"

    # Place
    def test_city(self):
        assert classify_entity_type("a city in Texas, United States") == "place"

    def test_country(self):
        assert classify_entity_type("a country in Western Europe") == "place"

    def test_launch_site(self):
        assert classify_entity_type("a launch site on the Texas Gulf Coast") == "place"

    # Concept
    def test_technology(self):
        # "spacecraft" would match vehicle first; use a definition without vehicle keywords
        assert classify_entity_type("a propulsion technology used in industry") == "technology"

    def test_system(self):
        # "satellite" would match vehicle; use a definition without vehicle keywords
        assert classify_entity_type("a distributed computing system and framework") == "technology"

    def test_publication(self):
        assert classify_entity_type("an American business magazine") == "publication"

    def test_product(self):
        assert classify_entity_type("a consumer product made by Tesla") == "product"

    def test_service(self):
        assert classify_entity_type("a satellite internet constellation operated by SpaceX") == "service"

    def test_platform_service(self):
        assert classify_entity_type("a subscription platform with broadband access") == "service"

    def test_cryptocurrency(self):
        assert classify_entity_type("a decentralized cryptocurrency") == "concept"

    # Other
    def test_empty_string(self):
        assert classify_entity_type("") == "other"

    def test_unknown_text(self):
        assert classify_entity_type("xkjhqwer zqmno plkrt") == "other"

    def test_whitespace_only(self):
        assert classify_entity_type("   ") == "other"

    # Priority: person wins over organization even if both keywords present
    def test_person_beats_organization(self):
        assert classify_entity_type("founder and businessman who built a company") == "person"

    # Priority: vehicle beats concept
    def test_vehicle_beats_concept(self):
        assert classify_entity_type("a rocket propulsion system") == "vehicle"


# ---------------------------------------------------------------------------
# Demo cases from spec
# ---------------------------------------------------------------------------

class TestSpecDemoCases:
    def test_elon_musk_person(self):
        assert classify_entity_type(
            "Elon Musk is a businessman and entrepreneur"
        ) == "person"

    def test_spacex_organization(self):
        assert classify_entity_type(
            "SpaceX is an American aerospace manufacturer and space transportation company"
        ) == "organization"

    def test_falcon_9_vehicle(self):
        assert classify_entity_type(
            "Falcon 9 is a reusable medium-lift orbital launch vehicle"
        ) == "vehicle"

    def test_project_mercury_program(self):
        assert classify_entity_type(
            "Project Mercury was the first American spaceflight program"
        ) == "program"

    def test_tesla_semi_vehicle(self):
        assert classify_entity_type(
            "The Tesla Semi is an all-electric semi truck"
        ) == "vehicle"

    def test_robert_zubrin_person(self):
        assert classify_entity_type(
            "Robert Zubrin is an aerospace engineer and author"
        ) == "person"


# ---------------------------------------------------------------------------
# infer_entity_type — overlay lookup
# ---------------------------------------------------------------------------

class TestInferEntityType:
    def test_overlay_entity_explicit_type(self):
        items = [
            {
                "overlay_type": "overlay_entity",
                "label": "SpaceX",
                "entity_type": "organization",
            }
        ]
        assert infer_entity_type("SpaceX", items) == "organization"

    def test_overlay_definition_lookup(self):
        items = [
            {
                "overlay_type": "overlay_definition",
                "subject": "SpaceX",
                "definition": "an American aerospace manufacturer and launch company",
            }
        ]
        assert infer_entity_type("SpaceX", items) == "organization"

    def test_overlay_relation_is_a(self):
        items = [
            {
                "overlay_type": "overlay_relation",
                "subject": "Falcon 9",
                "predicate": "is_a",
                "object": "medium-lift orbital launch vehicle",
            }
        ]
        assert infer_entity_type("Falcon 9", items) == "vehicle"

    def test_overlay_relation_type_of(self):
        items = [
            {
                "overlay_type": "overlay_relation",
                "subject": "Project Artemis",
                "predicate": "type_of",
                "object": "NASA spaceflight program",
            }
        ]
        assert infer_entity_type("Project Artemis", items) == "program"

    def test_case_insensitive_lookup(self):
        items = [
            {
                "overlay_type": "overlay_definition",
                "subject": "spacex",
                "definition": "an American aerospace company",
            }
        ]
        assert infer_entity_type("SpaceX", items) == "organization"

    def test_empty_overlay_returns_other(self):
        assert infer_entity_type("UnknownEntity", []) == "other"

    def test_entity_type_explicit_takes_priority_over_definition(self):
        items = [
            {
                "overlay_type": "overlay_entity",
                "label": "Elon Musk",
                "entity_type": "person",
            },
            {
                "overlay_type": "overlay_definition",
                "subject": "Elon Musk",
                "definition": "CEO of Tesla",
            },
        ]
        assert infer_entity_type("Elon Musk", items) == "person"

    def test_no_matching_subject_returns_other(self):
        items = [
            {
                "overlay_type": "overlay_definition",
                "subject": "Tesla",
                "definition": "an electric vehicle company",
            }
        ]
        assert infer_entity_type("SpaceX", items) == "other"

    def test_empty_definition_skipped(self):
        items = [
            {
                "overlay_type": "overlay_definition",
                "subject": "SpaceX",
                "definition": "",
            },
            {
                "overlay_type": "overlay_relation",
                "subject": "SpaceX",
                "predicate": "is_a",
                "object": "aerospace company",
            },
        ]
        assert infer_entity_type("SpaceX", items) == "organization"

    def test_empty_entity_type_field_skipped(self):
        items = [
            {
                "overlay_type": "overlay_entity",
                "label": "SpaceX",
                "entity_type": "",
            },
            {
                "overlay_type": "overlay_definition",
                "subject": "SpaceX",
                "definition": "an aerospace manufacturer",
            },
        ]
        assert infer_entity_type("SpaceX", items) == "organization"
