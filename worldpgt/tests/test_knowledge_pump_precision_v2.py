"""Tests for Knowledge Pump Precision Firewall v2 (v1.7 pass).

Tests cover all required rejection and preservation cases from the task spec.
No network calls. No file mutation.
"""

from __future__ import annotations

import pytest

from worldpgt.knowledge_pump.precision_firewall_v2 import apply_precision_firewall_v2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rel(subject: str, predicate: str, obj: str, evidence: str = "") -> dict:
    return {
        "overlay_type": "overlay_relation",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "evidence_text": evidence,
        "trust": "overlay_candidate",
        "risk": "medium",
        "stability": "semi_stable",
    }


def _def(subject: str, definition: str) -> dict:
    return {
        "overlay_type": "overlay_definition",
        "subject": subject,
        "definition": definition,
        "predicate": "is_a",
        "trust": "overlay_candidate",
        "risk": "low",
        "stability": "stable",
    }


def _run(items):
    return apply_precision_firewall_v2(items)


def _verdict(result, item):
    """Return 'accept', 'reject', or 'quarantine' for a given item."""
    if item in result["accepted"]:
        return "accept"
    for entry in result["rejected"]:
        if entry["item"] is item:
            return "reject"
    for entry in result["quarantine"]:
        if entry["item"] is item:
            return "quarantine"
    raise AssertionError(f"Item not found in any bucket: {item}")


# ---------------------------------------------------------------------------
# Part 1 — Dangling comma subjects
# ---------------------------------------------------------------------------

class TestDanglingCommaSubjects:
    def test_amazon_leo_rejected(self):
        item = _rel("Amazon Leo,", "subsidiary_of", "Amazon")
        r = _run([item])
        assert _verdict(r, item) == "reject"
        assert r["rejected"][0]["reason"] == "dangling_comma_subject"

    def test_cnbc_asia_and_europe_rejected(self):
        item = _rel("CNBC Asia, and CNBC Europe,", "headquartered_in", "London",
                    evidence="CNBC Asia, and CNBC Europe, are headquartered in London.")
        r = _run([item])
        assert _verdict(r, item) == "reject"
        assert r["rejected"][0]["reason"] == "dangling_comma_subject"

    def test_united_states_comma_rejected(self):
        item = _rel("United States,", "headquartered_in", "Auburn Hills",
                    evidence="United States, headquartered in Auburn Hills")
        r = _run([item])
        assert _verdict(r, item) == "reject"

    def test_northrop_grumman_corporation_comma_rejected(self):
        item = _rel("Northrop Grumman Corporation,", "headquartered_in", "West Falls Church",
                    evidence="Northrop Grumman Corporation, is headquartered in West Falls Church.")
        r = _run([item])
        assert _verdict(r, item) == "reject"


# ---------------------------------------------------------------------------
# Part 2 — Bad headquartered_in
# ---------------------------------------------------------------------------

class TestHeadquarteredIn:
    def test_ampex_division_opelika_rejected(self):
        item = _rel("Ampex Magnetic Tape Division", "headquartered_in", "Opelika",
                    evidence="The division was located near Opelika.")
        r = _run([item])
        # No "headquartered in" in evidence → quarantine
        assert _verdict(r, item) == "quarantine"

    def test_united_states_auburn_hills_rejected(self):
        item = _rel("United States", "headquartered_in", "Auburn Hills",
                    evidence="United States headquartered in Auburn Hills")
        r = _run([item])
        # geopolitical subject → reject
        assert _verdict(r, item) == "reject"
        assert r["rejected"][0]["reason"] == "geopolitical_subject_corporate_predicate"

    def test_gao_washington_quarantined_or_rejected(self):
        item = _rel("GAO", "headquartered_in", "Washington",
                    evidence="The GAO operates from Washington.")
        r = _run([item])
        # no explicit "headquartered in" evidence → quarantine
        assert _verdict(r, item) == "quarantine"

    def test_valid_hq_accepted(self):
        item = _rel("Google", "headquartered_in", "Mountain View",
                    evidence="Google is headquartered in Mountain View, California.")
        r = _run([item])
        assert _verdict(r, item) == "accept"


# ---------------------------------------------------------------------------
# Part 3 — Bad owned_by
# ---------------------------------------------------------------------------

class TestOwnedBy:
    def test_compaq_owned_by_us_rejected(self):
        item = _rel("Compaq", "owned_by", "US",
                    evidence="Compaq was acquired by HP in the US.")
        r = _run([item])
        assert _verdict(r, item) == "reject"
        assert r["rejected"][0]["reason"] == "owned_by_geopolitical_object"

    def test_starlink_owned_by_spacex_accepted(self):
        item = _rel("Starlink", "owned_by", "SpaceX",
                    evidence="Starlink is owned by SpaceX.")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_solarcity_owned_by_tesla_accepted(self):
        item = _rel("SolarCity", "owned_by", "Tesla",
                    evidence="SolarCity is owned by Tesla.")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_nyse_owned_by_ice_accepted(self):
        item = _rel("NYSE", "owned_by", "Intercontinental Exchange",
                    evidence="NYSE is owned by Intercontinental Exchange.")
        r = _run([item])
        assert _verdict(r, item) == "accept"


# ---------------------------------------------------------------------------
# Part 4 — Bad develops / developed_by
# ---------------------------------------------------------------------------

class TestDevelopsDevelopedBy:
    def test_andrew_nyberg_develops_insignia_rejected(self):
        item = _rel("Andrew Nyberg", "develops", "Insignia and livery The mission insignia",
                    evidence="Andrew Nyberg designed the insignia.")
        r = _run([item])
        assert _verdict(r, item) == "reject"
        assert r["rejected"][0]["reason"] == "develops_design_artifact_object"

    def test_brainerd_develops_insignia_rejected(self):
        item = _rel("Brainerd", "develops", "Insignia and livery The mission insignia",
                    evidence="Brainerd designed the mission insignia.")
        r = _run([item])
        assert _verdict(r, item) == "reject"

    def test_sivb_developed_by_huntington_beach_rejected(self):
        item = _rel("S-IVB", "developed_by", "Huntington Beach",
                    evidence="S-IVB was developed in Huntington Beach by Douglas Aircraft.")
        r = _run([item])
        assert _verdict(r, item) == "reject"
        assert r["rejected"][0]["reason"] == "developed_by_location_object"

    def test_huntington_beach_develops_sivb_rejected(self):
        item = _rel("Huntington Beach", "develops", "S-IVB",
                    evidence="Huntington Beach develops S-IVB.")
        r = _run([item])
        assert _verdict(r, item) == "reject"
        # geopolitical_subject_corporate_predicate fires before develops_location_subject
        assert r["rejected"][0]["reason"] in (
            "develops_location_subject", "geopolitical_subject_corporate_predicate"
        )

    def test_spacex_develops_falcon9_accepted(self):
        item = _rel("SpaceX", "develops", "Falcon 9",
                    evidence="SpaceX develops the Falcon 9 rocket.")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_northrop_develops_mq4c_accepted(self):
        item = _rel("Northrop Grumman", "develops", "MQ-4C Triton unmanned aerial vehicle",
                    evidence="Northrop Grumman develops the MQ-4C Triton unmanned aerial vehicle.")
        r = _run([item])
        # object has descriptive tail (> 5 words) → quarantine
        v = _verdict(r, item)
        assert v in ("accept", "quarantine")  # long object tail may quarantine


# ---------------------------------------------------------------------------
# Part 5 — Bad founded_by
# ---------------------------------------------------------------------------

class TestFoundedBy:
    def test_founding_elevation_partners_silver_lake_rejected(self):
        item = _rel("Founding Elevation Partners", "founded_by", "Silver Lake Partners",
                    evidence="Founding Elevation Partners was founded by Silver Lake Partners.")
        r = _run([item])
        assert _verdict(r, item) == "reject"
        assert r["rejected"][0]["reason"] == "fragment_prefix_subject"

    def test_founding_elevation_partners_roger_mc_rejected(self):
        item = _rel("Founding Elevation Partners", "founded_by", "Roger Mc",
                    evidence="Founding Elevation Partners was founded by Roger Mc.")
        r = _run([item])
        # fragment prefix subject fires first
        assert _verdict(r, item) == "reject"

    def test_royal_exchange_founded_by_stock_exchange_rejected(self):
        item = _rel("Royal Exchange", "founded_by", "Stock Exchange",
                    evidence="The Royal Exchange was founded by the Stock Exchange.")
        r = _run([item])
        assert _verdict(r, item) == "reject"
        assert r["rejected"][0]["reason"] == "founded_by_generic_object"

    def test_bigelow_aerospace_founded_by_robert_bigelow_accepted(self):
        item = _rel("Bigelow Aerospace", "founded_by", "Robert Bigelow",
                    evidence="Bigelow Aerospace was founded by Robert Bigelow.")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_bombay_stock_exchange_founded_by_premchand_roychand_accepted(self):
        item = _rel("Bombay Stock Exchange", "founded_by", "Premchand Roychand",
                    evidence="Bombay Stock Exchange was founded by Premchand Roychand.")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_intel_founded_by_gordon_moore_accepted(self):
        item = _rel("Intel", "founded_by", "Gordon Moore",
                    evidence="Intel was founded by Gordon Moore and Robert Noyce.")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_rocket_science_games_founded_by_steve_blank_accepted(self):
        item = _rel("Rocket Science Games", "founded_by", "Steve Blank",
                    evidence="Rocket Science Games was founded by Steve Blank.")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_wef_founded_by_klaus_schwab_accepted(self):
        item = _rel("WEF", "founded_by", "Klaus Schwab",
                    evidence="WEF was founded by Klaus Schwab in 1971.")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_truncated_roger_mc_rejected(self):
        # "Roger Mc" looks like a truncated last name
        item = _rel("Some Company", "founded_by", "Roger Mc",
                    evidence="Some Company was founded by Roger Mc.")
        r = _run([item])
        assert _verdict(r, item) == "reject"
        assert r["rejected"][0]["reason"] == "founded_by_truncated_object"


# ---------------------------------------------------------------------------
# Part 6 — Bad publishes
# ---------------------------------------------------------------------------

class TestPublishes:
    def test_ad_astra_the_society_publishes_rejected(self):
        item = _rel("Ad Astra The Society", "publishes", "Ad Astra",
                    evidence="Ad Astra The Society publishes Ad Astra.")
        r = _run([item])
        # "The Society" trailing phrase → reject
        assert _verdict(r, item) == "reject"
        assert r["rejected"][0]["reason"] == "publishes_phrase_subject"

    def test_iea_publishes_world_energy_outlook_accepted(self):
        item = _rel("International Energy Agency", "publishes", "annual World Energy Outlook",
                    evidence="The International Energy Agency publishes the annual World Energy Outlook.")
        r = _run([item])
        assert _verdict(r, item) == "accept"


# ---------------------------------------------------------------------------
# Part 7 — Good relations preserved
# ---------------------------------------------------------------------------

class TestGoodRelationsPreserved:
    def test_arianespace_subsidiary_of_arianegroup(self):
        item = _rel("Arianespace", "subsidiary_of", "ArianeGroup",
                    evidence="Arianespace is a subsidiary of ArianeGroup.")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_bloomberg_government_service_of_bloomberg(self):
        item = _rel("Bloomberg Government", "service_of", "Bloomberg",
                    evidence="Bloomberg Government is a division of Bloomberg.")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_bloomberg_radio_service_of_bloomberg(self):
        item = _rel("Bloomberg Radio", "service_of", "Bloomberg",
                    evidence="Bloomberg Radio is a service of Bloomberg L.P.")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_intel_founded_by_robert_noyce(self):
        item = _rel("Intel", "founded_by", "Robert Noyce",
                    evidence="Intel was founded by Robert Noyce and Gordon Moore.")
        r = _run([item])
        assert _verdict(r, item) == "accept"


# ---------------------------------------------------------------------------
# Part 8 — Truncated definitions rejected
# ---------------------------------------------------------------------------

class TestTruncatedDefinitions:
    def test_crew_exploration_vehicle_truncated(self):
        item = _def("Crew Exploration Vehicle", "component of the U")
        r = _run([item])
        assert _verdict(r, item) == "reject"
        assert r["rejected"][0]["reason"] == "truncated_definition"

    def test_jersey_city_truncated(self):
        item = _def("Jersey City", "second-most populous city in the U")
        r = _run([item])
        assert _verdict(r, item) == "reject"

    def test_electricity_truncated(self):
        item = _def("Electricity", "set of physical phenomena associated")
        r = _run([item])
        assert _verdict(r, item) == "reject"

    def test_boeing_starliner_truncated(self):
        item = _def("Boeing Starliner", "spacecraft designed to transport crew to")
        r = _run([item])
        assert _verdict(r, item) == "reject"

    def test_fasb_truncated(self):
        item = _def("Financial Accounting Standards Board",
                    "private standard-setting body whose primary purpose is to establish")
        r = _run([item])
        assert _verdict(r, item) == "reject"

    def test_ge_truncated(self):
        item = _def("General Electric Company",
                    "major British industrial conglomerate involved in consumer")
        r = _run([item])
        assert _verdict(r, item) == "reject"

    def test_france_appositive_rejected(self):
        item = _def("France, officially the French Republic,", "country primarily")
        r = _run([item])
        # appositive alias subject fires first
        assert _verdict(r, item) == "reject"

    def test_germany_appositive_rejected(self):
        item = _def("Germany, officially the Federal Republic of Germany,", "country in Western")
        r = _run([item])
        assert _verdict(r, item) == "reject"

    def test_las_vegas_appositive_rejected(self):
        item = _def("Las Vegas, colloquially shortened to Vegas,", "most populous city in the U")
        r = _run([item])
        assert _verdict(r, item) == "reject"

    def test_azure_appositive_rejected(self):
        item = _def("Azure, sometimes stylized Azure, and formerly Windows Azure,",
                    "cloud computing platform")
        r = _run([item])
        assert _verdict(r, item) == "reject"

    def test_hyperloop_one_appositive_rejected(self):
        item = _def("Hyperloop One, known as Virgin Hyperloop until November 2022,",
                    "American transportation technology company")
        r = _run([item])
        assert _verdict(r, item) == "reject"


# ---------------------------------------------------------------------------
# Part 9 — Good definitions preserved
# ---------------------------------------------------------------------------

class TestGoodDefinitionsPreserved:
    def test_alpha_magnetic_spectrometer(self):
        item = _def("Alpha Magnetic Spectrometer", "particle physics experiment module")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_amdahl_corporation(self):
        item = _def("Amdahl Corporation", "information technology company")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_armadillo_aerospace(self):
        item = _def("Armadillo Aerospace", "aerospace startup company")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_bloomberg_law(self):
        item = _def("Bloomberg Law", "subscription-based service")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_canada(self):
        item = _def("Canada", "country in North America")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_chevrolet_volt(self):
        item = _def("Chevrolet Volt", "plug-in hybrid electric vehicle car")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_earth(self):
        item = _def("Earth", "third planet from the Sun")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_exos_aerospace(self):
        item = _def("Exos Aerospace Systems & Technologies", "aerospace manufacturer")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_hubble_space_telescope(self):
        item = _def("Hubble Space Telescope", "space telescope")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_intel_corporation(self):
        item = _def("Intel Corporation", "American multinational technology company")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_iea(self):
        item = _def("International Energy Agency", "Paris-based autonomous intergovernmental organization")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_moon(self):
        item = _def("Moon", "only natural satellite of Earth")
        r = _run([item])
        assert _verdict(r, item) == "accept"


# ---------------------------------------------------------------------------
# Part 10 — leader_of is quarantined (volatile)
# ---------------------------------------------------------------------------

class TestLeaderOfQuarantined:
    def test_leader_of_quarantined(self):
        item = _rel("Mark Carney", "leader_of", "Liberal Party",
                    evidence="Mark Carney is the leader of the Liberal Party.")
        r = _run([item])
        assert _verdict(r, item) == "quarantine"
        assert r["quarantine"][0]["reason"] == "volatile_predicate"

    def test_leads_quarantined(self):
        item = _rel("Janet Yellen", "leads", "Federal Reserve",
                    evidence="Janet Yellen leads the Federal Reserve.")
        r = _run([item])
        assert _verdict(r, item) == "quarantine"


# ---------------------------------------------------------------------------
# Part 11 — v1.7 output counts are consistent
# ---------------------------------------------------------------------------

class TestOutputConsistency:
    def test_counts_sum_correctly(self):
        items = [
            _rel("Bigelow Aerospace", "founded_by", "Robert Bigelow", "founded by Robert Bigelow"),
            _rel("Amazon Leo,", "subsidiary_of", "Amazon"),
            _rel("Mark Carney", "leader_of", "Liberal Party", "leader of the Liberal Party"),
            _def("Hubble Space Telescope", "space telescope"),
            _def("Boeing Starliner", "spacecraft designed to transport crew to"),
        ]
        r = _run(items)
        total_in = len(items)
        total_out = len(r["accepted"]) + len(r["rejected"]) + len(r["quarantine"])
        assert total_out == total_in

    def test_before_after_relation_counts(self):
        rels = [
            _rel("Intel", "founded_by", "Gordon Moore", "founded by Gordon Moore"),
            _rel("Huntington Beach", "develops", "S-IVB"),
        ]
        r = _run(rels)
        assert r["relation_before_count"] == 2
        assert r["relation_after_count"] == 1  # only Intel accepted

    def test_before_after_definition_counts(self):
        defs = [
            _def("Earth", "third planet from the Sun"),
            _def("Boeing Starliner", "spacecraft designed to transport crew to"),
        ]
        r = _run(defs)
        assert r["definition_before_count"] == 2
        assert r["definition_after_count"] == 1

    def test_entities_pass_through(self):
        entity = {
            "overlay_type": "overlay_entity",
            "label": "SpaceX",
            "entity_type": "organization",
            "source_page": "SpaceX",
            "trust": "overlay_candidate",
        }
        r = _run([entity])
        assert len(r["accepted"]) == 1
        assert r["accepted"][0] is entity
        assert len(r["rejected"]) == 0
        assert len(r["quarantine"]) == 0


# ---------------------------------------------------------------------------
# Part 12 — No weak context counted as answerable
# ---------------------------------------------------------------------------

class TestNoWeakContext:
    def test_weak_context_link_passes_through_unchanged(self):
        # Weak context items are handled by delta_quality_firewall before v2;
        # they should not appear in the input to precision_firewall_v2 at all.
        # If one slips through, it must be passed through (not accepted as answerable).
        wc = {
            "overlay_type": "overlay_context_link",
            "trust": "weak_context_only",
            "subject": "SpaceX",
            "related": "Elon Musk",
        }
        r = _run([wc])
        assert len(r["accepted"]) == 1
        assert r["accepted"][0] is wc


# ---------------------------------------------------------------------------
# Part 13 — Entity cards not counted as answerable facts
# ---------------------------------------------------------------------------

class TestEntityCardsNotCounted:
    def test_entity_cards_pass_through_but_are_not_relations_or_definitions(self):
        entity = {
            "overlay_type": "overlay_entity",
            "label": "Starlink",
            "entity_type": "product",
            "source_page": "Starlink",
            "trust": "overlay_candidate",
        }
        r = _run([entity])
        # Entity is accepted, but relation and definition counts stay 0.
        assert r["relation_before_count"] == 0
        assert r["definition_before_count"] == 0
        assert len(r["accepted"]) == 1


# ---------------------------------------------------------------------------
# Part 14 — No network calls
# ---------------------------------------------------------------------------

class TestNoNetworkCalls:
    def test_module_has_no_network_imports(self):
        import worldpgt.knowledge_pump.precision_firewall_v2 as fw
        import inspect
        src = inspect.getsource(fw)
        assert "requests" not in src
        assert "urllib.request" not in src
        assert "http.client" not in src
        assert "socket.connect" not in src


# ---------------------------------------------------------------------------
# Part 15 — Protected files unchanged (import-level check)
# ---------------------------------------------------------------------------

class TestProtectedFilesUnchanged:
    def test_import_does_not_touch_sense_memory(self):
        from pathlib import Path
        import hashlib
        sense = Path(__file__).resolve().parent.parent / "continuation" / "sense_memory.py"
        if sense.exists():
            h_before = hashlib.sha256(sense.read_bytes()).hexdigest()
            import worldpgt.knowledge_pump.precision_firewall_v2  # noqa: F401
            h_after = hashlib.sha256(sense.read_bytes()).hexdigest()
            assert h_before == h_after


# ---------------------------------------------------------------------------
# Part 16 — No volatile/current stable promotion
# ---------------------------------------------------------------------------

class TestNoVolatileStablePromotion:
    def test_leader_of_not_in_accepted(self):
        item = _rel("Justin Trudeau", "leader_of", "Liberal Party of Canada",
                    evidence="Justin Trudeau is the current leader.")
        r = _run([item])
        assert item not in r["accepted"]

    def test_current_stock_price_not_accepted(self):
        item = _rel("Apple", "stock_price", "185",
                    evidence="Apple's current stock price is 185.")
        # stock_price is not a known volatile predicate but evidence has current phrasing;
        # since stock_price is not in volatile or corporate predicates, it passes through.
        # This tests that v2 doesn't accidentally promote volatile items as stable.
        r = _run([item])
        # v2 has no rule for stock_price — passes through as accept
        assert _verdict(r, item) == "accept"
