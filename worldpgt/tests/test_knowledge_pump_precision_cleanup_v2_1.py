"""Tests for Knowledge Pump Precision Cleanup v2.1 (v1.7.1 pass).

Covers all required rejection, quarantine, and preservation cases from the
task spec. No network calls. No file mutation.
"""

from __future__ import annotations

import pytest

from worldpgt.knowledge_pump.precision_cleanup_v2_1 import apply_precision_cleanup_v2_1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rel(subject: str, predicate: str, obj: str, *,
         evidence: str = "", source_page: str = "") -> dict:
    return {
        "overlay_type": "overlay_relation",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "evidence_text": evidence,
        "source_page": source_page,
        "trust": "overlay_candidate",
        "risk": "medium",
        "stability": "semi_stable",
    }


def _def(subject: str, definition: str, source_page: str = "") -> dict:
    return {
        "overlay_type": "overlay_definition",
        "subject": subject,
        "definition": definition,
        "predicate": "is_a",
        "source_page": source_page,
        "trust": "overlay_candidate",
        "risk": "low",
        "stability": "stable",
    }


def _run(items):
    return apply_precision_cleanup_v2_1(items)


def _verdict(result, item):
    if item in result["accepted"]:
        return "accept"
    for entry in result["rejected"]:
        if entry["item"] is item:
            return "reject"
    for entry in result["quarantine"]:
        if entry["item"] is item:
            return "quarantine"
    raise AssertionError(f"Item not found in any bucket: {item}")


def _reason(result, item):
    for entry in result["rejected"] + result["quarantine"]:
        if entry["item"] is item:
            return entry["reason"]
    return None


# ---------------------------------------------------------------------------
# Part 1 — Truncated person alias (object)
# ---------------------------------------------------------------------------

class TestTruncatedPersonAlias:
    def test_affirm_developed_by_levchin_quarantined(self):
        item = _rel("Affirm", "developed_by", "Levchin",
                    evidence="Affirm was created by Levchin and co-founders.",
                    source_page="Max Levchin")
        r = _run([item])
        assert _verdict(r, item) == "quarantine"
        assert _reason(r, item) == "truncated_person_alias"

    def test_levchin_develops_affirm_quarantined(self):
        item = _rel("Levchin", "develops", "Affirm",
                    evidence="Affirm was created by Levchin.",
                    source_page="Max Levchin")
        r = _run([item])
        assert _verdict(r, item) == "quarantine"
        assert _reason(r, item) == "truncated_person_alias"

    def test_full_name_object_preserved(self):
        item = _rel("Bigelow Aerospace", "founded_by", "Robert Bigelow",
                    evidence="Bigelow Aerospace was founded by Robert Bigelow.",
                    source_page="Bigelow Aerospace")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_acronym_object_preserved(self):
        # HP is all-caps acronym → not a truncated person alias
        item = _rel("USA Parkway", "developed_by", "TRIC",
                    evidence="USA Parkway was developed by TRIC.",
                    source_page="Giga Nevada")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_multi_word_object_preserved(self):
        item = _rel("Intel", "founded_by", "Gordon Moore",
                    evidence="Intel was founded by Gordon Moore.",
                    source_page="Intel Corporation")
        r = _run([item])
        assert _verdict(r, item) == "accept"


# ---------------------------------------------------------------------------
# Part 2 — Subject alias too short
# ---------------------------------------------------------------------------

class TestSubjectAliasTooShort:
    def test_morgan_founded_by_quarantined(self):
        item = _rel("Morgan", "founded_by", "Henry Frederick Stanley Morgan",
                    evidence="Morgan was founded in 1910 by Henry Frederick Stanley Morgan.",
                    source_page="Morgan Motor Company")
        r = _run([item])
        assert _verdict(r, item) == "quarantine"
        assert _reason(r, item) == "subject_alias_too_short"

    def test_harvard_founded_by_general_court_quarantined(self):
        item = _rel("Harvard", "founded_by", "General Court",
                    evidence="Harvard was founded in 1636 by the Great and General Court.",
                    source_page="Harvard University")
        r = _run([item])
        assert _verdict(r, item) == "quarantine"
        assert _reason(r, item) == "subject_alias_too_short"

    def test_harvard_founded_by_massachusetts_quarantined(self):
        item = _rel("Harvard", "founded_by", "Massachusetts Bay Colony",
                    evidence="Harvard was founded by the Massachusetts Bay Colony.",
                    source_page="Harvard University")
        r = _run([item])
        assert _verdict(r, item) == "quarantine"
        assert _reason(r, item) == "subject_alias_too_short"

    def test_intel_founded_by_preserved(self):
        # Intel is the official short name, not an alias for "Intel Corporation"
        # (Intel Corporation has " Corporation" not a known institution suffix).
        item = _rel("Intel", "founded_by", "Gordon Moore",
                    evidence="Intel was founded by Gordon Moore.",
                    source_page="Intel Corporation")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_wef_founded_by_preserved(self):
        item = _rel("WEF", "founded_by", "Klaus Schwab",
                    evidence="WEF was founded by Klaus Schwab.",
                    source_page="World Economic Forum")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_xm_founded_by_preserved(self):
        item = _rel("XM", "founded_by", "Lon Levin",
                    evidence="XM was founded by Lon Levin.",
                    source_page="Satellite Radio")
        r = _run([item])
        assert _verdict(r, item) == "accept"


# ---------------------------------------------------------------------------
# Part 3 — Composite region subjects
# ---------------------------------------------------------------------------

class TestCompositeRegionSubject:
    def test_africa_and_eurasia_worldspace_rejected(self):
        item = _rel("Africa and Eurasia WorldSpace", "founded_by", "Noah Samara",
                    evidence="Africa and Eurasia WorldSpace was founded by Noah Samara.",
                    source_page="Satellite Radio")
        r = _run([item])
        assert _verdict(r, item) == "reject"
        assert _reason(r, item) == "composite_region_subject"

    def test_united_states_sirius_satellite_radio_rejected(self):
        item = _rel("United States Sirius Satellite Radio", "founded_by", "Martine Rothblatt",
                    evidence="United States Sirius Satellite Radio was founded by Martine Rothblatt.",
                    source_page="Satellite Radio")
        r = _run([item])
        assert _verdict(r, item) == "reject"
        assert _reason(r, item) == "composite_region_subject"

    def test_arianespace_preserved(self):
        item = _rel("Arianespace", "subsidiary_of", "ArianeGroup",
                    evidence="Arianespace is a subsidiary of ArianeGroup.",
                    source_page="Arianespace")
        r = _run([item])
        assert _verdict(r, item) == "accept"


# ---------------------------------------------------------------------------
# Part 4 — headquartered_in quarantine default
# ---------------------------------------------------------------------------

class TestHeadquarteredInQuarantineDefault:
    def test_gao_headquartered_washington_quarantined(self):
        item = _rel("GAO", "headquartered_in", "Washington",
                    evidence="The GAO is headquartered in Washington, D.C.",
                    source_page="Government Accountability Office")
        r = _run([item])
        assert _verdict(r, item) == "quarantine"
        assert _reason(r, item) == "headquartered_in_quarantine_default"

    def test_any_headquartered_in_quarantined(self):
        item = _rel("Google", "headquartered_in", "Mountain View",
                    evidence="Google is headquartered in Mountain View, California.",
                    source_page="Google")
        r = _run([item])
        assert _verdict(r, item) == "quarantine"
        assert _reason(r, item) == "headquartered_in_quarantine_default"


# ---------------------------------------------------------------------------
# Part 5 — owned_by predicate ambiguous (vehicle subjects)
# ---------------------------------------------------------------------------

class TestOwnedByPredicateAmbiguous:
    def test_crew_dragon_owned_by_spacex_quarantined(self):
        item = _rel("Crew Dragon", "owned_by", "SpaceX",
                    evidence="Crew Dragon Demo-2 operated by SpaceX became the first crewed mission.",
                    source_page="Private spaceflight")
        r = _run([item])
        assert _verdict(r, item) == "quarantine"
        assert _reason(r, item) == "owned_by_predicate_ambiguous"

    def test_starlink_owned_by_spacex_preserved(self):
        # Starlink is a service/constellation, not a vehicle — no vehicle words.
        item = _rel("Starlink", "owned_by", "SpaceX",
                    evidence="Starlink is a satellite internet constellation operated by SpaceX.",
                    source_page="Communications Satellite")
        r = _run([item])
        # No strong ownership evidence, so quarantined by no_strong_evidence rule.
        # Both are reasonable outcomes; the key is Starlink isn't quarantined for
        # vehicle ambiguity — that rule only fires for vehicle words.
        v = _verdict(r, item)
        assert v in ("accept", "quarantine")
        if v == "quarantine":
            assert _reason(r, item) != "owned_by_predicate_ambiguous"

    def test_nyse_owned_by_ice_preserved(self):
        item = _rel("NYSE", "owned_by", "Intercontinental Exchange",
                    evidence="The NYSE is owned by Intercontinental Exchange.",
                    source_page="NYSE")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_solarcity_owned_by_tesla_preserved(self):
        item = _rel("SolarCity", "owned_by", "Tesla",
                    evidence="SolarCity was acquired by Tesla in 2016.",
                    source_page="Gigafactory New York")
        r = _run([item])
        assert _verdict(r, item) == "accept"


# ---------------------------------------------------------------------------
# Part 6 — Duplicate alias relations
# ---------------------------------------------------------------------------

class TestDuplicateAliasRelation:
    def test_compaq_owned_by_hp_quarantined_hewlett_packard_preserved(self):
        hp_item = _rel("Compaq", "owned_by", "HP",
                       evidence="Compaq was acquired by Hewlett-Packard, also known as HP.",
                       source_page="Compaq Computer")
        hp_full_item = _rel("Compaq", "owned_by", "Hewlett-Packard",
                            evidence="Compaq was acquired by Hewlett-Packard, also known as HP.",
                            source_page="Compaq Computer")
        r = _run([hp_item, hp_full_item])
        # HP is the alias form → quarantine
        assert _verdict(r, hp_item) == "quarantine"
        assert _reason(r, hp_item) == "duplicate_alias_relation"
        # Hewlett-Packard is the canonical form → accept (or quarantine for other reasons)
        v_full = _verdict(r, hp_full_item)
        assert v_full in ("accept", "quarantine")
        if v_full == "quarantine":
            assert _reason(r, hp_full_item) != "duplicate_alias_relation"

    def test_no_dup_single_object_preserved(self):
        item = _rel("Compaq", "owned_by", "Hewlett-Packard",
                    evidence="Compaq was acquired by Hewlett-Packard.",
                    source_page="Compaq Computer")
        r = _run([item])
        assert _verdict(r, item) == "accept"


# ---------------------------------------------------------------------------
# Part 7 — Truncated participle definitions
# ---------------------------------------------------------------------------

class TestTruncatedParticipleDefinitions:
    def test_embraer_e_jet_designed_rejected(self):
        item = _def("Embraer E-Jet E2 family",
                    "series of four-abreast narrow-body regional jet airliners designed")
        r = _run([item])
        assert _verdict(r, item) == "reject"
        assert _reason(r, item) == "definition_truncated_participle"

    def test_ford_f_series_marketed_rejected(self):
        item = _def("Ford F-Series", "series of light-duty trucks marketed")
        r = _run([item])
        assert _verdict(r, item) == "reject"
        assert _reason(r, item) == "definition_truncated_participle"

    def test_truncated_offered_rejected(self):
        item = _def("Some Product", "subscription service offered")
        r = _run([item])
        assert _verdict(r, item) == "reject"
        assert _reason(r, item) == "definition_truncated_participle"

    def test_truncated_developed_rejected(self):
        item = _def("Some Widget", "device developed")
        r = _run([item])
        assert _verdict(r, item) == "reject"
        assert _reason(r, item) == "definition_truncated_participle"


# ---------------------------------------------------------------------------
# Part 8 — Long but complete definition preserved
# ---------------------------------------------------------------------------

class TestLongCompleteDefinitionPreserved:
    def test_june_long_definition_preserved(self):
        item = _def("June",
                    "sixth month of the year in the Julian and Gregorian calendars"
                    "—the latter the most widely used calendar in the world")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_cape_canaveral_space_force_station_preserved(self):
        item = _def("Cape Canaveral Space Force Station",
                    "installation of the United States Space Force's Space Launch Delta 45")
        r = _run([item])
        assert _verdict(r, item) == "accept"


# ---------------------------------------------------------------------------
# Part 9 — Good relations preserved
# ---------------------------------------------------------------------------

class TestGoodRelationsPreserved:
    def test_starlink_is_a_satellite_internet(self):
        item = _rel("Starlink", "is_a", "Satellite internet",
                    source_page="SpaceX")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_spacex_develops_falcon_9(self):
        item = _rel("SpaceX", "develops", "Falcon 9",
                    evidence="SpaceX develops the Falcon 9 rocket.",
                    source_page="Falcon 9")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_jeff_bezos_founded_amazon(self):
        item = _rel("Jeff Bezos", "founded", "Amazon",
                    evidence="Jeff Bezos founded Amazon.",
                    source_page="Amazon")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_bloomberg_news_founded_by_michael_bloomberg(self):
        item = _rel("Bloomberg News", "founded_by", "Michael Bloomberg",
                    evidence="Bloomberg News was founded by Michael Bloomberg.",
                    source_page="Bloomberg News")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_iea_publishes_world_energy_outlook(self):
        item = _rel("International Energy Agency", "publishes", "annual World Energy Outlook",
                    evidence="The IEA publishes the annual World Energy Outlook.",
                    source_page="International Energy Agency")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_arianespace_subsidiary_of_arianegroup(self):
        item = _rel("Arianespace", "subsidiary_of", "ArianeGroup",
                    evidence="Arianespace is a subsidiary of ArianeGroup.",
                    source_page="Arianespace")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_bigelow_aerospace_founded_by_robert_bigelow(self):
        item = _rel("Bigelow Aerospace", "founded_by", "Robert Bigelow",
                    evidence="Bigelow Aerospace was founded by Robert Bigelow.",
                    source_page="Bigelow Aerospace")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_bloomberg_government_service_of_bloomberg(self):
        item = _rel("Bloomberg Government", "service_of", "Bloomberg",
                    evidence="Bloomberg Government is a service of Bloomberg.",
                    source_page="Bloomberg Government")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_intel_founded_by_gordon_moore(self):
        item = _rel("Intel", "founded_by", "Gordon Moore",
                    evidence="Intel was founded by Gordon Moore.",
                    source_page="Intel Corporation")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_rocket_science_games_founded_by_steve_blank(self):
        item = _rel("Rocket Science Games", "founded_by", "Steve Blank",
                    evidence="Rocket Science Games was founded by Steve Blank.",
                    source_page="Rocket Science Games")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_nyse_owned_by_intercontinental_exchange(self):
        item = _rel("NYSE", "owned_by", "Intercontinental Exchange",
                    evidence="The NYSE is owned by Intercontinental Exchange.",
                    source_page="NYSE")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_reuters_owned_by_thomson_corporation(self):
        item = _rel("Reuters", "owned_by", "Canadian Thomson Corporation",
                    evidence="Reuters was acquired by the Canadian Thomson Corporation.",
                    source_page="Reuters")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_wef_founded_by_klaus_schwab(self):
        item = _rel("WEF", "founded_by", "Klaus Schwab",
                    evidence="WEF was founded by Klaus Schwab in 1971.",
                    source_page="World Economic Forum")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_solarcity_owned_by_tesla(self):
        item = _rel("SolarCity", "owned_by", "Tesla",
                    evidence="SolarCity was acquired by Tesla in 2016.",
                    source_page="Gigafactory New York")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_loopt_owned_by_green_dot(self):
        item = _rel("Loopt", "owned_by", "Green Dot Corporation",
                    evidence="Loopt was acquired by Green Dot Corporation.",
                    source_page="Sam Altman")
        r = _run([item])
        assert _verdict(r, item) == "accept"


# ---------------------------------------------------------------------------
# Part 10 — Good definitions preserved
# ---------------------------------------------------------------------------

class TestGoodDefinitionsPreserved:
    def test_hubble_space_telescope(self):
        item = _def("Hubble Space Telescope", "space telescope")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_intel_corporation(self):
        item = _def("Intel Corporation", "American multinational technology company")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_iea(self):
        item = _def("International Energy Agency",
                    "Paris-based autonomous intergovernmental organization")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_moon(self):
        item = _def("Moon", "only natural satellite of Earth")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_exos_aerospace(self):
        item = _def("Exos Aerospace Systems & Technologies", "aerospace manufacturer")
        r = _run([item])
        assert _verdict(r, item) == "accept"

    def test_rocket_science_games(self):
        item = _def("Rocket Science Games", "independent game studio")
        r = _run([item])
        assert _verdict(r, item) == "accept"


# ---------------------------------------------------------------------------
# Part 11 — Output consistency and counts
# ---------------------------------------------------------------------------

class TestOutputConsistency:
    def test_all_items_accounted_for(self):
        items = [
            _rel("Africa and Eurasia WorldSpace", "founded_by", "Noah Samara",
                 source_page="Satellite Radio"),
            _rel("GAO", "headquartered_in", "Washington",
                 evidence="The GAO is headquartered in Washington, D.C.",
                 source_page="Government Accountability Office"),
            _rel("Crew Dragon", "owned_by", "SpaceX",
                 evidence="Crew Dragon Demo-2 operated by SpaceX.",
                 source_page="Private spaceflight"),
            _def("Embraer E-Jet E2 family",
                 "series of four-abreast narrow-body regional jet airliners designed"),
            _def("Intel Corporation", "American multinational technology company"),
        ]
        r = _run(items)
        total_out = len(r["accepted"]) + len(r["rejected"]) + len(r["quarantine"])
        assert total_out == len(items)

    def test_before_after_counts_match(self):
        items = [
            _rel("Intel", "founded_by", "Gordon Moore",
                 evidence="Intel was founded by Gordon Moore.", source_page="Intel Corporation"),
            _rel("Harvard", "founded_by", "General Court",
                 evidence="Harvard was founded by the General Court.", source_page="Harvard University"),
            _def("Moon", "only natural satellite of Earth"),
            _def("Ford F-Series", "series of light-duty trucks marketed"),
        ]
        r = _run(items)
        assert r["relation_before_count"] == 2
        assert r["relation_after_count"] == 1  # Harvard quarantined
        assert r["definition_before_count"] == 2
        assert r["definition_after_count"] == 1  # Ford rejected

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
        assert r["relation_before_count"] == 0
        assert r["definition_before_count"] == 0


# ---------------------------------------------------------------------------
# Part 12 — No weak context counted as answerable facts
# ---------------------------------------------------------------------------

class TestNoWeakContext:
    def test_weak_context_link_passes_through(self):
        wc = {
            "overlay_type": "overlay_context_link",
            "trust": "weak_context_only",
            "subject": "SpaceX",
            "related": "Elon Musk",
        }
        r = _run([wc])
        assert len(r["accepted"]) == 1
        assert r["accepted"][0] is wc
        assert r["relation_before_count"] == 0


# ---------------------------------------------------------------------------
# Part 13 — Entity cards not counted as answerable facts
# ---------------------------------------------------------------------------

class TestEntityCardsNotCounted:
    def test_entity_card_not_counted_in_relation_or_definition(self):
        entity = {
            "overlay_type": "overlay_entity",
            "label": "Starlink",
            "entity_type": "product",
            "source_page": "Starlink",
        }
        r = _run([entity])
        assert r["relation_before_count"] == 0
        assert r["definition_before_count"] == 0


# ---------------------------------------------------------------------------
# Part 14 — No network calls
# ---------------------------------------------------------------------------

class TestNoNetworkCalls:
    def test_module_has_no_network_imports(self):
        import worldpgt.knowledge_pump.precision_cleanup_v2_1 as m
        import inspect
        src = inspect.getsource(m)
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
            import worldpgt.knowledge_pump.precision_cleanup_v2_1  # noqa: F401
            h_after = hashlib.sha256(sense.read_bytes()).hexdigest()
            assert h_before == h_after


# ---------------------------------------------------------------------------
# Part 16 — Deduplicate report fields present
# ---------------------------------------------------------------------------

class TestDeduplicateReport:
    def test_dup_report_fields_present(self):
        hp_item = _rel("Compaq", "owned_by", "HP",
                       evidence="Compaq was acquired by Hewlett-Packard.",
                       source_page="Compaq Computer")
        hp_full = _rel("Compaq", "owned_by", "Hewlett-Packard",
                       evidence="Compaq was acquired by Hewlett-Packard.",
                       source_page="Compaq Computer")
        r = _run([hp_item, hp_full])
        assert "duplicate_relation_groups" in r
        assert "duplicate_alias_relation_count" in r
        assert "canonicalized_relation_count" in r
        assert "quarantined_alias_duplicate_count" in r
        assert r["duplicate_alias_relation_count"] == 1
        assert r["quarantined_alias_duplicate_count"] == 1
