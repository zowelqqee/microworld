"""Tests for the universal domain bootstrap (entity_bootstrapper + overlay)."""

from __future__ import annotations

import pytest

from worldpgt.schema_induction.entity_bootstrapper import (
    bootstrap_entities,
    _acronym,
    _is_code_like,
)
from worldpgt.schema_induction.domain_overlay_builder import (
    build_domain_overlay,
    _BootstrapResolver,
)
from worldpgt.schema_induction.entity_bootstrapper import BootstrappedEntity


_O1A_DOCS = [
    {"doc_id": "d1", "title": "O-1A", "url": "", "text":
        "The O-1A visa is a nonimmigrant visa for individuals with extraordinary "
        "ability in the sciences. U.S. Citizenship and Immigration Services (USCIS) "
        "adjudicates O-1A petitions. The O-1A visa requires sustained acclaim."},
    {"doc_id": "d2", "title": "O-1B", "url": "", "text":
        "The O-1B visa is a nonimmigrant visa for individuals with extraordinary "
        "ability in the arts. The O-1B visa requires evidence of distinction. "
        "The O-1A visa prohibits self-petition. USCIS reviews O-1B petitions."},
]


# ---------------------------------------------------------------------------
# Pure helper unit tests (no spaCy needed).
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_acronym_expands_dotted_abbreviation(self):
        assert _acronym("U.S. Citizenship and Immigration Services") == "USCIS"

    def test_acronym_skips_stopwords(self):
        assert _acronym("Department of Motor Vehicles") == "DMV"

    def test_code_like_detection(self):
        assert _is_code_like("O-1A")
        assert _is_code_like("I-129")
        assert not _is_code_like("visa")
        assert not _is_code_like("extraordinary ability")


class TestResolver:
    def _entities(self):
        return [
            BootstrappedEntity("bootstrap:o_1a", "O-1A", ("O-1A visa",), "product",
                               ("PRODUCT",), 5, ("d1",)),
            BootstrappedEntity("bootstrap:uscis", "U.S. Citizenship and Immigration Services",
                               ("USCIS",), "organization", ("ORG",), 3, ("d1",)),
        ]

    def test_resolves_alias_to_canonical(self):
        r = _BootstrapResolver(self._entities())
        assert r.resolve("USCIS") == "U.S. Citizenship and Immigration Services"

    def test_resolves_containment_for_subject(self):
        r = _BootstrapResolver(self._entities())
        # "The O-1A visa" should resolve to canonical "O-1A".
        assert r.resolve("The O-1A visa") == "O-1A"

    def test_high_coverage_blocks_descriptive_object(self):
        r = _BootstrapResolver(self._entities())
        # A long descriptive object that merely contains a short entity must NOT
        # collapse to that entity when min_coverage is high.
        obj = "self-petition without a U.S. agent and a peer consultation letter"
        assert r.resolve(obj, min_coverage=0.6) != "O-1A"

    def test_unknown_surface_returns_none(self):
        r = _BootstrapResolver(self._entities())
        assert r.resolve("something entirely unrelated") is None


# ---------------------------------------------------------------------------
# Pass 1 — entity bootstrap (uses spaCy if available; tolerant assertions).
# ---------------------------------------------------------------------------

class TestBootstrapEntities:
    def test_finds_domain_entities_without_prior_list(self):
        ents = bootstrap_entities([(d["doc_id"], d["text"]) for d in _O1A_DOCS])
        labels = {e.canonical_label for e in ents}
        all_surfaces = labels | {a for e in ents for a in e.aliases}
        # The cold-start system must discover the O-1A term with no prior list.
        assert any("O-1A" in s for s in all_surfaces)

    def test_acronym_clusters_with_full_name(self):
        ents = bootstrap_entities([(d["doc_id"], d["text"]) for d in _O1A_DOCS])
        # USCIS must cluster with its expansion (one entity, the other an alias).
        for e in ents:
            surfaces = {e.canonical_label, *e.aliases}
            if any("USCIS" in s for s in surfaces):
                assert any("Citizenship" in s for s in surfaces)
                break
        else:
            pytest.skip("USCIS not detected by this spaCy build")

    def test_entities_have_canonical_types(self):
        ents = bootstrap_entities([(d["doc_id"], d["text"]) for d in _O1A_DOCS])
        valid = {"person", "organization", "publication", "product", "service",
                 "vehicle", "program", "place", "concept", "technology", "other"}
        assert ents
        for e in ents:
            assert e.entity_type in valid


# ---------------------------------------------------------------------------
# Pass 3 — overlay construction + end-to-end shape.
# ---------------------------------------------------------------------------

class TestDomainOverlay:
    @pytest.fixture(scope="class")
    def built(self):
        return build_domain_overlay(_O1A_DOCS, domain="o1a", min_evidence=1, min_sources=1)

    def test_overlay_has_entities_definitions_relations(self, built):
        overlay = built["overlay"]
        kinds = {it["overlay_type"] for it in overlay}
        assert "overlay_entity" in kinds
        assert "overlay_definition" in kinds
        assert "overlay_relation" in kinds

    def test_definition_for_o1a(self, built):
        defs = [it for it in built["overlay"] if it["overlay_type"] == "overlay_definition"]
        o1a_def = [d for d in defs if "O-1A" in d["subject"]]
        assert o1a_def
        assert "visa" in o1a_def[0]["definition"].lower()

    def test_requires_relation_present(self, built):
        rels = [it for it in built["overlay"] if it["overlay_type"] == "overlay_relation"]
        requires = [r for r in rels if r["predicate"] == "requires"]
        assert requires
        # Subject resolves to a bootstrapped entity, object kept descriptive.
        assert any("O-1A" in r["subject"] for r in requires)

    def test_overlay_items_carry_source_trace(self, built):
        for it in built["overlay"]:
            if it["overlay_type"] in ("overlay_relation", "overlay_definition"):
                assert "evidence_text" in it
                assert it.get("bootstrap_source") == "schema_induction_domain_bootstrap"

    def test_no_control_char_artifacts(self, built):
        # The abbreviation-doubling sentence-splitter artifact must be sanitized.
        for it in built["overlay"]:
            blob = " ".join(str(v) for v in it.values())
            assert "\x00" not in blob
            assert "U.SU.S" not in blob
