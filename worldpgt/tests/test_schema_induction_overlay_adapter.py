"""Tests for schema_induction overlay_adapter and induced_precision_gate."""

from __future__ import annotations

import pytest

from worldpgt.schema_induction.promotion_gates import GateConfig
from worldpgt.schema_induction.run_schema_induction import run_induction
from worldpgt.schema_induction.overlay_adapter import (
    schema_result_to_overlay_items,
    family_to_overlay_items,
)
from worldpgt.schema_induction.induced_precision_gate import (
    apply_induced_precision_gate,
    dedup_and_merge,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VISA_DOCS = [
    {"doc_id": "d1", "title": "D7", "url": "",
     "text": "Portugal D7 visa requires proof of passive income. "
             "Portugal D7 visa requires accommodation."},
    {"doc_id": "d2", "title": "NLV", "url": "",
     "text": "Spain non-lucrative visa prohibits work. "
             "Spain non-lucrative visa requires proof of financial means. "
             "Spain non-lucrative visa prohibits any local employment."},
    {"doc_id": "d3", "title": "DNV", "url": "",
     "text": "Digital nomad visa allows remote work under conditions."},
]

_ANIMAL_DOCS = [
    {"doc_id": "d4", "title": "Giraffe", "url": "",
     "text": "Giraffes move seasonally in search of food and water."},
    {"doc_id": "d5", "title": "Wildebeest", "url": "",
     "text": "Wildebeest migrate toward areas with fresh grass."},
]


@pytest.fixture(scope="module")
def visa_result():
    return run_induction(_VISA_DOCS, GateConfig(min_evidence=2, min_sources=1))


@pytest.fixture(scope="module")
def animal_result():
    return run_induction(_ANIMAL_DOCS, GateConfig(min_evidence=1, min_sources=1))


# ---------------------------------------------------------------------------
# Step 1: overlay_adapter
# ---------------------------------------------------------------------------

class TestOverlayAdapter:
    def test_promoted_requires_becomes_overlay_relation(self, visa_result):
        items = schema_result_to_overlay_items(
            list(visa_result.families),
            list(visa_result.frames),
            list(visa_result.claims),
            include_generated=True,
        )
        relations = [i for i in items if i.get("overlay_type") == "overlay_relation"]
        assert relations, "expected at least one overlay_relation"

        requires = [r for r in relations if r.get("predicate") == "requires"]
        assert requires, "expected 'requires' predicate from promoted requires family"

        r = requires[0]
        assert r["subject"]
        assert r["object"]
        assert r["evidence_text"]
        assert r["pump_source_kind"] == "schema_induced"
        assert r["candidate_source"] == "pump_schema_induction"

    def test_prohibits_family_maps_to_prohibits_predicate(self, visa_result):
        items = schema_result_to_overlay_items(
            list(visa_result.families), list(visa_result.frames), list(visa_result.claims)
        )
        prohibits = [i for i in items if i.get("predicate") == "prohibits"]
        assert prohibits
        assert all(i["overlay_type"] == "overlay_relation" for i in prohibits)

    def test_promoted_family_gets_overlay_candidate_trust(self, visa_result):
        items = schema_result_to_overlay_items(
            list(visa_result.families), list(visa_result.frames), list(visa_result.claims)
        )
        promoted_items = [
            i for i in items if i.get("schema_promotion_status") == "promoted"
        ]
        assert promoted_items
        for item in promoted_items:
            assert item["trust"] == "overlay_candidate"

    def test_generated_family_gets_candidate_generated_trust(self, visa_result):
        items = schema_result_to_overlay_items(
            list(visa_result.families), list(visa_result.frames), list(visa_result.claims),
            include_generated=True,
        )
        generated_items = [
            i for i in items if i.get("schema_promotion_status") == "generated"
        ]
        assert generated_items
        for item in generated_items:
            assert item["trust"] == "overlay_candidate_generated"

    def test_source_trace_in_evidence_text(self, visa_result):
        items = schema_result_to_overlay_items(
            list(visa_result.families), list(visa_result.frames), list(visa_result.claims)
        )
        for item in items:
            assert item.get("evidence_text"), f"missing evidence_text: {item}"

    def test_include_generated_false_omits_generated(self, visa_result):
        all_items = schema_result_to_overlay_items(
            list(visa_result.families), list(visa_result.frames), list(visa_result.claims),
            include_generated=True,
        )
        promo_only = schema_result_to_overlay_items(
            list(visa_result.families), list(visa_result.frames), list(visa_result.claims),
            include_generated=False,
        )
        assert len(promo_only) <= len(all_items)
        for item in promo_only:
            assert item.get("schema_promotion_status") == "promoted"

    def test_destination_role_maps_to_located_in(self, animal_result):
        items = schema_result_to_overlay_items(
            list(animal_result.families), list(animal_result.frames), list(animal_result.claims)
        )
        located = [i for i in items if i.get("predicate") == "located_in"]
        assert located, "expected located_in from destination role"

    def test_schema_family_id_preserved(self, visa_result):
        items = schema_result_to_overlay_items(
            list(visa_result.families), list(visa_result.frames), list(visa_result.claims)
        )
        for item in items:
            assert item.get("schema_family_id"), "missing schema_family_id"


# ---------------------------------------------------------------------------
# Step 3: induced_precision_gate
# ---------------------------------------------------------------------------

class TestInducedPrecisionGate:
    def _make_item(self, status="promoted", src_count=2, predicate="requires",
                   subject="X", obj="Y"):
        return {
            "overlay_type": "overlay_relation",
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "evidence_text": "X requires Y.",
            "source_page": "d1",
            "trust": "overlay_candidate",
            "stability": "semi_stable",
            "pump_source_kind": "schema_induced",
            "schema_promotion_status": status,
            "schema_source_doc_count": src_count,
            "schema_family_id": "fam_test",
        }

    def test_promoted_item_passes_gate(self):
        item = self._make_item(status="promoted", src_count=1)
        result = apply_induced_precision_gate([item])
        assert len(result["accepted"]) == 1
        assert len(result["rejected"]) == 0

    def test_generated_with_sufficient_sources_passes(self):
        item = self._make_item(status="generated", src_count=2)
        result = apply_induced_precision_gate([item], min_source_docs_generated=2)
        assert len(result["accepted"]) == 1

    def test_generated_with_insufficient_sources_quarantined(self):
        item = self._make_item(status="generated", src_count=1)
        result = apply_induced_precision_gate([item], min_source_docs_generated=2)
        assert len(result["quarantine"]) == 1
        assert len(result["accepted"]) == 0

    def test_empty_subject_rejected(self):
        item = self._make_item()
        item["subject"] = ""
        result = apply_induced_precision_gate([item])
        assert any("empty_subject" in r.get("_gate_reason", "") for r in result["rejected"])

    def test_empty_object_rejected(self):
        item = self._make_item()
        item["object"] = "  "
        result = apply_induced_precision_gate([item])
        assert any("empty_object" in r.get("_gate_reason", "") for r in result["rejected"])

    def test_self_relation_rejected(self):
        item = self._make_item(subject="Portugal D7 visa", obj="Portugal D7 visa")
        result = apply_induced_precision_gate([item])
        assert any("self_relation" in r.get("_gate_reason", "") for r in result["rejected"])


# ---------------------------------------------------------------------------
# Dedup logic
# ---------------------------------------------------------------------------

class TestDedupAndMerge:
    def _make_relation(self, subject, predicate, obj, stability="semi_stable",
                       source="pump_extraction_yield_v2"):
        return {
            "overlay_type": "overlay_relation",
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "stability": stability,
            "candidate_source": source,
            "evidence_text": f"{subject} {predicate} {obj}",
        }

    def test_new_schema_item_added(self):
        path_a = [self._make_relation("X", "requires", "Y")]
        schema = [self._make_relation("X", "requires", "Z", source="pump_schema_induction")]
        result = dedup_and_merge(path_a, schema)
        accepted_keys = {(i["subject"], i["predicate"], i["object"]) for i in result["accepted"]}
        assert ("X", "requires", "Z") in accepted_keys
        assert ("X", "requires", "Y") in accepted_keys
        assert len(result["new_induced"]) == 1

    def test_duplicate_item_deduped(self):
        path_a = [self._make_relation("X", "requires", "Y", stability="stable")]
        # Same (subject, predicate, object) with lower stability → should dedup.
        schema = [self._make_relation("X", "requires", "Y", stability="semi_stable",
                                      source="pump_schema_induction")]
        result = dedup_and_merge(path_a, schema)
        # Path-A version kept (higher stability stable > semi_stable).
        assert result["deduped"]
        assert len(result["new_induced"]) == 0

    def test_schema_upgrades_lower_stability(self):
        path_a = [self._make_relation("X", "requires", "Y", stability="semi_stable")]
        schema_stable = [self._make_relation("X", "requires", "Y", stability="stable",
                                              source="pump_schema_induction")]
        result = dedup_and_merge(path_a, schema_stable)
        # Schema item is more stable → should replace.
        assert result["upgraded"]
        matching = [i for i in result["accepted"]
                    if i["subject"] == "X" and i["predicate"] == "requires" and i["object"] == "Y"]
        assert matching[0]["stability"] == "stable"

    def test_path_a_always_included(self):
        path_a = [self._make_relation("A", "founded by", "B")]
        schema: list[dict] = []
        result = dedup_and_merge(path_a, schema)
        assert len(result["accepted"]) == 1


# ---------------------------------------------------------------------------
# Integration smoke: existing pipeline not broken
# ---------------------------------------------------------------------------

class TestExistingPipelineUnchanged:
    def test_schema_result_to_overlay_does_not_touch_existing(self, visa_result):
        """Producing overlay items must not mutate the SchemaInductionResult."""
        families_before = list(visa_result.families)
        _ = schema_result_to_overlay_items(
            list(visa_result.families), list(visa_result.frames), list(visa_result.claims)
        )
        assert list(visa_result.families) == families_before

    def test_overlay_items_have_required_fields(self, visa_result):
        items = schema_result_to_overlay_items(
            list(visa_result.families), list(visa_result.frames), list(visa_result.claims)
        )
        required = {"overlay_type", "subject", "evidence_text", "trust",
                    "stability", "candidate_source", "pump_source_kind"}
        for item in items:
            missing = required - item.keys()
            assert not missing, f"item missing fields {missing}: {item}"
