"""Tests for the automated knowledge review gate v1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldpgt.knowledge.automated_memory_reviewer import AutomatedMemoryReviewer
from worldpgt.knowledge.auto_review_types import (
    SAFETY_CHECKS_BEFORE_APPLY,
    AutoReviewOutput,
    ReviewedProposal,
)

_PROPOSALS_FILE = (
    Path(__file__).parent.parent / "experiments"
    / "knowledge_ingestion_v1_memory_update_proposals.json"
)

_ALL_ITEM_TYPES = {
    "positive_cue", "anti_cue", "typical_action", "typical_location",
    "part", "object", "semantic_frame_hint", "phrase_candidate_hint",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_proposals() -> list[dict]:
    return json.loads(_PROPOSALS_FILE.read_text(encoding="utf-8"))


def _review(proposals: list[dict] | None = None) -> AutoReviewOutput:
    if proposals is None:
        proposals = _load_proposals()
    return AutomatedMemoryReviewer().review(proposals)


def _make_proposal(
    term: str,
    sense: str,
    *,
    positive_cues: list[str] | None = None,
    anti_cues: list[str] | None = None,
    typical_actions: list[str] | None = None,
    typical_locations: list[str] | None = None,
    semantic_frame_hints: list[str] | None = None,
    phrase_candidate_hints: list[str] | None = None,
    conflicting_cues: list[str] | None = None,
    pid: str = "prop_synthetic_01",
) -> dict:
    return {
        "proposal_id": pid,
        "term": term,
        "sense": sense,
        "risk_level": "medium",
        "proposed_update": {
            "positive_cues": positive_cues or [],
            "anti_cues": anti_cues or [],
            "typical_actions": typical_actions or [],
            "typical_locations": typical_locations or [],
            "parts": [],
            "objects": [],
            "semantic_frame_hints": semantic_frame_hints or [],
            "phrase_candidate_hints": phrase_candidate_hints or [],
        },
        "evidence": {
            "source_titles": [],
            "matched_phrases": [],
            "rejected_broad_cues": [],
            "conflicting_cues": conflicting_cues or [],
        },
    }


def _items_for(output: AutoReviewOutput, term: str, sense: str) -> list:
    for r in output.proposal_reviews:
        if r.term == term and r.sense == sense:
            return r.items
    return []


def _accepted_values(output: AutoReviewOutput, term: str, sense: str) -> set[str]:
    return {i.value for i in _items_for(output, term, sense) if i.decision == "accepted_auto"}


def _rejected_values(output: AutoReviewOutput, term: str, sense: str) -> set[str]:
    return {i.value for i in _items_for(output, term, sense) if i.decision == "rejected_auto"}


def _needs_review_values(output: AutoReviewOutput, term: str, sense: str) -> set[str]:
    return {i.value for i in _items_for(output, term, sense) if i.decision == "needs_review"}


# ---------------------------------------------------------------------------
# 1. Loads all 12 proposals
# ---------------------------------------------------------------------------

def test_review_loads_all_proposals():
    output = _review()
    assert output.summary.total_proposals == 12


def test_review_covers_all_term_sense_pairs():
    output = _review()
    pairs = {(r.term, r.sense) for r in output.proposal_reviews}
    expected = {
        ("bank", "financial_institution"), ("bank", "river_edge"),
        ("bat", "animal"), ("bat", "sports_equipment"),
        ("crane", "bird"), ("crane", "machine"),
        ("rock", "music"), ("rock", "stone"),
        ("seal", "animal"), ("seal", "closure_stamp"),
        ("spring", "coil"), ("spring", "season"),
    }
    assert pairs == expected


# ---------------------------------------------------------------------------
# 2. Every proposed item receives exactly one item-level decision
# ---------------------------------------------------------------------------

def test_every_item_gets_exactly_one_decision():
    proposals = _load_proposals()
    output = _review(proposals)
    for prop_raw, reviewed in zip(proposals, output.proposal_reviews):
        update = prop_raw["proposed_update"]
        expected_count = sum(len(v) for v in update.values())
        assert len(reviewed.items) == expected_count, (
            f"[{reviewed.term}:{reviewed.sense}] expected {expected_count} items, "
            f"got {len(reviewed.items)}"
        )


def test_every_item_decision_is_valid():
    output = _review()
    valid = {"accepted_auto", "rejected_auto", "needs_review"}
    for r in output.proposal_reviews:
        for item in r.items:
            assert item.decision in valid, (
                f"Invalid decision {item.decision!r} for [{r.term}:{r.sense}] {item.value!r}"
            )


def test_item_buckets_are_disjoint_and_complete():
    """accepted + rejected + needs_review fields must partition every value exactly once."""
    output = _review()
    for r in output.proposal_reviews:
        for field in (
            "positive_cues", "anti_cues", "typical_actions", "typical_locations",
            "parts", "objects", "semantic_frame_hints", "phrase_candidate_hints",
        ):
            accepted = set(getattr(r.accepted_update, field))
            rejected = set(getattr(r.rejected_items, field))
            needs = set(getattr(r.needs_review_items, field))
            # Disjoint
            assert not (accepted & rejected), f"accepted ∩ rejected non-empty in {field}"
            assert not (accepted & needs), f"accepted ∩ needs_review non-empty in {field}"
            assert not (rejected & needs), f"rejected ∩ needs_review non-empty in {field}"
            # Complete — union must equal the items' values for that field
            item_values = {
                i.value for i in r.items
                if _ALL_ITEM_TYPES and True  # all items
                   and i.item_type.replace(" ", "_") + "s" == field
                   or i.item_type == field.rstrip("s")
            }
            # Simpler: just verify the sum of bucket sizes matches decided items per field
            item_type_for_field = field[:-1] if field.endswith("s") else field
            actual_total = len(accepted) + len(rejected) + len(needs)
            expected_total = len([i for i in r.items
                                  if _field_matches(i.item_type, field)])
            assert actual_total == expected_total, (
                f"[{r.term}:{r.sense}] field {field}: bucket total={actual_total} "
                f"!= item count={expected_total}"
            )


def _field_matches(item_type: str, field_name: str) -> bool:
    mapping = {
        "positive_cues": "positive_cue",
        "anti_cues": "anti_cue",
        "typical_actions": "typical_action",
        "typical_locations": "typical_location",
        "parts": "part",
        "objects": "object",
        "semantic_frame_hints": "semantic_frame_hint",
        "phrase_candidate_hints": "phrase_candidate_hint",
    }
    return mapping.get(field_name) == item_type


# ---------------------------------------------------------------------------
# 3. Concrete cues / actions are accepted_auto (real proposals)
# ---------------------------------------------------------------------------

def test_seal_animal_concrete_cues_accepted():
    output = _review()
    acc = _accepted_values(output, "seal", "animal")
    for cue in ("fish", "flipper", "ocean", "coast", "marine", "mammal", "swim", "dive"):
        assert cue in acc, f"seal:animal concrete cue '{cue}' should be accepted_auto"


def test_seal_closure_stamp_concrete_cues_accepted():
    output = _review()
    acc = _accepted_values(output, "seal", "closure_stamp")
    for cue in ("wax", "envelope", "document", "parcel", "flap", "press", "break", "stamp"):
        assert cue in acc, f"seal:closure_stamp cue '{cue}' should be accepted_auto"


def test_bank_financial_institution_cues_accepted():
    output = _review()
    acc = _accepted_values(output, "bank", "financial_institution")
    for cue in ("account", "deposit", "loan", "teller", "cash"):
        assert cue in acc, f"bank:financial_institution cue '{cue}' should be accepted_auto"


def test_bank_river_edge_cues_accepted():
    output = _review()
    acc = _accepted_values(output, "bank", "river_edge")
    for cue in ("stream", "current", "reed", "grass", "bridge"):
        assert cue in acc, f"bank:river_edge cue '{cue}' should be accepted_auto"


def test_bat_animal_cues_accepted():
    output = _review()
    acc = _accepted_values(output, "bat", "animal")
    for cue in ("cave", "insect", "echolocation", "wing", "dusk", "hunt", "roost", "hang"):
        assert cue in acc, f"bat:animal cue '{cue}' should be accepted_auto"


def test_bat_sports_equipment_cues_accepted():
    output = _review()
    acc = _accepted_values(output, "bat", "sports_equipment")
    for cue in ("pitcher", "ball", "swing", "player", "batter"):
        assert cue in acc, f"bat:sports_equipment cue '{cue}' should be accepted_auto"


def test_crane_machine_concrete_cues_accepted():
    output = _review()
    acc = _accepted_values(output, "crane", "machine")
    for cue in ("hook", "load", "cable", "boom", "operator", "lift", "lower"):
        assert cue in acc, f"crane:machine cue '{cue}' should be accepted_auto"


def test_crane_bird_cues_accepted():
    output = _review()
    acc = _accepted_values(output, "crane", "bird")
    for cue in ("neck", "reed", "lake", "wing", "migration", "wade", "dance", "migrate"):
        assert cue in acc, f"crane:bird cue '{cue}' should be accepted_auto"


def test_rock_music_cues_accepted():
    output = _review()
    acc = _accepted_values(output, "rock", "music")
    for cue in ("band", "concert", "crowd", "stage", "guitar", "drum", "album"):
        assert cue in acc, f"rock:music cue '{cue}' should be accepted_auto"


def test_rock_stone_cues_accepted():
    output = _review()
    acc = _accepted_values(output, "rock", "stone")
    for cue in ("cliff", "boulder", "trail", "mineral", "geology", "ground"):
        assert cue in acc, f"rock:stone cue '{cue}' should be accepted_auto"


def test_spring_season_cues_accepted():
    output = _review()
    acc = _accepted_values(output, "spring", "season")
    for cue in ("thaw", "flower", "morning", "rain", "bloom", "warm"):
        assert cue in acc, f"spring:season cue '{cue}' should be accepted_auto"


def test_spring_coil_cues_accepted():
    output = _review()
    acc = _accepted_values(output, "spring", "coil")
    for cue in ("latch", "handle", "coil", "metal", "mattress", "snap", "compress", "stretch"):
        assert cue in acc, f"spring:coil cue '{cue}' should be accepted_auto"


# ---------------------------------------------------------------------------
# 4. Broad cues are rejected_auto (synthetic)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("broad", ["water", "light", "thing", "object", "place",
                                    "time", "people", "near", "under", "over",
                                    "good", "bad", "big", "small"])
def test_broad_cue_rejected_auto(broad):
    prop = _make_proposal("test", "sense_a", positive_cues=[broad])
    output = _review([prop])
    items = output.proposal_reviews[0].items
    assert len(items) == 1
    assert items[0].decision == "rejected_auto", (
        f"Broad cue '{broad}' should be rejected_auto, got {items[0].decision}"
    )
    assert items[0].risk_level == "high"


@pytest.mark.parametrize("broad", ["water", "light", "thing"])
def test_broad_anti_cue_rejected_auto(broad):
    prop = _make_proposal("test", "sense_a", anti_cues=[broad])
    output = _review([prop])
    assert output.proposal_reviews[0].items[0].decision == "rejected_auto"


@pytest.mark.parametrize("broad", ["water", "light", "thing", "object", "place", "people"])
def test_broad_cue_not_in_accepted_update(broad):
    prop = _make_proposal("test", "sense_a", positive_cues=[broad])
    output = _review([prop])
    r = output.proposal_reviews[0]
    assert broad not in r.accepted_update.positive_cues
    assert broad in r.rejected_items.positive_cues


# ---------------------------------------------------------------------------
# 5. Conflicting cues are not accepted_auto (synthetic)
# ---------------------------------------------------------------------------

def test_conflicting_cue_not_accepted():
    prop = _make_proposal(
        "crane", "bird",
        positive_cues=["wing"],
        conflicting_cues=["wing (conflicts: machine)"],
    )
    output = _review([prop])
    items = output.proposal_reviews[0].items
    assert items[0].decision != "accepted_auto", (
        "Conflicting cue 'wing' must not be accepted_auto"
    )


def test_conflicting_cue_goes_to_needs_review():
    prop = _make_proposal(
        "crane", "bird",
        positive_cues=["wing", "lake"],
        conflicting_cues=["wing (conflicts: machine)"],
    )
    output = _review([prop])
    r = output.proposal_reviews[0]
    by_value = {i.value: i.decision for i in r.items}
    assert by_value["wing"] == "needs_review"
    assert by_value["lake"] == "accepted_auto"


def test_conflicting_cue_not_in_accepted_update():
    prop = _make_proposal(
        "bat", "animal",
        positive_cues=["cave", "wing"],
        conflicting_cues=["wing (conflicts: sports_equipment)"],
    )
    output = _review([prop])
    r = output.proposal_reviews[0]
    assert "wing" not in r.accepted_update.positive_cues
    assert "cave" in r.accepted_update.positive_cues


# ---------------------------------------------------------------------------
# 6. Unknown semantic frame hints go to needs_review
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unknown_frame", [
    "heavy_machinery_operation",
    "music_performance",
    "natural_material",
    "mechanical_device",
    "seasonal_transition",
    "totally_new_frame_xyz",
])
def test_unknown_frame_needs_review(unknown_frame):
    prop = _make_proposal("test", "sense_a", semantic_frame_hints=[unknown_frame])
    output = _review([prop])
    items = output.proposal_reviews[0].items
    assert items[0].decision == "needs_review", (
        f"Unknown frame '{unknown_frame}' should be needs_review"
    )


@pytest.mark.parametrize("known_frame", [
    "animal_behavior",
    "bird_behavior",
    "document_closure",
    "financial_transaction",
    "natural_geography",
    "sports_equipment_use",
])
def test_known_safe_frame_accepted(known_frame):
    prop = _make_proposal("test", "sense_a", semantic_frame_hints=[known_frame])
    output = _review([prop])
    items = output.proposal_reviews[0].items
    assert items[0].decision == "accepted_auto", (
        f"Known frame '{known_frame}' should be accepted_auto"
    )


# ---------------------------------------------------------------------------
# 7. Generic phrase candidates are rejected_auto
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("generic_phrase", [
    "did something",
    "moved around",
    "stayed there",
    "was important",
    "continued",
])
def test_generic_phrase_rejected(generic_phrase):
    prop = _make_proposal("test", "sense_a", phrase_candidate_hints=[generic_phrase])
    output = _review([prop])
    items = output.proposal_reviews[0].items
    assert items[0].decision == "rejected_auto", (
        f"Generic phrase '{generic_phrase}' should be rejected_auto"
    )


# ---------------------------------------------------------------------------
# 8. Known-safe phrase candidates are accepted_auto
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("safe_phrase", [
    "dove below the surface",
    "opened an account",
    "snapped back into place",
    "sealed it shut",
])
def test_known_safe_phrase_accepted(safe_phrase):
    prop = _make_proposal("test", "sense_a", phrase_candidate_hints=[safe_phrase])
    output = _review([prop])
    items = output.proposal_reviews[0].items
    assert items[0].decision == "accepted_auto", (
        f"Known-safe phrase '{safe_phrase}' should be accepted_auto"
    )


# ---------------------------------------------------------------------------
# 9. Current 12 proposals → correct proposal_decision distribution
# ---------------------------------------------------------------------------

def test_proposal_decisions_are_partial_accept_or_better():
    output = _review()
    for r in output.proposal_reviews:
        assert r.proposal_decision in ("accepted_auto", "partial_accept"), (
            f"[{r.term}:{r.sense}] expected partial_accept or accepted_auto, "
            f"got {r.proposal_decision!r}"
        )


def test_seal_animal_is_fully_accepted():
    """seal:animal has only known-safe items including the known-safe phrase → accepted_auto."""
    output = _review()
    for r in output.proposal_reviews:
        if r.term == "seal" and r.sense == "animal":
            assert r.proposal_decision == "accepted_auto", (
                f"seal:animal should be accepted_auto (all items concrete + known-safe phrase), "
                f"got {r.proposal_decision!r}"
            )
            return
    pytest.fail("seal:animal proposal not found")


def test_no_proposal_is_rejected_auto():
    output = _review()
    for r in output.proposal_reviews:
        assert r.proposal_decision != "rejected_auto", (
            f"[{r.term}:{r.sense}] no real proposal should be entirely rejected"
        )


# ---------------------------------------------------------------------------
# 10. Summary counts match item decisions
# ---------------------------------------------------------------------------

def test_summary_counts_match_item_totals():
    output = _review()
    s = output.summary
    all_items = [i for r in output.proposal_reviews for i in r.items]
    assert s.total_items_reviewed == len(all_items)
    assert s.accepted_auto == sum(1 for i in all_items if i.decision == "accepted_auto")
    assert s.rejected_auto == sum(1 for i in all_items if i.decision == "rejected_auto")
    assert s.needs_review == sum(1 for i in all_items if i.decision == "needs_review")
    assert s.accepted_auto + s.rejected_auto + s.needs_review == s.total_items_reviewed


def test_summary_rates_sum_to_one():
    output = _review()
    s = output.summary
    total_rate = s.acceptance_rate + s.rejection_rate + s.needs_review_rate
    assert abs(total_rate - 1.0) < 0.01, (
        f"Rates sum to {total_rate:.4f}, expected ~1.0"
    )


def test_by_term_counts_consistent():
    output = _review()
    s = output.summary
    for term, counts in s.by_term.items():
        term_total = sum(counts.values())
        all_items_for_term = [
            i for r in output.proposal_reviews
            if r.term == term
            for i in r.items
        ]
        assert term_total == len(all_items_for_term), (
            f"by_term[{term}] total {term_total} != item count {len(all_items_for_term)}"
        )


# ---------------------------------------------------------------------------
# 11. Human-review workload is lower than total item count
# ---------------------------------------------------------------------------

def test_human_workload_less_than_total():
    output = _review()
    s = output.summary
    assert s.needs_review < s.total_items_reviewed, (
        "Human-review workload must be smaller than total item count"
    )


def test_acceptance_rate_exceeds_50_percent():
    output = _review()
    assert output.summary.acceptance_rate > 0.5, (
        f"Acceptance rate {output.summary.acceptance_rate:.1%} should exceed 50%"
    )


def test_needs_review_rate_less_than_25_percent():
    output = _review()
    assert output.summary.needs_review_rate < 0.25, (
        f"needs_review rate {output.summary.needs_review_rate:.1%} should be under 25%"
    )


# ---------------------------------------------------------------------------
# 12. Output is deterministic
# ---------------------------------------------------------------------------

def test_review_output_is_deterministic():
    o1 = _review()
    o2 = _review()
    assert o1.summary.accepted_auto == o2.summary.accepted_auto
    assert o1.summary.needs_review == o2.summary.needs_review
    assert o1.summary.rejected_auto == o2.summary.rejected_auto
    for r1, r2 in zip(o1.proposal_reviews, o2.proposal_reviews):
        assert r1.proposal_id == r2.proposal_id
        assert r1.proposal_decision == r2.proposal_decision
        d1 = [(i.item_type, i.value, i.decision) for i in r1.items]
        d2 = [(i.item_type, i.value, i.decision) for i in r2.items]
        assert d1 == d2, f"Non-deterministic items in [{r1.term}:{r1.sense}]"


# ---------------------------------------------------------------------------
# 13. CLI writes JSON and CSV
# ---------------------------------------------------------------------------

def test_cli_writes_json_and_csv(tmp_path):
    from worldpgt.experiments.auto_review_knowledge_ingestion_v1 import main
    json_out = tmp_path / "review.json"
    csv_out = tmp_path / "review.csv"
    main([
        "--proposals", str(_PROPOSALS_FILE),
        "--output-json", str(json_out),
        "--output-csv", str(csv_out),
    ])
    assert json_out.exists() and json_out.stat().st_size > 0
    assert csv_out.exists() and csv_out.stat().st_size > 0

    data = json.loads(json_out.read_text())
    assert "summary" in data
    assert "policy" in data
    assert "proposal_reviews" in data
    assert data["summary"]["total_proposals"] == 12
    assert data["policy"]["auto_apply"] is False
    assert data["policy"]["sense_memory_modified"] is False
    assert data["policy"]["generation_behavior_changed"] is False

    csv_text = csv_out.read_text()
    assert "proposal_id" in csv_text
    assert "item_decision" in csv_text


def test_cli_json_policy_is_correct(tmp_path):
    from worldpgt.experiments.auto_review_knowledge_ingestion_v1 import main
    json_out = tmp_path / "review.json"
    main(["--proposals", str(_PROPOSALS_FILE), "--output-json", str(json_out)])
    data = json.loads(json_out.read_text())
    policy = data["policy"]
    assert policy["mode"] == "automated_review_only"
    assert policy["auto_apply"] is False
    assert policy["generation_behavior_changed"] is False
    assert policy["sense_memory_modified"] is False
    assert policy["human_review_required_for_uncertain_items"] is True


# ---------------------------------------------------------------------------
# 14. Does not modify sense_memory.py
# ---------------------------------------------------------------------------

def test_review_does_not_modify_sense_memory():
    sm_path = Path(__file__).parent.parent / "continuation" / "sense_memory.py"
    mtime_before = sm_path.stat().st_mtime
    _review()
    mtime_after = sm_path.stat().st_mtime
    assert mtime_before == mtime_after, "sense_memory.py mtime changed during review"


# ---------------------------------------------------------------------------
# 15. Does not modify benchmark outputs
# ---------------------------------------------------------------------------

def test_review_does_not_modify_benchmark_outputs():
    bench = Path(__file__).parent.parent / "experiments" / "continuation_prompts_v1.csv"
    before = bench.read_text() if bench.exists() else None
    _review()
    after = bench.read_text() if bench.exists() else None
    assert before == after, "Benchmark output was modified by auto-review"


# ---------------------------------------------------------------------------
# 16. No decision suggests threshold lowering
# ---------------------------------------------------------------------------

def test_no_decision_suggests_threshold_lowering():
    output = _review()
    for r in output.proposal_reviews:
        for item in r.items:
            assert "threshold" not in item.reason.lower()
            assert "threshold" not in item.value.lower()
        for field in ("positive_cues", "anti_cues", "typical_actions",
                      "typical_locations", "semantic_frame_hints", "phrase_candidate_hints"):
            for val in getattr(r.accepted_update, field):
                assert "threshold" not in val.lower()


# ---------------------------------------------------------------------------
# 17. No decision suggests validator weakening
# ---------------------------------------------------------------------------

def test_no_decision_suggests_validator_weakening():
    output = _review()
    for r in output.proposal_reviews:
        for item in r.items:
            assert "validator" not in item.reason.lower()
            assert "weaken" not in item.reason.lower()


# ---------------------------------------------------------------------------
# 18. No decision suggests generic trusted fallback
# ---------------------------------------------------------------------------

def test_no_decision_suggests_generic_fallback():
    output = _review()
    for r in output.proposal_reviews:
        for item in r.items:
            assert "fallback" not in item.reason.lower()
            assert "generic_trusted" not in item.reason.lower()


# ---------------------------------------------------------------------------
# 19. No neural / GPT / training imports or strings in the review modules
# ---------------------------------------------------------------------------

def test_no_neural_imports_in_review_modules():
    knowledge_dir = Path(__file__).parent.parent / "knowledge"
    forbidden = [
        "torch", "transformers", "openai", "backprop", "fine-tun", "finetun",
        "gradient", "weight tensor", "neural network", "model.train", "model.eval",
        "sklearn", "tensorflow", "keras",
    ]
    for py_file in knowledge_dir.glob("*.py"):
        src = py_file.read_text().lower()
        for term in forbidden:
            assert term not in src, f"Forbidden ML term {term!r} in {py_file.name}"


# ---------------------------------------------------------------------------
# Extra: safety_checks_required_before_apply always present
# ---------------------------------------------------------------------------

def test_safety_checks_present_in_every_proposal():
    output = _review()
    for r in output.proposal_reviews:
        for check in SAFETY_CHECKS_BEFORE_APPLY:
            assert check in r.safety_checks_required_before_apply, (
                f"Safety check '{check}' missing from [{r.term}:{r.sense}]"
            )


# ---------------------------------------------------------------------------
# Extra: accepted_update never contains broad words
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("broad", ["water", "light", "thing", "object", "place",
                                    "time", "people", "good", "bad"])
def test_accepted_update_never_contains_broad_word(broad):
    output = _review()
    for r in output.proposal_reviews:
        for field in ("positive_cues", "anti_cues"):
            assert broad not in getattr(r.accepted_update, field), (
                f"Broad word '{broad}' found in accepted_update.{field} "
                f"for [{r.term}:{r.sense}]"
            )


# ---------------------------------------------------------------------------
# Extra: narrow multiword exceptions reach accepted_update as positive_cues
# ---------------------------------------------------------------------------

def test_narrow_multiword_construction_site_accepted():
    output = _review()
    acc = _accepted_values(output, "crane", "machine")
    assert "construction site" in acc


def test_narrow_multiword_river_bank_accepted():
    output = _review()
    acc = _accepted_values(output, "bank", "river_edge")
    assert "river bank" in acc


def test_narrow_multiword_steel_beams_accepted():
    output = _review()
    acc = _accepted_values(output, "crane", "machine")
    assert "steel beams" in acc


def test_narrow_multiword_home_run_accepted():
    output = _review()
    acc = _accepted_values(output, "bat", "sports_equipment")
    assert "home run" in acc
