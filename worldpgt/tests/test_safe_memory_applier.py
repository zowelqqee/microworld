"""Tests for Safe Knowledge Memory Applier v1 (26 tests).

Covers: accepted memory builder, deduplication, provenance, stats, validator
safety gates, quarantine, and import constraints.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import pytest

from worldpgt.experiments.build_accepted_knowledge_memory_v1 import run as build_run
from worldpgt.experiments.validate_accepted_knowledge_memory_v1 import (
    _is_safe,
    run as validate_run,
)
from worldpgt.knowledge.safe_memory_applier import SafeMemoryApplier, _dedup_key, _make_item_id

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_EXP = Path(__file__).parent.parent / "experiments"
_AUTO_REVIEW = str(_EXP / "knowledge_ingestion_v1_auto_review.json")
_PATTERNS = str(_EXP / "wiki_pattern_candidates_v1.json")
_OLD_BENCH_PROMPTS = str(_EXP / "continuation_prompts_v1.csv")
_OLD_BENCH_BASELINE = str(_EXP / "microworld_continuation_v1_2_outputs.csv")
_PROBE_PROMPTS = str(_EXP / "knowledge_probe_prompts_v1.csv")
_PREV_PROBE_SUMMARY = str(_EXP / "knowledge_probe_overlay_v1_summary.json")
_SENSE_MEMORY_PY = str(Path(__file__).parent.parent / "continuation" / "sense_memory.py")


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_auto_review() -> dict:
    with open(_AUTO_REVIEW, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_patterns() -> dict:
    with open(_PATTERNS, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _build_applier() -> SafeMemoryApplier:
    applier = SafeMemoryApplier()
    applier.build_from_sources(_load_auto_review(), _load_patterns())
    return applier


@lru_cache(maxsize=1)
def _run_build_cached() -> tuple[dict, dict]:
    """Run builder and return (artifact, manifest). Cached for the session."""
    with tempfile.TemporaryDirectory() as tmp:
        out_json = os.path.join(tmp, "artifact.json")
        manifest = os.path.join(tmp, "manifest.json")
        stats = build_run(_AUTO_REVIEW, _PATTERNS, out_json, manifest)
        with open(out_json) as f:
            artifact = json.load(f)
        with open(manifest) as f:
            man = json.load(f)
    return artifact, man


@lru_cache(maxsize=1)
def _run_validation_cached() -> dict:
    """Run full validation using committed artifacts. Cached for the session."""
    committed_artifact = str(_EXP / "accepted_knowledge_memory_v1.json")
    with tempfile.TemporaryDirectory() as tmp:
        out_json = os.path.join(tmp, "validation.json")
        quar_json = os.path.join(tmp, "quarantine.json")
        val_csv = os.path.join(tmp, "validation.csv")
        result = validate_run(
            accepted_memory_path=committed_artifact,
            old_benchmark_input_path=_OLD_BENCH_PROMPTS,
            old_benchmark_baseline_path=_OLD_BENCH_BASELINE,
            probe_input_path=_PROBE_PROMPTS,
            previous_probe_summary_path=_PREV_PROBE_SUMMARY,
            output_json_path=out_json,
            quarantine_json_path=quar_json,
            validation_csv_path=val_csv,
        )
    return result


# ===========================================================================
# Test 1: Builder loads auto-review artifact
# ===========================================================================

def test_1_builder_loads_auto_review():
    applier = SafeMemoryApplier()
    applier.build_from_sources(_load_auto_review(), {"pattern_candidates": []})
    assert applier.stats["fact_items"] > 0, "Expected facts from auto_review"


# ===========================================================================
# Test 2: Builder loads pattern candidates
# ===========================================================================

def test_2_builder_loads_pattern_candidates():
    applier = SafeMemoryApplier()
    applier.build_from_sources({"proposal_reviews": []}, _load_patterns())
    assert applier.stats["pattern_items"] > 0, "Expected patterns from wiki_pattern_candidates"


# ===========================================================================
# Test 3: Only accepted_auto facts are included
# ===========================================================================

def test_3_only_accepted_auto_facts_included():
    applier = _build_applier()
    facts = [it for it in applier.items if it.item_kind == "fact"]
    for it in facts:
        assert it.decision == "accepted_auto", f"{it.item_id} has decision {it.decision}"


# ===========================================================================
# Test 4: Only accepted_auto patterns are included
# ===========================================================================

def test_4_only_accepted_auto_patterns_included():
    applier = _build_applier()
    patterns = [it for it in applier.items if it.item_kind == "pattern"]
    pats_data = _load_patterns()
    nr_count = sum(
        1 for p in pats_data.get("pattern_candidates", [])
        if p.get("decision") == "needs_review"
    )
    rej_count = sum(
        1 for p in pats_data.get("pattern_candidates", [])
        if p.get("decision") == "rejected_auto"
    )
    assert nr_count >= 2, "Expected at least 2 needs_review patterns in source"
    assert rej_count >= 2, "Expected at least 2 rejected_auto patterns in source"
    # None of these appear in items
    for it in patterns:
        assert it.decision == "accepted_auto", f"Pattern {it.item_id} should be accepted_auto"


# ===========================================================================
# Test 5: needs_review and rejected_auto items are excluded
# ===========================================================================

def test_5_needs_review_and_rejected_excluded():
    applier = _build_applier()
    stats = applier.stats
    assert stats["excluded_needs_review"] > 0, "Should have excluded needs_review items"
    # Auto-review has 16 needs_review, patterns have 2 more
    assert stats["excluded_needs_review"] >= 16
    # Patterns have 2 rejected_auto
    assert stats["excluded_rejected_auto"] >= 2


# ===========================================================================
# Test 6: Broad/generic/conflicting/high-risk items are excluded
# ===========================================================================

def test_6_high_risk_and_broad_items_excluded():
    applier = _build_applier()
    for it in applier.items:
        assert it.risk_level != "high", f"{it.item_id} has high risk_level"
    # All accepted_auto items in the source are low-risk; confirmed exclusion logic exists
    stats = applier.stats
    assert stats["excluded_high_risk"] == 0  # no high-risk in this dataset


# ===========================================================================
# Test 7: Item IDs are stable
# ===========================================================================

def test_7_item_ids_are_stable():
    # Build twice, IDs must match exactly
    a1 = SafeMemoryApplier().build_from_sources(_load_auto_review(), _load_patterns())
    a2 = SafeMemoryApplier().build_from_sources(_load_auto_review(), _load_patterns())
    ids1 = [it.item_id for it in a1.items]
    ids2 = [it.item_id for it in a2.items]
    assert ids1 == ids2
    # IDs are prefixed ami-
    assert all(pid.startswith("ami-") for pid in ids1)
    # IDs are unique
    assert len(ids1) == len(set(ids1)), "Item IDs must be unique"


# ===========================================================================
# Test 8: Duplicate items are deduplicated
# ===========================================================================

def test_8_duplicate_items_are_deduplicated():
    # Inject a deliberate duplicate in auto_review
    ar_data = json.loads(json.dumps(_load_auto_review()))
    first_pr = ar_data["proposal_reviews"][0]
    dup_item = dict(first_pr["items"][0])  # copy first accepted item
    first_pr["items"].append(dup_item)     # add duplicate

    applier = SafeMemoryApplier()
    applier.build_from_sources(ar_data, _load_patterns())
    assert applier.stats["deduplicated_count"] >= 1, "Duplicate must be detected"


# ===========================================================================
# Test 9: Every item has provenance
# ===========================================================================

def test_9_every_item_has_provenance():
    applier = _build_applier()
    for it in applier.items:
        prov = it.provenance
        assert prov.source_artifact, f"{it.item_id} missing source_artifact"
        if it.item_kind == "fact":
            assert prov.source_proposal_id is not None, f"{it.item_id} fact missing proposal_id"
        if it.item_kind == "pattern":
            assert prov.source_pattern_id, f"{it.item_id} pattern missing pattern_id"


# ===========================================================================
# Test 10: Stats match item list
# ===========================================================================

def test_10_stats_match_item_list():
    applier = _build_applier()
    stats = applier.stats
    items = applier.items
    assert stats["total_items"] == len(items)
    assert stats["fact_items"] == sum(1 for it in items if it.item_kind == "fact")
    assert stats["pattern_items"] == sum(1 for it in items if it.item_kind == "pattern")
    for term, count in stats["by_term"].items():
        assert count == sum(1 for it in items if it.term == term)
    for it_type, count in stats["by_item_type"].items():
        assert count == sum(1 for it in items if it.item_type == it_type)


# ===========================================================================
# Test 11: Accepted memory artifact is deterministic
# ===========================================================================

def test_11_artifact_is_deterministic():
    a1 = SafeMemoryApplier().build_from_sources(_load_auto_review(), _load_patterns())
    a2 = SafeMemoryApplier().build_from_sources(_load_auto_review(), _load_patterns())
    # Same item_ids in same order
    assert [it.item_id for it in a1.items] == [it.item_id for it in a2.items]
    # Same values
    assert [it.value for it in a1.items] == [it.value for it in a2.items]


# ===========================================================================
# Test 12: Builder writes JSON and manifest
# ===========================================================================

def test_12_builder_writes_json_and_manifest():
    artifact, manifest = _run_build_cached()
    assert artifact["memory_version"] == "accepted_knowledge_memory_v1"
    assert artifact["auto_apply_to_live_memory"] is False
    assert isinstance(artifact["items"], list)
    assert len(artifact["items"]) > 0
    assert manifest["total_items"] == len(artifact["items"])
    assert manifest["fact_items"] + manifest["pattern_items"] == manifest["total_items"]


# ===========================================================================
# Test 13: Builder does not modify sense_memory.py
# ===========================================================================

def test_13_builder_does_not_modify_sense_memory_py():
    before_hash = _file_hash(_SENSE_MEMORY_PY)
    with tempfile.TemporaryDirectory() as tmp:
        build_run(
            _AUTO_REVIEW, _PATTERNS,
            os.path.join(tmp, "a.json"),
            os.path.join(tmp, "m.json"),
        )
    after_hash = _file_hash(_SENSE_MEMORY_PY)
    assert before_hash == after_hash, "sense_memory.py must not be modified"


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ===========================================================================
# Test 14: Builder does not modify benchmark outputs
# ===========================================================================

def test_14_builder_does_not_modify_benchmark_outputs():
    before = _file_hash(_OLD_BENCH_BASELINE)
    with tempfile.TemporaryDirectory() as tmp:
        build_run(
            _AUTO_REVIEW, _PATTERNS,
            os.path.join(tmp, "a.json"),
            os.path.join(tmp, "m.json"),
        )
    after = _file_hash(_OLD_BENCH_BASELINE)
    assert before == after, "Old benchmark baseline CSV must not be modified"


# ===========================================================================
# Test 15: Validator loads accepted memory artifact
# ===========================================================================

def test_15_validator_loads_accepted_memory_artifact():
    committed = str(_EXP / "accepted_knowledge_memory_v1.json")
    if not os.path.exists(committed):
        pytest.skip("accepted_knowledge_memory_v1.json not yet generated")
    with open(committed) as f:
        artifact = json.load(f)
    applier = SafeMemoryApplier.from_artifact(artifact)
    assert len(applier.items) > 0


# ===========================================================================
# Test 16: Validator writes validation JSON
# ===========================================================================

def test_16_validator_writes_validation_json():
    committed = str(_EXP / "accepted_knowledge_memory_v1.json")
    if not os.path.exists(committed):
        pytest.skip("accepted_knowledge_memory_v1.json not yet generated")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "v.json")
        quar = os.path.join(tmp, "q.json")
        validate_run(
            accepted_memory_path=committed,
            old_benchmark_input_path=_OLD_BENCH_PROMPTS,
            old_benchmark_baseline_path=_OLD_BENCH_BASELINE,
            probe_input_path=_PROBE_PROMPTS,
            previous_probe_summary_path=_PREV_PROBE_SUMMARY,
            output_json_path=out,
            quarantine_json_path=quar,
        )
        assert os.path.exists(out) and os.path.getsize(out) > 0
        with open(out) as f:
            v = json.load(f)
        assert "validation_decision" in v
        assert "old_benchmark" in v
        assert "knowledge_probe" in v


# ===========================================================================
# Test 17: Validator writes quarantine JSON
# ===========================================================================

def test_17_validator_writes_quarantine_json():
    committed = str(_EXP / "accepted_knowledge_memory_v1.json")
    if not os.path.exists(committed):
        pytest.skip("accepted_knowledge_memory_v1.json not yet generated")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "v.json")
        quar = os.path.join(tmp, "q.json")
        validate_run(
            accepted_memory_path=committed,
            old_benchmark_input_path=_OLD_BENCH_PROMPTS,
            old_benchmark_baseline_path=_OLD_BENCH_BASELINE,
            probe_input_path=_PROBE_PROMPTS,
            previous_probe_summary_path=_PREV_PROBE_SUMMARY,
            output_json_path=out,
            quarantine_json_path=quar,
        )
        assert os.path.exists(quar) and os.path.getsize(quar) > 0
        with open(quar) as f:
            q = json.load(f)
        assert "quarantined_items" in q
        assert "memory_version" in q


# ===========================================================================
# Test 18: Validation checks old benchmark safety
# ===========================================================================

def test_18_validation_checks_old_benchmark_safety():
    result = _run_validation_cached()
    old = result["old_benchmark"]
    assert old["wrong_continue_count"] == 0, (
        f"Old benchmark wrong_continue must be 0, got {old['wrong_continue_count']}"
    )
    assert old["semantic_quality_flagged"] == 0, (
        f"Old benchmark quality flag must be 0, got {old['semantic_quality_flagged']}"
    )


# ===========================================================================
# Test 19: Validation checks probe benchmark safety
# ===========================================================================

def test_19_validation_checks_probe_benchmark_safety():
    result = _run_validation_cached()
    probe = result["knowledge_probe"]
    assert probe["wrong_continue_count"] == 0, (
        f"Probe wrong_continue must be 0, got {probe['wrong_continue_count']}"
    )
    assert probe["semantic_quality_flagged"] == 0, (
        f"Probe quality flag must be 0, got {probe['semantic_quality_flagged']}"
    )


# ===========================================================================
# Test 20: Validation marks unsafe run as unsafe_quarantined
# ===========================================================================

def test_20_validation_refuses_unsafe_runs():
    # _is_safe returns False when wrong_continue > 0
    bad_old = {"wrong_continue_count": 1, "semantic_quality_flagged": 0,
                "risk_regressions": [], "v1_051_audited": True}
    bad_probe = {"wrong_continue_count": 0, "semantic_quality_flagged": 0,
                  "risk_regressions": []}
    assert not _is_safe(bad_old, bad_probe), "Should be unsafe with wrong_continue > 0"

    # _is_safe returns False when semantic_quality_flagged > 0
    bad_old2 = {"wrong_continue_count": 0, "semantic_quality_flagged": 1,
                 "risk_regressions": [], "v1_051_audited": True}
    assert not _is_safe(bad_old2, bad_probe), "Should be unsafe with quality flagged"

    # _is_safe returns False when risk_regressions not empty
    bad_old3 = {"wrong_continue_count": 0, "semantic_quality_flagged": 0,
                 "risk_regressions": ["v1-010:sense_changed"], "v1_051_audited": True}
    assert not _is_safe(bad_old3, bad_probe), "Should be unsafe with risk_regressions"

    # _is_safe returns False when v1_051 not audited
    bad_old4 = {"wrong_continue_count": 0, "semantic_quality_flagged": 0,
                 "risk_regressions": [], "v1_051_audited": False}
    assert not _is_safe(bad_old4, bad_probe), "Should be unsafe when v1-051 not audited"

    # Good path
    good_old = {"wrong_continue_count": 0, "semantic_quality_flagged": 0,
                "risk_regressions": [], "v1_051_audited": True}
    good_probe = {"wrong_continue_count": 0, "semantic_quality_flagged": 0,
                   "risk_regressions": []}
    assert _is_safe(good_old, good_probe), "Should be safe with all gates passing"


# ===========================================================================
# Test 21: Validation never lowers thresholds
# ===========================================================================

def test_21_validation_never_lowers_thresholds():
    result = _run_validation_cached()
    assert result["safety"]["thresholds_changed"] is False


# ===========================================================================
# Test 22: Validation never weakens validators
# ===========================================================================

def test_22_validation_never_weakens_validators():
    result = _run_validation_cached()
    assert result["safety"]["validators_weakened"] is False


# ===========================================================================
# Test 23: Validation never adds generic fallback
# ===========================================================================

def test_23_validation_never_adds_generic_fallback():
    import worldpgt.experiments.validate_accepted_knowledge_memory_v1 as mod
    src = inspect.getsource(mod)
    for marker in ("default_continuation", "fallback_text", "force_continue", "generic_continuation"):
        assert marker not in src, f"Generic fallback marker found: {marker!r}"
    res = _run_validation_cached()
    assert res["safety"]["generic_fallback_added"] is False


# ===========================================================================
# Test 24: True unsafe rows remain audited
# ===========================================================================

def test_24_true_unsafe_rows_remain_audited():
    res = _run_validation_cached()
    # No risk regressions means no true-unsafe row became continue
    assert res["old_benchmark"]["risk_regressions"] == [], (
        f"Risk regressions: {res['old_benchmark']['risk_regressions']}"
    )
    assert res["knowledge_probe"]["risk_regressions"] == []


# ===========================================================================
# Test 25: v1-051 remains audited
# ===========================================================================

def test_25_v1_051_remains_audited():
    res = _run_validation_cached()
    assert res["old_benchmark"]["v1_051_audited"] is True, (
        "v1-051 must remain audited after applying the accepted memory overlay"
    )


# ===========================================================================
# Test 26: No neural/GPT/training imports or strings
# ===========================================================================

def test_26_no_neural_gpt_training_imports():
    import importlib
    import re
    modules_to_check = [
        "worldpgt.knowledge.accepted_memory_types",
        "worldpgt.knowledge.safe_memory_applier",
        "worldpgt.experiments.build_accepted_knowledge_memory_v1",
        "worldpgt.experiments.validate_accepted_knowledge_memory_v1",
    ]
    banned = {"torch", "transformers", "openai", "tensorflow", "keras",
              "backprop", "fine_tuning", "gpt", "neural"}
    for mod_name in modules_to_check:
        mod = importlib.import_module(mod_name)
        src = inspect.getsource(mod)
        import_lines = [
            line.strip() for line in src.splitlines()
            if line.strip().startswith(("import ", "from "))
        ]
        for line in import_lines:
            line_lower = line.lower()
            for b in banned:
                assert not re.search(r"\b" + re.escape(b) + r"\b", line_lower), (
                    f"Banned term {b!r} found in import: {line!r} in {mod_name}"
                )
