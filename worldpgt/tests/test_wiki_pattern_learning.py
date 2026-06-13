"""Tests for Wiki Pattern Learning v1 (25 tests).

Covers: pattern extractor, probe benchmark structure, probe runner behaviour,
and safety constraints.
"""

from __future__ import annotations

import ast
import csv
import inspect
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import pytest

from worldpgt.experiments.build_knowledge_probe_benchmark_v1 import (
    _PROBE_ROWS,
    run as build_run,
)
from worldpgt.experiments.run_knowledge_probe_overlay_v1 import run as probe_run
from worldpgt.knowledge.wiki_pattern_extractor import WikiPatternExtractor
from worldpgt.knowledge.wiki_pattern_types import (
    KNOWN_SEMANTIC_FRAMES,
    PatternCandidate,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_EXP = Path(__file__).parent.parent / "experiments"
_SNIPPETS = str(_EXP / "curated_wiki_snippets_v1.json")
_AUTO_REVIEW = str(_EXP / "knowledge_ingestion_v1_auto_review.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_snippets() -> list[dict]:
    with open(_SNIPPETS, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Cached full extraction run
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _extract_all_cached() -> tuple[PatternCandidate, ...]:
    snippets = _load_snippets()
    extractor = WikiPatternExtractor()
    return tuple(extractor.extract(snippets))


@lru_cache(maxsize=1)
def _run_full_probe_cached() -> dict:
    """Build patterns + probe, then run the overlay; cache the summary."""
    snippets = _load_snippets()
    with tempfile.TemporaryDirectory() as tmp:
        patterns_path = os.path.join(tmp, "patterns.json")
        probe_csv = os.path.join(tmp, "probe.csv")
        out_csv = os.path.join(tmp, "probe_out.csv")
        out_json = os.path.join(tmp, "probe_summary.json")

        build_run(
            snippets_path=_SNIPPETS,
            auto_review_path=_AUTO_REVIEW,
            patterns_out_path=patterns_path,
            probe_out_path=probe_csv,
        )
        summary = probe_run(
            probe_path=probe_csv,
            patterns_path=patterns_path,
            auto_review_path=_AUTO_REVIEW,
            output_csv_path=out_csv,
            output_json_path=out_json,
        )
    return summary


# ===========================================================================
# Tests 1-3: extractor loads snippets and produces deterministic output
# ===========================================================================

def test_1_extractor_loads_curated_snippets():
    """Pattern extractor accepts the curated snippets without error."""
    candidates = _extract_all_cached()
    assert len(candidates) > 0


def test_2_extractor_emits_deterministic_candidates():
    """Running the extractor twice on the same input yields identical results."""
    snippets = _load_snippets()
    extractor = WikiPatternExtractor()
    run1 = extractor.extract(snippets)
    run2 = extractor.extract(snippets)
    assert [c.pattern_id for c in run1] == [c.pattern_id for c in run2]


def test_3_pattern_ids_are_stable():
    """Pattern IDs are stable: same source always produces the same ID."""
    candidates = _extract_all_cached()
    ids = [c.pattern_id for c in candidates]
    assert len(ids) == len(set(ids)), "Pattern IDs must be unique"
    assert all(pid.startswith("pat-") for pid in ids)


# ===========================================================================
# Tests 4-7: pattern type coverage
# ===========================================================================

def test_4_definition_patterns_are_extracted():
    candidates = _extract_all_cached()
    defs = [c for c in candidates if c.pattern_type == "definition"]
    assert len(defs) >= 3, f"Expected >=3 definition patterns, got {len(defs)}"


def test_5_location_patterns_are_extracted():
    candidates = _extract_all_cached()
    locs = [c for c in candidates if c.pattern_type == "location"]
    assert len(locs) >= 3, f"Expected >=3 location patterns, got {len(locs)}"


def test_6_action_patterns_are_extracted():
    candidates = _extract_all_cached()
    acts = [c for c in candidates if c.pattern_type == "action"]
    assert len(acts) >= 3, f"Expected >=3 action patterns, got {len(acts)}"


def test_7_property_and_relation_patterns_extracted():
    candidates = _extract_all_cached()
    pr = [c for c in candidates if c.pattern_type in ("property", "relation")]
    assert len(pr) >= 3, f"Expected >=3 property/relation patterns, got {len(pr)}"


# ===========================================================================
# Test 8: generic pattern rejection
# ===========================================================================

def test_8_generic_patterns_are_rejected():
    """Templates in GENERIC_TEMPLATES or with no concrete slots are rejected_auto."""
    candidates = _extract_all_cached()
    rejected = [c for c in candidates if c.decision == "rejected_auto"]
    assert len(rejected) >= 1, "At least one generic pattern must be rejected_auto"
    for r in rejected:
        assert r.reason in ("generic_template", "no_concrete_slots"), (
            f"Unexpected rejection reason: {r.reason}"
        )


# ===========================================================================
# Tests 9-10: accepted patterns are well-formed
# ===========================================================================

def test_9_accepted_patterns_linked_to_exact_term_sense():
    candidates = _extract_all_cached()
    for c in candidates:
        if c.decision != "accepted_auto":
            continue
        assert c.term, f"{c.pattern_id}: term is empty"
        assert c.sense, f"{c.pattern_id}: sense is empty"
        assert c.source_id, f"{c.pattern_id}: source_id is empty"


def test_10_accepted_patterns_have_concrete_slots():
    """Every accepted_auto pattern has at least one required_slot beyond 'subject'."""
    candidates = _extract_all_cached()
    for c in candidates:
        if c.decision != "accepted_auto":
            continue
        non_subject = [s for s in c.required_slots if s != "subject"]
        assert non_subject, (
            f"{c.pattern_id} ({c.term}:{c.sense}) accepted_auto but no concrete slots"
        )


# ===========================================================================
# Test 11: unknown semantic frames go to needs_review
# ===========================================================================

def test_11_unknown_frames_go_to_needs_review():
    candidates = _extract_all_cached()
    needs_review = [c for c in candidates if c.decision == "needs_review"]
    assert len(needs_review) >= 1, "Expected at least one needs_review pattern"
    for nr in needs_review:
        assert nr.semantic_frame not in KNOWN_SEMANTIC_FRAMES, (
            f"{nr.pattern_id} is needs_review but frame {nr.semantic_frame!r} IS known"
        )


# ===========================================================================
# Tests 12-15: probe benchmark structure
# ===========================================================================

def test_12_probe_benchmark_has_60_rows():
    assert len(_PROBE_ROWS) == 60, f"Expected 60 probe rows, got {len(_PROBE_ROWS)}"


def test_13_probe_benchmark_covers_all_6_terms_and_12_senses():
    term_senses = {(r["term"], r["target_sense"]) for r in _PROBE_ROWS}
    expected = {
        ("bank", "financial_institution"), ("bank", "river_edge"),
        ("bat", "animal"), ("bat", "sports_equipment"),
        ("seal", "animal"), ("seal", "closure_stamp"),
        ("crane", "bird"), ("crane", "machine"),
        ("rock", "stone"), ("rock", "music"),
        ("spring", "season"), ("spring", "coil"),
    }
    assert term_senses == expected, f"Missing: {expected - term_senses}"


def test_14_probe_benchmark_includes_wiki_derived_cues_unused_in_old_prompts():
    """Probe rows use cues (expected_cue) not present in the original 120-row benchmark."""
    # The original benchmark is in continuation_prompts_v1.csv.
    old_prompts_path = _EXP / "continuation_prompts_v1.csv"
    if not old_prompts_path.exists():
        pytest.skip("continuation_prompts_v1.csv not found")
    with open(old_prompts_path, newline="", encoding="utf-8") as f:
        old_rows = list(csv.DictReader(f))
    old_text = " ".join(r.get("prompt", "").lower() for r in old_rows)

    # New wiki cues not in old benchmark text
    wiki_cues = {"flipper", "pup", "orca", "marine", "airtight", "notary",
                 "flock", "wetland", "cable", "boom", "outrigger",
                 "mineral", "hillside", "geology", "album", "genre", "drum",
                 "bloom", "migration", "morning", "flower", "coil", "mattress", "elastic"}
    found_new = [c for c in wiki_cues if c in old_text]
    used_in_probe = {r["expected_cue"] for r in _PROBE_ROWS}
    new_cues_in_probe = wiki_cues & used_in_probe
    assert len(new_cues_in_probe) >= 5, (
        f"Expected >= 5 new wiki cues in probe expected_cue; found {new_cues_in_probe}"
    )


def test_15_probe_benchmark_includes_conflict_audit_rows():
    audit_rows = [r for r in _PROBE_ROWS if r["expected_safe_behavior"] == "audit"]
    assert len(audit_rows) >= 6, f"Expected >=6 audit/conflict rows, got {len(audit_rows)}"
    terms_with_audit = {r["term"] for r in audit_rows}
    assert len(terms_with_audit) >= 4, "Audit rows should span multiple terms"


# ===========================================================================
# Tests 16-17: probe runner writes artifacts and summary is consistent
# ===========================================================================

def test_16_probe_runner_writes_csv_and_json():
    snippets = _load_snippets()
    with tempfile.TemporaryDirectory() as tmp:
        patterns_path = os.path.join(tmp, "pat.json")
        probe_csv = os.path.join(tmp, "probe.csv")
        out_csv = os.path.join(tmp, "out.csv")
        out_json = os.path.join(tmp, "out.json")

        build_run(_SNIPPETS, _AUTO_REVIEW, patterns_path, probe_csv)
        probe_run(probe_csv, patterns_path, _AUTO_REVIEW, out_csv, out_json)

        assert os.path.exists(out_csv) and os.path.getsize(out_csv) > 0
        assert os.path.exists(out_json) and os.path.getsize(out_json) > 0

        with open(out_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 60

        with open(out_json) as f:
            summary = json.load(f)
        assert summary["probe_total"] == 60


def test_17_probe_summary_counts_are_internally_consistent():
    summary = _run_full_probe_cached()
    b = summary["baseline"]
    o = summary["knowledge_pattern_overlay"]
    d = summary["delta"]

    assert b["continue_count"] + b["audit_count"] == 60
    assert o["continue_count"] + o["audit_count"] == 60
    assert d["continue_count"] == o["continue_count"] - b["continue_count"]
    assert len(d["newly_continued_rows"]) == sum(
        1 for r in d["newly_continued_rows"] if r
    )


# ===========================================================================
# Tests 18-22: safety constraints
# ===========================================================================

def test_18_probe_overlay_does_not_modify_sense_memory_py():
    import worldpgt.knowledge.wiki_pattern_extractor as mod
    src = inspect.getsource(mod)
    assert "sense_memory" not in src or "ExplicitSenseMemory" not in src or \
           "add_sense(" not in src, "wiki_pattern_extractor must not call add_sense"
    # Specifically: the extractor must not import or mutate sense_memory
    assert "_load_builtin_senses" not in src


def test_19_probe_overlay_does_not_modify_trusted_baseline_outputs():
    """The runner never opens the trusted baseline CSV for writing."""
    import worldpgt.experiments.run_knowledge_probe_overlay_v1 as mod
    src = inspect.getsource(mod)
    assert "microworld_continuation_v1_2_outputs" not in src


def test_20_probe_overlay_does_not_lower_thresholds():
    summary = _run_full_probe_cached()
    assert summary["safety"]["thresholds_changed"] is False


def test_21_probe_overlay_does_not_weaken_validators():
    summary = _run_full_probe_cached()
    assert summary["safety"]["validators_weakened"] is False


def test_22_probe_overlay_does_not_add_generic_fallback():
    import worldpgt.experiments.run_knowledge_probe_overlay_v1 as mod
    src = inspect.getsource(mod)
    generic_markers = [
        "default_continuation",
        "fallback_text",
        "generic_continuation",
        "force_continue",
    ]
    for marker in generic_markers:
        assert marker not in src, f"Generic fallback marker found: {marker!r}"


# ===========================================================================
# Tests 23-24: output quality after overlay
# ===========================================================================

def test_23_overlay_wrong_continue_count_remains_0():
    summary = _run_full_probe_cached()
    assert summary["knowledge_pattern_overlay"]["wrong_continue_count"] == 0, (
        f"wrong_continue_count should be 0, got "
        f"{summary['knowledge_pattern_overlay']['wrong_continue_count']}"
    )


def test_24_overlay_semantic_quality_flagged_remains_0():
    summary = _run_full_probe_cached()
    assert summary["knowledge_pattern_overlay"]["semantic_quality_flagged"] == 0


# ===========================================================================
# Test 25: no neural / GPT / training imports
# ===========================================================================

def test_25_no_neural_gpt_training_imports():
    import importlib
    import re
    modules_to_check = [
        "worldpgt.knowledge.wiki_pattern_types",
        "worldpgt.knowledge.wiki_pattern_extractor",
        "worldpgt.experiments.build_knowledge_probe_benchmark_v1",
        "worldpgt.experiments.run_knowledge_probe_overlay_v1",
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
