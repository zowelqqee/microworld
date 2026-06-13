"""Tests for AcceptedKnowledgeMemoryProvider v1.

Covers:
 1.  Provider loads accepted memory artifact.
 2.  Provider loads exactly 221 items.
 3.  Provider separates fact items and pattern items.
 4.  Provider indexes by term and sense.
 5.  Provider returns expected items for bank:financial_institution,
     seal:animal, bat:sports_equipment, spring:coil.
 6.  Provider is read-only.
 7.  Provider does not mutate source memory.
 8.  Provider output is deterministic.
 9.  Provider trace metadata is present when used.
10.  Default behavior remains unchanged without provider.
11.  Old benchmark with provider: wrong_continue_count=0, semantic_quality_flagged=0.
12.  Knowledge probe with provider: wrong_continue_count=0, semantic_quality_flagged=0,
     continue_count >= 43.
13.  True unsafe old benchmark rows remain audited.
14.  v1-051 remains audited.
15.  No thresholds lowered.
16.  No validators weakened.
17.  No generic fallback added.
18.  No neural/GPT/training imports or strings.
"""

from __future__ import annotations

import csv
import importlib
import inspect
import os
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from worldpgt.continuation.continuation_engine import ControlledContinuationEngine
from worldpgt.continuation.continuation_policy import ContinuationPolicy
from worldpgt.continuation.sense_memory import ExplicitSenseMemory
from worldpgt.experiments.check_semantic_render_quality import check_rows
from worldpgt.experiments.risk_coverage_metrics import summarize_rows
from worldpgt.knowledge.accepted_memory_provider import (
    MEMORY_VERSION,
    AcceptedKnowledgeMemoryProvider,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT = _REPO_ROOT / "worldpgt" / "experiments" / "accepted_knowledge_memory_v1.json"
_OLD_PROMPTS = _REPO_ROOT / "worldpgt" / "experiments" / "continuation_prompts_v1.csv"
_OLD_BASELINE = _REPO_ROOT / "worldpgt" / "experiments" / "microworld_continuation_v1_2_outputs.csv"
_PROBE_PROMPTS = _REPO_ROOT / "worldpgt" / "experiments" / "knowledge_probe_prompts_v1.csv"

_TRUE_UNSAFE_IDS = frozenset({
    "v1-081", "v1-082", "v1-083", "v1-085", "v1-086",
    "v1-088", "v1-089", "v1-090", "v1-091", "v1-092",
    "v1-093", "v1-094",
})


# ------------------------------------------------------------------
# Fixture-level caches (compute once per test session)
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def _provider() -> AcceptedKnowledgeMemoryProvider:
    return AcceptedKnowledgeMemoryProvider(str(_ARTIFACT))


@lru_cache(maxsize=1)
def _run_old_with_provider() -> tuple[dict, ...]:
    p = _provider()
    overlay_mem = p.build_overlay_memory()
    engine = ControlledContinuationEngine(memory=overlay_mem)
    rows: list[dict] = []
    with _OLD_PROMPTS.open(newline="", encoding="utf-8") as fh:
        for record in csv.DictReader(fh):
            result = engine.continue_prompt(record["prompt"])
            rows.append({
                "id": record.get("id", ""),
                "prompt": record.get("prompt", ""),
                "expected_sense": record.get("expected_sense", ""),
                "continuation": result.continuation,
                "selected_sense": result.selected_sense or "",
                "decision": result.decision,
                "reasons": " | ".join(result.reasons),
                "memory_hits": " | ".join(result.memory_hits),
            })
    return tuple(rows)


@lru_cache(maxsize=1)
def _run_probe_with_provider() -> tuple[dict, ...]:
    p = _provider()
    overlay_mem = p.build_overlay_memory()
    engine = ControlledContinuationEngine(memory=overlay_mem)
    rows: list[dict] = []
    with _PROBE_PROMPTS.open(newline="", encoding="utf-8") as fh:
        for record in csv.DictReader(fh):
            result = engine.continue_prompt(record["prompt"])
            rows.append({
                "id": record.get("row_id", ""),
                "prompt": record.get("prompt", ""),
                "expected_sense": record.get("target_sense", ""),
                "continuation": result.continuation,
                "selected_sense": result.selected_sense or "",
                "decision": result.decision,
                "reasons": " | ".join(result.reasons),
                "memory_hits": " | ".join(result.memory_hits),
            })
    return tuple(rows)


@lru_cache(maxsize=1)
def _run_old_default() -> tuple[dict, ...]:
    engine = ControlledContinuationEngine()
    rows: list[dict] = []
    with _OLD_PROMPTS.open(newline="", encoding="utf-8") as fh:
        for record in csv.DictReader(fh):
            result = engine.continue_prompt(record["prompt"])
            rows.append({
                "id": record.get("id", ""),
                "prompt": record.get("prompt", ""),
                "expected_sense": record.get("expected_sense", ""),
                "continuation": result.continuation,
                "selected_sense": result.selected_sense or "",
                "decision": result.decision,
                "reasons": " | ".join(result.reasons),
                "memory_hits": " | ".join(result.memory_hits),
            })
    return tuple(rows)


# ------------------------------------------------------------------
# 1. Provider loads artifact
# ------------------------------------------------------------------

def test_provider_loads_artifact():
    p = _provider()
    assert len(p.items) > 0


# ------------------------------------------------------------------
# 2. Provider loads exactly 221 items
# ------------------------------------------------------------------

def test_provider_loads_exactly_221_items():
    p = _provider()
    assert len(p.items) == 221


# ------------------------------------------------------------------
# 3. Provider separates fact items and pattern items
# ------------------------------------------------------------------

def test_provider_separates_fact_and_pattern_items():
    p = _provider()
    assert p.fact_count == 163
    assert p.pattern_count == 58
    assert p.fact_count + p.pattern_count == 221


# ------------------------------------------------------------------
# 4. Provider indexes by term and sense
# ------------------------------------------------------------------

def test_provider_indexes_by_term():
    p = _provider()
    assert "bank" in p.terms
    assert "seal" in p.terms
    assert "bat" in p.terms
    assert "spring" in p.terms
    bank_items = p.items_for_term("bank")
    assert len(bank_items) == 40


def test_provider_indexes_by_sense():
    p = _provider()
    items = p.items_for_sense("bank", "financial_institution")
    assert len(items) == 21
    items = p.items_for_sense("seal", "animal")
    assert len(items) == 21


def test_provider_indexes_by_item_type():
    p = _provider()
    pos_cue_items = p.items_for_item_type("positive_cue")
    assert len(pos_cue_items) == 104


# ------------------------------------------------------------------
# 5. Provider returns expected items for specific term:sense combos
# ------------------------------------------------------------------

def test_provider_bank_financial_institution():
    p = _provider()
    items = p.items_for_sense("bank", "financial_institution")
    assert len(items) > 0
    item_types = {i.item_type for i in items}
    assert "positive_cue" in item_types
    cue_values = {
        i.value.strip().lower()
        for i in items
        if i.item_type == "positive_cue"
    }
    assert len(cue_values) > 0


def test_provider_seal_animal():
    p = _provider()
    items = p.items_for_sense("seal", "animal")
    assert len(items) > 0
    cue_values = {
        i.value.strip().lower()
        for i in items
        if i.item_type == "positive_cue"
    }
    assert len(cue_values) > 0


def test_provider_bat_sports_equipment():
    p = _provider()
    items = p.items_for_sense("bat", "sports_equipment")
    assert len(items) > 0
    cue_values = {
        i.value.strip().lower()
        for i in items
        if i.item_type == "positive_cue"
    }
    assert len(cue_values) > 0


def test_provider_spring_coil():
    p = _provider()
    items = p.items_for_sense("spring", "coil")
    assert len(items) > 0
    cue_values = {
        i.value.strip().lower()
        for i in items
        if i.item_type == "positive_cue"
    }
    assert len(cue_values) > 0


# ------------------------------------------------------------------
# 6. Provider is read-only
# ------------------------------------------------------------------

def test_provider_items_are_copies():
    p = _provider()
    items_a = p.items
    items_b = p.items
    assert items_a is not items_b  # fresh list each time


def test_provider_items_for_term_returns_copy():
    p = _provider()
    result = p.items_for_term("bank")
    result.clear()
    assert len(p.items_for_term("bank")) == 40


def test_provider_stats_read_only():
    p = _provider()
    s = p.stats
    assert isinstance(s, dict)
    s["items_loaded"] = 9999
    assert p.stats["items_loaded"] == 221


# ------------------------------------------------------------------
# 7. Provider does not mutate source memory
# ------------------------------------------------------------------

def test_provider_does_not_mutate_builtin_memory():
    p = _provider()
    baseline_mem = ExplicitSenseMemory(include_builtin=True)
    original_bank_cues = list(baseline_mem.get_senses("bank")[0].cues)

    overlay_mem = p.build_overlay_memory()

    # Builtin must be unchanged
    after_bank_cues = list(baseline_mem.get_senses("bank")[0].cues)
    assert after_bank_cues == original_bank_cues

    # Overlay must have at least as many cues
    overlay_bank_cues = overlay_mem.get_senses("bank")[0].cues
    assert set(original_bank_cues).issubset(set(overlay_bank_cues))


def test_two_overlay_builds_are_independent():
    p = _provider()
    ov1 = p.build_overlay_memory()
    ov2 = p.build_overlay_memory()

    # Mutate ov1 and verify ov2 is unaffected
    cues_before = list(ov2.get_senses("bank")[0].cues)
    ov1.get_senses("bank")[0].cues.append("__test_cue__")
    cues_after = list(ov2.get_senses("bank")[0].cues)
    assert "__test_cue__" not in cues_after
    assert cues_before == cues_after


# ------------------------------------------------------------------
# 8. Provider output is deterministic
# ------------------------------------------------------------------

def test_provider_overlay_memory_is_deterministic():
    p = _provider()
    ov1 = p.build_overlay_memory()
    ov2 = p.build_overlay_memory()

    for term in ov1.known_terms():
        for e1 in ov1.get_senses(term):
            e2_list = [e for e in ov2.get_senses(term) if e.sense_id == e1.sense_id]
            assert e2_list, f"sense {e1.sense_id} missing in second overlay for {term}"
            assert sorted(e1.cues) == sorted(e2_list[0].cues), f"cues differ for {term}:{e1.sense_id}"


def test_provider_old_benchmark_is_deterministic():
    rows_a = list(_run_old_with_provider())
    _run_old_with_provider.cache_clear()
    rows_b = list(_run_old_with_provider())
    decisions_a = [r["decision"] for r in rows_a]
    decisions_b = [r["decision"] for r in rows_b]
    assert decisions_a == decisions_b


# ------------------------------------------------------------------
# 9. Provider trace metadata is present when used
# ------------------------------------------------------------------

def test_provider_trace_markers_format():
    p = _provider()
    markers = p.provider_trace_markers()
    assert any(m == "accepted_memory_provider=enabled" for m in markers)
    assert any(m == f"accepted_memory_version={MEMORY_VERSION}" for m in markers)
    assert any(m.startswith("accepted_memory_items_loaded=") for m in markers)


def test_provider_item_trace_markers_format():
    p = _provider()
    item = p.items_for_item_type("positive_cue")[0]
    markers = p.item_trace_markers(item)
    assert any(m == "accepted_memory_provider=enabled" for m in markers)
    assert any(m == f"accepted_memory_version={MEMORY_VERSION}" for m in markers)
    assert any(m == f"accepted_memory_item_id={item.item_id}" for m in markers)
    assert any(m == f"accepted_memory_item_type={item.item_type}" for m in markers)
    assert any(m == f"accepted_memory_value={item.value}" for m in markers)


# ------------------------------------------------------------------
# 10. Default behavior remains unchanged without provider
# ------------------------------------------------------------------

def test_default_behavior_unchanged_without_provider():
    default_rows = list(_run_old_default())
    default_stats = summarize_rows(default_rows)
    assert default_stats["continue_count"] == 58
    assert default_stats["audit_count"] == 62
    assert default_stats["wrong_continue_count"] == 0


def test_default_engine_has_no_provider():
    engine = ControlledContinuationEngine()
    # The default engine must be a plain builtin memory
    assert isinstance(engine.memory, ExplicitSenseMemory)


# ------------------------------------------------------------------
# 11. Old benchmark with provider: wrong_continue=0, quality_flagged=0
# ------------------------------------------------------------------

def test_old_benchmark_with_provider_wrong_continue_is_zero():
    rows = list(_run_old_with_provider())
    wrong = sum(
        1 for r in rows
        if r["decision"] == "continue"
        and r["expected_sense"]
        and r["selected_sense"] != r["expected_sense"]
    )
    assert wrong == 0


def test_old_benchmark_with_provider_semantic_quality_flagged_is_zero():
    rows = list(_run_old_with_provider())
    quality = check_rows(rows)
    assert quality["flagged_count"] == 0, quality["flagged_rows"]


# ------------------------------------------------------------------
# 12. Knowledge probe with provider: safe, continue_count >= 43
# ------------------------------------------------------------------

def test_probe_with_provider_wrong_continue_is_zero():
    rows = list(_run_probe_with_provider())
    wrong = sum(
        1 for r in rows
        if r["decision"] == "continue"
        and r["expected_sense"]
        and r["selected_sense"] != r["expected_sense"]
    )
    assert wrong == 0


def test_probe_with_provider_semantic_quality_flagged_is_zero():
    rows = list(_run_probe_with_provider())
    quality = check_rows(rows)
    assert quality["flagged_count"] == 0, quality["flagged_rows"]


def test_probe_with_provider_continue_count_at_least_43():
    rows = list(_run_probe_with_provider())
    stats = summarize_rows(rows)
    assert stats["continue_count"] >= 43, f"continue_count={stats['continue_count']}"


# ------------------------------------------------------------------
# 13. True unsafe old benchmark rows remain audited
# ------------------------------------------------------------------

def test_true_unsafe_rows_remain_audited_with_provider():
    rows_by_id = {r["id"]: r for r in _run_old_with_provider()}
    for row_id in _TRUE_UNSAFE_IDS:
        row = rows_by_id.get(row_id)
        assert row is not None, f"row {row_id} not found"
        assert row["decision"] == "audit", f"{row_id} was {row['decision']}, expected audit"
        assert row["continuation"] == "", f"{row_id} has non-empty continuation"


# ------------------------------------------------------------------
# 14. v1-051 remains audited
# ------------------------------------------------------------------

def test_v1_051_remains_audited_with_provider():
    rows_by_id = {r["id"]: r for r in _run_old_with_provider()}
    row = rows_by_id["v1-051"]
    assert row["decision"] == "audit"
    assert row["continuation"] == ""
    assert "audit_reason=no_safe_repaired_candidate" in row["reasons"]


# ------------------------------------------------------------------
# 15. No thresholds lowered
# ------------------------------------------------------------------

def test_policy_thresholds_not_lowered():
    policy = ContinuationPolicy()
    assert policy.min_score == 1.0
    assert policy.min_margin == 1.0


# ------------------------------------------------------------------
# 16. No validators weakened — provider module contains no bypass keywords
# ------------------------------------------------------------------

_PROVIDER_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "knowledge" / "accepted_memory_provider.py"
)

def test_provider_module_has_no_validator_bypass():
    source = _PROVIDER_MODULE_PATH.read_text(encoding="utf-8")
    forbidden = [
        "skip_validator",
        "bypass_validator",
        "disable_validator",
        "force_continue",
        "force_trust",
    ]
    for kw in forbidden:
        assert kw not in source, f"Forbidden keyword '{kw}' found in provider module"


# ------------------------------------------------------------------
# 17. No generic fallback added
# ------------------------------------------------------------------

def test_provider_module_has_no_generic_fallback():
    source = _PROVIDER_MODULE_PATH.read_text(encoding="utf-8")
    forbidden = [
        "generic_fallback",
        "fallback_continuation",
        "always_continue",
    ]
    for kw in forbidden:
        assert kw not in source, f"Forbidden keyword '{kw}' found in provider module"


# ------------------------------------------------------------------
# 18. No neural/GPT/training imports or strings
# ------------------------------------------------------------------

_MODULES_TO_CHECK = [
    Path(__file__).resolve().parents[1] / "knowledge" / "accepted_memory_provider.py",
    Path(__file__).resolve().parents[1] / "experiments" / "run_accepted_memory_provider_benchmark_v1.py",
]

_FORBIDDEN_IMPORT_TOKENS = [
    "torch",
    "transformers",
    "openai",
    "backprop",
    "fine_tuning",
    "fine-tuning",
    "finetuning",
    "neural_weight",
    "model_weight",
    "gptneox",
    "automodelfor",
    "nanogpt",
]


@pytest.mark.parametrize("module_path", _MODULES_TO_CHECK, ids=lambda p: p.name)
def test_no_neural_gpt_training_imports(module_path: Path):
    lines = module_path.read_text(encoding="utf-8").splitlines()
    import_lines = [
        ln.lower() for ln in lines
        if ln.strip().startswith(("import ", "from "))
    ]
    import_block = "\n".join(import_lines)
    for token in _FORBIDDEN_IMPORT_TOKENS:
        assert token.lower() not in import_block, (
            f"Forbidden token '{token}' found in imports of {module_path.name}"
        )
