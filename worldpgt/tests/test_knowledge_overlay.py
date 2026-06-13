"""Tests for knowledge overlay dry-run v1 (16 spec tests + extras).

Test numbering follows the task specification:
1.  Overlay loads only accepted_auto items.
2.  Overlay ignores needs_review items.
3.  Overlay ignores rejected_auto items.
4.  Overlay object is deterministic.
5.  Overlay does not modify sense_memory.py.
6.  Overlay does not modify baseline benchmark output.
7.  Overlay does not lower thresholds.
8.  Overlay does not weaken validators.
9.  Overlay does not add generic output paths (code inspection).
10. Overlay trace markers are present for overlay-influenced rows.
11. Overlay run writes separate output files with correct structure.
12. Baseline metrics in summary match trusted baseline (58/62/0/0).
13. Overlay summary counts are internally consistent (continue+audit=120).
14. True unsafe rows remain audited in overlay.
15. v1-051 remains audited in overlay.
16. No semantic-quality flagged regressions in overlay.
+   No neural/GPT/training imports in overlay modules.
+   Safety flags in summary are set correctly.
+   No wrong continues introduced by overlay.
"""

from __future__ import annotations

import csv
import json
import tempfile
from functools import lru_cache
from pathlib import Path

from worldpgt.continuation.continuation_policy import ContinuationPolicy
from worldpgt.continuation.sense_memory import ExplicitSenseMemory
from worldpgt.knowledge.knowledge_overlay import KnowledgeOverlay

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
_AUTO_REVIEW = _EXPERIMENTS / "knowledge_ingestion_v1_auto_review.json"
_PROMPTS = _EXPERIMENTS / "continuation_prompts_v1.csv"
_BASELINE = _EXPERIMENTS / "microworld_continuation_v1_2_outputs.csv"
_SENSE_MEMORY_PY = Path(__file__).resolve().parents[1] / "continuation" / "sense_memory.py"

_TRUE_UNSAFE_IDS = {
    "v1-081", "v1-082", "v1-083", "v1-085", "v1-086",
    "v1-088", "v1-089", "v1-090", "v1-091", "v1-092",
    "v1-093", "v1-094",
}
_NO_SAFE_REPAIRED_IDS = {"v1-051"}


def _load_auto_review() -> dict:
    return json.loads(_AUTO_REVIEW.read_text())


@lru_cache(maxsize=1)
def _run_overlay_cached() -> tuple[tuple[dict, ...], dict]:
    """Run overlay benchmark exactly once and cache rows + summary in memory."""
    import os
    from worldpgt.experiments.run_knowledge_overlay_benchmark_v1 import run

    with tempfile.TemporaryDirectory() as tmp:
        out_csv = os.path.join(tmp, "overlay.csv")
        out_json = os.path.join(tmp, "overlay.json")
        delta_csv = os.path.join(tmp, "delta.csv")
        summary = run(
            input_path=str(_PROMPTS),
            baseline_output_path=str(_BASELINE),
            auto_review_path=str(_AUTO_REVIEW),
            output_csv_path=out_csv,
            output_json_path=out_json,
            delta_csv_path=delta_csv,
        )
        with open(out_csv, newline="", encoding="utf-8") as f:
            rows = tuple(csv.DictReader(f))

    return rows, summary


def _overlay_rows() -> list[dict]:
    return list(_run_overlay_cached()[0])


def _overlay_summary() -> dict:
    return _run_overlay_cached()[1]


# ---------------------------------------------------------------------------
# 1. Overlay loads only accepted_auto items
# ---------------------------------------------------------------------------

def test_overlay_loads_only_accepted_auto_items():
    data = _load_auto_review()
    overlay = KnowledgeOverlay(data)
    expected = sum(
        1 for pr in data["proposal_reviews"]
        for item in pr["items"]
        if item.get("decision") == "accepted_auto"
    )
    assert overlay.stats["accepted_auto_items_loaded"] == expected


# ---------------------------------------------------------------------------
# 2. Overlay ignores needs_review items
# ---------------------------------------------------------------------------

def test_overlay_ignores_needs_review_items():
    synthetic = {
        "proposal_reviews": [{
            "term": "bank",
            "sense": "financial_institution",
            "items": [
                {"item_type": "positive_cue", "value": "vault",  "decision": "needs_review"},
                {"item_type": "positive_cue", "value": "ledger", "decision": "accepted_auto"},
            ],
        }]
    }
    overlay = KnowledgeOverlay(synthetic)
    assert overlay.stats["accepted_auto_items_loaded"] == 1
    cues = overlay.new_cues_for("bank", "financial_institution")
    assert "ledger" in cues
    assert "vault" not in cues


# ---------------------------------------------------------------------------
# 3. Overlay ignores rejected_auto items
# ---------------------------------------------------------------------------

def test_overlay_ignores_rejected_auto_items():
    synthetic = {
        "proposal_reviews": [{
            "term": "bat",
            "sense": "animal",
            "items": [
                {"item_type": "positive_cue", "value": "sonar", "decision": "rejected_auto"},
                {"item_type": "positive_cue", "value": "dusk",  "decision": "accepted_auto"},
            ],
        }]
    }
    overlay = KnowledgeOverlay(synthetic)
    assert overlay.stats["accepted_auto_items_loaded"] == 1
    cues = overlay.new_cues_for("bat", "animal")
    assert "dusk" in cues
    assert "sonar" not in cues


def test_overlay_ignores_all_non_accepted_decisions():
    synthetic = {
        "proposal_reviews": [{
            "term": "seal",
            "sense": "animal",
            "items": [
                {"item_type": "positive_cue", "value": "a", "decision": "needs_review"},
                {"item_type": "positive_cue", "value": "b", "decision": "rejected_auto"},
                {"item_type": "positive_cue", "value": "c", "decision": "pending"},
                {"item_type": "positive_cue", "value": "d", "decision": "accepted_auto"},
            ],
        }]
    }
    overlay = KnowledgeOverlay(synthetic)
    assert overlay.stats["accepted_auto_items_loaded"] == 1
    cues = overlay.new_cues_for("seal", "animal")
    assert cues == ["d"]


# ---------------------------------------------------------------------------
# 4. Overlay object is deterministic
# ---------------------------------------------------------------------------

def test_overlay_is_deterministic():
    data = _load_auto_review()
    o1 = KnowledgeOverlay(data)
    o2 = KnowledgeOverlay(data)
    assert o1.stats == o2.stats

    m1 = o1.build_overlay_memory()
    m2 = o2.build_overlay_memory()
    for term in m1.known_terms():
        for e1 in m1.get_senses(term):
            e2_list = [e for e in m2.get_senses(term) if e.sense_id == e1.sense_id]
            assert e2_list, f"Sense {term}:{e1.sense_id} missing from second overlay memory"
            assert sorted(e1.cues) == sorted(e2_list[0].cues), (
                f"Cue sets differ for {term}:{e1.sense_id} between two overlay builds"
            )


def test_overlay_memory_cue_sets_are_superset_of_builtin():
    data = _load_auto_review()
    overlay = KnowledgeOverlay(data)
    baseline = ExplicitSenseMemory(include_builtin=True)
    overlay_mem = overlay.build_overlay_memory()

    for term in baseline.known_terms():
        for base_entry in baseline.get_senses(term):
            ov_list = [e for e in overlay_mem.get_senses(term) if e.sense_id == base_entry.sense_id]
            assert ov_list, f"Overlay memory lost {term}:{base_entry.sense_id}"
            base_set = set(base_entry.cues)
            ov_set = set(ov_list[0].cues)
            assert base_set <= ov_set, (
                f"Overlay removed builtin cues for {term}:{base_entry.sense_id}"
            )


# ---------------------------------------------------------------------------
# 5. Overlay does not modify sense_memory.py
# ---------------------------------------------------------------------------

def test_overlay_does_not_modify_sense_memory_py_mtime():
    mtime_before = _SENSE_MEMORY_PY.stat().st_mtime
    data = _load_auto_review()
    overlay = KnowledgeOverlay(data)
    _ = overlay.build_overlay_memory()
    mtime_after = _SENSE_MEMORY_PY.stat().st_mtime
    assert mtime_before == mtime_after, "sense_memory.py mtime changed during overlay build"


def test_overlay_module_does_not_import_or_write_sense_memory_source():
    import worldpgt.knowledge.knowledge_overlay as mod
    src = Path(mod.__file__).read_text()
    assert "write_text" not in src
    assert ".write(" not in src or "f.write" not in src
    assert "open(" not in src or "sense_memory" not in src


# ---------------------------------------------------------------------------
# 6. Overlay does not modify baseline benchmark output
# ---------------------------------------------------------------------------

def test_overlay_run_never_opens_baseline_for_writing():
    import worldpgt.experiments.run_knowledge_overlay_benchmark_v1 as mod
    src = Path(mod.__file__).read_text()
    for line in src.splitlines():
        stripped = line.strip()
        if "baseline_output_path" in stripped and "open(" in stripped:
            assert '"w"' not in stripped and "'w'" not in stripped, (
                f"Baseline path opened for writing: {stripped!r}"
            )


def test_overlay_does_not_change_baseline_csv_content():
    content_before = _BASELINE.read_text()
    _ = _overlay_rows()
    content_after = _BASELINE.read_text()
    assert content_before == content_after, "Overlay run modified the trusted baseline CSV"


# ---------------------------------------------------------------------------
# 7. Overlay does not lower thresholds
# ---------------------------------------------------------------------------

def test_overlay_does_not_lower_thresholds():
    policy = ContinuationPolicy()
    assert policy.min_score == 1.0, f"min_score was lowered to {policy.min_score}"
    assert policy.min_margin == 1.0, f"min_margin was lowered to {policy.min_margin}"


def test_overlay_benchmark_module_contains_no_threshold_lowering_code():
    import worldpgt.experiments.run_knowledge_overlay_benchmark_v1 as mod
    src = Path(mod.__file__).read_text()
    forbidden = [
        "min_score = 0",
        "min_margin = 0",
        "lower_threshold",
        "reduce_threshold",
        "CONFIDENCE_THRESHOLD =",
        "TRUST_THRESHOLD =",
    ]
    for pat in forbidden:
        assert pat not in src, f"Threshold-lowering code {pat!r} found in benchmark runner"


# ---------------------------------------------------------------------------
# 8. Overlay does not weaken validators
# ---------------------------------------------------------------------------

def test_overlay_module_imports_no_validator_modules():
    import worldpgt.knowledge.knowledge_overlay as mod
    src = Path(mod.__file__).read_text()
    forbidden_imports = [
        "from worldpgt.continuation.surface_validator",
        "from worldpgt.continuation.subject_action_validator",
        "from worldpgt.continuation.prompt_tail_validator",
        "from worldpgt.continuation.continuation_policy",
        "import surface_validator",
        "import subject_action_validator",
        "import prompt_tail_validator",
        "import continuation_policy",
    ]
    for imp in forbidden_imports:
        assert imp not in src, f"Forbidden import {imp!r} in knowledge_overlay.py"


def test_overlay_benchmark_contains_no_validator_weakening():
    import worldpgt.experiments.run_knowledge_overlay_benchmark_v1 as mod
    src = Path(mod.__file__).read_text()
    # Check for code-level patterns that would indicate validator weakening
    # (docstring mentions like "not weakened" are intentionally excluded)
    forbidden_code = [
        "weaken_validator(",
        "bypass_validator(",
        "skip_validator(",
        "disable_validator(",
        "validator = None",
        ".validate = lambda",
    ]
    for tok in forbidden_code:
        assert tok not in src, f"Validator-weakening code {tok!r} found in benchmark runner"


# ---------------------------------------------------------------------------
# 9. Overlay does not add a generic output path (no unconditional continuation)
# ---------------------------------------------------------------------------

def test_overlay_module_contains_no_unconditional_continuation_code():
    import worldpgt.knowledge.knowledge_overlay as mod
    src = Path(mod.__file__).read_text()
    # The overlay only adds cues; it must NOT generate text or force continuations
    assert "continue_prompt" not in src
    assert "generate(" not in src


def test_overlay_benchmark_never_forces_continue_decision():
    import worldpgt.experiments.run_knowledge_overlay_benchmark_v1 as mod
    src = Path(mod.__file__).read_text()
    # Must not force or override the engine's decision
    assert 'decision = "continue"' not in src
    assert "verdict.decision = 'continue'" not in src
    assert 'row["decision"] = "continue"' not in src


# ---------------------------------------------------------------------------
# 10. Overlay trace markers are present for overlay-influenced rows
# ---------------------------------------------------------------------------

def test_overlay_trace_markers_present_for_influenced_rows():
    rows = _overlay_rows()
    for row in rows:
        hits = row.get("memory_hits", "")
        if "knowledge_overlay=enabled" not in hits:
            continue
        assert "overlay_source=knowledge_ingestion_v1_auto_review" in hits, (
            f"Row {row['id']}: missing overlay_source marker"
        )
        assert "overlay_item_count=" in hits, (
            f"Row {row['id']}: missing overlay_item_count marker"
        )
        assert "overlay_new_cues_matched_in_prompt=" in hits, (
            f"Row {row['id']}: missing overlay_new_cues_matched_in_prompt marker"
        )


def test_overlay_trace_markers_have_nonzero_item_count():
    rows = _overlay_rows()
    for row in rows:
        hits = row.get("memory_hits", "")
        if "knowledge_overlay=enabled" not in hits:
            continue
        # overlay_item_count must be > 0 for influenced rows
        for part in hits.split(" | "):
            if part.startswith("overlay_item_count="):
                count = int(part.split("=", 1)[1])
                assert count > 0, f"Row {row['id']}: overlay_item_count=0 on influenced row"


# ---------------------------------------------------------------------------
# 11. Overlay run writes separate output files with correct structure
# ---------------------------------------------------------------------------

def test_overlay_run_writes_separate_output_files(tmp_path):
    from worldpgt.experiments.run_knowledge_overlay_benchmark_v1 import run

    out_csv = tmp_path / "overlay.csv"
    out_json = tmp_path / "overlay.json"
    delta_csv = tmp_path / "delta.csv"

    summary = run(
        input_path=str(_PROMPTS),
        baseline_output_path=str(_BASELINE),
        auto_review_path=str(_AUTO_REVIEW),
        output_csv_path=str(out_csv),
        output_json_path=str(out_json),
        delta_csv_path=str(delta_csv),
    )

    assert out_csv.exists(), "Overlay CSV was not written"
    assert out_json.exists(), "Overlay summary JSON was not written"
    assert delta_csv.exists(), "Delta CSV was not written"

    csv_rows = list(csv.DictReader(out_csv.open()))
    assert len(csv_rows) == 120, f"Overlay CSV should have 120 rows, got {len(csv_rows)}"

    json_data = json.loads(out_json.read_text())
    for key in ("baseline", "overlay", "delta", "safety", "overlay_stats"):
        assert key in json_data, f"Missing key {key!r} in overlay summary JSON"

    delta_rows = list(csv.DictReader(delta_csv.open()))
    assert len(delta_rows) == 120, f"Delta CSV should have 120 rows, got {len(delta_rows)}"

    # The overlay output must NOT be the same path as the baseline output
    assert str(out_csv) != str(_BASELINE), "Overlay CSV overwrote the baseline"


# ---------------------------------------------------------------------------
# 12. Baseline metrics in summary match the trusted baseline (58/62/0/0)
# ---------------------------------------------------------------------------

def test_baseline_continue_count_is_58():
    assert _overlay_summary()["baseline"]["continue_count"] == 58


def test_baseline_audit_count_is_62():
    assert _overlay_summary()["baseline"]["audit_count"] == 62


def test_baseline_wrong_continue_count_is_0():
    assert _overlay_summary()["baseline"]["wrong_continue_count"] == 0


def test_baseline_semantic_quality_flagged_is_0():
    assert _overlay_summary()["baseline"]["semantic_quality_flagged"] == 0


# ---------------------------------------------------------------------------
# 13. Overlay summary counts are internally consistent (continue+audit=120)
# ---------------------------------------------------------------------------

def test_overlay_continue_plus_audit_equals_120():
    o = _overlay_summary()["overlay"]
    total = o["continue_count"] + o["audit_count"]
    assert total == 120, f"overlay continue+audit should be 120, got {total}"


# ---------------------------------------------------------------------------
# 14. True unsafe rows remain audited in overlay
# ---------------------------------------------------------------------------

def test_true_unsafe_rows_remain_audited_in_overlay():
    rows_by_id = {r["id"]: r for r in _overlay_rows()}
    for row_id in _TRUE_UNSAFE_IDS:
        row = rows_by_id.get(row_id)
        assert row is not None, f"Row {row_id} missing from overlay output"
        assert row["decision"] == "audit", (
            f"True-unsafe row {row_id} became {row['decision']!r} in overlay — safety regression"
        )
        assert row["continuation"] == "", (
            f"True-unsafe row {row_id} has non-empty continuation in overlay"
        )


def test_safety_flag_true_unsafe_rows_still_audited_is_true():
    assert _overlay_summary()["safety"]["true_unsafe_rows_still_audited"] is True


# ---------------------------------------------------------------------------
# 15. v1-051 (no_safe_repaired_candidate) remains audited in overlay
# ---------------------------------------------------------------------------

def test_v1_051_remains_audited_in_overlay():
    rows_by_id = {r["id"]: r for r in _overlay_rows()}
    row = rows_by_id["v1-051"]
    assert row["decision"] == "audit", "v1-051 became continue in overlay — safety regression"
    assert row["continuation"] == "", "v1-051 has non-empty continuation in overlay"


def test_safety_flag_no_safe_repaired_still_audited_is_true():
    assert _overlay_summary()["safety"]["no_safe_repaired_candidate_rows_still_audited"] is True


# ---------------------------------------------------------------------------
# 16. No semantic-quality flagged regressions in overlay
# ---------------------------------------------------------------------------

def test_overlay_semantic_quality_flagged_is_zero():
    assert _overlay_summary()["overlay"]["semantic_quality_flagged"] == 0, (
        "Overlay introduced semantic quality regressions — unsafe overlay"
    )


# ---------------------------------------------------------------------------
# Extra: No wrong continues introduced
# ---------------------------------------------------------------------------

def test_overlay_wrong_continue_count_is_zero():
    assert _overlay_summary()["overlay"]["wrong_continue_count"] == 0, (
        "Overlay produced wrong-sense continuations — unsafe overlay"
    )


def test_overlay_risk_regressions_list_is_empty():
    regressions = _overlay_summary()["delta"]["risk_regressions"]
    assert regressions == [], f"Risk regressions detected: {regressions}"


# ---------------------------------------------------------------------------
# Extra: Safety flags in summary are set correctly
# ---------------------------------------------------------------------------

def test_overlay_safety_flags_all_correct():
    safety = _overlay_summary()["safety"]
    assert safety["thresholds_changed"] is False
    assert safety["validators_weakened"] is False
    assert safety["sense_memory_modified"] is False
    assert safety["trusted_baseline_modified"] is False
    assert safety["true_unsafe_rows_still_audited"] is True
    assert safety["no_safe_repaired_candidate_rows_still_audited"] is True


# ---------------------------------------------------------------------------
# Extra: No neural / GPT / training imports in overlay modules
# ---------------------------------------------------------------------------

def test_no_neural_gpt_training_imports_in_overlay_knowledge_module():
    import worldpgt.knowledge.knowledge_overlay as mod
    src = Path(mod.__file__).read_text().lower()
    forbidden = [
        "torch", "transformers", "openai", "backprop",
        "fine-tun", "finetun", "gradient", "weight tensor",
        "neural network", "model.train", "model.eval",
        "sklearn", "tensorflow", "keras",
    ]
    for term in forbidden:
        assert term not in src, f"Forbidden ML/neural term {term!r} in knowledge_overlay.py"


def test_no_neural_gpt_training_imports_in_overlay_benchmark():
    import worldpgt.experiments.run_knowledge_overlay_benchmark_v1 as mod
    src = Path(mod.__file__).read_text().lower()
    forbidden = [
        "torch", "transformers", "openai", "backprop",
        "fine-tun", "finetun", "gradient", "neural network",
        "model.train", "model.eval", "sklearn", "tensorflow", "keras",
    ]
    for term in forbidden:
        assert term not in src, (
            f"Forbidden ML/neural term {term!r} in run_knowledge_overlay_benchmark_v1.py"
        )


# ---------------------------------------------------------------------------
# Extra: Overlay stats are plausible
# ---------------------------------------------------------------------------

def test_overlay_stats_accepted_auto_items_loaded_is_163():
    # Verified from knowledge_ingestion_v1_auto_review.json summary
    stats = _overlay_summary()["overlay_stats"]
    assert stats["accepted_auto_items_loaded"] == 163


def test_overlay_stats_positive_cues_loaded_is_104():
    stats = _overlay_summary()["overlay_stats"]
    assert stats["positive_cues_loaded"] == 104


def test_overlay_stats_counts_are_positive():
    stats = _overlay_summary()["overlay_stats"]
    assert stats["accepted_auto_items_loaded"] > 0
    assert stats["positive_cues_loaded"] > 0
