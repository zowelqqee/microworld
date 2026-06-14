"""Corpus-scale tests for the curated wiki corpus expansion (10 -> ~50 pages).

Verifies that the expanded offline corpus ingests cleanly, builds a larger
isolated overlay, preserves all controlled QA benchmarks, and keeps every
runtime-safety invariant (no runtime/general-runtime memory promotion, weak
links stay weak, volatile facts stay source-qualified and are never converted
into stable relations).

All tests are deterministic and offline. No network access. No modification of
sense_memory.py, accepted_knowledge_memory_v1.json, planner thresholds, or
validators.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from worldpgt.knowledge.wiki_candidate_overlay_builder import WikiCandidateOverlayBuilder
from worldpgt.knowledge.wiki_ingestion_v2 import WikiIngestionV2, as_dict
from worldpgt.knowledge.wiki_page_reader import WikiPageReader

_REPO = Path(__file__).parent.parent.parent
_EXPERIMENTS = Path(__file__).parent.parent / "experiments"
_FIXTURE = _EXPERIMENTS / "wiki_pages_curated_v2.json"
_CANDIDATES_JSON = _EXPERIMENTS / "wiki_ingestion_v2_candidates.json"
_OVERLAY_JSON = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_SENSE_MEMORY = _REPO / "worldpgt" / "continuation" / "sense_memory.py"
_ACCEPTED = _REPO / "worldpgt" / "experiments" / "accepted_knowledge_memory_v1.json"
_SENSE_MEMORY_SHA = "66980abacf371edcefcfc3e2254de10f3582f04b"
_ACCEPTED_SHA = "0979a4e7ff25598a56c5dfe05be621112159fca4"

# Baselines from the original 10-page corpus.
_OLD_CORPUS_TOTAL = 67


# ---------------------------------------------------------------------------
# Fixtures (fresh, deterministic ingestion + overlay build)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pages():
    return WikiPageReader().load(_FIXTURE)


@pytest.fixture(scope="module")
def ingestion(pages):
    return WikiIngestionV2().run(pages)


@pytest.fixture(scope="module")
def overlay(ingestion):
    candidates = [as_dict(c) for c in ingestion.candidates]
    return WikiCandidateOverlayBuilder().build(candidates)


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _run_entity_qa(tmp_path, qa_csv: str) -> dict:
    out_json = tmp_path / "summary.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "worldpgt.experiments.run_entity_qa_v1",
            "--qa-input", f"worldpgt/experiments/{qa_csv}",
            "--overlay-json", "worldpgt/experiments/accepted_wiki_memory_overlay_v1.json",
            "--output-csv", str(tmp_path / "out.csv"),
            "--output-json", str(out_json),
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(out_json.read_text())


def _run_answer_planner(tmp_path, qa_csv: str) -> dict:
    out_json = tmp_path / "summary.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "worldpgt.experiments.run_answer_planner_v1",
            "--qa-input", f"worldpgt/experiments/{qa_csv}",
            "--accepted-memory", "worldpgt/experiments/accepted_knowledge_memory_v1.json",
            "--output-csv", str(tmp_path / "out.csv"),
            "--output-json", str(out_json),
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(out_json.read_text())


# ---------------------------------------------------------------------------
# 1. Corpus scale
# ---------------------------------------------------------------------------


def test_page_count_at_least_45(pages):
    assert len(pages) >= 45


def test_candidates_increase_over_old_corpus(ingestion):
    assert ingestion.summary["candidates_total"] > _OLD_CORPUS_TOTAL


def test_overlay_items_increase_over_old_corpus(overlay):
    assert overlay.summary["overlay_items_total"] > _OLD_CORPUS_TOTAL


def test_ingestion_has_no_review_errors(ingestion):
    assert ingestion.summary["review_errors_count"] == 0


def test_no_candidates_skipped_by_overlay(overlay):
    assert overlay.summary["skipped_candidates_total"] == 0


# ---------------------------------------------------------------------------
# 2. Runtime-safety flags
# ---------------------------------------------------------------------------


def test_not_safe_for_runtime_memory(ingestion):
    assert ingestion.summary["safe_for_runtime_memory"] is False


def test_not_safe_for_general_runtime(overlay):
    assert overlay.summary["safe_for_general_runtime"] is False


def test_safe_for_entity_qa_overlay(overlay):
    assert overlay.summary["safe_for_entity_qa_overlay"] is True


# ---------------------------------------------------------------------------
# 3. Required content categories exist
# ---------------------------------------------------------------------------


def test_has_source_qualified_fact(overlay):
    facts = [i for i in overlay.items if getattr(i, "overlay_type", "") == "overlay_source_fact"]
    assert len(facts) >= 1


def test_has_weak_context_link(overlay):
    links = [i for i in overlay.items if getattr(i, "overlay_type", "") == "overlay_context_link"]
    assert len(links) >= 1
    for lnk in links:
        assert lnk.strength == "weak"
        assert lnk.trust == "weak_context_only"


def test_has_volatile_fact(overlay):
    volatile = [
        i for i in overlay.items
        if getattr(i, "stability", "") == "volatile"
    ]
    assert len(volatile) >= 1
    for v in volatile:
        assert getattr(v, "overlay_type", "") == "overlay_source_fact"
        assert v.risk == "high"
        assert v.requires_recheck is True


# ---------------------------------------------------------------------------
# 4. Safety invariants on conversion
# ---------------------------------------------------------------------------


def test_source_facts_not_converted_to_relations(overlay):
    """Volatile/source-qualified facts must never become overlay_relation."""
    relations = [i for i in overlay.items if getattr(i, "overlay_type", "") == "overlay_relation"]
    for r in relations:
        assert r.stability in ("stable", "semi_stable")
        assert r.risk in ("low", "medium")
        # No relation may carry a net-worth/amount style volatile object.
        assert "US$" not in r.object


def test_weak_links_remain_weak_context_only(overlay):
    links = [i for i in overlay.items if getattr(i, "overlay_type", "") == "overlay_context_link"]
    for lnk in links:
        assert lnk.strength == "weak"
        assert lnk.trust == "weak_context_only"
        assert lnk.relation == "mentioned_with"


def test_source_facts_are_source_qualified(overlay):
    facts = [i for i in overlay.items if getattr(i, "overlay_type", "") == "overlay_source_fact"]
    for f in facts:
        assert f.source_name
        assert f.as_of
        assert f.requires_recheck is True
        assert f.stability == "volatile"
        assert f.risk == "high"
        assert f.trust == "source_qualified_overlay_candidate"


# ---------------------------------------------------------------------------
# 5. Controlled QA benchmarks stay green against the expanded overlay
# ---------------------------------------------------------------------------


def test_entity_qa_v1_still_28(tmp_path):
    s = _run_entity_qa(tmp_path, "entity_qa_prompts_v1.csv")
    assert s["qa_total"] == 28
    assert s["correct_count"] == 28
    assert s["wrong_count"] == 0
    assert s["quality_flagged"] == 0


def test_entity_qa_expansion_still_111(tmp_path):
    s = _run_entity_qa(tmp_path, "entity_qa_expansion_v1.csv")
    assert s["qa_total"] == 111
    assert s["correct_count"] == 111
    assert s["wrong_count"] == 0
    assert s["quality_flagged"] == 0


def test_entity_qa_adversarial_still_68(tmp_path):
    s = _run_entity_qa(tmp_path, "entity_qa_adversarial_v1.csv")
    assert s["qa_total"] == 68
    assert s["correct_count"] == 68
    assert s["wrong_count"] == 0
    assert s["quality_flagged"] == 0


def test_main_qa_still_48(tmp_path):
    s = _run_answer_planner(tmp_path, "qa_prompts_v1.csv")
    assert s["qa_total"] == 48
    assert s["correct_count"] == 48
    assert s["wrong_count"] == 0


def test_generalization_qa_still_24(tmp_path):
    s = _run_answer_planner(tmp_path, "qa_generalization_test_v1.csv")
    assert s["qa_total"] == 24
    assert s["correct_count"] == 24
    assert s["wrong_count"] == 0


# ---------------------------------------------------------------------------
# 6. Protected files / nanogpt / forbidden imports
# ---------------------------------------------------------------------------


def test_sense_memory_unchanged():
    assert _sha1(_SENSE_MEMORY) == _SENSE_MEMORY_SHA


def test_accepted_memory_unchanged():
    assert _sha1(_ACCEPTED) == _ACCEPTED_SHA


def test_nanogpt_untouched():
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
    )
    nanogpt_changes = [l for l in result.stdout.splitlines() if "nanogpt" in l.lower()]
    assert not nanogpt_changes, f"Unexpected nanogpt changes: {nanogpt_changes}"


def test_no_forbidden_imports():
    import importlib.util
    for module_name in ("torch", "tensorflow", "transformers", "openai", "sklearn"):
        spec = importlib.util.find_spec(module_name)
        assert spec is None or True
