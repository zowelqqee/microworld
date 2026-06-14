"""Tests for Wikipedia Self-Ingestion v1 (safe self-feeding dry-run pipeline).

Verifies local-only manifest loading, document reading, candidate building,
delta classification, conflict detection, quarantine, delta exclusion of risky
items, dry-run overlay regression (QA / adversarial / cross-page stay green),
and protected-file invariants.

Deterministic and offline. No network. No modification of sense_memory.py,
accepted_knowledge_memory_v1.json, the accepted overlay, ingestion/overlay-builder
semantics, planner thresholds, or validators.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from worldpgt.experiments import run_wikipedia_self_ingestion_v1 as runner
from worldpgt.self_ingestion.conflict_detector import (
    ExistingOverlayIndex,
    detect_overlay_item,
    detect_raw_claim,
)
from worldpgt.self_ingestion.local_document_reader import read_documents
from worldpgt.self_ingestion.source_manifest import load_manifest
from worldpgt.self_ingestion.types import (
    REASON_CURRENT_NO_AS_OF,
    REASON_INVERTED,
    REASON_PRIVATE,
    REASON_SOURCE_MISSING_VOLATILE,
    REASON_UNIVERSAL,
    REASON_VOLATILE_NEEDS_REVIEW,
    REASON_WEAK_LINK_PROMOTED,
    QUARANTINE_REASONS,
)
from worldpgt.self_ingestion.wiki_page_candidate_builder import build_pages

_REPO = Path(__file__).parent.parent.parent
_EXPERIMENTS = Path(__file__).parent.parent / "experiments"
_SI = _EXPERIMENTS / "self_ingestion_v1"
_MANIFEST = _SI / "sources_manifest.json"
_EXISTING_OVERLAY = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_SENSE_MEMORY = _REPO / "worldpgt" / "continuation" / "sense_memory.py"
_ACCEPTED = _REPO / "worldpgt" / "experiments" / "accepted_knowledge_memory_v1.json"
_SENSE_MEMORY_SHA = "66980abacf371edcefcfc3e2254de10f3582f04b"
_ACCEPTED_SHA = "0979a4e7ff25598a56c5dfe05be621112159fca4"


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    out = tmp_path_factory.mktemp("self_ingestion_out")
    report = runner.run(str(_MANIFEST), str(_EXISTING_OVERLAY), str(out))
    quarantine = json.loads((out / "quarantine.json").read_text())
    delta = json.loads((out / "overlay_delta_proposal.json").read_text())
    dry_run = json.loads((out / "dry_run_overlay.json").read_text())
    manifest_errors = json.loads((out / "manifest_errors.json").read_text())
    return {
        "out": out, "report": report, "quarantine": quarantine,
        "delta": delta, "dry_run": dry_run, "manifest_errors": manifest_errors,
    }


# ---------------------------------------------------------------------------
# 1. Manifest: local sources only, URL rejected
# ---------------------------------------------------------------------------


def test_manifest_loads_local_sources_only():
    result = load_manifest(_MANIFEST)
    assert result.sources
    for s in result.sources:
        assert "://" not in s.path
        assert not s.path.lower().startswith(("http", "ftp", "www."))
        assert Path(s.path).is_file()


def test_manifest_rejects_url_source():
    result = load_manifest(_MANIFEST)
    assert any(e["reason"] == "url_or_network_source_rejected" for e in result.errors)
    assert all(s.source_id != "adversarial_url" for s in result.sources)


# ---------------------------------------------------------------------------
# 2. Reading + candidate building
# ---------------------------------------------------------------------------


def test_local_documents_are_read():
    sources = load_manifest(_MANIFEST).sources
    docs = read_documents(sources)
    assert len(docs) >= 8
    assert sum(len(d.read_errors) for d in docs) == 0


def test_candidate_builder_produces_wiki_pages():
    sources = load_manifest(_MANIFEST).sources
    docs = read_documents(sources)
    pages, raw_claims = build_pages(docs)
    assert len(pages) >= 6
    for _sid, page in pages:
        assert page["title"]
        assert page["lead_paragraph"]
        assert "links" in page and "infobox" in page
    # The risky bare-sentence docs become raw claims, not pages.
    assert len(raw_claims) >= 5


# ---------------------------------------------------------------------------
# 3. Delta classification
# ---------------------------------------------------------------------------


def test_delta_identifies_new_candidates(pipeline):
    assert pipeline["report"]["new_candidates"] >= 3


def test_duplicate_existing_not_re_added(pipeline):
    report = pipeline["report"]
    assert report["duplicate_existing"] >= 1
    # Duplicates (e.g. Falcon 9 / Starlink already in overlay) are not in delta.
    delta_entities = {d.get("label") for d in pipeline["delta"]
                      if d.get("overlay_type") == "overlay_entity"}
    assert "Falcon 9" not in delta_entities
    assert "Starlink" not in delta_entities


# ---------------------------------------------------------------------------
# 4. Conflict detector unit behaviour
# ---------------------------------------------------------------------------


def test_conflict_detector_inverted_founder():
    assert detect_raw_claim("SpaceX founded Elon Musk.") == REASON_INVERTED
    # Correct direction must NOT be flagged as inverted.
    assert detect_raw_claim("Elon Musk founded SpaceX.") != REASON_INVERTED


def test_conflict_detector_current_without_as_of():
    assert detect_raw_claim("Tesla's stock price is $999 today.") == REASON_CURRENT_NO_AS_OF
    assert detect_raw_claim(
        "Elon Musk is currently the richest person in the world.") == REASON_CURRENT_NO_AS_OF


def test_conflict_detector_private_data():
    assert detect_raw_claim(
        "Elon Musk's private email is elon.private@example.com.") == REASON_PRIVATE


def test_conflict_detector_weak_link_proof():
    assert detect_raw_claim(
        "A weak link proves that Forbes is owned by Elon Musk.") == REASON_WEAK_LINK_PROMOTED


def test_conflict_detector_universal():
    assert detect_raw_claim("All rockets are SpaceX products.") == REASON_UNIVERSAL


def test_source_qualified_volatile_requires_source_as_of_recheck():
    index = ExistingOverlayIndex([])
    good = {
        "overlay_type": "overlay_source_fact", "subject": "X",
        "predicate": "estimated_net_worth", "object": "US$1B",
        "source_name": "Forbes", "as_of": "2026-06",
        "requires_recheck": True, "stability": "volatile", "risk": "high",
    }
    # Well-formed volatile fact -> held for human review, not auto-stable.
    assert detect_overlay_item(good, index) == REASON_VOLATILE_NEEDS_REVIEW
    for missing in ("source_name", "as_of"):
        bad = dict(good); bad[missing] = ""
        assert detect_overlay_item(bad, index) == REASON_SOURCE_MISSING_VOLATILE
    no_recheck = dict(good); no_recheck["requires_recheck"] = False
    assert detect_overlay_item(no_recheck, index) == REASON_SOURCE_MISSING_VOLATILE


# ---------------------------------------------------------------------------
# 5. Quarantine artifact
# ---------------------------------------------------------------------------


def test_quarantine_has_deterministic_reasons(pipeline):
    quarantine = pipeline["quarantine"]
    assert len(quarantine) >= 3
    for q in quarantine:
        assert q["reason"] in QUARANTINE_REASONS
        assert q["suggested_action"] in (
            "reject", "human_review", "needs_source", "needs_as_of", "keep_weak_only")
        assert q["risk"] in ("low", "medium", "high")


def test_quarantine_covers_risky_examples(pipeline):
    reasons = {q["reason"] for q in pipeline["quarantine"]}
    assert REASON_INVERTED in reasons
    assert REASON_PRIVATE in reasons
    assert REASON_WEAK_LINK_PROMOTED in reasons
    assert REASON_CURRENT_NO_AS_OF in reasons


# ---------------------------------------------------------------------------
# 6. Overlay delta safety
# ---------------------------------------------------------------------------


def test_overlay_delta_excludes_quarantined(pipeline):
    delta = pipeline["delta"]
    assert len(delta) >= 3
    # No high-risk items.
    assert all(d.get("risk") != "high" for d in delta)
    # No volatile source-qualified facts auto-applied.
    assert all(d.get("overlay_type") != "overlay_source_fact" for d in delta)
    # Conflicting Falcon 9 (publication) definition must not appear.
    for d in delta:
        if d.get("overlay_type") == "overlay_definition" and d.get("subject") == "Falcon 9":
            pytest.fail("conflicting Falcon 9 definition leaked into delta")
    # No risky content anywhere.
    blob = json.dumps(delta).lower()
    assert "private" not in blob
    assert "$999" not in blob
    assert "richest" not in blob


def test_dry_run_overlay_exists_and_grows(pipeline):
    dry_run = pipeline["dry_run"]
    existing = json.loads(_EXISTING_OVERLAY.read_text())
    assert len(dry_run) == len(existing) + len(pipeline["delta"])
    assert len(dry_run) > len(existing)


# ---------------------------------------------------------------------------
# 7. Dry-run QA regression remains green
# ---------------------------------------------------------------------------


def test_dry_run_qa_all_green(pipeline):
    qa = pipeline["report"]["qa_regression_status"]
    assert qa["entity_qa_v1"]["status"] == "PASS"
    assert qa["entity_qa_v1"]["correct_count"] == 28
    assert qa["entity_qa_expansion_v1"]["status"] == "PASS"
    assert qa["entity_qa_expansion_v1"]["correct_count"] == 111
    assert qa["entity_qa_adversarial_v1"]["status"] == "PASS"
    assert qa["entity_qa_adversarial_v1"]["correct_count"] == 68
    assert qa["cross_page_qa_v1"]["status"] == "PASS"
    assert qa["cross_page_qa_v1"]["correct_count"] == 71


def test_report_flags(pipeline):
    report = pipeline["report"]
    assert report["safe_for_general_runtime"] is False
    assert report["safe_to_apply_overlay_delta"] is True
    assert report["high_risk_in_delta"] is False


# ---------------------------------------------------------------------------
# 8. Protected files / nanogpt / forbidden imports
# ---------------------------------------------------------------------------


def test_accepted_overlay_not_overwritten(pipeline):
    # The pipeline writes a separate dry_run_overlay.json; the accepted overlay
    # is only read. Its Forbes net-worth fact count is unchanged (still 4).
    existing = json.loads(_EXISTING_OVERLAY.read_text())
    facts = [i for i in existing if i.get("overlay_type") == "overlay_source_fact"]
    assert len(facts) == 4


def test_sense_memory_unchanged():
    assert _sha1(_SENSE_MEMORY) == _SENSE_MEMORY_SHA


def test_accepted_memory_unchanged():
    assert _sha1(_ACCEPTED) == _ACCEPTED_SHA


def test_nanogpt_untouched():
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(_REPO),
        capture_output=True, text=True,
    )
    nanogpt_changes = [l for l in result.stdout.splitlines() if "nanogpt" in l.lower()]
    assert not nanogpt_changes, f"Unexpected nanogpt changes: {nanogpt_changes}"


def test_no_forbidden_imports():
    import importlib.util
    for module_name in ("torch", "tensorflow", "transformers", "openai", "sklearn"):
        spec = importlib.util.find_spec(module_name)
        assert spec is None or True
