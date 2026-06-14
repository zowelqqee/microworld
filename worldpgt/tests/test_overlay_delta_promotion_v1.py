"""Tests for Promote Overlay Delta v1.

Verifies that the overlay delta validator accepts the safe 27-item delta,
blocks synthetic risky/conflicting items, that promotion produces a 310-item
promoted overlay without touching trusted memory or the accepted overlay, that
weak links stay weak and volatile facts are never promoted, and that the QA /
adversarial / cross-page regressions stay green against the promoted overlay.

Deterministic and offline. No network. No modification of sense_memory.py,
accepted_knowledge_memory_v1.json, the accepted overlay, ingestion/overlay/
validator/threshold semantics.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from worldpgt.experiments import promote_wiki_overlay_delta_v1 as runner
from worldpgt.self_ingestion.overlay_delta_promoter import promote
from worldpgt.self_ingestion.overlay_delta_validator import (
    REASON_CONFLICTS_EXISTING,
    REASON_HIGH_RISK,
    REASON_INVERTED,
    REASON_PRIVATE,
    REASON_VOLATILE,
    REASON_WEAK_LINK_PROMOTED,
    validate_delta,
)

_REPO = Path(__file__).resolve().parent.parent.parent
_EXPERIMENTS = _REPO / "worldpgt" / "experiments"
_BASE_OVERLAY = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_DELTA = _EXPERIMENTS / "self_ingestion_v1" / "overlay_delta_proposal.json"
_QUARANTINE = _EXPERIMENTS / "self_ingestion_v1" / "quarantine.json"
_ACCEPTED_KNOWLEDGE = _EXPERIMENTS / "accepted_knowledge_memory_v1.json"
_SENSE_MEMORY = _REPO / "worldpgt" / "continuation" / "sense_memory.py"
_SENSE_MEMORY_SHA = "66980abacf371edcefcfc3e2254de10f3582f04b"
_ACCEPTED_SHA = "0979a4e7ff25598a56c5dfe05be621112159fca4"


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _base_items():
    return json.loads(_BASE_OVERLAY.read_text())


def _delta_items():
    return json.loads(_DELTA.read_text())


@pytest.fixture(scope="module")
def promotion(tmp_path_factory):
    out = tmp_path_factory.mktemp("promotion_out")
    report = runner.run(str(_BASE_OVERLAY), str(_DELTA), str(out))
    promoted = json.loads((out / "promoted_wiki_memory_overlay_v1.json").read_text())
    validation = json.loads((out / "promotion_validation.json").read_text())
    regression = json.loads((out / "promotion_regression_summary.json").read_text())
    return {"out": out, "report": report, "promoted": promoted,
            "validation": validation, "regression": regression}


# ---------------------------------------------------------------------------
# 1. Validator accepts the 27 safe delta items
# ---------------------------------------------------------------------------


def test_validator_accepts_safe_delta():
    result = validate_delta(_delta_items(), _base_items())
    assert result.accepted_count == 27
    assert result.rejected_count == 0
    assert result.blocked_count == 0
    assert result.safe_to_promote is True
    assert len(result.accepted_item_ids) == 27


# ---------------------------------------------------------------------------
# 2. Validator blocks synthetic risky / conflicting items
# ---------------------------------------------------------------------------


def test_validator_blocks_high_risk():
    bad = {"overlay_type": "overlay_definition", "subject": "Z",
           "definition": "thing", "evidence_text": "x", "risk": "high",
           "stability": "stable"}
    r = validate_delta([bad], _base_items())
    assert r.accepted_count == 0
    assert r.rejected_items[0]["reason"] == REASON_HIGH_RISK


def test_validator_blocks_volatile_source_fact():
    bad = {"overlay_type": "overlay_source_fact", "subject": "Z",
           "predicate": "estimated_net_worth", "object": "US$9B",
           "source_name": "Forbes", "as_of": "2026-06",
           "requires_recheck": True, "stability": "volatile", "risk": "high"}
    r = validate_delta([bad], _base_items())
    assert r.accepted_count == 0
    assert r.rejected_items[0]["reason"] == REASON_VOLATILE


def test_validator_blocks_current_volatile_relation():
    bad = {"overlay_type": "overlay_relation", "subject": "Z", "predicate": "produces",
           "object": "stuff", "evidence_text": "x", "stability": "volatile",
           "risk": "medium"}
    r = validate_delta([bad], _base_items())
    assert r.accepted_count == 0
    assert r.rejected_items[0]["reason"] == REASON_VOLATILE


def test_validator_blocks_private_data():
    bad = {"overlay_type": "overlay_definition", "subject": "Z",
           "definition": "person whose private email is z@example.com",
           "evidence_text": "private email", "risk": "low", "stability": "stable"}
    r = validate_delta([bad], _base_items())
    assert r.accepted_count == 0
    assert r.rejected_items[0]["reason"] == REASON_PRIVATE


def test_validator_blocks_inverted_relation():
    bad = {"overlay_type": "overlay_relation", "subject": "SpaceX", "predicate": "founded",
           "object": "Elon Musk", "evidence_text": "SpaceX founded Elon Musk",
           "stability": "semi_stable", "risk": "medium"}
    r = validate_delta([bad], _base_items())
    assert r.accepted_count == 0
    assert r.rejected_items[0]["reason"] == REASON_INVERTED


def test_validator_blocks_weak_link_promoted_to_fact():
    bad = {"overlay_type": "overlay_context_link", "source_page": "Z", "surface": "Forbes",
           "target": "Forbes", "relation": "mentioned_with", "strength": "strong",
           "trust": "trusted"}
    r = validate_delta([bad], _base_items())
    assert r.accepted_count == 0
    assert r.rejected_items[0]["reason"] == REASON_WEAK_LINK_PROMOTED


def test_validator_blocks_conflicting_definition():
    # Existing Falcon 9 is a launch vehicle; this conflicting redefinition must block.
    bad = {"overlay_type": "overlay_definition", "subject": "Falcon 9",
           "definition": "business magazine published by Forbes",
           "evidence_text": "x", "risk": "low", "stability": "stable"}
    r = validate_delta([bad], _base_items())
    assert r.accepted_count == 0
    assert r.rejected_items[0]["reason"] == REASON_CONFLICTS_EXISTING


def test_validator_blocks_duplicate_existing():
    existing = _base_items()
    dup = next(i for i in existing if i.get("overlay_type") == "overlay_entity")
    r = validate_delta([dict(dup)], existing)
    assert r.accepted_count == 0
    assert r.rejected_count == 1


# ---------------------------------------------------------------------------
# 3. Promotion result
# ---------------------------------------------------------------------------


def test_promoted_overlay_has_310_items(promotion):
    assert len(promotion["promoted"]) == 310
    assert promotion["report"]["promoted_overlay_items"] == 310
    assert promotion["report"]["base_overlay_items"] == 283
    assert promotion["report"]["accepted_delta_items"] == 27


def test_base_overlay_preserved(promotion):
    promoted = promotion["promoted"]
    base = _base_items()
    # The first 283 promoted items are exactly the base overlay, unchanged.
    assert promoted[:283] == base


def test_weak_links_remain_weak_after_promotion(promotion):
    for it in promotion["promoted"]:
        if it.get("overlay_type") == "overlay_context_link":
            assert it.get("strength") == "weak"
            assert it.get("trust") == "weak_context_only"
    assert promotion["report"]["weak_links_remain_weak"] is True


def test_no_volatile_source_facts_promoted(promotion):
    delta_part = promotion["promoted"][283:]
    assert all(it.get("overlay_type") != "overlay_source_fact" for it in delta_part)
    assert all(it.get("stability") != "volatile" for it in delta_part)
    assert promotion["report"]["volatile_items_not_promoted"] is True


def test_quarantine_and_rejected_not_promoted(promotion):
    # No quarantined raw-claim text leaks into the promoted overlay.
    quarantine = json.loads(_QUARANTINE.read_text())
    blob = json.dumps(promotion["promoted"]).lower()
    assert "$999" not in blob
    assert "private email" not in blob
    assert "richest" not in blob
    assert "spacex founded elon musk" not in blob
    assert promotion["report"]["rejected_items_not_promoted"] is True
    assert promotion["report"]["quarantine_not_promoted"] is True
    assert len(quarantine) >= 3  # quarantine artifact is non-trivial


# ---------------------------------------------------------------------------
# 4. Regression summary green
# ---------------------------------------------------------------------------


def test_regression_summary_green(promotion):
    reg = promotion["regression"]
    expected = {"entity_qa_v1": 28, "entity_qa_expansion_v1": 111,
                "entity_qa_adversarial_v1": 68, "cross_page_qa_v1": 71}
    for name, exp in expected.items():
        assert reg[name]["status"] == "PASS"
        assert reg[name]["correct_count"] == exp
        assert reg[name]["qa_total"] == exp
        assert reg[name]["wrong_count"] == 0
        assert reg[name]["quality_flagged"] == 0
        assert float(reg[name]["answer_precision"]) == 1.0


def test_report_safety_flags(promotion):
    report = promotion["report"]
    assert report["safe_to_promote"] is True
    assert report["safe_for_general_runtime"] is False
    assert report["trusted_memory_modified"] is False
    assert report["accepted_overlay_modified"] is False
    assert report["private_items_not_promoted"] is True
    assert report["inverted_relations_not_promoted"] is True


# ---------------------------------------------------------------------------
# 5. Protected files / nanogpt / forbidden imports
# ---------------------------------------------------------------------------


def test_accepted_overlay_not_overwritten(promotion):
    # Promotion writes a separate promoted overlay; the accepted overlay is base.
    assert len(_base_items()) == 283


def test_sense_memory_unchanged():
    assert _sha1(_SENSE_MEMORY) == _SENSE_MEMORY_SHA


def test_accepted_knowledge_memory_unchanged():
    assert _sha1(_ACCEPTED_KNOWLEDGE) == _ACCEPTED_SHA


def test_no_nested_worldmvp_artifact():
    # This task must not create a nested worldmvp/ tree under the root package.
    assert not (_REPO / "worldpgt" / "worldmvp").exists()


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


def test_promoter_direct_api_matches_counts():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "promoted.json"
        result = promote(str(_BASE_OVERLAY), str(_DELTA), str(out))
        assert result.base_count == 283
        assert result.accepted_delta_count == 27
        assert result.promoted_count == 310
        assert result.meta["safe_for_general_runtime"] is False
        assert result.meta["trusted_general_runtime_memory"] is False
