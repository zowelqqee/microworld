"""Tests for Working Context Pack v1 (read-only probe layer).

Deterministic and offline. No network. Confirms entity matching, stable-path vs
weak-only vs source-qualified separation, safety blocking of
inversion/universal/private/current-live cases, validator catching synthetic
unsafe packs, and that no protected file / runtime behavior is touched.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from worldpgt.experiments import run_context_pack_probe_v1 as runner
from worldpgt.context_pack.context_pack_builder import (
    ContextPackBuilder,
    load_knowledge_requests,
)
from worldpgt.context_pack.context_pack_validator import validate_pack
from worldpgt.context_pack.entity_matcher import match_entities
from worldpgt.context_pack.context_pack_builder import _OverlayAccess
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider
from worldpgt.context_pack.types import (
    OVERLAY_ACCEPTED,
    OVERLAY_PROMOTED,
    ContextRelation,
    ContextSourceFact,
    WorkingContextPack,
)

_REPO = Path(__file__).resolve().parent.parent.parent
_EXPERIMENTS = _REPO / "worldpgt" / "experiments"
_ACCEPTED = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_PROMOTED = (
    _EXPERIMENTS / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json"
)
_KREQUESTS = _EXPERIMENTS / "knowledge_requests_v1" / "candidate_knowledge_requests.json"

_PROTECTED_FILES = [
    _EXPERIMENTS / "accepted_knowledge_memory_v1.json",
    _ACCEPTED,
    _PROMOTED,
    _REPO / "worldpgt" / "continuation" / "sense_memory.py",
]

_FORBIDDEN_MODULES = {
    "torch",
    "tensorflow",
    "numpy",
    "transformers",
    "openai",
    "requests",
    "urllib",
    "urllib2",
    "urllib3",
    "http",
    "socket",
    "nanogpt",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _builder():
    return ContextPackBuilder(
        str(_ACCEPTED),
        overlay_mode=OVERLAY_ACCEPTED,
        knowledge_requests=load_knowledge_requests(_KREQUESTS),
    )


@pytest.fixture(scope="module")
def builder():
    return _builder()


@pytest.fixture(scope="module")
def overlay_access():
    return _OverlayAccess(WikiMemoryOverlayProvider(str(_ACCEPTED)))


@pytest.fixture(scope="module")
def probe_result(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("context_pack_out")
    return runner.run_probe(overlay_mode=OVERLAY_ACCEPTED, out_dir=out_dir, write=True)


# --------------------------------------------------------------------------- #
# 1-2. Entity matcher.
# --------------------------------------------------------------------------- #
def test_matcher_detects_known_entities(overlay_access):
    names = {m.name for m in match_entities("Who founded SpaceX?", overlay_access)}
    assert "SpaceX" in names


def test_matcher_avoids_tiny_generic_tokens(overlay_access):
    matched = match_entities("Is a to of the current price?", overlay_access)
    assert matched == []


# --------------------------------------------------------------------------- #
# 3. Stable/semi-stable path for SpaceX/rockets.
# --------------------------------------------------------------------------- #
def test_rockets_path_has_explicit_stable_path(builder):
    pack = builder.build("How is Elon Musk connected to rockets?")
    assert pack.candidate_paths
    path = pack.candidate_paths[0]
    assert path.nodes[0].lower().startswith("elon musk")
    assert path.nodes[-1].lower() == "rockets"
    assert pack.safe_for_answering is True


# --------------------------------------------------------------------------- #
# 4-5. Starlink / Forbes weak links stay weak and block weak-only inference.
# --------------------------------------------------------------------------- #
def test_starlink_weak_only_blocked(builder):
    pack = builder.build("How is Elon Musk connected to Starlink?")
    assert any(b.reason == "weak_only_path" for b in pack.blocked_paths)
    assert pack.candidate_paths == []
    assert pack.safe_for_answering is False
    # Starlink only appears via a weak context link, never a direct relation.
    assert all("starlink" not in r.object.lower() for r in pack.direct_relations)


def test_forbes_link_stays_weak(builder):
    pack = builder.build("Why is Forbes linked to Elon Musk?")
    assert pack.weak_links
    assert any("forbes" in (w.source_page + w.surface + w.target).lower() for w in pack.weak_links)
    # No stable relation between Forbes and Elon Musk is inferred.
    assert pack.safe_for_answering is False


# --------------------------------------------------------------------------- #
# 6. Forbes net-worth estimate stays source-qualified volatile with recheck.
# --------------------------------------------------------------------------- #
def test_forbes_estimate_source_qualified(builder):
    pack = builder.build("What does Forbes estimate about Elon Musk?")
    assert pack.source_facts
    sf = pack.source_facts[0]
    assert sf.stability == "volatile"
    assert sf.requires_recheck is True
    assert sf.source_name
    assert pack.safe_for_answering is False
    assert any("source_qualified_volatile_fact" in n for n in pack.safety_notes)


# --------------------------------------------------------------------------- #
# 7. Tesla current stock price not answered as stable context.
# --------------------------------------------------------------------------- #
def test_tesla_current_stock_price_not_stable(builder):
    pack = builder.build("What is Tesla's current stock price?")
    assert pack.safe_for_answering is False
    assert any("current_or_live_question" in n for n in pack.safety_notes)


# --------------------------------------------------------------------------- #
# 8. SpaceX founded Elon Musk inversion blocked/safety-noted.
# --------------------------------------------------------------------------- #
def test_inversion_blocked(builder):
    pack = builder.build("Did SpaceX found Elon Musk?")
    assert any(b.reason == "relation_inversion" for b in pack.blocked_paths)
    assert pack.candidate_paths == []
    assert pack.safe_for_answering is False


# --------------------------------------------------------------------------- #
# 9. Unsupported universal claim blocked/safety-noted.
# --------------------------------------------------------------------------- #
def test_universal_blocked(builder):
    pack = builder.build("All rockets are SpaceX products.")
    assert any(b.reason == "unsupported_universal_claim" for b in pack.blocked_paths)
    assert pack.safe_for_answering is False


# --------------------------------------------------------------------------- #
# 10. Private/sensitive question produces no stable context.
# --------------------------------------------------------------------------- #
def test_private_question_no_stable_context(builder):
    pack = builder.build("What is Elon Musk's home address?")
    assert pack.direct_relations == []
    assert pack.candidate_paths == []
    assert pack.safe_for_answering is False
    assert any("private_or_sensitive_question" in n for n in pack.safety_notes)


# --------------------------------------------------------------------------- #
# 11-12. Weak links never make answering safe; source facts never stable.
# --------------------------------------------------------------------------- #
def test_weak_links_never_safe(probe_result):
    for pack, _validation, _rendered in probe_result["packs"]:
        if pack.weak_links and not pack.candidate_paths and not any(
            r.stability in ("stable", "semi_stable") and r.trust != "weak_context_only"
            for r in pack.direct_relations
        ):
            assert pack.safe_for_answering is False


def test_source_facts_never_in_direct_relations(probe_result):
    for pack, _v, _r in probe_result["packs"]:
        for r in pack.direct_relations:
            assert r.trust != "weak_context_only"
            assert r.stability != "volatile"
            assert not r.trust.startswith("source_qualified")


# --------------------------------------------------------------------------- #
# 13. Missing knowledge hints attach when matching candidate requests.
# --------------------------------------------------------------------------- #
def test_missing_knowledge_hints_attach(builder):
    pack = builder.build("How is Elon Musk connected to Starlink?")
    assert pack.missing_knowledge_hints
    assert all(h.must_remain_audit_until_supported for h in pack.missing_knowledge_hints)


# --------------------------------------------------------------------------- #
# 14-15. Both overlay modes work.
# --------------------------------------------------------------------------- #
def test_accepted_mode_works(probe_result):
    assert probe_result["summary"]["overlay_mode"] == OVERLAY_ACCEPTED
    assert probe_result["summary"]["probe_total"] == 10


def test_promoted_mode_works(tmp_path):
    res = runner.run_probe(overlay_mode=OVERLAY_PROMOTED, out_dir=tmp_path / "out", write=True)
    assert res["summary"]["overlay_mode"] == OVERLAY_PROMOTED
    assert res["summary"]["validation_passed_count"] == res["summary"]["probe_total"]


# --------------------------------------------------------------------------- #
# 16-17. Validator catches synthetic unsafe packs.
# --------------------------------------------------------------------------- #
def test_validator_catches_weak_link_as_stable():
    bad = WorkingContextPack(question="x", overlay_mode=OVERLAY_ACCEPTED)
    bad.direct_relations.append(
        ContextRelation(
            subject="SpaceX",
            predicate="linked",
            object="Starlink",
            stability="semi_stable",
            risk="low",
            trust="weak_context_only",  # weak link smuggled into stable relations
        )
    )
    result = validate_pack(bad)
    assert not result.passed
    assert any("weak link" in f for f in result.failures)


def test_validator_catches_volatile_as_stable():
    bad = WorkingContextPack(question="x", overlay_mode=OVERLAY_ACCEPTED)
    bad.direct_relations.append(
        ContextRelation(
            subject="Elon Musk",
            predicate="estimated_net_worth",
            object="US$1.1 trillion",
            stability="volatile",  # volatile fact smuggled into stable relations
            risk="high",
            trust="source_qualified_overlay_candidate",
        )
    )
    result = validate_pack(bad)
    assert not result.passed
    assert any("volatile" in f for f in result.failures)


# --------------------------------------------------------------------------- #
# 18-20. Summary safety flags.
# --------------------------------------------------------------------------- #
def test_summary_safe_for_general_runtime_false(probe_result):
    assert probe_result["summary"]["safe_for_general_runtime"] is False
    assert probe_result["summary"]["runtime_behavior_modified"] is False
    assert probe_result["summary"]["network_calls"] is False


def test_summary_trusted_memory_unchanged(probe_result):
    assert probe_result["summary"]["trusted_memory_modified"] is False


def test_summary_overlays_unchanged(probe_result):
    assert probe_result["summary"]["accepted_overlay_modified"] is False
    assert probe_result["summary"]["promoted_overlay_modified"] is False


def test_artifacts_written(tmp_path):
    out_dir = tmp_path / "out"
    runner.run_probe(overlay_mode=OVERLAY_ACCEPTED, out_dir=out_dir, write=True)
    for fname in (
        "context_pack_probe_outputs.json",
        "context_pack_probe_outputs.csv",
        "context_pack_probe_summary.json",
    ):
        path = out_dir / fname
        assert path.is_file() and path.stat().st_size > 0


# --------------------------------------------------------------------------- #
# 21. No protected files modified by a run.
# --------------------------------------------------------------------------- #
def test_protected_files_unchanged(tmp_path):
    before = {p: _hash(p) for p in _PROTECTED_FILES}
    runner.run_probe(overlay_mode=OVERLAY_ACCEPTED, out_dir=tmp_path / "a", write=True)
    runner.run_probe(overlay_mode=OVERLAY_PROMOTED, out_dir=tmp_path / "b", write=True)
    after = {p: _hash(p) for p in _PROTECTED_FILES}
    assert before == after


# --------------------------------------------------------------------------- #
# 22-23. No forbidden imports; nanogpt untouched / unreferenced.
# --------------------------------------------------------------------------- #
def _imported_modules(src: Path) -> set:
    tree = ast.parse(src.read_text(encoding="utf-8"))
    mods: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.add(node.module.split(".")[0])
    return mods


def test_no_forbidden_imports():
    package = _REPO / "worldpgt" / "context_pack"
    sources = list(package.glob("*.py"))
    sources.append(_EXPERIMENTS / "run_context_pack_probe_v1.py")
    for src in sources:
        offending = _imported_modules(src) & _FORBIDDEN_MODULES
        assert not offending, f"forbidden imports {offending} in {src.name}"
