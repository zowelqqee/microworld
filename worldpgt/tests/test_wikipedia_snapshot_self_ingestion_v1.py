"""Tests for Wikipedia Snapshot Self-Ingestion v1.

Offline and deterministic. These tests exercise the proposal pipeline without
network calls and without writing accepted/promoted runtime memory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from worldpgt.experiments import run_wikipedia_snapshot_self_ingestion_v1 as runner
from worldpgt.wiki_snapshot_ingestion.ready_snapshot_loader import load_ready_snapshot_docs
from worldpgt.wiki_snapshot_ingestion.snapshot_batch_ingestor import ingest_snapshot_batch
from worldpgt.wiki_snapshot_ingestion.snapshot_delta_builder import classify_snapshot_overlay_items
from worldpgt.wiki_snapshot_ingestion.snapshot_page_adapter import adapt_snapshot_doc, split_snapshot_doc
from worldpgt.wiki_snapshot_ingestion.types import ReadySnapshotDoc, SnapshotRegressionResult
from worldpgt.wiki_snapshot_ingestion.snapshot_ingestion_report import build_summary

_REPO = Path(__file__).resolve().parent.parent.parent
_WORLD = _REPO / "worldpgt"
_PROTECTED = [
    _WORLD / "experiments" / "accepted_knowledge_memory_v1.json",
    _WORLD / "experiments" / "accepted_wiki_memory_overlay_v1.json",
    _WORLD / "experiments" / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json",
    _WORLD / "continuation" / "sense_memory.py",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_doc(path: Path, title: str, body: str) -> None:
    text = (
        f"# {title}\n\n"
        f"Source: https://en.wikipedia.org/wiki/{title.replace(' ', '_')}\n"
        "Retrieved at: 2026-06-15T00:00:00Z\n"
        "Revision ID: 1\n"
        "Raw text SHA256: abc123\n"
        "Status: LOCAL_WIKIPEDIA_SNAPSHOT\n"
        "Safe for accepted memory: false\n"
        "Requires ingestion/quarantine/promotion/regression: true\n\n"
        f"{body}\n"
    )
    path.write_text(text, encoding="utf-8")


def _fixtures(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    good = docs / "SpaceX.md"
    short = docs / "Amazon.md"
    missing = docs / "Missing.md"
    _write_doc(good, "SpaceX", "SpaceX is an aerospace manufacturer. SpaceX manufactures rockets.")
    _write_doc(short, "Amazon", "Amazon may refer to:")
    readiness = [
        {"title": "SpaceX", "ready_for_self_ingestion": True, "reasons": []},
        {"title": "Amazon", "ready_for_self_ingestion": False, "reasons": ["disambiguation_like_page"]},
        {"title": "Missing", "ready_for_self_ingestion": False, "reasons": ["fetch_status:missing"]},
    ]
    manifest = [
        {
            "title": "SpaceX",
            "normalized_title": "SpaceX",
            "source_url": "https://en.wikipedia.org/wiki/SpaceX",
            "retrieved_at": "2026-06-15T00:00:00Z",
            "revision_id": 1,
            "raw_text_sha256": "hash-spacex",
            "normalized_doc_path": str(good),
            "fetch_status": "success",
        },
        {
            "title": "Amazon",
            "normalized_title": "Amazon",
            "source_url": "https://en.wikipedia.org/wiki/Amazon",
            "retrieved_at": "2026-06-15T00:00:00Z",
            "revision_id": 2,
            "raw_text_sha256": "hash-amazon",
            "normalized_doc_path": str(short),
            "fetch_status": "success",
        },
        {
            "title": "Missing",
            "normalized_title": "Missing",
            "source_url": "https://en.wikipedia.org/wiki/Missing",
            "retrieved_at": "2026-06-15T00:00:00Z",
            "revision_id": None,
            "raw_text_sha256": "",
            "normalized_doc_path": str(missing),
            "fetch_status": "missing",
        },
    ]
    readiness_path = tmp_path / "readiness.json"
    manifest_path = tmp_path / "manifest.json"
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return readiness_path, manifest_path, good


def _ready_doc(tmp_path: Path, title: str, body: str, digest: str = "hash") -> ReadySnapshotDoc:
    path = tmp_path / f"{title.replace(' ', '_')}.md"
    _write_doc(path, title, body)
    return ReadySnapshotDoc(
        title=title,
        normalized_title=title,
        source_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        retrieved_at="2026-06-15T00:00:00Z",
        revision_id=1,
        raw_text_sha256=digest,
        normalized_doc_path=str(path),
    )


def test_ready_snapshot_loader_selects_only_ready_true(tmp_path):
    readiness, manifest, _doc = _fixtures(tmp_path)
    selected, skipped = load_ready_snapshot_docs(readiness, manifest)
    assert [doc.title for doc in selected] == ["SpaceX"]
    assert len(skipped) == 2


def test_loader_skips_disambiguation_short_missing_error_docs(tmp_path):
    readiness, manifest, _doc = _fixtures(tmp_path)
    _selected, skipped = load_ready_snapshot_docs(readiness, manifest)
    reasons = {s["title"]: s["reasons"] for s in skipped}
    assert "disambiguation_like_page" in reasons["Amazon"]
    assert "fetch_status:missing" in reasons["Missing"]


def test_loader_preserves_source_url_retrieved_hash_revision(tmp_path):
    readiness, manifest, _doc = _fixtures(tmp_path)
    selected, _skipped = load_ready_snapshot_docs(readiness, manifest)
    doc = selected[0]
    assert doc.source_url.endswith("/SpaceX")
    assert doc.retrieved_at == "2026-06-15T00:00:00Z"
    assert doc.raw_text_sha256 == "hash-spacex"
    assert doc.revision_id == 1


def test_snapshot_page_adapter_preserves_provenance_header(tmp_path):
    doc = _ready_doc(tmp_path, "SpaceX", "SpaceX is an aerospace manufacturer.")
    header, _body = split_snapshot_doc(Path(doc.normalized_doc_path).read_text())
    page = adapt_snapshot_doc(doc, known_titles=["SpaceX"])
    assert header["source"].endswith("/SpaceX")
    assert page.source.source_type == "local_wikipedia_snapshot"
    assert page.source.source_url == doc.source_url


def test_snapshot_page_adapter_does_not_fabricate_body_text(tmp_path):
    doc = _ready_doc(tmp_path, "Empty", "")
    try:
        adapt_snapshot_doc(doc, known_titles=[])
    except ValueError as exc:
        assert "no body text" in str(exc)
    else:
        raise AssertionError("empty snapshot body should fail adaptation")


def test_batch_ingestor_continues_after_one_bad_doc(tmp_path):
    good = _ready_doc(tmp_path, "SpaceX", "SpaceX is an aerospace manufacturer.", "hash1")
    bad = _ready_doc(tmp_path, "Bad", "", "hash2")
    candidates, _overlay, failures, _by_type, status = ingest_snapshot_batch([good, bad])
    assert candidates
    assert len(failures) == 1
    assert status["succeeded"] == 1
    assert status["failed"] == 1


def test_not_ready_docs_are_not_ingested(tmp_path):
    readiness, manifest, _doc = _fixtures(tmp_path)
    selected, _skipped = load_ready_snapshot_docs(readiness, manifest)
    candidates, _overlay, _failures, _by_type, status = ingest_snapshot_batch(selected)
    assert status["attempted"] == 1
    assert {c.source_doc_title for c in candidates} == {"SpaceX"}


def test_candidate_outputs_preserve_source_doc_title_hash(tmp_path):
    doc = _ready_doc(tmp_path, "SpaceX", "SpaceX is an aerospace manufacturer.", "hash-source")
    candidates, _overlay, _failures, _by_type, _status = ingest_snapshot_batch([doc])
    assert candidates[0].source_doc_title == "SpaceX"
    assert candidates[0].source_doc_hash == "hash-source"
    assert candidates[0].candidate["snapshot_raw_text_sha256"] == "hash-source"


def test_weak_context_links_remain_weak(tmp_path):
    doc = _ready_doc(tmp_path, "SpaceX", "SpaceX is an aerospace manufacturer. Tesla is mentioned.", "hash")
    tesla = _ready_doc(tmp_path, "Tesla", "Tesla is a company.", "hash-tesla")
    _candidates, overlay, _failures, _by_type, _status = ingest_snapshot_batch([doc, tesla])
    links = [item for item in overlay if item.get("overlay_type") == "overlay_context_link"]
    assert links
    assert all(item["strength"] == "weak" and item["trust"] == "weak_context_only" for item in links)


def test_volatile_current_claims_are_quarantined_not_stable_delta():
    item = {
        "overlay_type": "overlay_source_fact",
        "subject": "Elon Musk",
        "predicate": "estimated_net_worth",
        "object": "US$1 billion",
        "source_name": "Forbes",
        "as_of": "2026-06",
        "requires_recheck": True,
        "stability": "volatile",
        "risk": "high",
        "snapshot_source_title": "Elon Musk",
    }
    delta, *_rest, quarantine, _tainted = classify_snapshot_overlay_items([item], [], [])
    assert delta == []
    assert quarantine[0].reason == "volatile_requires_source"


def test_relation_inversion_is_quarantined_rejected():
    item = {
        "overlay_type": "overlay_definition",
        "subject": "SpaceX",
        "definition": "Elon Musk",
        "risk": "low",
        "evidence_text": "SpaceX founded Elon Musk.",
        "snapshot_source_title": "SpaceX",
    }
    delta, _da, _dp, _conflicts, quarantine, _tainted = classify_snapshot_overlay_items([item], [], [])
    assert delta == []
    assert quarantine[0].reason == "inverted_relation"


def test_private_sensitive_data_is_quarantined_rejected():
    item = {
        "overlay_type": "overlay_relation",
        "subject": "Elon Musk",
        "predicate": "related_to",
        "object": "private email elon@example.com",
        "risk": "medium",
        "stability": "semi_stable",
        "snapshot_source_title": "Elon Musk",
    }
    delta, _da, _dp, _conflicts, quarantine, _tainted = classify_snapshot_overlay_items([item], [], [])
    assert delta == []
    assert quarantine[0].reason == "private_or_sensitive_data"


def test_unsupported_universal_claim_is_quarantined_rejected():
    item = {
        "overlay_type": "overlay_definition",
        "subject": "Rocket",
        "definition": "all vehicles",
        "risk": "low",
        "snapshot_source_title": "Rocket",
    }
    delta, _da, _dp, _conflicts, quarantine, _tainted = classify_snapshot_overlay_items([item], [], [])
    assert delta == []
    assert quarantine[0].reason == "unsupported_universal_claim"


def test_duplicate_accepted_overlay_item_is_not_added_to_delta():
    item = {"overlay_type": "overlay_entity", "label": "SpaceX", "entity_type": "organization", "snapshot_source_title": "SpaceX"}
    delta, dup_a, _dup_p, _conflicts, _q, _tainted = classify_snapshot_overlay_items([item], [item], [])
    assert delta == []
    assert len(dup_a) == 1


def test_duplicate_promoted_overlay_item_is_not_added_to_delta():
    item = {"overlay_type": "overlay_entity", "label": "SpaceX", "entity_type": "organization", "snapshot_source_title": "SpaceX"}
    delta, _dup_a, dup_p, _conflicts, _q, _tainted = classify_snapshot_overlay_items([item], [], [item])
    assert delta == []
    assert len(dup_p) == 1


def test_conflict_with_accepted_promoted_overlay_is_quarantined():
    item = {
        "overlay_type": "overlay_definition",
        "subject": "SpaceX",
        "definition": "manufacturer",
        "risk": "low",
        "snapshot_source_title": "SpaceX",
    }
    existing = [{"overlay_type": "overlay_definition", "subject": "SpaceX", "definition": "company"}]
    delta, _da, _dp, conflicts, quarantine, _tainted = classify_snapshot_overlay_items([item], existing, [])
    assert delta == []
    assert conflicts
    assert quarantine


def test_dry_run_overlay_is_written_separately():
    path = _WORLD / "experiments" / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json"
    assert path.is_file()


def test_accepted_overlay_file_is_unchanged(tmp_path):
    before = _sha(_PROTECTED[1])
    runner.run_snapshot_self_ingestion(out_dir=tmp_path, run_regressions=False)
    assert _sha(_PROTECTED[1]) == before


def test_promoted_overlay_file_is_unchanged(tmp_path):
    before = _sha(_PROTECTED[2])
    runner.run_snapshot_self_ingestion(out_dir=tmp_path, run_regressions=False)
    assert _sha(_PROTECTED[2]) == before


def test_trusted_memory_file_is_unchanged(tmp_path):
    before = _sha(_PROTECTED[0])
    runner.run_snapshot_self_ingestion(out_dir=tmp_path, run_regressions=False)
    assert _sha(_PROTECTED[0]) == before


def test_summary_says_auto_ingest_false():
    summary = build_summary(0, 0, 0, {}, 0, {}, 0, 0, 0, 0, 0, 0, 0, 0, [])
    assert summary.auto_ingest is False


def test_summary_says_auto_promote_false():
    summary = build_summary(0, 0, 0, {}, 0, {}, 0, 0, 0, 0, 0, 0, 0, 0, [])
    assert summary.auto_promote is False


def test_summary_says_runtime_behavior_modified_false():
    summary = build_summary(0, 0, 0, {}, 0, {}, 0, 0, 0, 0, 0, 0, 0, 0, [])
    assert summary.runtime_behavior_modified is False


def test_summary_says_network_calls_false():
    summary = build_summary(0, 0, 0, {}, 0, {}, 0, 0, 0, 0, 0, 0, 0, 0, [])
    assert summary.network_calls is False


def test_summary_says_safe_for_general_runtime_false():
    summary = build_summary(0, 0, 0, {}, 0, {}, 0, 0, 0, 0, 0, 0, 0, 0, [])
    assert summary.safe_for_general_runtime is False


def test_regression_summary_is_written():
    path = _WORLD / "experiments" / "wiki_snapshot_ingestion_v1" / "snapshot_regression_summary.json"
    assert path.is_file()
    assert json.loads(path.read_text())


def test_regression_runner_can_report_not_run_requires_adapter():
    result = SnapshotRegressionResult("context_pack_qa_consistency_v1", "not_run_requires_adapter", {}, "adapter needed")
    assert result.to_dict()["status"] == "not_run_requires_adapter"


def test_no_protected_files_modified(tmp_path):
    before = {path: _sha(path) for path in _PROTECTED}
    runner.run_snapshot_self_ingestion(out_dir=tmp_path, run_regressions=False)
    after = {path: _sha(path) for path in _PROTECTED}
    assert after == before


def test_no_neural_gpt_training_embedding_network_imports_in_runtime_packages():
    roots = [_WORLD / "qa", _WORLD / "entity_qa", _WORLD / "cross_page_qa", _WORLD / "context_pack"]
    forbidden = ("import torch", "import openai", "import requests", "urllib.request", "import embeddings", "backprop")
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for marker in forbidden:
                assert marker not in text


def test_nanogpt_untouched():
    assert not (_WORLD / "nanogpt").exists()
    assert not (_WORLD / "worldmvp").exists()
