"""Tests for Wikipedia Snapshot Collector v1.

Offline and deterministic. The collector is a source-snapshot layer only: no
accepted memory writes, no overlay writes, no runtime network wiring, no
ingestion, and no promotion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from worldpgt.experiments import collect_wikipedia_snapshots_v1 as runner
from worldpgt.wiki_snapshots.mediawiki_client import MediaWikiClient
from worldpgt.wiki_snapshots.page_title_loader import dedupe_titles, load_page_titles
from worldpgt.wiki_snapshots.snapshot_manifest import build_manifest_row, write_manifest
from worldpgt.wiki_snapshots.snapshot_normalizer import build_normalized_doc, write_normalized_doc
from worldpgt.wiki_snapshots.snapshot_readiness import evaluate_snapshot_readiness
from worldpgt.wiki_snapshots.types import PageSnapshot

_REPO = Path(__file__).resolve().parent.parent.parent
_WORLD = _REPO / "worldpgt"
_PROTECTED = [
    _WORLD / "experiments" / "accepted_knowledge_memory_v1.json",
    _WORLD / "experiments" / "accepted_wiki_memory_overlay_v1.json",
    _WORLD / "experiments" / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json",
    _WORLD / "continuation" / "sense_memory.py",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(
    title: str = "Example",
    raw_text: str | None = None,
    fetch_status: str = "success",
    error: str = "",
) -> PageSnapshot:
    text = raw_text if raw_text is not None else ("Example page paragraph. " * 40)
    return PageSnapshot(
        title=title,
        normalized_title=title,
        pageid=123,
        revision_id=456,
        timestamp="2025-01-01T00:00:00Z",
        source_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        api_url="https://en.wikipedia.org/w/api.php?titles=Example",
        retrieved_at="2026-01-01T00:00:00Z",
        raw_text=text,
        raw_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest() if text else "",
        license_note="test",
        fetch_status=fetch_status,
        error=error,
    )


class FakeClient:
    def __init__(self, allowed_titles, allow_network, user_agent, delay_sec, timeout_sec):
        assert allow_network is True
        self.allowed_titles = set(allowed_titles)
        self.network_calls = 0

    def fetch_page(self, title: str) -> PageSnapshot:
        assert title in self.allowed_titles
        self.network_calls += 1
        return _snapshot(title=title)


def test_page_title_loader_reads_allowlist_and_deduplicates(tmp_path):
    path = tmp_path / "titles.json"
    path.write_text(json.dumps(["Elon Musk", "SpaceX", " elon musk ", "", "Tesla, Inc."]))
    assert load_page_titles(path) == ["Elon Musk", "SpaceX", "Tesla, Inc."]


def test_title_loader_enforces_limit_100_by_default():
    titles = [f"Title {idx}" for idx in range(120)]
    loaded = dedupe_titles(titles)
    assert len(loaded) == 100
    assert loaded[-1] == "Title 99"


def test_mediawiki_client_refuses_network_without_explicit_allow_flag():
    client = MediaWikiClient(["Elon Musk"], allow_network=False)
    with pytest.raises(PermissionError):
        client.fetch_page("Elon Musk")


@pytest.mark.parametrize("user_agent", ["", "python-urllib/3.13", "requests", "bot"])
def test_mediawiki_client_refuses_generic_or_missing_user_agent(user_agent):
    with pytest.raises(ValueError):
        MediaWikiClient(["Elon Musk"], allow_network=True, user_agent=user_agent)


def test_mediawiki_client_builds_url_only_for_allowlisted_titles():
    client = MediaWikiClient(["Elon Musk"], allow_network=False)
    url = client.build_api_url("Elon Musk")
    assert "titles=Elon+Musk" in url
    with pytest.raises(ValueError):
        client.build_api_url("Not Allowlisted")


def test_snapshot_normalizer_preserves_provenance_header():
    doc = build_normalized_doc(_snapshot(title="Elon Musk"))
    assert doc.startswith("# Elon Musk")
    assert "Source: https://en.wikipedia.org/wiki/Elon_Musk" in doc
    assert "Safe for accepted memory: false" in doc
    assert "Requires ingestion/quarantine/promotion/regression: true" in doc


def test_snapshot_normalizer_does_not_fabricate_body_text():
    doc = build_normalized_doc(_snapshot(raw_text=""))
    assert "Example page paragraph" not in doc
    assert doc.count("Status: LOCAL_WIKIPEDIA_SNAPSHOT") == 1


def test_manifest_writes_json_and_csv(tmp_path):
    snapshot = _snapshot()
    doc_path = write_normalized_doc(snapshot, tmp_path / "docs")
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(snapshot.to_dict()))
    row = build_manifest_row(snapshot, raw_path, doc_path)
    write_manifest([row], tmp_path / "manifest.json", tmp_path / "manifest.csv")
    assert json.loads((tmp_path / "manifest.json").read_text())[0]["title"] == "Example"
    assert "ready_for_self_ingestion" in (tmp_path / "manifest.csv").read_text()


def test_readiness_accepts_valid_fake_snapshot(tmp_path):
    snapshot = _snapshot()
    doc_path = write_normalized_doc(snapshot, tmp_path)
    result = evaluate_snapshot_readiness(snapshot, doc_path)
    assert result.ready_for_self_ingestion is True
    assert result.requires_quarantine is True
    assert result.safe_for_general_runtime is False


def test_readiness_rejects_missing_error_or_short_snapshot(tmp_path):
    short = _snapshot(raw_text="too short", fetch_status="success")
    doc_path = write_normalized_doc(short, tmp_path)
    assert evaluate_snapshot_readiness(short, doc_path).ready_for_self_ingestion is False
    error = _snapshot(raw_text="", fetch_status="error", error="boom")
    assert evaluate_snapshot_readiness(error, doc_path).ready_for_self_ingestion is False


def test_disambiguation_like_page_is_marked_not_ready(tmp_path):
    snapshot = _snapshot(title="Mercury", raw_text="Mercury may refer to:\n\n* Mercury (planet)" * 40)
    doc_path = write_normalized_doc(snapshot, tmp_path)
    result = evaluate_snapshot_readiness(snapshot, doc_path)
    assert result.ready_for_self_ingestion is False
    assert "disambiguation_like_page" in result.reasons


def test_default_runner_mode_performs_no_network_calls(tmp_path):
    result = runner.run_collection(out_dir=tmp_path, dry_run=True)
    assert result["summary"]["allow_network"] is False
    assert result["summary"]["network_calls"] == 0
    assert result["summary"]["fetched_count"] == 0
    assert (tmp_path / "snapshot_collection_report.json").is_file()


def test_allow_network_path_can_be_tested_with_mocked_client(tmp_path):
    titles = tmp_path / "titles.json"
    titles.write_text(json.dumps(["Elon Musk", "SpaceX"]))
    result = runner.run_collection(
        out_dir=tmp_path,
        titles_file=titles,
        allow_network=True,
        limit=2,
        user_agent="MicroworldResearchBot/0.1 (local research; contact: test@example.com)",
        client_factory=FakeClient,
    )
    summary = result["summary"]
    assert summary["fetched_count"] == 2
    assert summary["success_count"] == 2
    assert summary["ready_for_self_ingestion_count"] == 2
    assert summary["network_calls"] == 2


def test_summary_says_auto_ingest_false(tmp_path):
    assert runner.run_collection(out_dir=tmp_path, dry_run=True)["summary"]["auto_ingest"] is False


def test_summary_says_auto_promote_false(tmp_path):
    assert runner.run_collection(out_dir=tmp_path, dry_run=True)["summary"]["auto_promote"] is False


def test_summary_says_trusted_memory_unchanged(tmp_path):
    assert runner.run_collection(out_dir=tmp_path, dry_run=True)["summary"]["trusted_memory_modified"] is False


def test_summary_says_accepted_overlay_unchanged(tmp_path):
    assert runner.run_collection(out_dir=tmp_path, dry_run=True)["summary"]["accepted_overlay_modified"] is False


def test_summary_says_promoted_overlay_unchanged(tmp_path):
    assert runner.run_collection(out_dir=tmp_path, dry_run=True)["summary"]["promoted_overlay_modified"] is False


def test_summary_says_safe_for_general_runtime_false(tmp_path):
    assert runner.run_collection(out_dir=tmp_path, dry_run=True)["summary"]["safe_for_general_runtime"] is False


def test_no_protected_files_modified(tmp_path):
    before = {path: _sha256(path) for path in _PROTECTED}
    runner.run_collection(out_dir=tmp_path, dry_run=True)
    after = {path: _sha256(path) for path in _PROTECTED}
    assert after == before


def test_no_neural_gpt_training_embedding_imports():
    files = list((_WORLD / "wiki_snapshots").glob("*.py")) + [
        _WORLD / "experiments" / "collect_wikipedia_snapshots_v1.py"
    ]
    forbidden = ("import torch", "import openai", "import requests", "embedding", "backprop")
    for path in files:
        text = path.read_text(encoding="utf-8").lower()
        for marker in forbidden:
            assert marker not in text


def test_no_runtime_qa_planner_context_pack_network_imports():
    runtime_files = []
    for rel in ("qa", "entity_qa", "cross_page_qa", "context_pack"):
        root = _WORLD / rel
        if root.exists():
            runtime_files.extend(root.glob("*.py"))
    forbidden = ("import urllib.request", "import requests", "from requests", "mediawiki", "wiki_snapshots")
    for path in runtime_files:
        text = path.read_text(encoding="utf-8").lower()
        for marker in forbidden:
            assert marker not in text


def test_nanogpt_untouched_by_runner(tmp_path):
    nested = _WORLD / "nanogpt"
    before_exists = nested.exists()
    runner.run_collection(out_dir=tmp_path, dry_run=True)
    assert nested.exists() is before_exists
    assert not (_WORLD / "worldmvp").exists()
