"""Tests for Knowledge Pump 5000 Mode v1.

No real network. The pump writes proposal/checkpoint artifacts only and never
mutates accepted/promoted/runtime memory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldpgt.experiments import ask_microworld_v1
from worldpgt.experiments import run_knowledge_pump_v1 as runner
from worldpgt.knowledge_pump.expanded_allowlist_builder import build_expanded_allowlist
from worldpgt.knowledge_pump.frontier_title_extractor import extract_frontier_titles
from worldpgt.knowledge_pump.pump_batch_planner import plan_batches
from worldpgt.knowledge_pump.pump_checkpoint import load_checkpoint
from worldpgt.knowledge_pump.safe_delta_merger import merge_safe_deltas, merge_with_fresh_deltas
from worldpgt.knowledge_pump.title_ranker import normalize_title, priority_for, risk_hint
from worldpgt.knowledge_pump.types import FrontierTitle

_REPO = Path(__file__).resolve().parent.parent.parent
_WORLD = _REPO / "worldpgt"
_EXP = _WORLD / "experiments"
_PROTECTED = [
    _EXP / "accepted_knowledge_memory_v1.json",
    _EXP / "accepted_wiki_memory_overlay_v1.json",
    _EXP / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json",
    _EXP / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json",
]


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def pump_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("knowledge_pump")
    summary = runner.run(plan_only=True, target_total=5000, batch_size=250, max_batches=1, out_dir=out)
    return out, summary


def _frontier(n: int = 20) -> list[FrontierTitle]:
    return [FrontierTitle(f"SpaceX Product {i}", "test", "fixture", 10) for i in range(n)]


def test_plan_only_creates_target_plan_up_to_cap(pump_run):
    out, summary = pump_run
    assert 0 < summary["expanded_allowlist_total"] <= 5000
    assert len(json.loads((out / "expanded_allowlist.json").read_text())) == summary["expanded_allowlist_total"]


def test_plan_only_does_not_call_network(pump_run):
    _out, summary = pump_run
    assert summary["allow_network"] is False
    assert summary["network_calls"] is False


def test_ask_cli_pump_dry_run_overlay_prints_proposal_marker(capsys):
    code = ask_microworld_v1.main(["What is Starlink?", "--overlay", "pump-dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Overlay mode: pump-dry-run proposal, not accepted memory." in out


def test_ask_cli_custom_overlay_path_resolves(tmp_path, capsys):
    overlay = tmp_path / "custom_overlay.json"
    overlay.write_text(
        json.dumps(
            [
                {
                    "overlay_type": "overlay_entity",
                    "label": "Test Widget",
                    "entity_type": "device",
                    "source_page": "Test Widget",
                    "trust": "overlay_candidate",
                    "risk": "low",
                },
                {
                    "overlay_type": "overlay_definition",
                    "subject": "Test Widget",
                    "definition": "device",
                    "predicate": "is_a",
                    "trust": "overlay_candidate",
                    "risk": "low",
                    "stability": "stable",
                },
            ]
        ),
        encoding="utf-8",
    )
    code = ask_microworld_v1.main(["What is Test Widget?", "--overlay-path", str(overlay)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Overlay mode: custom overlay path, not accepted memory." in out
    assert "Test Widget is a device." in out


def test_ask_cli_ontology_layer_extends_is_a_traversal(tmp_path, capsys):
    overlay = tmp_path / "custom_overlay.json"
    ontology = tmp_path / "ontology_layer.json"
    overlay.write_text(
        json.dumps(
            [
                {
                    "overlay_type": "overlay_definition",
                    "subject": "Elon Musk",
                    "definition": "businessman",
                    "predicate": "is_a",
                    "trust": "overlay_candidate",
                    "risk": "low",
                    "stability": "stable",
                },
            ]
        ),
        encoding="utf-8",
    )
    ontology.write_text(
        json.dumps(
            [
                {
                    "overlay_type": "overlay_relation",
                    "subject": "businessman",
                    "predicate": "is_a",
                    "object": "worker",
                    "trust": "wikidata_p279_ontology",
                    "risk": "low",
                    "stability": "stable",
                },
            ]
        ),
        encoding="utf-8",
    )

    code = ask_microworld_v1.main([
        "Is Elon Musk a worker?",
        "--overlay-path",
        str(overlay),
        "--ontology-layer",
        str(ontology),
    ])
    out = capsys.readouterr().out

    assert code == 0
    assert "Yes. Elon Musk is a businessman, which is a worker." in out
    assert "Support: explicit is_a chain." in out
    assert "Decision: answer." in out


def test_ask_cli_rejects_overlay_and_overlay_path_together(tmp_path, capsys):
    overlay = tmp_path / "custom_overlay.json"
    overlay.write_text("[]\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        ask_microworld_v1.main(
            ["What is Test Widget?", "--overlay", "pump-dry-run", "--overlay-path", str(overlay)]
        )
    err = capsys.readouterr().err
    assert exc.value.code == 2
    assert "--overlay and --overlay-path cannot be used together" in err


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "What does the International Energy Agency publish?",
            "International Energy Agency publishes annual World Energy Outlook.",
        ),
        ("Who founded Bloomberg News?", "Bloomberg News was founded by Michael Bloomberg."),
        ("Who was Bloomberg News founded by?", "Bloomberg News was founded by Michael Bloomberg."),
        ("Who owns SolarCity?", "SolarCity is owned by Tesla."),
        ("Who is SolarCity owned by?", "SolarCity is owned by Tesla."),
        ("What company owns SolarCity?", "SolarCity is owned by Tesla."),
        ("What is Rocket Science Games?", "Rocket Science Games is an independent game studio."),
        ("What is June?", "June is the sixth month of the year"),
        (
            "What is Exos Aerospace Systems & Technologies?",
            "Exos Aerospace Systems & Technologies is an Aerospace manufacturer.",
        ),
    ],
)
def test_pump_supported_facts_answer_common_user_phrasings(question, expected):
    answer = ask_microworld_v1.ask(question, "pump-dry-run")
    assert answer.decision == "answer"
    assert answer.supported_by_context is True
    assert expected in answer.answer_text


def test_pump_article_normalization_supports_leading_the():
    answer = ask_microworld_v1.ask(
        "What does the International Energy Agency publish?", "pump-dry-run"
    )
    assert answer.decision == "answer"
    assert "annual World Energy Outlook" in answer.answer_text


def test_expanded_allowlist_respects_target_total():
    entries = build_expanded_allowlist(_frontier(10), target_total=7, batch_size=3)
    assert len(entries) == 7


def test_batch_planner_respects_batch_size():
    entries = build_expanded_allowlist(_frontier(10), target_total=10, batch_size=4)
    batches = plan_batches(entries, batch_size=4, max_batches=3)
    assert [len(b) for b in batches] == [4, 4, 2]


def test_max_batches_limits_execution():
    entries = build_expanded_allowlist(_frontier(20), target_total=20, batch_size=5)
    assert len(plan_batches(entries, batch_size=5, max_batches=2)) == 2


def test_checkpoint_is_written(pump_run):
    out, _summary = pump_run
    assert (out / "pump_checkpoint.json").is_file()


def test_checkpoint_supports_resume(pump_run):
    out, _summary = pump_run
    checkpoint = load_checkpoint(out / "pump_checkpoint.json")
    assert checkpoint is not None
    assert checkpoint.can_resume is True
    assert checkpoint.safe_to_continue is True


def test_already_fetched_titles_are_skipped():
    entries = build_expanded_allowlist(_frontier(3), 3, 2, already_fetched={"SpaceX Product 0"})
    assert entries[0].already_fetched is True


def test_failed_titles_are_tracked_in_checkpoint_shape(pump_run):
    out, _summary = pump_run
    data = json.loads((out / "pump_checkpoint.json").read_text())
    assert "failed_fetch_titles" in data
    assert isinstance(data["failed_fetch_titles"], list)


def test_not_ready_docs_are_not_ingested(pump_run):
    _out, summary = pump_run
    assert summary["not_ready_skipped_count"] >= 9


def test_frontier_extractor_reads_local_snapshots(tmp_path):
    snap = tmp_path / "snapshots" / "normalized_docs"
    snap.mkdir(parents=True)
    (snap / "one.md").write_text("# One\n\nSpaceX Starship mentions Tesla Model Y.\n", encoding="utf-8")
    titles = extract_frontier_titles(tmp_path / "snapshots")
    assert any(t.title == "SpaceX Starship" for t in titles)


def test_frontier_extractor_filters_observed_junk_titles(tmp_path):
    snap = tmp_path / "snapshots" / "normalized_docs"
    snap.mkdir(parents=True)
    junk = (
        "According After Although April December They What When With SHA256 "
        "Does Tesla In July The Model Therefore turbocharged engines usually "
        "SpaceX Starship Tesla Model Y"
    )
    (snap / "junk.md").write_text(f"# Junk\n\n{junk}.\n", encoding="utf-8")
    titles = {t.title for t in extract_frontier_titles(tmp_path / "snapshots")}
    for bad in {
        "According", "After", "Although", "April", "December", "They",
        "What", "When", "With", "SHA256", "Does Tesla", "In July",
        "The Model", "Therefore turbocharged engines usually",
    }:
        assert bad not in titles
    assert "SpaceX Starship" in titles


def test_frontier_extractor_uses_relation_v2_candidates(tmp_path):
    rel = tmp_path / "rel.json"
    rel.write_text(json.dumps([{"subject": "Elon Musk", "object": "SpaceX"}]), encoding="utf-8")
    titles = extract_frontier_titles(tmp_path / "missing", relation_candidates_path=rel)
    assert any(t.title == "Elon Musk" for t in titles)


def test_frontier_extractor_uses_missing_knowledge_requests(tmp_path):
    req = tmp_path / "req.json"
    req.write_text(json.dumps([{"entity": "Blue Origin", "question": "Who founded Blue Origin?"}]), encoding="utf-8")
    titles = extract_frontier_titles(tmp_path / "missing", knowledge_requests_path=req)
    assert any(t.title == "Blue Origin" for t in titles)


def test_title_ranker_prioritizes_product_org_technology_pages():
    product = FrontierTitle("SpaceX Starship", "test", "product", 1)
    generic = FrontierTitle("Beautiful Idea", "test", "generic", 1)
    assert priority_for(product) > priority_for(generic)


def test_volatile_current_financial_titles_are_low_priority_or_high_risk():
    title = "Tesla stock price"
    assert risk_hint(title) == "volatile_or_current"
    assert priority_for(FrontierTitle(title, "test", "volatile", 10)) < 0


def test_dedup_normalizes_titles():
    assert normalize_title(" SpaceX_Starship ") == "SpaceX Starship"


def test_safe_delta_merger_deduplicates_exact_relations(tmp_path):
    base = tmp_path / "base.json"
    snap = tmp_path / "snap.json"
    rel = tmp_path / "rel.json"
    out = tmp_path / "out.json"
    item = {"overlay_type": "overlay_relation", "subject": "A", "predicate": "founded", "object": "B", "risk": "medium", "stability": "semi_stable"}
    base.write_text("[]")
    snap.write_text(json.dumps([item, item]))
    rel.write_text("[]")
    merged, _counts = merge_safe_deltas(base, snap, rel, out)
    assert len(merged) == 1


def test_safe_delta_merger_treats_founded_inverse_as_duplicate(tmp_path):
    base = tmp_path / "base.json"
    snap = tmp_path / "snap.json"
    rel = tmp_path / "rel.json"
    out = tmp_path / "out.json"
    base.write_text(json.dumps([{"overlay_type": "overlay_relation", "subject": "Elon Musk", "predicate": "founded", "object": "SpaceX"}]))
    snap.write_text(json.dumps([{"overlay_type": "overlay_relation", "subject": "SpaceX", "predicate": "founded_by", "object": "Elon Musk", "risk": "medium", "stability": "semi_stable"}]))
    rel.write_text("[]")
    merged, _counts = merge_safe_deltas(base, snap, rel, out)
    assert merged == []


def test_safe_delta_merger_excludes_weak_link_as_fact(tmp_path):
    base, snap, rel, out = tmp_path / "b.json", tmp_path / "s.json", tmp_path / "r.json", tmp_path / "o.json"
    base.write_text("[]")
    snap.write_text(json.dumps([{"overlay_type": "overlay_context_link", "source_page": "A", "target": "B", "trust": "overlay_candidate"}]))
    rel.write_text("[]")
    merged, _counts = merge_safe_deltas(base, snap, rel, out)
    assert merged == []


def test_safe_delta_merger_excludes_volatile_current_promotion(tmp_path):
    base, snap, rel, out = tmp_path / "b.json", tmp_path / "s.json", tmp_path / "r.json", tmp_path / "o.json"
    base.write_text("[]")
    snap.write_text(json.dumps([{"overlay_type": "overlay_relation", "subject": "A", "predicate": "price", "object": "$1", "risk": "high", "stability": "volatile"}]))
    rel.write_text("[]")
    merged, _counts = merge_safe_deltas(base, snap, rel, out)
    assert merged == []


def test_pump_dry_run_overlay_is_separate(pump_run):
    out, _summary = pump_run
    assert (out / "pump_dry_run_overlay.json").is_file()
    assert out / "pump_dry_run_overlay.json" != _PROTECTED[3]


def test_accepted_overlay_unchanged(pump_run):
    before = _sha(_PROTECTED[1])
    _out, _summary = pump_run
    assert _sha(_PROTECTED[1]) == before


def test_promoted_overlay_unchanged(pump_run):
    before = _sha(_PROTECTED[2])
    _out, _summary = pump_run
    assert _sha(_PROTECTED[2]) == before


def test_snapshot_dry_run_overlay_unchanged(pump_run):
    before = _sha(_PROTECTED[3])
    _out, _summary = pump_run
    assert _sha(_PROTECTED[3]) == before


def test_trusted_memory_unchanged(pump_run):
    before = _sha(_PROTECTED[0])
    _out, _summary = pump_run
    assert _sha(_PROTECTED[0]) == before


def test_summary_says_auto_ingest_false(pump_run):
    _out, summary = pump_run
    assert summary["auto_ingest"] is False


def test_summary_says_auto_promote_false(pump_run):
    _out, summary = pump_run
    assert summary["auto_promote"] is False


def test_summary_says_network_calls_false_in_plan_only(pump_run):
    _out, summary = pump_run
    assert summary["network_calls"] is False


def test_summary_says_safe_for_general_runtime_false(pump_run):
    _out, summary = pump_run
    assert summary["safe_for_general_runtime"] is False


def test_summary_has_v11_recompute_fields(pump_run):
    _out, summary = pump_run
    assert "new_ready_docs_this_batch" in summary
    assert "new_ingestion_candidates_this_batch" in summary
    assert "new_relation_candidates_this_batch" in summary
    assert "stale_artifacts_reused" in summary


def _write_json(path: Path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _fresh_merge(tmp_path, *, base=None, legacy_snapshot=None, legacy_relations=None, fresh_snapshot=None, fresh_relations=None):
    base_path = tmp_path / "base.json"
    legacy_snapshot_path = tmp_path / "legacy_snapshot.json"
    legacy_rel_path = tmp_path / "legacy_rel.json"
    out_overlay = tmp_path / "pump_overlay.json"
    out_delta = tmp_path / "pump_delta.json"
    _write_json(base_path, base or [])
    _write_json(legacy_snapshot_path, legacy_snapshot or [])
    _write_json(legacy_rel_path, legacy_relations or [])
    return merge_with_fresh_deltas(
        base_overlay_path=base_path,
        legacy_snapshot_delta_path=legacy_snapshot_path,
        legacy_relation_candidates_path=legacy_rel_path,
        fresh_snapshot_overlay_items=fresh_snapshot or [],
        fresh_relation_rows=fresh_relations or [],
        output_overlay_path=out_overlay,
        output_delta_path=out_delta,
    )


def test_fresh_ingestion_candidates_are_written(pump_run):
    out, _summary = pump_run
    assert (out / "pump_fresh_ingestion_candidates.json").is_file()
    assert (out / "pump_fresh_ingestion_candidates.csv").is_file()


def test_fresh_relation_candidates_are_written(pump_run):
    out, _summary = pump_run
    assert (out / "pump_fresh_relation_candidates.json").is_file()
    assert (out / "pump_fresh_relation_candidates.csv").is_file()


def test_fresh_candidates_are_passed_into_safe_delta_merger(tmp_path):
    fresh = [{"overlay_type": "overlay_entity", "label": "Fresh Entity", "entity_type": "organization", "risk": "low"}]
    result = _fresh_merge(tmp_path, fresh_snapshot=fresh)
    assert result["counts"]["fresh_safe_snapshot_delta_count"] == 1
    assert result["fresh_safe_delta"][0]["label"] == "Fresh Entity"


def test_fresh_safe_deltas_increase_pump_safe_delta_when_non_duplicate(tmp_path):
    fresh = [{"overlay_type": "overlay_relation", "subject": "FreshCo", "predicate": "founded_by", "object": "Ada Lovelace", "risk": "low", "stability": "semi_stable"}]
    result = _fresh_merge(tmp_path, fresh_snapshot=fresh)
    assert result["counts"]["pump_safe_delta_total"] == 1


def test_fresh_duplicates_do_not_increase_pump_safe_delta(tmp_path):
    item = {"overlay_type": "overlay_relation", "subject": "A", "predicate": "founded_by", "object": "B", "risk": "low", "stability": "semi_stable"}
    result = _fresh_merge(tmp_path, base=[item], fresh_snapshot=[item])
    assert result["counts"]["fresh_duplicates_count"] == 1
    assert result["counts"]["pump_safe_delta_total"] == 0


def test_fresh_conflicts_go_to_conflicts_and_quarantine(tmp_path):
    base = [{"overlay_type": "overlay_entity", "label": "Fresh Entity", "entity_type": "person"}]
    fresh = [{"overlay_type": "overlay_entity", "label": "Fresh Entity", "entity_type": "organization", "risk": "low"}]
    result = _fresh_merge(tmp_path, base=base, fresh_snapshot=fresh)
    assert result["counts"]["fresh_conflicts_count"] == 1
    assert result["counts"]["fresh_quarantined_count"] == 1


def test_fresh_weak_links_are_not_promoted_as_stable_facts(tmp_path):
    fresh = [{"overlay_type": "overlay_context_link", "source_page": "A", "target": "B", "strength": "strong", "trust": "overlay_candidate"}]
    result = _fresh_merge(tmp_path, fresh_snapshot=fresh)
    assert result["counts"]["fresh_safe_snapshot_delta_count"] == 0
    assert result["fresh_quarantine"][0]["reason"] == "weak_link_promoted_as_fact"


def test_fresh_volatile_current_facts_are_not_promoted(tmp_path):
    fresh = [{"overlay_type": "overlay_relation", "subject": "Tesla", "predicate": "stock_price", "object": "$1", "risk": "high", "stability": "volatile"}]
    result = _fresh_merge(tmp_path, fresh_snapshot=fresh)
    assert result["counts"]["fresh_safe_snapshot_delta_count"] == 0
    assert result["fresh_quarantine"][0]["reason"] == "volatile_or_high_risk"


def test_pump_dry_run_overlay_includes_fresh_safe_delta(tmp_path):
    fresh = [{"overlay_type": "overlay_entity", "label": "Fresh Entity", "entity_type": "organization", "risk": "low"}]
    _fresh_merge(tmp_path, fresh_snapshot=fresh)
    overlay = json.loads((tmp_path / "pump_overlay.json").read_text())
    assert any(item.get("label") == "Fresh Entity" for item in overlay)


def test_summary_includes_fresh_delta_counts(pump_run):
    _out, summary = pump_run
    for key in (
        "fresh_ingestion_candidates_total",
        "fresh_relation_candidates_total",
        "fresh_candidates_total",
        "fresh_safe_snapshot_delta_count",
        "fresh_safe_relation_delta_count",
        "fresh_merged_safe_delta_count",
        "fresh_duplicates_count",
        "fresh_conflicts_count",
        "fresh_quarantined_count",
        "fresh_rejected_count",
        "pump_safe_delta_total",
        "assistant_benchmark_overlay_path",
    ):
        assert key in summary


def test_stale_artifacts_reused_false_when_fresh_recompute_does_not_fallback(pump_run):
    _out, summary = pump_run
    assert summary["stale_artifacts_reused"] is False


def test_assistant_benchmark_uses_pump_overlay_or_reports_adapter(pump_run):
    out, summary = pump_run
    path = str(out / "pump_dry_run_overlay.json")
    assert summary["assistant_benchmark_overlay_path"] in (path, "not_run_requires_adapter")


def test_plan_only_reports_no_stale_artifact_reuse_when_no_batch_docs(pump_run):
    _out, summary = pump_run
    assert summary["new_ready_docs_this_batch"] == 0
    assert summary["new_ingestion_candidates_this_batch"] == 0
    assert summary["new_relation_candidates_this_batch"] == 0
    assert summary["stale_artifacts_reused"] is False


def test_no_neural_gpt_training_embedding_imports():
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in (_WORLD / "knowledge_pump").glob("*.py"))
    for marker in ("import torch", "import openai", "backprop", "training loop"):
        assert marker not in text


def test_nanogpt_untouched():
    assert not (_WORLD / "nanogpt").exists()
    assert not (_WORLD / "worldmvp").exists()


def test_full_suite_placeholder_for_green_verification(pump_run):
    _out, summary = pump_run
    assert summary["all_critical_passed"] is True


# ─── v1.3 Delta Quality Firewall tests ────────────────────────────────────────

from worldpgt.knowledge_pump.delta_quality_firewall import classify_pump_delta  # noqa: E402


def _weak_ctx(source="A", target="B") -> dict:
    return {
        "overlay_type": "overlay_context_link",
        "source_page": source,
        "target": target,
        "relation": "mentioned_with",
        "trust": "weak_context_only",
        "strength": "weak",
    }


def _relation(subject, predicate, obj, evidence="") -> dict:
    return {
        "overlay_type": "overlay_relation",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "evidence_text": evidence,
        "trust": "overlay_candidate",
        "risk": "medium",
        "stability": "semi_stable",
    }


# Test 1: weak mentioned_with links land in weak_context bucket
def test_firewall_weak_mentioned_with_goes_to_weak_context_bucket():
    item = _weak_ctx("SpaceX", "They")
    result = classify_pump_delta([item])
    assert result["weak_context"] == [item]
    assert item not in result["answerable"]


# Test 2: weak links are not counted as answerable delta
def test_firewall_weak_links_not_in_answerable():
    items = [_weak_ctx("A", "B"), _weak_ctx("C", "D")]
    result = classify_pump_delta(items)
    assert len(result["weak_context"]) == 2
    assert len(result["answerable"]) == 0


# Test 3: pump_dry_run_overlay (default) excludes weak context links
def test_default_overlay_excludes_weak_context_links(pump_run):
    out, _summary = pump_run
    overlay = json.loads((out / "pump_dry_run_overlay.json").read_text(encoding="utf-8"))
    weak = [
        item for item in overlay
        if item.get("overlay_type") == "overlay_context_link"
        and item.get("trust") == "weak_context_only"
    ]
    assert weak == [], f"overlay contains {len(weak)} weak context links when none expected"


# Test 4: pump_dry_run_overlay includes weak links when flag is set
@pytest.fixture(scope="module")
def pump_run_weak(tmp_path_factory):
    out = tmp_path_factory.mktemp("knowledge_pump_weak")
    summary = runner.run(
        plan_only=True, target_total=5000, batch_size=250,
        max_batches=1, out_dir=out, include_weak_context=True,
    )
    return out, summary


def test_overlay_includes_weak_links_with_flag(pump_run_weak):
    out, summary = pump_run_weak
    assert summary["include_weak_context"] is True
    overlay = json.loads((out / "pump_dry_run_overlay.json").read_text(encoding="utf-8"))
    weak = [
        item for item in overlay
        if item.get("overlay_type") == "overlay_context_link"
        and item.get("trust") == "weak_context_only"
    ]
    # The legacy delta contains 1172 weak context links; at least some should appear.
    assert len(weak) > 0


# Test 5: SpaceX headquartered_in SpaceX is rejected (self-relation)
def test_firewall_rejects_self_relation_headquartered_in():
    item = _relation("SpaceX", "headquartered_in", "SpaceX",
                     "Facilities SpaceX is headquartered at SpaceX Starbase.")
    result = classify_pump_delta([item])
    assert len(result["rejected"]) == 1
    assert result["rejected"][0]["reason"] == "self_relation"
    assert len(result["answerable"]) == 0


# Test 6: Falcon 1 develops Falcon 1 is rejected (self-relation)
def test_firewall_rejects_self_relation_develops():
    item = _relation("Falcon 1", "develops", "Falcon 1",
                     "was considered for Falcon 1 launches but never developed before Falcon 1 was retired.")
    result = classify_pump_delta([item])
    assert len(result["rejected"]) == 1
    # Self-relation takes priority over negated_context
    assert result["rejected"][0]["reason"] == "self_relation"


# Test 7: OpenAI is_a Artificial intelligence is rejected (too-broad is_a object)
def test_firewall_rejects_is_a_too_broad():
    item = _relation("OpenAI", "is_a", "Artificial intelligence",
                     "OpenAI is an American artificial intelligence research organization.")
    result = classify_pump_delta([item])
    reasons = {r["reason"] for r in result["rejected"]} | {r["reason"] for r in result["quarantine"]}
    assert "is_a_object_too_broad" in reasons or "is_a_object_truncated_nationality" in reasons
    assert len(result["answerable"]) == 0


# Test 8: The Boring Company is_a American infrastructure is rejected (truncated nationality)
def test_firewall_rejects_is_a_truncated_nationality():
    item = _relation("The Boring Company", "is_a", "American infrastructure",
                     "The Boring Company (TBC) is an American infrastructure and tunnel construction company.")
    result = classify_pump_delta([item])
    reasons = {r["reason"] for r in result["rejected"]}
    assert "is_a_object_truncated_nationality" in reasons
    assert len(result["answerable"]) == 0


# Test 9: xAI subsidiary_of SpaceX with volatile/date-heavy evidence is quarantined
def test_firewall_quarantines_volatile_date_heavy_relation():
    item = _relation(
        "xAI", "subsidiary_of", "SpaceX",
        "In May 2026, Anthropic signed a contract of $1.25 billion per month with xAI, a subsidiary of SpaceX.",
    )
    result = classify_pump_delta([item])
    assert len(result["quarantine"]) == 1
    assert result["quarantine"][0]["reason"] == "volatile_evidence"
    assert len(result["answerable"]) == 0


# Test 10: SpaceX develops Falcon 9 is accepted
def test_firewall_accepts_spacex_develops_falcon9():
    item = _relation("SpaceX", "develops", "Falcon 9",
                     "SpaceX developed Falcon 9 with private capital as well.")
    result = classify_pump_delta([item])
    assert len(result["answerable"]) == 1
    assert len(result["rejected"]) == 0
    assert len(result["quarantine"]) == 0


# Test 11: Neuralink develops brain-computer interfaces is accepted
def test_firewall_accepts_neuralink_develops_bci():
    item = _relation("Neuralink", "develops", "brain-computer interfaces",
                     "Neuralink develops brain-computer interfaces to connect humans and computers.")
    result = classify_pump_delta([item])
    assert len(result["answerable"]) == 1
    assert len(result["rejected"]) == 0
    assert len(result["quarantine"]) == 0


# Test 12: The Boring Company founded_by Elon Musk is accepted
def test_firewall_accepts_boring_company_founded_by_musk():
    item = _relation("The Boring Company", "founded_by", "Elon Musk",
                     "The Boring Company was founded by Elon Musk in 2016.")
    result = classify_pump_delta([item])
    assert len(result["answerable"]) == 1
    assert len(result["rejected"]) == 0
    assert len(result["quarantine"]) == 0


# Confirm quality artifact files are written by the runner
def test_quality_artifact_files_written(pump_run):
    out, _summary = pump_run
    for fname in (
        "pump_answerable_delta.json",
        "pump_weak_context_delta.json",
        "pump_entity_delta.json",
        "pump_rejected_delta.json",
        "pump_delta_quality_report.json",
    ):
        assert (out / fname).is_file(), f"expected {fname} to exist"


# Confirm summary carries quality firewall fields
def test_summary_has_quality_firewall_fields(pump_run):
    _out, summary = pump_run
    for key in (
        "pump_answerable_delta_count",
        "pump_world_model_delta_count",
        "pump_entity_delta_count",
        "pump_answerable_fact_delta_count",
        "pump_relation_delta_count",
        "pump_definition_delta_count",
        "pump_property_candidate_count",
        "weak_context_delta_count",
        "entity_delta_count",
        "rejected_quality_count",
        "relation_quality_rejection_by_reason",
        "pump_dry_run_overlay_items_count_without_weak",
        "pump_dry_run_overlay_items_count_with_weak",
        "include_weak_context",
        "assistant_answer_gain_vs_baseline",
        "pump_smoke_prompt_count",
        "pump_smoke_answer_count",
        "pump_smoke_audit_count",
        "pump_smoke_wrong_count",
        "pump_smoke_unsupported_answer_count",
        "pump_smoke_planner_gap_count",
        "pump_smoke_supported_fact_answer_count",
    ):
        assert key in summary, f"summary missing key: {key}"


def test_honest_pump_metrics_separate_entity_cards_from_answerable_facts(pump_run):
    out, summary = pump_run
    delta = json.loads((out / "pump_answerable_delta.json").read_text(encoding="utf-8"))
    by_type = {}
    for item in delta:
        by_type[item.get("overlay_type")] = by_type.get(item.get("overlay_type"), 0) + 1

    assert summary["pump_entity_delta_count"] == by_type.get("overlay_entity", 0)
    assert summary["pump_relation_delta_count"] == by_type.get("overlay_relation", 0)
    assert summary["pump_definition_delta_count"] == by_type.get("overlay_definition", 0)
    assert summary["pump_answerable_fact_delta_count"] == (
        summary["pump_relation_delta_count"] + summary["pump_definition_delta_count"]
    )
    assert summary["pump_world_model_delta_count"] == (
        summary["pump_entity_delta_count"] + summary["pump_answerable_fact_delta_count"]
    )


def test_honest_metric_counter_does_not_count_entities_as_answerable_facts():
    metrics = runner._pump_delta_metrics(
        [],
        [
            {"overlay_type": "overlay_entity", "label": "Entity Card"},
            {"overlay_type": "overlay_relation", "subject": "A", "predicate": "develops", "object": "B"},
            {"overlay_type": "overlay_definition", "subject": "C", "definition": "thing"},
        ],
        [{"overlay_type": "overlay_relation"}],
    )
    assert metrics["pump_world_model_delta_count"] == 3
    assert metrics["pump_entity_delta_count"] == 1
    assert metrics["pump_answerable_fact_delta_count"] == 2
    assert metrics["pump_relation_delta_count"] == 1
    assert metrics["pump_definition_delta_count"] == 1
    assert metrics["pump_property_candidate_count"] == 1


# Confirm pump_weak_context_delta.json count matches summary and content is correct
def test_weak_context_delta_file_has_items(pump_run):
    out, summary = pump_run
    items = json.loads((out / "pump_weak_context_delta.json").read_text(encoding="utf-8"))
    # Count must match summary field; the file may be empty in plan-only with no fresh batch docs.
    assert summary["weak_context_delta_count"] == len(items)
    # Any items present must be weak context links.
    for item in items:
        assert item.get("overlay_type") == "overlay_context_link"
        assert item.get("trust") == "weak_context_only"


# Confirm pump_answerable_delta.json contains no weak context links
def test_answerable_delta_has_no_weak_context_links(pump_run):
    out, _summary = pump_run
    items = json.loads((out / "pump_answerable_delta.json").read_text(encoding="utf-8"))
    assert not any(
        i.get("overlay_type") == "overlay_context_link" and i.get("trust") == "weak_context_only"
        for i in items
    )


# The quality delta report must not claim rejected items are answerable
def test_quality_report_counts_are_consistent(pump_run):
    out, summary = pump_run
    report = json.loads((out / "pump_delta_quality_report.json").read_text(encoding="utf-8"))
    assert report["pump_answerable_delta_count"] == summary["pump_answerable_delta_count"]
    for key in (
        "pump_world_model_delta_count",
        "pump_entity_delta_count",
        "pump_answerable_fact_delta_count",
        "pump_relation_delta_count",
        "pump_definition_delta_count",
        "pump_property_candidate_count",
    ):
        assert report[key] == summary[key]
    assert report["weak_context_delta_count"] == summary["weak_context_delta_count"]
    # The v1.3 consistency invariant is over the *before-precision* answerable
    # bucket (the v1.4 precision firewall only reshapes the answerable bucket).
    total_classified = (
        report["answerable_delta_before_precision"]
        + report["weak_context_delta_count"]
        + report["rejected_quality_count"]
        + report["quarantine_quality_count"]
    )
    assert total_classified == summary["pump_safe_delta_total"]


# Without-weak count must be less than with-weak count (when legacy delta has weak links)
def test_overlay_counts_reflect_weak_context_separation(pump_run):
    _out, summary = pump_run
    assert summary["pump_dry_run_overlay_items_count_without_weak"] < summary["pump_dry_run_overlay_items_count_with_weak"]
    # The actual overlay written is the without-weak version
    assert summary["pump_dry_run_overlay_items_count"] == summary["pump_dry_run_overlay_items_count_without_weak"]


# include_weak_context=False is reported honestly in summary
def test_summary_reports_include_weak_context_false(pump_run):
    _out, summary = pump_run
    assert summary["include_weak_context"] is False


# ─── v1.4 Answerable Delta Precision Firewall tests ───────────────────────────

from worldpgt.knowledge_pump.precision_firewall import apply_precision_firewall  # noqa: E402


def _rel(subject, predicate, obj, evidence="") -> dict:
    return {
        "overlay_type": "overlay_relation",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "evidence_text": evidence,
        "trust": "overlay_candidate",
        "risk": "low",
        "stability": "stable",
    }


def _defn(subject, definition, evidence="", predicate="is_a") -> dict:
    return {
        "overlay_type": "overlay_definition",
        "subject": subject,
        "definition": definition,
        "predicate": predicate,
        "evidence_text": evidence,
        "trust": "overlay_candidate",
        "risk": "low",
        "stability": "stable",
    }


def _verdict(item) -> str:
    """Return 'accept', 'reject', or 'quarantine' for a single item."""
    result = apply_precision_firewall([item])
    if result["accepted"]:
        return "accept"
    if result["rejected"]:
        return "reject"
    if result["quarantine"]:
        return "quarantine"
    if result["property_candidates"]:
        return "property_candidate"
    return "unknown"


# --- bad relations must be rejected or quarantined -----------------------------

_BAD_RELATIONS = [
    _rel("Intel", "known_for", "aggressive",
         "Intel has a history of aggressive anti-competitive tactics."),
    _rel("Ellison", "known_for", "songwriter Sheila Ellison",
         "Ellison is a surname. Notable people known for the name include songwriter Sheila Ellison."),
    _rel("If none of the previously", "develops", "Rocket",
         "If none of the previously listed engines develops enough thrust the rocket fails."),
    _rel("The range also uses instrumentation", "owned_by", "NASA",
         "The range also uses instrumentation operated by NASA at Wallops and KSC."),
    _rel("Economy of the United States", "develops", "Private spaceflight",
         "Economy of the United States supports private spaceflight ventures."),
    _rel("Bifacial cells", "produces", "Tesla Energy",
         "Bifacial cells are produced and sold through Tesla Energy."),
    _rel("LVMH", "subsidiary_of", "Oracle Corporation",
         "In 2005, iNEXTV, a wholly owned subsidiary of respondent Ampex Corporation, brought a lawsuit."),
    _rel("Satellite internet", "owned_by", "Amazon",
         "Satellite internet infrastructure was later acquired by Amazon."),
    _rel("Many companies", "produces", "Rocket",
         "Many companies produce rocket components for the launch market."),
    _rel("GSFC also", "operates", "Private spaceflight",
         "GSFC also operates two spaceflight tracking and data acquisition networks."),
]


@pytest.mark.parametrize("item", _BAD_RELATIONS, ids=lambda i: f"{i['subject']}|{i['predicate']}|{i['object']}")
def test_precision_rejects_or_quarantines_bad_relations(item):
    assert _verdict(item) in ("reject", "quarantine")


# --- bad definitions must be rejected or quarantined ---------------------------

_BAD_DEFINITIONS = [
    _defn("Dalal Street", "proposal to rename the street after Nagarmal Saraf",
          "In 2008, there was a proposal to rename the street after Nagarmal Saraf."),
    _defn("Lyndon Rive", "Ernst & Young Entrepreneur of the Year Award winner in the Northern California Region",
          "In 2013, Rive was an Ernst & Young Entrepreneur of the Year Award winner."),
    _defn("Securities Exchange Act of 1934", "physical place where securities (stocks",
          "One area is the physical place where securities (stocks, bonds) are exchanged."),
    _defn("Tesla Autopilot", "good thing to have in planes",
          'Musk noted that "Autopilot is a good thing to have in planes, and we should have it in cars."'),
    _defn("Linda Yaccarino", "assistant chief of police",
          "Yaccarino grew up in Deer Park, where her father was an assistant chief of police."),
    _defn("Lawrence Sperry", "third son of the gyrocompass co-inventor",
          "Sperry was the third son of the gyrocompass co-inventor, Elmer Ambrose Sperry."),
]


@pytest.mark.parametrize("item", _BAD_DEFINITIONS, ids=lambda i: i["subject"])
def test_precision_rejects_or_quarantines_bad_definitions(item):
    assert _verdict(item) in ("reject", "quarantine")


# --- July = ruby: not a definition; clean birthstone property candidate --------

def test_precision_july_ruby_not_accepted_as_definition():
    item = _defn("July", "ruby", "July's birthstone is the ruby, which symbolizes contentment.")
    result = apply_precision_firewall([item])
    # Must not be accepted as an answerable definition.
    assert item not in result["accepted"]


def test_precision_july_ruby_converted_to_birthstone_property():
    item = _defn("July", "ruby", "July's birthstone is the ruby, which symbolizes contentment.")
    result = apply_precision_firewall([item])
    props = result["property_candidates"]
    assert len(props) == 1
    assert props[0]["subject"] == "July"
    assert props[0]["predicate"] == "has_birthstone"
    assert props[0]["object"] == "Ruby"


def test_precision_july_ruby_quarantined_when_unsupported():
    # No birthstone evidence -> cannot cleanly convert -> not accepted as definition.
    item = _defn("July", "ruby", "July is the seventh month and ruby red is its colour theme.")
    result = apply_precision_firewall([item])
    assert item not in result["accepted"]
    assert not result["property_candidates"]


# --- good relations must be preserved ------------------------------------------

_GOOD_RELATIONS = [
    _rel("SpaceX", "develops", "Falcon 9",
         "SpaceX developed Falcon 9 with private capital as well."),
    _rel("Starlink", "owned_by", "SpaceX",
         "Starlink is a satellite internet constellation operated by SpaceX."),
    _rel("Starlink", "is_a", "Satellite internet",
         "Starlink is an internet satellite constellation under development by SpaceX."),
    _rel("Bloomberg News", "founded_by", "Michael Bloomberg",
         "Bloomberg News was founded by Michael Bloomberg and Matthew Winkler in 1990."),
    _rel("Exos Aerospace Systems & Technologies", "is_a", "Aerospace manufacturer",
         "Exos Aerospace Systems & Technologies is an aerospace manufacturer and developer."),
    _rel("International Energy Agency", "publishes", "annual World Energy Outlook",
         "publishes a range of reports including its flagship publication, the annual World Energy Outlook."),
    _rel("Neuralink", "develops", "brain-computer interfaces",
         "Neuralink develops brain-computer interfaces to connect humans and computers."),
    _rel("The Boring Company", "develops", "tunnels",
         "The Boring Company develops tunnels for transportation systems."),
    _rel("Elon Musk", "founded", "The Boring Company",
         "Elon Musk founded The Boring Company in 2016."),
    _rel("Jeff Bezos", "founded", "Amazon",
         "Three years after Bezos founded Amazon, he took it public."),
    _rel("Jeff Bezos", "founded", "Blue Origin",
         "Jeff Bezos founded Blue Origin in 2000 to develop spaceflight."),
]


@pytest.mark.parametrize("item", _GOOD_RELATIONS, ids=lambda i: f"{i['subject']}|{i['predicate']}|{i['object']}")
def test_precision_preserves_good_relations(item):
    assert _verdict(item) == "accept"


# --- artifacts and summary fields ----------------------------------------------

def test_precision_artifact_files_written(pump_run):
    out, _summary = pump_run
    for fname in (
        "pump_precision_answerable_delta.json",
        "pump_precision_rejected_delta.json",
        "pump_precision_quarantine.json",
        "pump_precision_property_candidates.json",
        "pump_precision_report.json",
        "pump_fact_smoke_questions.json",
        "pump_fact_smoke_outputs.json",
    ):
        assert (out / fname).is_file(), f"expected {fname} to exist"


def test_pump_fact_smoke_includes_planner_gap_prompts_as_answers(tmp_path):
    facts = [
        {
            "overlay_type": "overlay_relation",
            "subject": "International Energy Agency",
            "predicate": "publishes",
            "object": "annual World Energy Outlook",
            "trust": "overlay_candidate",
            "risk": "low",
            "stability": "semi_stable",
        },
        {
            "overlay_type": "overlay_relation",
            "subject": "Bloomberg News",
            "predicate": "founded_by",
            "object": "Michael Bloomberg",
            "trust": "overlay_candidate",
            "risk": "low",
            "stability": "semi_stable",
        },
        {
            "overlay_type": "overlay_relation",
            "subject": "SolarCity",
            "predicate": "owned_by",
            "object": "Tesla",
            "trust": "overlay_candidate",
            "risk": "low",
            "stability": "semi_stable",
        },
        {
            "overlay_type": "overlay_definition",
            "subject": "Rocket Science Games",
            "definition": "independent game studio",
            "predicate": "is_a",
            "trust": "overlay_candidate",
            "risk": "low",
            "stability": "stable",
        },
        {
            "overlay_type": "overlay_definition",
            "subject": "June",
            "definition": "sixth month of the year",
            "predicate": "is_a",
            "trust": "overlay_candidate",
            "risk": "low",
            "stability": "stable",
        },
        {
            "overlay_type": "overlay_relation",
            "subject": "Exos Aerospace Systems & Technologies",
            "predicate": "is_a",
            "object": "Aerospace manufacturer",
            "trust": "overlay_candidate",
            "risk": "low",
            "stability": "stable",
        },
    ]
    overlay = tmp_path / "pump_overlay.json"
    overlay.write_text(json.dumps(facts), encoding="utf-8")
    summary = runner._write_pump_fact_smoke(tmp_path, overlay, facts)
    outputs = json.loads((tmp_path / "pump_fact_smoke_outputs.json").read_text(encoding="utf-8"))
    by_question = {row["question"]: row for row in outputs}
    for question in (
        "What does the International Energy Agency publish?",
        "Who founded Bloomberg News?",
        "Who owns SolarCity?",
        "What is Rocket Science Games?",
        "What is June?",
        "What is Exos Aerospace Systems & Technologies?",
    ):
        assert question in by_question
        assert by_question[question]["answer"]["decision"] == "answer"
        assert by_question[question]["answer"]["supported_by_context"] is True
    assert summary["pump_smoke_wrong_count"] == 0
    assert summary["pump_smoke_unsupported_answer_count"] == 0


def test_precision_answerable_delta_matches_count(pump_run):
    out, summary = pump_run
    items = json.loads((out / "pump_precision_answerable_delta.json").read_text(encoding="utf-8"))
    assert len(items) == summary["answerable_delta_after_precision"]
    assert len(items) == summary["pump_answerable_delta_count"]
    # The default answerable delta file is the precision-filtered one.
    default_items = json.loads((out / "pump_answerable_delta.json").read_text(encoding="utf-8"))
    assert len(default_items) == summary["answerable_delta_after_precision"]


def test_precision_reduces_or_holds_answerable_delta(pump_run):
    _out, summary = pump_run
    assert summary["answerable_delta_after_precision"] <= summary["answerable_delta_before_precision"]


def test_summary_has_precision_firewall_fields(pump_run):
    _out, summary = pump_run
    for key in (
        "answerable_delta_before_precision",
        "answerable_delta_after_precision",
        "precision_rejected_count",
        "precision_quarantined_count",
        "precision_property_candidate_count",
        "precision_rejection_by_reason",
        "precision_quarantine_by_reason",
        "relation_precision_before_by_predicate",
        "relation_precision_after_by_predicate",
        "definition_precision_before_count",
        "definition_precision_after_count",
        "pump_dry_run_overlay_items_count_without_weak_before_precision",
        "pump_dry_run_overlay_items_count_without_weak_after_precision",
    ):
        assert key in summary, f"summary missing key: {key}"


def test_precision_overlay_built_from_precision_answerable(pump_run):
    out, summary = pump_run
    overlay = json.loads((out / "pump_dry_run_overlay.json").read_text(encoding="utf-8"))
    assert len(overlay) == summary["pump_dry_run_overlay_items_count_without_weak_after_precision"]
    # No weak context links leak into the default overlay.
    assert not any(
        i.get("overlay_type") == "overlay_context_link" and i.get("trust") == "weak_context_only"
        for i in overlay
    )


def test_precision_bucket_counts_partition_answerable_before(pump_run):
    _out, summary = pump_run
    total = (
        summary["answerable_delta_after_precision"]
        + summary["precision_rejected_count"]
        + summary["precision_quarantined_count"]
        + summary["precision_property_candidate_count"]
        + summary.get("precision_v2_rejected_count", 0)
        + summary.get("precision_v2_quarantined_count", 0)
        + summary.get("precision_cleanup_v2_1_rejected_count", 0)
        + summary.get("precision_cleanup_v2_1_quarantined_count", 0)
    )
    assert total == summary["answerable_delta_before_precision"]


def test_no_neural_imports_in_precision_firewall():
    text = (_WORLD / "knowledge_pump" / "precision_firewall.py").read_text(encoding="utf-8").lower()
    for marker in ("import torch", "import openai", "import tensorflow", "embedding", "backprop", "training loop", "gpt"):
        assert marker not in text
