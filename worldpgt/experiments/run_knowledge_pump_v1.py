"""Knowledge Pump 5000 Mode v1.

Checkpointed, batch-based, local-first Wikipedia snapshot expansion planner.
Network fetches are disabled by default and only run with ``--allow-network``.

This runner writes only under ``worldpgt/experiments/knowledge_pump_v1/`` and
never mutates accepted memory, accepted/promoted overlays, snapshot dry-run
overlay, runtime behavior, planner thresholds, validators, ingestion semantics,
or overlay builder semantics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

print("[INIT] importing worldpgt modules...", flush=True)
from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
print("[INIT] answer_orchestrator OK", flush=True)
from worldpgt.knowledge_pump.expanded_allowlist_builder import (
    build_expanded_allowlist,
    write_allowlist,
)
print("[INIT] expanded_allowlist_builder OK", flush=True)
from worldpgt.knowledge_pump.dynamic_frontier import (
    load_dynamic_frontier,
    merge_frontiers,
    update_dynamic_frontier_from_fetch_rows,
)
print("[INIT] dynamic_frontier OK", flush=True)
from worldpgt.knowledge_pump.frontier_title_extractor import extract_frontier_titles
print("[INIT] frontier_title_extractor OK", flush=True)
from worldpgt.knowledge_pump.pump_batch_planner import plan_batches
from worldpgt.knowledge_pump.pump_batch_runner import run_fetch_batch
from worldpgt.knowledge_pump.pump_checkpoint import (
    append_history,
    load_checkpoint,
    load_history,
    utc_now,
    write_checkpoint,
)
print("[INIT] pump_checkpoint OK", flush=True)
from worldpgt.knowledge_pump.delta_quality_firewall import classify_pump_delta, is_weak_context
print("[INIT] delta_quality_firewall OK", flush=True)
from worldpgt.knowledge_pump.precision_firewall import apply_precision_firewall
from worldpgt.knowledge_pump.precision_firewall_v2 import (
    apply_precision_firewall_v2,
    write_precision_v2_artifacts,
)
from worldpgt.knowledge_pump.precision_cleanup_v2_1 import (
    apply_precision_cleanup_v2_1,
    write_cleanup_v2_1_artifacts,
)
print("[INIT] precision firewalls OK", flush=True)
from worldpgt.knowledge_pump.pump_regression_runner import run_assistant_regression
print("[INIT] pump_regression_runner OK", flush=True)
from worldpgt.knowledge_pump.pump_report import build_summary, write_frontier, write_json
from worldpgt.knowledge_pump.safe_delta_merger import merge_with_fresh_deltas
from worldpgt.knowledge_pump.title_ranker import normalize_title
from worldpgt.knowledge_pump.types import ExpandedAllowlistEntry, PumpCheckpoint
from worldpgt.knowledge_pump.extraction_yield_v2 import (
    extract_yield_v2,
    write_extraction_yield_v2_artifacts,
)
print("[INIT] extraction_yield_v2 OK", flush=True)
from worldpgt.schema_induction.run_schema_induction import run_induction as _si_run_induction
from worldpgt.schema_induction.overlay_adapter import schema_result_to_overlay_items as _si_to_overlay
from worldpgt.schema_induction.induced_precision_gate import (
    apply_induced_precision_gate as _si_precision_gate,
    dedup_and_merge as _si_dedup_merge,
)
print("[INIT] schema_induction OK", flush=True)
from worldpgt.knowledge_pump.pump_summary_qa_preserver import merge_qa_preservation
from worldpgt.knowledge_pump.yield_ranked_frontier import run_yield_ranked_frontier
from worldpgt.knowledge_pump.yield_ranked_frontier import normalize_page
print("[INIT] yield_ranked_frontier OK", flush=True)
from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex
print("[INIT] entity_surface_index OK", flush=True)
from worldpgt.relation_extraction_v2.relation_candidate_extractor import extract_all_candidates
print("[INIT] relation_candidate_extractor OK", flush=True)
from worldpgt.relation_extraction_v2.relation_candidate_validator import validate_candidates
from worldpgt.wiki_snapshot_ingestion.snapshot_batch_ingestor import ingest_snapshot_batch, write_candidates
from worldpgt.wiki_snapshot_ingestion.types import ReadySnapshotDoc
from worldpgt.wiki_snapshots.snapshot_normalizer import safe_title_filename
from worldpgt.wiki_snapshots.snapshot_readiness import evaluate_snapshot_readiness
from worldpgt.wiki_snapshots.types import PageSnapshot
print("[INIT] all imports done", flush=True)

_EXPERIMENTS = Path(__file__).resolve().parent
_OUT = _EXPERIMENTS / "knowledge_pump_v1"
_SNAPSHOTS = _EXPERIMENTS / "wiki_snapshots_v1"
_SNAPSHOT_INGESTION = _EXPERIMENTS / "wiki_snapshot_ingestion_v1"
_RELATION_V2 = _EXPERIMENTS / "relation_extraction_v2"
_KNOWLEDGE_REQUESTS = _EXPERIMENTS / "knowledge_requests_v1" / "candidate_knowledge_requests.json"
_ACCEPTED = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_PROMOTED = _EXPERIMENTS / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json"
_SNAPSHOT_DRY_RUN = _SNAPSHOT_INGESTION / "snapshot_dry_run_overlay.json"
_TRUSTED = _EXPERIMENTS / "accepted_knowledge_memory_v1.json"
_SENSE_MEMORY = _ROOT / "worldpgt" / "continuation" / "sense_memory.py"

_CHECKPOINT = _OUT / "pump_checkpoint.json"
_HISTORY = _OUT / "pump_batch_history.json"
_ALLOWLIST = _OUT / "expanded_allowlist.json"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING"


def _protected_hashes() -> dict[str, str]:
    return {
        "trusted_memory": _hash(_TRUSTED),
        "accepted_overlay": _hash(_ACCEPTED),
        "promoted_overlay": _hash(_PROMOTED),
        "snapshot_dry_run_overlay": _hash(_SNAPSHOT_DRY_RUN),
        "sense_memory": _hash(_SENSE_MEMORY),
    }


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _pump_delta_metrics(
    _entity_delta: list[dict],
    precision_delta: list[dict],
    property_candidates: list[dict],
) -> dict[str, int]:
    # The honest split is by overlay_type in the precision delta.
    entity_count = sum(
        1 for item in precision_delta if item.get("overlay_type") == "overlay_entity"
    )
    relation_count = sum(
        1 for item in precision_delta if item.get("overlay_type") == "overlay_relation"
    )
    definition_count = sum(
        1 for item in precision_delta if item.get("overlay_type") == "overlay_definition"
    )
    return {
        "pump_world_model_delta_count": entity_count + relation_count + definition_count,
        "pump_entity_delta_count": entity_count,
        "pump_answerable_fact_delta_count": relation_count + definition_count,
        "pump_relation_delta_count": relation_count,
        "pump_definition_delta_count": definition_count,
        "pump_property_candidate_count": len(property_candidates),
    }


def _question_for_fact(item: dict) -> tuple[str, str] | None:
    typ = item.get("overlay_type")
    subject = str(item.get("subject", "")).strip()
    if typ == "overlay_definition":
        definition = str(item.get("definition", "")).strip()
        if subject and definition:
            return f"What is {subject}?", definition
        return None
    if typ != "overlay_relation":
        return None
    predicate = str(item.get("predicate", "")).strip()
    obj = str(item.get("object", "")).strip()
    if not subject or not predicate or not obj:
        return None
    if predicate == "develops":
        return f"What does {subject} develop?", obj
    if predicate == "produces":
        return f"What does {subject} produce?", obj
    if predicate == "publishes":
        return f"What does {subject} publish?", obj
    if predicate == "founded":
        return f"What did {subject} found?", obj
    if predicate == "founded_by":
        return f"Who founded {subject}?", obj
    if predicate == "owned_by":
        return f"Who owns {subject}?", obj
    if predicate == "is_a":
        return f"What is {subject}?", obj
    return None


def _write_pump_fact_smoke(
    out_dir: Path,
    overlay_path: Path,
    answerable_delta: list[dict],
) -> dict[str, int]:
    questions: list[dict] = []
    seen: set[str] = set()

    def _add_question(question: str, expected: str, item: dict) -> None:
        if question in seen:
            return
        seen.add(question)
        questions.append(
            {
                "id": f"pump-fact-{len(questions) + 1:03d}",
                "question": question,
                "expected_support_text": expected,
                "source_fact": item,
            }
        )

    for idx, item in enumerate(answerable_delta, start=1):
        pair = _question_for_fact(item)
        if pair is None:
            continue
        question, expected = pair
        _add_question(question, expected, item)

    required_questions = [
        ("International Energy Agency", "publishes", "What does the International Energy Agency publish?"),
        ("Bloomberg News", "founded_by", "Who founded Bloomberg News?"),
        ("SolarCity", "owned_by", "Who owns SolarCity?"),
        ("Rocket Science Games", "definition", "What is Rocket Science Games?"),
        ("June", "definition", "What is June?"),
        ("Exos Aerospace Systems & Technologies", "is_a", "What is Exos Aerospace Systems & Technologies?"),
    ]
    for subject, predicate, question in required_questions:
        for item in answerable_delta:
            if item.get("subject") != subject:
                continue
            if predicate == "definition" and item.get("overlay_type") == "overlay_definition":
                _add_question(question, str(item.get("definition", "")), item)
                break
            if item.get("overlay_type") == "overlay_relation" and item.get("predicate") == predicate:
                _add_question(question, str(item.get("object", "")), item)
                break

    write_json(out_dir / "pump_fact_smoke_questions.json", questions)

    orchestrator = AnswerOrchestrator("pump-dry-run", overlay_path=str(overlay_path))
    outputs: list[dict] = []
    counts = Counter()
    for row in questions:
        answer = orchestrator.answer(row["question"])
        answer_text = answer.answer_text or ""
        expected = str(row["expected_support_text"])
        expected_seen = expected.lower() in answer_text.lower()
        if answer.decision == "answer":
            counts["answer"] += 1
            if not answer.supported_by_context:
                counts["unsupported_answer"] += 1
            elif expected_seen:
                counts["supported_fact_answer"] += 1
            else:
                counts["wrong"] += 1
        else:
            counts["audit"] += 1
            counts["planner_gap"] += 1
        outputs.append(
            {
                **row,
                "answer": answer.to_dict(),
                "expected_support_seen": expected_seen,
                "classification": "answer" if answer.decision == "answer" else "planner_gap",
            }
        )

    write_json(out_dir / "pump_fact_smoke_outputs.json", outputs)
    return {
        "pump_smoke_prompt_count": len(questions),
        "pump_smoke_answer_count": counts["answer"],
        "pump_smoke_audit_count": counts["audit"],
        "pump_smoke_wrong_count": counts["wrong"],
        "pump_smoke_unsupported_answer_count": counts["unsupported_answer"],
        "pump_smoke_planner_gap_count": counts["planner_gap"],
        "pump_smoke_supported_fact_answer_count": counts["supported_fact_answer"],
    }


def _load_allowlist(path: Path) -> list[ExpandedAllowlistEntry]:
    rows = _read_json(path, [])
    return [ExpandedAllowlistEntry(**row) for row in rows]


def _frontier_and_allowlist(
    target_total: int,
    batch_size: int,
    already_fetched: set[str],
    out_dir: Path = _OUT,
) -> tuple[list, list[ExpandedAllowlistEntry]]:
    static_frontier = extract_frontier_titles(
        _SNAPSHOTS,
        _RELATION_V2 / "relation_extraction_v2_candidates.json",
        _KNOWLEDGE_REQUESTS,
        [_ACCEPTED, _PROMOTED, _SNAPSHOT_DRY_RUN],
    )
    dynamic_frontier = load_dynamic_frontier(out_dir / "dynamic_frontier_titles.json")
    frontier = merge_frontiers([static_frontier, dynamic_frontier])
    allowlist = build_expanded_allowlist(frontier, target_total, batch_size, already_fetched)
    return frontier, allowlist


def _write_batch_plan(path: Path, batches: list[list[ExpandedAllowlistEntry]]) -> None:
    write_json(
        path,
        [
            {
                "batch_index": idx,
                "titles": [e.normalized_title for e in batch],
                "count": len(batch),
            }
            for idx, batch in enumerate(batches)
        ],
    )


def _copy_existing_summaries(out: Path) -> tuple[dict, dict]:
    ingestion_summary = _read_json(_SNAPSHOT_INGESTION / "snapshot_ingestion_summary.json", {})
    relation_summary = _read_json(_RELATION_V2 / "relation_extraction_v2_summary.json", {})
    write_json(out / "pump_snapshot_ingestion_summary.json", ingestion_summary)
    write_json(out / "pump_relation_extraction_summary.json", relation_summary)
    return ingestion_summary, relation_summary


def _collection_summary(history, allow_network: bool) -> dict[str, Any]:
    return {
        "allow_network": allow_network,
        "batches_completed": sum(1 for h in history if h.status == "completed"),
        "fetched_count_total": sum(len(h.titles_fetched) for h in history),
        "fetch_success_count_total": sum(len(h.fetch_success) for h in history),
        "fetch_failed_count_total": sum(len(h.fetch_failed) for h in history),
        "ready_for_ingestion_count_total": sum(h.ready_count for h in history),
        "not_ready_count_total": sum(h.not_ready_count for h in history),
        "network_calls": sum(h.network_calls for h in history),
    }


def _ready_batch_docs(batch_dir: Path) -> list[ReadySnapshotDoc]:
    raw_dir = batch_dir / "raw_snapshots"
    doc_dir = batch_dir / "normalized_docs"
    docs: list[ReadySnapshotDoc] = []
    if not raw_dir.exists():
        return docs
    for raw_path in sorted(raw_dir.glob("*.json")):
        row = _read_json(raw_path, {})
        title = str(row.get("normalized_title") or row.get("title") or "")
        doc_path = doc_dir / f"{safe_title_filename(title)}.md"
        snapshot = PageSnapshot(
            title=str(row.get("title") or title),
            normalized_title=title,
            pageid=row.get("pageid"),
            revision_id=row.get("revision_id"),
            timestamp=str(row.get("timestamp") or ""),
            source_url=str(row.get("source_url") or ""),
            api_url=str(row.get("api_url") or ""),
            retrieved_at=str(row.get("retrieved_at") or ""),
            raw_text=str(row.get("raw_text") or ""),
            raw_text_sha256=str(row.get("raw_text_sha256") or ""),
            license_note=str(row.get("license_note") or ""),
            fetch_status=str(row.get("fetch_status") or "error"),
            error=str(row.get("error") or ""),
            links=list(row.get("links") or []),
        )
        if not evaluate_snapshot_readiness(snapshot, doc_path).ready_for_self_ingestion:
            continue
        docs.append(
            ReadySnapshotDoc(
                title=str(row.get("title") or title),
                normalized_title=title,
                source_url=str(row.get("source_url")),
                retrieved_at=str(row.get("retrieved_at")),
                revision_id=row.get("revision_id"),
                raw_text_sha256=str(row.get("raw_text_sha256")),
                normalized_doc_path=str(doc_path),
                manifest_row=dict(row),
            )
        )
    return docs


def _load_overlay_items() -> list[dict]:
    items: list[dict] = []
    for path in (_ACCEPTED, _PROMOTED, _SNAPSHOT_DRY_RUN):
        data = _read_json(path, [])
        if isinstance(data, list):
            items.extend(data)
    return items


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_relation_candidates(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_json(out_dir / "pump_fresh_relation_candidates.json", rows)
    _write_csv(
        out_dir / "pump_fresh_relation_candidates.csv",
        rows,
        [
            "id", "subject", "relation", "object", "confidence", "stability",
            "risk", "source_title", "pattern_id", "safe_for_overlay_delta",
        ],
    )


def _load_gap_frontier_signals(out_dir: Path) -> tuple[set[str], dict[str, dict]]:
    """Load audit-driven frontier boost signals from ``gap_report.json``.

    ``run_audit_driven_pump_v1.py`` is responsible for writing the production
    gap report with eligibility filtering. The pump runner consumes that report
    so the actual fetch batch uses the same cleaned acquisition targets.
    """

    report = _read_json(out_dir / "gap_report.json", {})
    if not isinstance(report, dict):
        return set(), {}

    gap_entities = {
        normalize_page(row.get("entity", ""))
        for row in report.get("acquisition_candidates", [])
        if isinstance(row, dict)
        and row.get("entity")
        and row.get("entity") != "[unknown entity]"
    }
    gap_entities = {entity for entity in gap_entities if entity}

    stale_signals: dict[str, dict] = {}
    for row in report.get("stale_candidates", []):
        if not isinstance(row, dict):
            continue
        key = normalize_page(row.get("subject", ""))
        if not key:
            continue
        previous = stale_signals.get(key)
        if previous is None or row.get("staleness_ratio", 0) > previous.get("staleness_ratio", 0):
            stale_signals[key] = row
    return gap_entities, stale_signals


def _write_empty_fresh_artifacts(out_dir: Path) -> None:
    write_json(out_dir / "pump_fresh_ingestion_candidates.json", [])
    _write_csv(out_dir / "pump_fresh_ingestion_candidates.csv", [], ["candidate_id", "source_doc_title", "source_doc_hash", "item_type"])
    _write_relation_candidates(out_dir, [])
    for name in (
        "pump_fresh_quarantine.json",
        "pump_fresh_duplicates.json",
        "pump_fresh_conflicts.json",
        "pump_fresh_safe_delta.json",
    ):
        write_json(out_dir / name, [])
    write_json(
        out_dir / "pump_fresh_delta_report.json",
        {
            "summary": {
                "fresh_ingestion_candidates_total": 0,
                "fresh_relation_candidates_total": 0,
                "fresh_candidates_total": 0,
                "fresh_safe_snapshot_delta_count": 0,
                "fresh_safe_relation_delta_count": 0,
                "fresh_merged_safe_delta_count": 0,
                "fresh_duplicates_count": 0,
                "fresh_conflicts_count": 0,
                "fresh_quarantined_count": 0,
                "fresh_rejected_count": 0,
            },
            "explanation": "No current batch ready docs were available for fresh recomputation.",
        },
    )


def _run_schema_induction_on_docs(
    ready_docs: "list[ReadySnapshotDoc]",
    path_a_items: list[dict[str, Any]],
    out_dir: Path,
    *,
    min_source_docs_generated: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run schema_induction on ready_docs and produce overlay items.

    Returns (merged_items, schema_summary) where merged_items already
    includes path_a_items with schema-induced items deduplicated/merged.
    """
    from worldpgt.schema_induction.promotion_gates import GateConfig
    from worldpgt.schema_induction.types import DocumentRecord
    from worldpgt.relation_extraction_v2.sentence_splitter import extract_full_body

    docs = []
    for rd in ready_docs:
        p = Path(rd.normalized_doc_path)
        if not p.exists():
            continue
        raw_text = p.read_text(encoding="utf-8")
        # Strip metadata header (Source:, Retrieved at:, etc.) — same as extraction_yield_v2.
        body = extract_full_body(raw_text)
        if not body.strip():
            continue
        docs.append(DocumentRecord(
            doc_id=rd.normalized_title or rd.title,
            title=rd.title,
            url=rd.source_url or "",
            text=body,
        ))
    docs = [d for d in docs if d.text.strip()]
    if not docs:
        return path_a_items, {"schema_induction_docs": 0, "schema_induction_raw_items": 0,
                               "schema_induction_after_gate": 0, "schema_induction_new_induced": 0,
                               "schema_induction_deduped": 0, "schema_induction_quarantined": 0}

    si_cfg = GateConfig(min_evidence=2, min_sources=1)
    si_result = _si_run_induction([{"doc_id": d.doc_id, "title": d.title, "url": d.url, "text": d.text} for d in docs], si_cfg)

    raw_overlay = _si_to_overlay(
        list(si_result.families),
        list(si_result.frames),
        list(si_result.claims),
        include_generated=True,
    )

    # Apply induced precision gate.
    gated = _si_precision_gate(raw_overlay, min_source_docs_generated=min_source_docs_generated)
    gated_items = gated["accepted"]

    # Dedup against Path-A.
    merge_result = _si_dedup_merge(path_a_items, gated_items)
    merged = merge_result["accepted"]

    si_out_dir = out_dir / "schema_induction_run"
    si_out_dir.mkdir(parents=True, exist_ok=True)
    from worldpgt.schema_induction import schema_store
    schema_store.write_result(si_result, si_out_dir)
    write_json(si_out_dir / "schema_induced_overlay_raw.json", raw_overlay)
    write_json(si_out_dir / "schema_induced_overlay_gated.json", gated_items)
    write_json(si_out_dir / "schema_induced_overlay_new.json", merge_result["new_induced"])

    schema_summary = {
        "schema_induction_enabled": True,
        "schema_induction_docs": len(docs),
        "schema_induction_families_generated": si_result.summary.get("relation_families_generated", 0),
        "schema_induction_families_promoted": si_result.summary.get("relation_families_promoted", 0),
        "schema_induction_raw_items": len(raw_overlay),
        "schema_induction_after_gate": len(gated_items),
        "schema_induction_quarantined": len(gated["quarantine"]),
        "schema_induction_rejected": len(gated["rejected"]),
        "schema_induction_new_induced": len(merge_result["new_induced"]),
        "schema_induction_deduped": len(merge_result["deduped"]),
        "schema_induction_upgraded": len(merge_result["upgraded"]),
        "schema_induction_new_by_predicate": _count_by_field(merge_result["new_induced"], "predicate"),
    }
    return merged, schema_summary


def _count_by_field(items: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        val = str(item.get(field) or "unknown")
        counts[val] = counts.get(val, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _recompute_batch_derived(
    out_dir: Path,
    *,
    enable_extraction_yield_v2: bool = True,
    enable_spacy: bool = False,
    enable_schema_induction: bool = False,
) -> dict[str, Any]:
    """Recompute derived counts over existing pump batch snapshots.

    This is intentionally local and read-only. If no batch snapshots exist,
    report zero new docs and no stale reuse.
    """

    print("[RECOMPUTE] loading ready_docs...", flush=True)
    ready_docs = _ready_batch_docs(out_dir / "batch_snapshots")
    print(f"[RECOMPUTE] ready_docs={len(ready_docs)}", flush=True)
    if not ready_docs:
        _write_empty_fresh_artifacts(out_dir)
        summary = {
            "new_ready_docs_this_batch": 0,
            "new_ingestion_candidates_this_batch": 0,
            "new_relation_candidates_this_batch": 0,
            "stale_artifacts_reused": False,
            "recompute_status": "no_batch_ready_docs",
            "extraction_yield_v2_enabled": enable_extraction_yield_v2,
            "extraction_yield_v2_candidate_count": 0,
            "extraction_yield_v2_candidates_count": 0,
            "extraction_yield_v2_rejected_count": 0,
            "extraction_yield_v2_quarantined_count": 0,
            "extraction_yield_v2_candidate_by_pattern": {},
            "_fresh_snapshot_overlay_items": [],
            "_extraction_yield_v2_items": [],
            "_extraction_yield_v2_stats": {"network_calls": False},
            "_fresh_relation_rows": [],
            "_fresh_relation_quarantine": [],
            "_fresh_ingestion_failures": [],
        }
        write_json(out_dir / "pump_batch_recompute_summary.json", {k: v for k, v in summary.items() if not k.startswith("_")})
        return summary

    try:
        print("[RECOMPUTE] ingest_snapshot_batch...", flush=True)
        candidates, overlay_items, failures, by_type, doc_status = ingest_snapshot_batch(ready_docs)
        print(f"[RECOMPUTE] ingest done: {len(candidates)} candidates", flush=True)
        write_candidates(
            candidates,
            out_dir / "pump_fresh_ingestion_candidates.json",
            out_dir / "pump_fresh_ingestion_candidates.csv",
        )
        print("[RECOMPUTE] EntitySurfaceIndex...", flush=True)
        index = EntitySurfaceIndex(_ACCEPTED, _PROMOTED, _SNAPSHOT_DRY_RUN, _SNAPSHOTS / "snapshot_manifest.json")
        print("[RECOMPUTE] extract_all_candidates...", flush=True)
        raw_relations, sentences, errors = extract_all_candidates(ready_docs, index, enable_spacy=enable_spacy)
        print(f"[RECOMPUTE] extract_all_candidates done: {len(raw_relations)} raw", flush=True)
        safe_relations, quarantine, duplicate_ids, conflict_ids = validate_candidates(
            raw_relations, index, _load_overlay_items()
        )
        relation_rows = [c.to_dict() for c in safe_relations]
        _write_relation_candidates(out_dir, relation_rows)

        v2_items: list[dict[str, Any]] = []
        v2_stats: dict[str, Any] = {"network_calls": False}
        v2_summary: dict[str, Any] = {
            "extraction_yield_v2_enabled": enable_extraction_yield_v2,
            "extraction_yield_v2_candidate_count": 0,
            "extraction_yield_v2_rejected_count": 0,
            "extraction_yield_v2_quarantined_count": 0,
            "extraction_yield_v2_candidate_by_pattern": {},
        }
        if enable_extraction_yield_v2:
            print("[RECOMPUTE] extract_yield_v2...", flush=True)
            v2_items, v2_stats = extract_yield_v2(ready_docs, index)
            print(f"[RECOMPUTE] extract_yield_v2 done: {len(v2_items)} items", flush=True)
            if v2_items:
                overlay_items = list(overlay_items) + v2_items
            v2_artifact_dir = out_dir / "extraction_yield_v2"
            v2_summary = write_extraction_yield_v2_artifacts(
                v2_artifact_dir, v2_items, v2_stats
            )
            write_json(out_dir / "pump_fresh_relation_candidates_v2.json", v2_items)
            write_json(out_dir / "pump_fresh_relation_candidates_merged.json", relation_rows + v2_items)
        else:
            write_json(out_dir / "pump_fresh_relation_candidates_v2.json", [])
            write_json(out_dir / "pump_fresh_relation_candidates_merged.json", relation_rows)

        # --- Schema induction parallel path ---
        si_summary: dict[str, Any] = {"schema_induction_enabled": False}
        if enable_schema_induction:
            print("[RECOMPUTE] schema_induction path...", flush=True)
            try:
                overlay_items, si_summary = _run_schema_induction_on_docs(
                    ready_docs, list(overlay_items), out_dir
                )
                print(
                    f"[RECOMPUTE] schema_induction done: "
                    f"{si_summary.get('schema_induction_new_induced', 0)} new items",
                    flush=True,
                )
            except Exception as si_exc:
                print(f"[RECOMPUTE] schema_induction ERROR (non-fatal): {si_exc}", flush=True)
                si_summary = {"schema_induction_enabled": True, "schema_induction_error": str(si_exc)}

        summary = {
            "new_ready_docs_this_batch": len(ready_docs),
            "new_ingestion_candidates_this_batch": len(candidates),
            "new_relation_candidates_this_batch": len(safe_relations),
            "stale_artifacts_reused": False,
            "recompute_status": "recomputed",
            "new_ingestion_candidates_by_type": by_type,
            "new_ingestion_docs_succeeded": doc_status.get("succeeded", 0),
            "new_ingestion_docs_failed": doc_status.get("failed", 0),
            "new_ingestion_failures": [f.to_dict() for f in failures[:20]],
            "new_relation_raw_candidates_this_batch": len(raw_relations),
            "new_relation_quarantined_this_batch": len(quarantine),
            "new_relation_duplicate_ids_this_batch": len(duplicate_ids),
            "new_relation_conflict_ids_this_batch": len(conflict_ids),
            "new_relation_sentences_scanned_this_batch": sentences,
            "new_relation_errors": errors[:20],
            "extraction_yield_v2_enabled": enable_extraction_yield_v2,
            "extraction_yield_v2_candidate_count": v2_summary.get("extraction_yield_v2_candidate_count", 0),
            "extraction_yield_v2_candidates_count": v2_summary.get("extraction_yield_v2_candidate_count", 0),
            "extraction_yield_v2_rejected_count": v2_summary.get("extraction_yield_v2_rejected_count", 0),
            "extraction_yield_v2_quarantined_count": v2_summary.get("extraction_yield_v2_quarantined_count", 0),
            "extraction_yield_v2_candidate_by_pattern": v2_summary.get("extraction_yield_v2_candidate_by_pattern", {}),
            "extraction_yield_v2_relation_candidate_count": v2_summary.get("extraction_yield_v2_relation_candidate_count", 0),
            "extraction_yield_v2_definition_candidate_count": v2_summary.get("extraction_yield_v2_definition_candidate_count", 0),
            "extraction_yield_v2_screened_out_sentence_count": v2_summary.get("screened_out_sentence_count", 0),
            "_fresh_snapshot_overlay_items": overlay_items,
            "_extraction_yield_v2_items": v2_items,
            "_extraction_yield_v2_stats": v2_stats,
            "_fresh_relation_rows": relation_rows,
            "_fresh_relation_quarantine": [q.to_dict() for q in quarantine],
            "_fresh_ingestion_failures": [f.to_dict() for f in failures],
            **si_summary,
        }
    except Exception as exc:
        _write_empty_fresh_artifacts(out_dir)
        summary = {
            "new_ready_docs_this_batch": len(ready_docs),
            "new_ingestion_candidates_this_batch": 0,
            "new_relation_candidates_this_batch": 0,
            "stale_artifacts_reused": True,
            "recompute_status": "failed",
            "recompute_error": str(exc),
            "extraction_yield_v2_enabled": enable_extraction_yield_v2,
            "extraction_yield_v2_candidates_count": 0,
            "_fresh_snapshot_overlay_items": [],
            "_fresh_relation_rows": [],
            "_fresh_relation_quarantine": [],
            "_fresh_ingestion_failures": [],
        }
    write_json(out_dir / "pump_batch_recompute_summary.json", {k: v for k, v in summary.items() if not k.startswith("_")})
    return summary


def _yield_ranked_batches(
    allowlist: list[ExpandedAllowlistEntry],
    yield_titles: list[str] | list[dict],
    batch_size: int,
    max_batches: int,
) -> list[list[ExpandedAllowlistEntry]]:
    """Build batches in yield-ranked plan order.

    The default allowlist is capped by ``target_total`` and may omit most
    high-scoring yield-ranked titles. In yield-ranked mode the plan itself is
    the control surface, so keep existing allowlist entries when present and
    synthesize read-only batch entries for otherwise eligible plan titles.
    No network is touched here; fetch happens later through the normal runner.
    """
    by_key: dict[str, ExpandedAllowlistEntry] = {}
    for entry in allowlist:
        key = normalize_title(entry.normalized_title or entry.title).casefold()
        by_key.setdefault(key, entry)
    selected: list[ExpandedAllowlistEntry] = []
    seen: set[str] = set()
    for idx, plan_item in enumerate(yield_titles):
        if isinstance(plan_item, dict):
            title = str(plan_item.get("title", ""))
            reason = "; ".join(str(r) for r in plan_item.get("positive_reasons", []))
            risk_hint = str(plan_item.get("expected_yield_class") or "yield_ranked")
        else:
            title = str(plan_item)
            reason = "yield-ranked frontier plan"
            risk_hint = "yield_ranked"
        if not title:
            continue
        key = normalize_title(title).casefold()
        if key in seen:
            continue
        seen.add(key)
        entry = by_key.get(key)
        if entry is None:
            entry = ExpandedAllowlistEntry(
                title=title,
                normalized_title=normalize_title(title),
                priority=idx,
                reason=reason or "yield-ranked frontier plan",
                source="yield_ranked_frontier",
                risk_hint=risk_hint,
                already_fetched=False,
                selected_for_batch=True,
                batch_index=len(selected) // batch_size,
            )
        selected.append(entry)
    if not selected:
        return []
    batches = [selected[i:i + batch_size] for i in range(0, len(selected), batch_size)]
    return batches[:max_batches] if max_batches else batches


def run(
    *,
    plan_only: bool = True,
    allow_network: bool = False,
    target_total: int = 5000,
    batch_size: int = 250,
    max_batches: int = 1,
    resume: bool = False,
    status: bool = False,
    user_agent: str = "MicroworldResearchBot/0.1 (local research; contact: YOUR_EMAIL)",
    delay_sec: float = 0.5,
    out_dir: Path = _OUT,
    include_weak_context: bool = False,
    enable_extraction_yield_v2: bool = True,
    enable_spacy: bool = False,
    enable_schema_induction: bool = False,
    frontier_policy: str = "default",
    force_low_yield_fetch: bool = False,
) -> dict[str, Any]:
    print("[RUN] start", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[RUN] _protected_hashes...", flush=True)
    before = _protected_hashes()
    print("[RUN] _protected_hashes done", flush=True)

    if status:
        checkpoint = load_checkpoint(out_dir / "pump_checkpoint.json")
        summary = _read_json(out_dir / "pump_summary.json", {})
        return {"checkpoint": checkpoint.to_dict() if checkpoint else None, "summary": summary}

    checkpoint = load_checkpoint(out_dir / "pump_checkpoint.json") if resume else None
    already_fetched = set(checkpoint.already_fetched_titles if checkpoint else [])
    fetched_existing = {
        normalize_title(str(row.get("normalized_title") or row.get("title") or ""))
        for row in _read_json(_SNAPSHOTS / "snapshot_manifest.json", [])
    }
    already_fetched |= fetched_existing

    print("[RUN] _frontier_and_allowlist...", flush=True)
    frontier, allowlist = _frontier_and_allowlist(target_total, batch_size, already_fetched, out_dir)
    print(f"[RUN] frontier={len(frontier)} allowlist={len(allowlist)}", flush=True)
    write_frontier(frontier, out_dir / "frontier_titles.json", out_dir / "frontier_titles.csv")
    write_allowlist(allowlist, out_dir / "expanded_allowlist.json", out_dir / "expanded_allowlist.csv")
    dynamic_frontier_initial_total = len(load_dynamic_frontier(out_dir / "dynamic_frontier_titles.json"))
    dynamic_frontier_summary: dict[str, Any] = {
        "dynamic_frontier_initial_total": dynamic_frontier_initial_total,
        "dynamic_frontier_previous_total": dynamic_frontier_initial_total,
        "dynamic_frontier_added_this_run": 0,
        "dynamic_frontier_total": dynamic_frontier_initial_total,
        "dynamic_frontier_pages_processed": 0,
        "dynamic_frontier_candidate_count": 0,
        "dynamic_frontier_rejected_already_fetched": 0,
        "dynamic_frontier_rejected_current_allowlist": 0,
        "dynamic_frontier_rejected_hygiene": 0,
        "dynamic_frontier_added_sample": [],
    }

    start_batch = checkpoint.current_batch_index if checkpoint else 0
    print("[RUN] plan_batches...", flush=True)
    default_batches = plan_batches(allowlist, batch_size, max_batches, already_fetched, start_batch)
    print(f"[RUN] plan_batches done: {len(default_batches)} batches", flush=True)

    # --- v1.9 Yield-ranked frontier + incremental yield gate ---
    # Pure selection/control: reads prior pump artifacts, never fetches, never
    # mutates accepted memory/overlays. Produces a gate recommendation and an
    # optional yield-ranked batch selection.
    yield_gate_info: dict[str, Any] = {}
    yield_summary: dict[str, Any] = {}
    yield_ranked_selected_count = 0
    yield_ranked_available_count = 0
    frontier_policy_fallback_reason = ""
    batches = default_batches
    try:
        print("[RUN] _load_gap_frontier_signals...", flush=True)
        gap_entities, stale_fact_signals = _load_gap_frontier_signals(out_dir)
        print(f"[RUN] gap_entities={len(gap_entities)} stale_signals={len(stale_fact_signals)}", flush=True)
        print("[RUN] run_yield_ranked_frontier...", flush=True)
        yield_result = run_yield_ranked_frontier(
            out_dir,
            out_dir / "yield_ranked_frontier_v1",
            snapshots_dir=_SNAPSHOTS,
            force=force_low_yield_fetch,
            frontier_policy=frontier_policy,
            gap_entities=gap_entities,
            stale_fact_signals=stale_fact_signals,
        )
        print("[RUN] run_yield_ranked_frontier done", flush=True)
        yield_gate_info = yield_result["gate"]
        yield_summary = yield_result["incremental_summary"]
        yield_ranked_available_count = yield_summary.get("yield_ranked_available_count", 0)
        if frontier_policy == "yield-ranked":
            yield_batches = _yield_ranked_batches(
                allowlist, yield_result["plan"], batch_size, max_batches
            )
            if yield_batches:
                batches = yield_batches
                yield_ranked_selected_count = sum(len(b) for b in yield_batches)
            else:
                frontier_policy_fallback_reason = "no_yield_ranked_candidates_in_allowlist"
        # Surface the gate warning when a blind network fetch is about to run
        # despite collapsed yield.
        if allow_network and not plan_only and frontier_policy == "default":
            if yield_gate_info.get("blind_fetch_unproductive") and not force_low_yield_fetch:
                for warning in yield_gate_info.get("warnings", []):
                    print(f"WARNING [yield-gate]: {warning}")
    except Exception as exc:  # pragma: no cover - defensive; keep pump robust
        print(f"[RUN] yield_ranked_frontier ERROR: {exc}", flush=True)
        frontier_policy_fallback_reason = f"yield_ranked_frontier_error:{exc}"

    print("[RUN] _write_batch_plan...", flush=True)
    _write_batch_plan(out_dir / "pump_batch_plan.json", batches)

    if checkpoint is None:
        checkpoint = PumpCheckpoint(
            target_total=target_total,
            batch_size=batch_size,
            current_batch_index=start_batch,
            selected_titles_total=len(allowlist),
            already_fetched_titles=sorted(already_fetched),
            last_completed_stage="planned",
            last_run_at=utc_now(),
        )

    print("[RUN] load_history...", flush=True)
    history = load_history(out_dir / "pump_batch_history.json")
    print(f"[RUN] history loaded ({len(history)} records)", flush=True)
    batch_rows: list[dict] = []

    if allow_network and not plan_only:
        for batch in batches:
            batch_index = checkpoint.current_batch_index
            record, rows = run_fetch_batch(batch_index, batch, out_dir / "batch_snapshots", True, user_agent, delay_sec)
            history = append_history(out_dir / "pump_batch_history.json", record)
            batch_rows.extend(rows)
            checkpoint.current_batch_index += 1
            checkpoint.already_fetched_titles = sorted(set(checkpoint.already_fetched_titles) | set(record.titles_fetched))
            checkpoint.successful_fetch_titles = sorted(set(checkpoint.successful_fetch_titles) | set(record.fetch_success))
            checkpoint.failed_fetch_titles = sorted(set(checkpoint.failed_fetch_titles) | set(record.fetch_failed))
            ready_titles = [r.get("normalized_title") or r.get("title") for r in rows if r.get("ready_for_self_ingestion")]
            not_ready_titles = [r.get("normalized_title") or r.get("title") for r in rows if not r.get("ready_for_self_ingestion")]
            checkpoint.ready_titles = sorted(set(checkpoint.ready_titles) | set(ready_titles))
            checkpoint.not_ready_titles = sorted(set(checkpoint.not_ready_titles) | set(not_ready_titles))
            checkpoint.last_completed_stage = "batch_fetch_completed"
            write_checkpoint(out_dir / "pump_checkpoint.json", checkpoint)
            batch_dynamic_summary = update_dynamic_frontier_from_fetch_rows(
                rows,
                dynamic_frontier_path=out_dir / "dynamic_frontier_titles.json",
                dynamic_frontier_csv_path=out_dir / "dynamic_frontier_titles.csv",
                already_fetched_titles=set(checkpoint.already_fetched_titles),
                current_allowlist=allowlist,
            )
            dynamic_frontier_summary["dynamic_frontier_added_this_run"] += batch_dynamic_summary[
                "dynamic_frontier_added_this_run"
            ]
            dynamic_frontier_summary["dynamic_frontier_total"] = batch_dynamic_summary["dynamic_frontier_total"]
            dynamic_frontier_summary["dynamic_frontier_pages_processed"] += batch_dynamic_summary[
                "dynamic_frontier_pages_processed"
            ]
            dynamic_frontier_summary["dynamic_frontier_candidate_count"] += batch_dynamic_summary[
                "dynamic_frontier_candidate_count"
            ]
            dynamic_frontier_summary["dynamic_frontier_rejected_already_fetched"] += batch_dynamic_summary[
                "dynamic_frontier_rejected_already_fetched"
            ]
            dynamic_frontier_summary["dynamic_frontier_rejected_current_allowlist"] += batch_dynamic_summary[
                "dynamic_frontier_rejected_current_allowlist"
            ]
            dynamic_frontier_summary["dynamic_frontier_rejected_hygiene"] += batch_dynamic_summary[
                "dynamic_frontier_rejected_hygiene"
            ]
            dynamic_frontier_summary["dynamic_frontier_added_sample"].extend(
                batch_dynamic_summary["dynamic_frontier_added_sample"]
            )
        if batch_rows:
            frontier, allowlist = _frontier_and_allowlist(
                target_total,
                batch_size,
                set(checkpoint.already_fetched_titles),
                out_dir,
            )
            write_frontier(frontier, out_dir / "frontier_titles.json", out_dir / "frontier_titles.csv")
            write_allowlist(allowlist, out_dir / "expanded_allowlist.json", out_dir / "expanded_allowlist.csv")
    else:
        checkpoint.last_completed_stage = "planned"
        checkpoint.can_resume = True
        checkpoint.safe_to_continue = True
        write_checkpoint(out_dir / "pump_checkpoint.json", checkpoint)
        if not (out_dir / "pump_batch_history.json").exists():
            write_json(out_dir / "pump_batch_history.json", [])

    dynamic_frontier_summary["dynamic_frontier_added_sample"] = dynamic_frontier_summary[
        "dynamic_frontier_added_sample"
    ][:20]
    write_json(out_dir / "dynamic_frontier_summary.json", dynamic_frontier_summary)

    history = load_history(out_dir / "pump_batch_history.json")
    collection_summary = _collection_summary(history, allow_network)
    if batch_rows:
        collection_summary["latest_batch_manifest_rows"] = len(batch_rows)
    write_json(out_dir / "pump_snapshot_collection_summary.json", collection_summary)

    print("[RUN] _recompute_batch_derived...", flush=True)
    recompute_summary = _recompute_batch_derived(
        out_dir,
        enable_extraction_yield_v2=enable_extraction_yield_v2,
        enable_spacy=enable_spacy,
        enable_schema_induction=enable_schema_induction,
    )
    print("[RUN] _recompute_batch_derived done", flush=True)
    ingestion_summary, relation_summary = _copy_existing_summaries(out_dir)
    print("[RUN] merge_with_fresh_deltas...", flush=True)
    merge_result = merge_with_fresh_deltas(
        base_overlay_path=_SNAPSHOT_DRY_RUN,
        legacy_snapshot_delta_path=_SNAPSHOT_INGESTION / "snapshot_overlay_delta_proposal.json",
        legacy_relation_candidates_path=_RELATION_V2 / "relation_extraction_v2_candidates.json",
        fresh_snapshot_overlay_items=recompute_summary.get("_fresh_snapshot_overlay_items", []),
        fresh_relation_rows=recompute_summary.get("_fresh_relation_rows", []),
        output_overlay_path=out_dir / "pump_dry_run_overlay.json",
        output_delta_path=out_dir / "pump_safe_delta.json",
    )
    print("[RUN] merge_with_fresh_deltas done", flush=True)
    merge_counts = merge_result["counts"]
    v2_safe_delta = [
        item for item in merge_result["fresh_safe_delta"]
        if item.get("candidate_source") == "pump_extraction_yield_v2"
    ]
    write_json(out_dir / "pump_extraction_yield_v2_safe_delta.json", v2_safe_delta)

    # --- v1.3 Delta Quality Firewall ---
    print("[RUN] classify_pump_delta...", flush=True)
    classified = classify_pump_delta(merge_result["pump_delta"])
    print("[RUN] classify_pump_delta done", flush=True)
    v13_answerable = classified["answerable"]
    pump_weak_context_delta = classified["weak_context"]
    pump_entity_delta = classified["entity"]
    all_quality_rejected = classified["rejected"] + classified["quarantine"]

    # --- v1.4 Answerable Delta Precision Firewall ---
    # Runs after the v1.3 quality firewall. The v1.3 answerable bucket still
    # contained semantic-extraction garbage (fragment subjects, adjective-only
    # objects, cross-page apposition, opinion/award "definitions"). This stricter
    # pass keeps only clean, supported, high-precision knowledge. Precision is
    # preferred over coverage; the answerable count is expected to drop sharply.
    precision = apply_precision_firewall(v13_answerable)
    v14_answerable = precision["accepted"]
    precision_rejected = precision["rejected"]
    precision_quarantine = precision["quarantine"]
    precision_property_candidates = precision["property_candidates"]

    answerable_before = len(v13_answerable)
    answerable_after_v14 = len(v14_answerable)

    # --- v1.7 Precision Firewall v2 ---
    # Runs after the v1.4 pass. Targets extraction-yield-v2-specific noise:
    # dangling-comma subjects, appositive-alias subjects, geopolitical subjects
    # for corporate predicates, location-as-developer, design-artifact objects,
    # truncated-name founded_by objects, volatile predicates, truncated definitions,
    # and appositive-alias definition subjects.
    precision_v2 = apply_precision_firewall_v2(v14_answerable)
    pump_answerable_delta = precision_v2["accepted"]
    precision_v2_rejected = precision_v2["rejected"]
    precision_v2_quarantine = precision_v2["quarantine"]

    answerable_after = len(pump_answerable_delta)

    # Write v1.7 precision artifacts.
    precision_v2_out = out_dir / "precision_firewall_v2"
    precision_v2_summary = write_precision_v2_artifacts(
        precision_v2_out,
        precision_v2,
        answerable_before=answerable_after_v14,
        answerable_after=answerable_after,
    )

    # --- v1.7.1 Precision Cleanup v2.1 ---
    # Runs after the v1.7 pass. Removes remaining alias/truncation/predicate-
    # ambiguity problems: truncated person aliases, subject aliases too short,
    # composite region subjects, headquartered_in quarantine default, owned_by
    # vehicle-type subjects, duplicate alias relations, and truncated-participle
    # definitions.
    answerable_after_v17 = answerable_after
    cleanup_v2_1 = apply_precision_cleanup_v2_1(pump_answerable_delta)
    pump_answerable_delta = cleanup_v2_1["accepted"]
    cleanup_v2_1_rejected = cleanup_v2_1["rejected"]
    cleanup_v2_1_quarantine = cleanup_v2_1["quarantine"]

    answerable_after = len(pump_answerable_delta)

    cleanup_v2_1_out = out_dir / "precision_cleanup_v2_1"
    cleanup_v2_1_summary = write_cleanup_v2_1_artifacts(
        cleanup_v2_1_out, cleanup_v2_1,
        answerable_before=answerable_after_v17,
        answerable_after=answerable_after,
    )

    honest_delta_metrics = _pump_delta_metrics(
        pump_entity_delta,
        pump_answerable_delta,
        precision_property_candidates,
    )
    v2_precision_answerable_delta = [
        item for item in pump_answerable_delta
        if item.get("candidate_source") == "pump_extraction_yield_v2"
    ]
    write_json(
        out_dir / "pump_extraction_yield_v2_precision_answerable_delta.json",
        v2_precision_answerable_delta,
    )
    non_v2_answerable_delta = [
        item for item in pump_answerable_delta
        if item.get("candidate_source") != "pump_extraction_yield_v2"
    ]
    before_v2_metrics = _pump_delta_metrics([], non_v2_answerable_delta, [])
    after_v2_metrics = honest_delta_metrics

    write_json(out_dir / "pump_answerable_delta.json", pump_answerable_delta)
    write_json(out_dir / "pump_weak_context_delta.json", pump_weak_context_delta)
    write_json(out_dir / "pump_entity_delta.json", pump_entity_delta)
    write_json(out_dir / "pump_rejected_delta.json", all_quality_rejected)

    # v1.4 precision artifacts.
    write_json(out_dir / "pump_precision_answerable_delta.json", pump_answerable_delta)
    write_json(out_dir / "pump_precision_rejected_delta.json", precision_rejected)
    write_json(out_dir / "pump_precision_quarantine.json", precision_quarantine)
    write_json(out_dir / "pump_precision_property_candidates.json", precision_property_candidates)
    precision_report = {
        "answerable_delta_before_precision": answerable_before,
        "answerable_delta_after_precision": answerable_after,
        "precision_rejected_count": len(precision_rejected),
        "precision_quarantined_count": len(precision_quarantine),
        "precision_property_candidate_count": len(precision_property_candidates),
        "precision_rejection_by_reason": precision["rejection_by_reason"],
        "precision_quarantine_by_reason": precision["quarantine_by_reason"],
        "relation_precision_before_by_predicate": precision["relation_before_by_predicate"],
        "relation_precision_after_by_predicate": precision["relation_after_by_predicate"],
        "definition_precision_before_count": precision["definition_before_count"],
        "definition_precision_after_count": precision["definition_after_count"],
        **honest_delta_metrics,
        "include_weak_context": include_weak_context,
        "precision_v2_enabled": True,
        "precision_v2_answerable_before_count": answerable_after_v14,
        "precision_v2_answerable_after_count": answerable_after,
        "precision_v2_relation_before_count": precision_v2["relation_before_count"],
        "precision_v2_relation_after_count": precision_v2["relation_after_count"],
        "precision_v2_definition_before_count": precision_v2["definition_before_count"],
        "precision_v2_definition_after_count": precision_v2["definition_after_count"],
        "precision_v2_rejected_count": len(precision_v2_rejected),
        "precision_v2_quarantined_count": len(precision_v2_quarantine),
        "precision_v2_rejection_by_reason": precision_v2["rejection_by_reason"],
        "precision_v2_quarantine_by_reason": precision_v2["quarantine_by_reason"],
        "precision_rejected_examples": precision_rejected[:10],
        "precision_quarantine_examples": precision_quarantine[:10],
        "precision_property_candidate_examples": precision_property_candidates[:10],
        "precision_v2_rejected_examples": precision_v2_rejected[:10],
        "precision_v2_quarantine_examples": precision_v2_quarantine[:10],
        "precision_cleanup_v2_1_enabled": True,
        "precision_cleanup_v2_1_answerable_before_count": answerable_after_v17,
        "precision_cleanup_v2_1_answerable_after_count": answerable_after,
        "precision_cleanup_v2_1_relation_before_count": cleanup_v2_1["relation_before_count"],
        "precision_cleanup_v2_1_relation_after_count": cleanup_v2_1["relation_after_count"],
        "precision_cleanup_v2_1_definition_before_count": cleanup_v2_1["definition_before_count"],
        "precision_cleanup_v2_1_definition_after_count": cleanup_v2_1["definition_after_count"],
        "precision_cleanup_v2_1_rejected_count": len(cleanup_v2_1_rejected),
        "precision_cleanup_v2_1_quarantined_count": len(cleanup_v2_1_quarantine),
        "precision_cleanup_v2_1_rejection_by_reason": cleanup_v2_1["rejection_by_reason"],
        "precision_cleanup_v2_1_quarantine_by_reason": cleanup_v2_1["quarantine_by_reason"],
        "precision_cleanup_v2_1_duplicate_alias_relation_count": cleanup_v2_1["duplicate_alias_relation_count"],
        "precision_cleanup_v2_1_quarantined_alias_duplicate_count": cleanup_v2_1["quarantined_alias_duplicate_count"],
        "answerable_relation_examples": [
            x for x in pump_answerable_delta if x.get("overlay_type") == "overlay_relation"
        ][:10],
    }
    write_json(out_dir / "pump_precision_report.json", precision_report)

    write_json(
        out_dir / "pump_delta_quality_report.json",
        {
            "pump_answerable_delta_count": answerable_after,
            **honest_delta_metrics,
            "answerable_delta_before_precision": answerable_before,
            "answerable_delta_after_precision": answerable_after,
            "weak_context_delta_count": len(pump_weak_context_delta),
            "entity_delta_count": len(pump_entity_delta),
            "rejected_quality_count": len(classified["rejected"]),
            "quarantine_quality_count": len(classified["quarantine"]),
            "relation_quality_rejection_by_reason": classified["rejection_by_reason"],
            "precision_rejected_count": len(precision_rejected),
            "precision_quarantined_count": len(precision_quarantine),
            "precision_property_candidate_count": len(precision_property_candidates),
            "precision_rejection_by_reason": precision["rejection_by_reason"],
            "include_weak_context": include_weak_context,
            "rejected_examples": classified["rejected"][:10],
            "quarantine_examples": classified["quarantine"][:10],
            "answerable_relation_examples": [
                x for x in pump_answerable_delta if x.get("overlay_type") == "overlay_relation"
            ][:10],
        },
    )

    # Rebuild pump_dry_run_overlay from the precision-filtered answerable delta
    # (plus weak context if opted in). The base overlay (snapshot_dry_run_overlay)
    # may itself contain weak context links; filter those out in the default
    # (non-weak) view so the pump overlay represents clean, high-precision knowledge.
    base_overlay = _read_json(_SNAPSHOT_DRY_RUN, [])
    base_answerable = [item for item in base_overlay if not is_weak_context(item)]
    overlay_without_weak = base_answerable + pump_answerable_delta
    overlay_with_weak = list(base_overlay) + pump_answerable_delta + pump_weak_context_delta
    final_overlay = overlay_with_weak if include_weak_context else overlay_without_weak
    pump_overlay_path = out_dir / "pump_dry_run_overlay.json"
    write_json(pump_overlay_path, final_overlay)
    print("[RUN] _write_pump_fact_smoke...", flush=True)
    smoke_metrics = _write_pump_fact_smoke(out_dir, pump_overlay_path, pump_answerable_delta)
    print("[RUN] _write_pump_fact_smoke done", flush=True)

    # Overlay-without-weak sizes before vs after the precision firewall.
    overlay_without_weak_before_precision = len(base_answerable) + answerable_before
    overlay_without_weak_after_precision = len(overlay_without_weak)
    # --- end v1.4 Answerable Delta Precision Firewall ---

    fresh_quarantine = (
        list(merge_result["fresh_quarantine"])
        + list(merge_result["fresh_rejected"])
        + list(recompute_summary.get("_fresh_relation_quarantine", []))
        + list(recompute_summary.get("_fresh_ingestion_failures", []))
    )
    fresh_quarantine_by_reason = dict(
        sorted(Counter(str(row.get("reason", "unknown")) for row in fresh_quarantine).items())
    )
    write_json(out_dir / "pump_fresh_safe_delta.json", merge_result["fresh_safe_delta"])
    write_json(out_dir / "pump_fresh_duplicates.json", merge_result["fresh_duplicates"])
    write_json(out_dir / "pump_fresh_conflicts.json", merge_result["fresh_conflicts"])
    write_json(out_dir / "pump_fresh_quarantine.json", fresh_quarantine)
    fresh_delta_explanation = []
    if merge_counts["fresh_merged_safe_delta_count"] == 0:
        fresh_delta_explanation.append(
            "Fresh candidates produced no new pump-safe delta after duplicate, conflict, quarantine, and safety filters."
        )
    if merge_counts["fresh_duplicates_count"]:
        fresh_delta_explanation.append("Some fresh candidates duplicate the base snapshot dry-run overlay or existing pump delta.")
    if merge_counts["fresh_quarantined_count"] or recompute_summary.get("_fresh_relation_quarantine"):
        fresh_delta_explanation.append("Some fresh candidates require quarantine or source-qualified review before promotion.")
    write_json(
        out_dir / "pump_fresh_delta_report.json",
        {
            "summary": {
                "fresh_ingestion_candidates_total": recompute_summary.get("new_ingestion_candidates_this_batch", 0),
                "fresh_relation_candidates_total": recompute_summary.get("new_relation_candidates_this_batch", 0),
                "fresh_candidates_total": (
                    recompute_summary.get("new_ingestion_candidates_this_batch", 0)
                    + recompute_summary.get("new_relation_candidates_this_batch", 0)
                ),
                **{
                    key: merge_counts[key]
                    for key in (
                        "fresh_safe_snapshot_delta_count",
                        "fresh_safe_relation_delta_count",
                        "fresh_merged_safe_delta_count",
                        "fresh_duplicates_count",
                        "fresh_conflicts_count",
                        "fresh_rejected_count",
                        "pump_safe_delta_total",
                        "pump_dry_run_overlay_items_count",
                    )
                },
                "fresh_quarantined_count": len(fresh_quarantine),
                "fresh_quarantine_by_reason": fresh_quarantine_by_reason,
                "stale_artifacts_reused": recompute_summary.get("stale_artifacts_reused", False),
            },
            "fresh_delta_examples": merge_result["fresh_safe_delta"][:10],
            "fresh_duplicate_examples": merge_result["fresh_duplicates"][:10],
            "fresh_conflict_examples": merge_result["fresh_conflicts"][:10],
            "fresh_quarantine_examples": fresh_quarantine[:10],
            "explanation": fresh_delta_explanation,
        },
    )

    print("[RUN] run_assistant_regression...", flush=True)
    assistant_summary = run_assistant_regression(out_dir, overlay_path=pump_overlay_path)
    print("[RUN] run_assistant_regression done", flush=True)
    baseline = _read_json(_EXPERIMENTS / "assistant_surface_v1" / "assistant_surface_summary.json", {})
    answer_gain = assistant_summary.get("answer_count", 0) - baseline.get("answer_count", assistant_summary.get("answer_count", 0))

    after = _protected_hashes()
    false_support_counts = {
        "weak_link_false_support_count": assistant_summary.get("weak_link_false_support_count", 0),
        "volatile_false_stable_count": assistant_summary.get("volatile_false_stable_count", 0),
        "current_live_false_support_count": assistant_summary.get("current_live_false_support_count", 0),
        "private_false_support_count": assistant_summary.get("private_false_support_count", 0),
        "inversion_false_support_count": assistant_summary.get("inversion_false_support_count", 0),
        "universal_false_support_count": assistant_summary.get("universal_false_support_count", 0),
    }
    fact_qa_summary = _read_json(out_dir / "pump_fact_qa_v1" / "pump_fact_qa_summary.json", {})
    fact_qa_count_after = fact_qa_summary.get(
        "pump_fact_qa_fact_count", after_v2_metrics["pump_answerable_fact_delta_count"]
    )

    summary = build_summary(
        plan_only=plan_only,
        allow_network=allow_network,
        target_total=target_total,
        batch_size=batch_size,
        max_batches=max_batches,
        frontier_titles_total=len(frontier),
        expanded_allowlist_total=len(allowlist),
        selected_for_batch_count=len(batches[0]) if batches else 0,
        batches_planned=len(batches),
        batches_completed=collection_summary["batches_completed"],
        already_fetched_skipped_count=sum(1 for e in allowlist if e.already_fetched),
        not_ready_skipped_count=len(_read_json(_SNAPSHOTS / "snapshot_readiness_report.json", [])) - ingestion_summary.get("ready_docs_selected", 0),
        fetched_count_total=collection_summary["fetched_count_total"],
        fetch_success_count_total=collection_summary["fetch_success_count_total"],
        fetch_failed_count_total=collection_summary["fetch_failed_count_total"],
        ready_for_ingestion_count_total=collection_summary["ready_for_ingestion_count_total"] or ingestion_summary.get("ready_docs_selected", 0),
        ingestion_candidates_total=ingestion_summary.get("candidates_total", 0),
        relation_v2_candidates_total=relation_summary.get("accepted_relation_candidates_total", 0),
        new_ready_docs_this_batch=recompute_summary.get("new_ready_docs_this_batch", 0),
        new_ingestion_candidates_this_batch=recompute_summary.get("new_ingestion_candidates_this_batch", 0),
        new_relation_candidates_this_batch=recompute_summary.get("new_relation_candidates_this_batch", 0),
        stale_artifacts_reused=recompute_summary.get("stale_artifacts_reused", False),
        fresh_ingestion_candidates_total=recompute_summary.get("new_ingestion_candidates_this_batch", 0),
        fresh_relation_candidates_total=recompute_summary.get("new_relation_candidates_this_batch", 0),
        fresh_candidates_total=(
            recompute_summary.get("new_ingestion_candidates_this_batch", 0)
            + recompute_summary.get("new_relation_candidates_this_batch", 0)
        ),
        extraction_yield_v2_enabled=recompute_summary.get("extraction_yield_v2_enabled", enable_extraction_yield_v2),
        extraction_yield_v2_candidate_count=recompute_summary.get("extraction_yield_v2_candidate_count", 0),
        extraction_yield_v2_candidates_count=recompute_summary.get("extraction_yield_v2_candidate_count", 0),
        extraction_yield_v2_rejected_count=recompute_summary.get("extraction_yield_v2_rejected_count", 0),
        extraction_yield_v2_quarantined_count=recompute_summary.get("extraction_yield_v2_quarantined_count", 0),
        extraction_yield_v2_candidate_by_pattern=recompute_summary.get("extraction_yield_v2_candidate_by_pattern", {}),
        extraction_yield_v2_relation_candidate_count=recompute_summary.get("extraction_yield_v2_relation_candidate_count", 0),
        extraction_yield_v2_definition_candidate_count=recompute_summary.get("extraction_yield_v2_definition_candidate_count", 0),
        extraction_yield_v2_safe_delta_count=len(v2_safe_delta),
        extraction_yield_v2_precision_accepted_count=len(v2_precision_answerable_delta),
        pump_answerable_fact_delta_count_before_v2=before_v2_metrics["pump_answerable_fact_delta_count"],
        pump_answerable_fact_delta_count_after_v2=after_v2_metrics["pump_answerable_fact_delta_count"],
        pump_relation_delta_count_before_v2=before_v2_metrics["pump_relation_delta_count"],
        pump_relation_delta_count_after_v2=after_v2_metrics["pump_relation_delta_count"],
        pump_definition_delta_count_before_v2=before_v2_metrics["pump_definition_delta_count"],
        pump_definition_delta_count_after_v2=after_v2_metrics["pump_definition_delta_count"],
        pump_world_model_delta_count_before_v2=before_v2_metrics["pump_world_model_delta_count"],
        pump_world_model_delta_count_after_v2=after_v2_metrics["pump_world_model_delta_count"],
        pump_fact_qa_fact_count_before_v2=before_v2_metrics["pump_answerable_fact_delta_count"],
        pump_fact_qa_fact_count_after_v2=fact_qa_count_after,
        pump_fact_qa_all_critical_passed_after_v2=fact_qa_summary.get("pump_fact_qa_all_critical_passed"),
        safe_snapshot_delta_count=merge_counts["safe_snapshot_delta_count"],
        safe_relation_v2_delta_count=merge_counts["safe_relation_v2_delta_count"],
        merged_safe_delta_count=merge_counts["merged_safe_delta_count"],
        fresh_safe_snapshot_delta_count=merge_counts["fresh_safe_snapshot_delta_count"],
        fresh_safe_relation_delta_count=merge_counts["fresh_safe_relation_delta_count"],
        fresh_merged_safe_delta_count=merge_counts["fresh_merged_safe_delta_count"],
        fresh_duplicates_count=merge_counts["fresh_duplicates_count"],
        fresh_conflicts_count=merge_counts["fresh_conflicts_count"],
        fresh_quarantined_count=len(fresh_quarantine),
        fresh_rejected_count=merge_counts["fresh_rejected_count"],
        fresh_delta_examples=merge_result["fresh_safe_delta"][:5],
        fresh_quarantine_by_reason=fresh_quarantine_by_reason,
        pump_safe_delta_total=merge_counts["pump_safe_delta_total"],
        pump_dry_run_overlay_items_count=len(final_overlay),
        pump_answerable_delta_count=len(pump_answerable_delta),
        **honest_delta_metrics,
        **smoke_metrics,
        weak_context_delta_count=len(pump_weak_context_delta),
        entity_delta_count=len(pump_entity_delta),
        rejected_quality_count=len(classified["rejected"]),
        quarantine_quality_count=len(classified["quarantine"]),
        relation_quality_rejection_by_reason=classified["rejection_by_reason"],
        answerable_delta_before_precision=answerable_before,
        answerable_delta_after_precision=answerable_after,
        precision_rejected_count=len(precision_rejected),
        precision_quarantined_count=len(precision_quarantine),
        precision_property_candidate_count=len(precision_property_candidates),
        precision_rejection_by_reason=precision["rejection_by_reason"],
        precision_quarantine_by_reason=precision["quarantine_by_reason"],
        relation_precision_before_by_predicate=precision["relation_before_by_predicate"],
        relation_precision_after_by_predicate=precision["relation_after_by_predicate"],
        definition_precision_before_count=precision["definition_before_count"],
        definition_precision_after_count=precision["definition_after_count"],
        precision_v2_enabled=True,
        precision_v2_answerable_before_count=answerable_after_v14,
        precision_v2_answerable_after_count=answerable_after,
        precision_v2_relation_before_count=precision_v2["relation_before_count"],
        precision_v2_relation_after_count=precision_v2["relation_after_count"],
        precision_v2_definition_before_count=precision_v2["definition_before_count"],
        precision_v2_definition_after_count=precision_v2["definition_after_count"],
        precision_v2_rejected_count=len(precision_v2_rejected),
        precision_v2_quarantined_count=len(precision_v2_quarantine),
        precision_v2_rejection_by_reason=precision_v2["rejection_by_reason"],
        precision_v2_quarantine_by_reason=precision_v2["quarantine_by_reason"],
        precision_cleanup_v2_1_enabled=True,
        precision_cleanup_v2_1_answerable_before_count=answerable_after_v17,
        precision_cleanup_v2_1_answerable_after_count=answerable_after,
        precision_cleanup_v2_1_relation_before_count=cleanup_v2_1["relation_before_count"],
        precision_cleanup_v2_1_relation_after_count=cleanup_v2_1["relation_after_count"],
        precision_cleanup_v2_1_definition_before_count=cleanup_v2_1["definition_before_count"],
        precision_cleanup_v2_1_definition_after_count=cleanup_v2_1["definition_after_count"],
        precision_cleanup_v2_1_rejected_count=len(cleanup_v2_1_rejected),
        precision_cleanup_v2_1_quarantined_count=len(cleanup_v2_1_quarantine),
        precision_cleanup_v2_1_rejection_by_reason=cleanup_v2_1["rejection_by_reason"],
        precision_cleanup_v2_1_quarantine_by_reason=cleanup_v2_1["quarantine_by_reason"],
        precision_cleanup_v2_1_duplicate_alias_relation_count=cleanup_v2_1["duplicate_alias_relation_count"],
        precision_cleanup_v2_1_quarantined_alias_duplicate_count=cleanup_v2_1["quarantined_alias_duplicate_count"],
        pump_dry_run_overlay_items_count_without_weak_before_precision=overlay_without_weak_before_precision,
        pump_dry_run_overlay_items_count_without_weak_after_precision=overlay_without_weak_after_precision,
        pump_dry_run_overlay_items_count_without_weak=len(overlay_without_weak),
        pump_dry_run_overlay_items_count_with_weak=len(overlay_with_weak),
        include_weak_context=include_weak_context,
        assistant_benchmark_overlay_path=assistant_summary.get("benchmark_overlay_path", str(pump_overlay_path)),
        assistant_prompt_total=assistant_summary.get("prompt_total", 0),
        assistant_answer_count=assistant_summary.get("answer_count", 0),
        assistant_audit_count=assistant_summary.get("audit_count", 0),
        assistant_answer_gain_vs_baseline=answer_gain,
        unsafe_answer_count=assistant_summary.get("unsafe_answer_count", 0),
        answer_without_context_support_count=assistant_summary.get("answer_without_context_support_count", 0),
        false_support_counts=false_support_counts,
        all_critical_passed=bool(assistant_summary.get("all_critical_passed", False)),
        network_calls=collection_summary["network_calls"] if allow_network else False,
        trusted_memory_modified=before["trusted_memory"] != after["trusted_memory"],
        accepted_overlay_modified=before["accepted_overlay"] != after["accepted_overlay"],
        promoted_overlay_modified=before["promoted_overlay"] != after["promoted_overlay"],
        snapshot_dry_run_overlay_modified=before["snapshot_dry_run_overlay"] != after["snapshot_dry_run_overlay"],
        runtime_behavior_modified=False,
    )
    # --- v1.9 yield-aware frontier control summary fields ---
    summary.update({
        "frontier_policy": frontier_policy,
        "dynamic_frontier_enabled": True,
        "dynamic_frontier_initial_total": dynamic_frontier_summary.get("dynamic_frontier_initial_total"),
        "dynamic_frontier_total": dynamic_frontier_summary.get("dynamic_frontier_total"),
        "dynamic_frontier_added_this_run": dynamic_frontier_summary.get("dynamic_frontier_added_this_run"),
        "dynamic_frontier_pages_processed": dynamic_frontier_summary.get("dynamic_frontier_pages_processed"),
        "dynamic_frontier_candidate_count": dynamic_frontier_summary.get("dynamic_frontier_candidate_count"),
        "dynamic_frontier_rejected_already_fetched": dynamic_frontier_summary.get(
            "dynamic_frontier_rejected_already_fetched"
        ),
        "dynamic_frontier_rejected_current_allowlist": dynamic_frontier_summary.get(
            "dynamic_frontier_rejected_current_allowlist"
        ),
        "dynamic_frontier_rejected_hygiene": dynamic_frontier_summary.get("dynamic_frontier_rejected_hygiene"),
        "dynamic_frontier_added_sample": dynamic_frontier_summary.get("dynamic_frontier_added_sample"),
        "yield_gate_enabled": True,
        "recent_new_answerable_facts": yield_summary.get("recent_new_answerable_facts"),
        "recent_answerable_yield_per_ready_doc": yield_summary.get(
            "recent_answerable_yield_per_ready_doc"
        ),
        "zero_yield_batch_count": yield_summary.get("zero_yield_batch_count"),
        "negative_yield_detected": yield_summary.get("negative_yield_detected"),
        "blind_fetch_recommended": yield_gate_info.get("blind_fetch_recommended"),
        "yield_ranked_fetch_recommended": yield_gate_info.get("yield_ranked_fetch_recommended"),
        "force_required_for_blind_fetch": yield_gate_info.get("force_required_for_blind_fetch"),
        "yield_ranked_selected_count": yield_ranked_selected_count,
        "yield_ranked_available_count": yield_ranked_available_count,
        "frontier_policy_fallback_reason": frontier_policy_fallback_reason,
        "force_low_yield_fetch": force_low_yield_fetch,
    })
    print("[RUN] merge_qa_preservation...", flush=True)
    merge_qa_preservation(
        summary,
        out_dir,
        current_fact_count=summary.get("pump_answerable_fact_delta_count", 0),
    )
    print("[RUN] writing final artifacts...", flush=True)
    write_json(out_dir / "pump_summary.json", summary)
    write_json(
        out_dir / "pump_report.json",
        {
            "summary": summary,
            "checkpoint": checkpoint.to_dict(),
            "source_artifacts": [
                str(_SNAPSHOTS),
                str(_SNAPSHOT_INGESTION),
                str(_RELATION_V2),
                str(_ACCEPTED),
                str(_PROMOTED),
                str(_SNAPSHOT_DRY_RUN),
            ],
            "fresh_delta_report": _read_json(out_dir / "pump_fresh_delta_report.json", {}),
            "recommended_next_actions": [
                "inspect expanded_allowlist and frontier_titles",
                "run one bounded network batch with --allow-network",
                "inspect pump_safe_delta and pump_dry_run_overlay",
                "rerun assistant/regression gates before any promotion",
            ],
        },
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Knowledge Pump v1")
    parser.add_argument("--plan-only", action="store_true", default=False)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--target-total", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--delay-sec", type=float, default=0.5)
    parser.add_argument("--user-agent", default="MicroworldResearchBot/0.1 (local research; contact: YOUR_EMAIL)")
    parser.add_argument("--include-weak-context", action="store_true", default=False,
                        help="Include weak context links in pump_dry_run_overlay (default: excluded)")
    parser.set_defaults(enable_extraction_yield_v2=True)
    parser.add_argument("--enable-extraction-yield-v2", action="store_true",
                        dest="enable_extraction_yield_v2",
                        help="Run Extraction Yield v2 to extract additional relation/definition "
                             "candidates using improved pattern matching (default: enabled)")
    parser.add_argument("--disable-extraction-yield-v2", action="store_false",
                        dest="enable_extraction_yield_v2",
                        help="Disable Extraction Yield v2 for before/after diagnostics.")
    parser.add_argument("--enable-spacy", action="store_true", default=False,
                        dest="enable_spacy",
                        help="Enable spaCy dependency-parse extraction path (default: off). "
                             "Adds passive-voice, coordination, and copula candidates on top of "
                             "the regex path. Use to compare yield with and without spaCy.")
    parser.add_argument("--frontier-policy", choices=["default", "yield-ranked"],
                        default="default",
                        help="Frontier selection policy. 'default' keeps the "
                             "backward-compatible blind frontier; 'yield-ranked' "
                             "selects titles by expected answerable yield.")
    parser.add_argument("--force-low-yield-fetch", action="store_true", default=False,
                        help="Override the yield gate and allow a blind network "
                             "fetch even when recent yield has collapsed.")
    parser.add_argument("--enable-schema-induction", action="store_true", default=False,
                        dest="enable_schema_induction",
                        help="Run schema_induction as a parallel extraction path. "
                             "Produces induced relation families from the same docs "
                             "and merges them with extraction_yield_v2 candidates. "
                             "Generated (non-promoted) families require "
                             "min 2 source documents to pass the precision gate. "
                             "(default: off)")
    args = parser.parse_args(argv)

    plan_only = args.plan_only or not args.allow_network
    print("[MAIN] calling run()...", flush=True)
    result = run(
        plan_only=plan_only,
        allow_network=args.allow_network,
        target_total=args.target_total,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        resume=args.resume,
        status=args.status,
        user_agent=args.user_agent,
        delay_sec=args.delay_sec,
        include_weak_context=args.include_weak_context,
        enable_extraction_yield_v2=args.enable_extraction_yield_v2,
        enable_spacy=args.enable_spacy,
        enable_schema_induction=args.enable_schema_induction,
        frontier_policy=args.frontier_policy,
        force_low_yield_fetch=args.force_low_yield_fetch,
    )
    if args.status:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    print("Knowledge Pump 5000 Mode v1")
    for key in (
        "plan_only", "allow_network", "include_weak_context", "target_total", "batch_size",
        "batches_completed", "frontier_titles_total", "expanded_allowlist_total",
        "selected_for_batch_count", "fetched_count_total",
        "fetch_success_count_total", "ready_for_ingestion_count_total",
        "schema_induction_enabled",
        "schema_induction_docs",
        "schema_induction_families_generated",
        "schema_induction_families_promoted",
        "schema_induction_raw_items",
        "schema_induction_after_gate",
        "schema_induction_quarantined",
        "schema_induction_new_induced",
        "schema_induction_deduped",
        "extraction_yield_v2_enabled",
        "extraction_yield_v2_candidate_count",
        "extraction_yield_v2_rejected_count",
        "extraction_yield_v2_quarantined_count",
        "extraction_yield_v2_safe_delta_count",
        "extraction_yield_v2_precision_accepted_count",
        "safe_snapshot_delta_count", "safe_relation_v2_delta_count",
        "merged_safe_delta_count", "pump_safe_delta_total",
        "pump_answerable_fact_delta_count_before_v2",
        "pump_answerable_fact_delta_count_after_v2",
        "pump_relation_delta_count_before_v2",
        "pump_relation_delta_count_after_v2",
        "pump_definition_delta_count_before_v2",
        "pump_definition_delta_count_after_v2",
        "pump_world_model_delta_count", "pump_entity_delta_count",
        "pump_answerable_fact_delta_count", "pump_relation_delta_count",
        "pump_definition_delta_count", "pump_property_candidate_count",
        "pump_answerable_delta_count", "weak_context_delta_count",
        "entity_delta_count", "rejected_quality_count", "quarantine_quality_count",
        "answerable_delta_before_precision", "answerable_delta_after_precision",
        "precision_rejected_count", "precision_quarantined_count",
        "precision_property_candidate_count",
        "precision_v2_enabled",
        "precision_v2_answerable_before_count",
        "precision_v2_answerable_after_count",
        "precision_v2_relation_before_count",
        "precision_v2_relation_after_count",
        "precision_v2_definition_before_count",
        "precision_v2_definition_after_count",
        "precision_v2_rejected_count",
        "precision_v2_quarantined_count",
        "precision_cleanup_v2_1_enabled",
        "precision_cleanup_v2_1_answerable_before_count",
        "precision_cleanup_v2_1_answerable_after_count",
        "precision_cleanup_v2_1_relation_before_count",
        "precision_cleanup_v2_1_relation_after_count",
        "precision_cleanup_v2_1_definition_before_count",
        "precision_cleanup_v2_1_definition_after_count",
        "precision_cleanup_v2_1_rejected_count",
        "precision_cleanup_v2_1_quarantined_count",
        "precision_cleanup_v2_1_duplicate_alias_relation_count",
        "precision_cleanup_v2_1_quarantined_alias_duplicate_count",
        "pump_dry_run_overlay_items_count_without_weak_before_precision",
        "pump_dry_run_overlay_items_count_without_weak_after_precision",
        "pump_dry_run_overlay_items_count_without_weak",
        "pump_dry_run_overlay_items_count_with_weak",
        "pump_dry_run_overlay_items_count",
        "assistant_prompt_total", "assistant_answer_count",
        "assistant_audit_count", "assistant_answer_gain_vs_baseline",
        "pump_smoke_prompt_count", "pump_smoke_answer_count",
        "pump_smoke_audit_count", "pump_smoke_wrong_count",
        "pump_smoke_unsupported_answer_count", "pump_smoke_planner_gap_count",
        "pump_smoke_supported_fact_answer_count",
        "unsafe_answer_count",
        "answer_without_context_support_count", "all_critical_passed",
        "frontier_policy", "yield_gate_enabled",
        "recent_new_answerable_facts", "recent_answerable_yield_per_ready_doc",
        "zero_yield_batch_count", "negative_yield_detected",
        "blind_fetch_recommended", "yield_ranked_fetch_recommended",
        "force_required_for_blind_fetch", "yield_ranked_selected_count",
        "yield_ranked_available_count", "frontier_policy_fallback_reason",
        "network_calls",
    ):
        print(f"  {key}: {result.get(key)}")
    return 0 if result.get("all_critical_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
