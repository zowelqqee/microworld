"""Multi-hop QA v1 experiment runner.

Builds a 2-hop relation graph from the pump dry-run overlay, runs a
curated set of connection and chain questions, validates safety constraints,
and writes artifacts to multihop_qa_v1/.

Read-only. No network. No mutation of accepted memory, accepted/promoted
overlays, or snapshot dry-run overlay. The pump overlay stays a
proposal/dry-run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider
from worldpgt.multihop_qa.answer_renderer import render
from worldpgt.multihop_qa.path_planner import MultihopPlanner
from worldpgt.multihop_qa.question_analyzer import analyze
from worldpgt.multihop_qa.relation_graph import RelationGraph
from worldpgt.multihop_qa.types import MultihopResult, MultihopSummary

_EXPERIMENTS = Path(__file__).resolve().parent
_PUMP_OUT = _EXPERIMENTS / "knowledge_pump_v1"
_PUMP_OVERLAY = _PUMP_OUT / "pump_dry_run_overlay.json"
_OUT = _EXPERIMENTS / "multihop_qa_v1"

_ACCEPTED = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_PROMOTED = _EXPERIMENTS / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json"
_SNAPSHOT_DRY_RUN = _EXPERIMENTS / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json"
_TRUSTED = _EXPERIMENTS / "accepted_knowledge_memory_v1.json"
_SENSE_MEMORY = _ROOT / "worldpgt" / "continuation" / "sense_memory.py"


# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------

# (question, expected_decision, chain_type_hint, notes)
_SAFE_CHAIN_QUESTIONS: list[tuple[str, str, str, str]] = [
    # 2-hop: owned_by then develops — real pump chains
    (
        "How is Starlink connected to Falcon 9?",
        "answer", "connection",
        "Starlink owned_by SpaceX; SpaceX develops Falcon 9",
    ),
    (
        "How is Starlink connected to rockets?",
        "answer", "connection",
        "Starlink owned_by SpaceX; SpaceX develops rockets",
    ),
    (
        "How is SolarCity connected to electric cars?",
        "answer", "connection",
        "SolarCity owned_by Tesla; Tesla produces electric cars",
    ),
    (
        "What links SolarCity to electric cars?",
        "answer", "connection",
        "SolarCity owned_by Tesla; Tesla produces Electric cars",
    ),
    # Through-node explicit
    (
        "Is Starlink connected to Falcon 9 through SpaceX?",
        "answer", "through_node",
        "Starlink -> SpaceX -> Falcon 9",
    ),
    (
        "Is SolarCity connected to electric cars through Tesla?",
        "answer", "through_node",
        "SolarCity -> Tesla -> electric cars",
    ),
]

# Safety prompts — must all audit.
_SAFETY_AUDIT_QUESTIONS: list[tuple[str, str, str, str]] = [
    # Current-sensitive predicate (leader_of) — must audit in v1.1.
    (
        "How is Elon Musk connected to Falcon 9?",
        "audit", "connection",
        "first path uses leader_of — current-sensitive; multi-hop must audit",
    ),
    (
        "How is Jeff Bezos connected to rockets?",
        "audit", "connection",
        "first path uses leader_of — current-sensitive; multi-hop must audit",
    ),
    # Direct founder — must audit; cannot use transitive chain.
    (
        "Who founded SolarCity?",
        "audit", "direct_only",
        "direct founder — transitive founder leak guard",
    ),
    (
        "Who founded Starlink?",
        "audit", "direct_only",
        "direct founder — transitive founder leak guard",
    ),
    (
        "Who was Starlink founded by?",
        "audit", "direct_only",
        "direct founded_by — transitive founder leak guard",
    ),
    # Direct owner — must audit.
    (
        "Who owns SolarCity?",
        "audit", "direct_only",
        "direct owner — transitive owner leak guard",
    ),
    (
        "What company owns Starlink?",
        "audit", "direct_only",
        "direct owner — transitive owner leak guard",
    ),
    # Missing-hop — one hop exists, other missing; must audit.
    (
        "How is Starlink connected to Ada Stone?",
        "audit", "connection",
        "missing second hop — Ada Stone not in overlay",
    ),
    (
        "How is UnknownCorp connected to SpaceX?",
        "audit", "connection",
        "missing first hop — UnknownCorp not in overlay",
    ),
    # Unsupported pattern — must audit.
    (
        "What is SpaceX?",
        "audit", "unsupported",
        "single-hop definition — not a connection form",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fields})


def _categorise_block_reason(reason: str) -> str:
    """Map a path_validator reason string to a summary counter name."""
    if reason and "volatile_relation:" in reason:
        return "volatile"
    if reason and "high_risk_relation:" in reason:
        return "high_risk"
    if reason and "current_sensitive_relation:" in reason:
        return "current_sensitive"
    return "other"


def _run_prompt(
    row_id: str,
    question: str,
    expected_decision: str,
    chain_type_hint: str,
    notes: str,
    planner: MultihopPlanner,
) -> tuple[MultihopResult, bool, str]:
    """Run one prompt; return (result, is_correct, block_reason)."""
    mq = analyze(question)
    plan = planner.plan(mq)
    answer_text = render(plan)

    hop1_str = plan.path.hop1.display() if plan.path else None
    hop2_str = plan.path.hop2.display() if plan.path else None
    hop1_detail = plan.path.hop1.to_detail_dict() if plan.path else None
    hop2_detail = plan.path.hop2.to_detail_dict() if plan.path else None

    is_safe = True
    unsafe_reason = None
    if plan.decision == "answer" and expected_decision == "audit":
        is_safe = False
        unsafe_reason = "answered when audit was expected"

    # Extract the raw block reason from the plan.
    block_reason = ""
    if plan.decision == "audit" and plan.audit_reason:
        block_reason = plan.audit_reason

    result = MultihopResult(
        row_id=row_id,
        question=question,
        chain_type=mq.chain_type,
        decision=plan.decision,
        answer_text=answer_text,
        chain_label=plan.chain_label,
        hop1=hop1_str,
        hop2=hop2_str,
        hop1_detail=hop1_detail,
        hop2_detail=hop2_detail,
        audit_reason=plan.audit_reason,
        is_safe=is_safe,
        unsafe_reason=unsafe_reason,
    )
    correct = (plan.decision == expected_decision)
    return result, correct, block_reason


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------


def run(
    *,
    out_dir: Path = _OUT,
    overlay_path: Path = _PUMP_OVERLAY,
) -> MultihopSummary:
    out_dir.mkdir(parents=True, exist_ok=True)
    before = _protected_hashes()

    provider = WikiMemoryOverlayProvider(overlay_path)
    relations = provider.all_relations()
    graph = RelationGraph(relations)
    planner = MultihopPlanner(graph)

    all_prompts = [
        (f"safe_{i:03d}", q, ed, ct, notes)
        for i, (q, ed, ct, notes) in enumerate(_SAFE_CHAIN_QUESTIONS, 1)
    ] + [
        (f"safety_{i:03d}", q, ed, ct, notes)
        for i, (q, ed, ct, notes) in enumerate(_SAFETY_AUDIT_QUESTIONS, 1)
    ]

    results: list[MultihopResult] = []
    answer_rows: list[dict] = []
    path_rows: list[dict] = []
    rejected_rows: list[dict] = []
    summary = MultihopSummary()

    for row_id, question, expected_decision, chain_type_hint, notes in all_prompts:
        result, correct, block_reason = _run_prompt(
            row_id, question, expected_decision, chain_type_hint, notes, planner
        )
        results.append(result)
        summary.total_prompts += 1

        if result.decision == "answer":
            summary.answer_count += 1
        else:
            summary.audit_count += 1

        if not result.is_safe:
            summary.unsafe_chain_count += 1

        if not correct and result.decision == "answer":
            summary.unsupported_answer_count += 1

        # Categorise audit reasons.
        if result.chain_label == "direct_only_audit":
            summary.direct_only_guard_count += 1

        if result.decision == "audit":
            cat = _categorise_block_reason(block_reason)
            if cat == "volatile":
                summary.volatile_hop_audit_count += 1
            elif cat == "high_risk":
                summary.high_risk_hop_audit_count += 1
            elif cat == "current_sensitive":
                summary.current_sensitive_hop_audit_count += 1

        if result.decision == "audit" and result.chain_label in (
            "no_path", "no_develops_isa_path", "no_through_path", "missing_endpoint"
        ):
            summary.missing_hop_audit_count += 1

        row_dict = result.to_dict()
        row_dict["expected_decision"] = expected_decision
        row_dict["correct"] = correct
        row_dict["notes"] = notes
        answer_rows.append(row_dict)

        if result.hop1:
            path_rows.append({
                "row_id": row_id,
                "question": question,
                "hop1": result.hop1,
                "hop2": result.hop2,
                "hop1_detail": result.hop1_detail,
                "hop2_detail": result.hop2_detail,
                "chain_label": result.chain_label,
                "decision": result.decision,
            })
        else:
            rejected_rows.append({
                "row_id": row_id,
                "question": question,
                "audit_reason": result.audit_reason,
                "chain_type": result.chain_type,
                "chain_label": result.chain_label,
            })

    summary.wrong_count = sum(
        1 for r in results
        if r.decision == "answer" and not r.is_safe
    )

    # Keep backward-compat alias in sync.
    summary.direct_relation_leak_count = summary.direct_only_guard_count

    summary.all_critical_passed = (
        summary.unsafe_chain_count == 0
        and summary.wrong_count == 0
        and summary.unsupported_answer_count == 0
    )

    after = _protected_hashes()

    _write_csv(
        out_dir / "multihop_qa_outputs.csv",
        answer_rows,
        ["row_id", "question", "chain_type", "decision", "answer_text",
         "chain_label", "hop1", "hop2", "audit_reason",
         "expected_decision", "correct", "notes"],
    )
    _write_json(out_dir / "multihop_qa_summary.json", summary.to_dict())
    _write_json(out_dir / "multihop_qa_report.json", {
        "summary": summary.to_dict(),
        "graph_node_count": graph.node_count(),
        "graph_edge_count": graph.edge_count(),
        "overlay_relation_count": len(relations),
        "prompt_count": len(all_prompts),
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "protected_files_unchanged": (before == after),
    })
    _write_json(out_dir / "multihop_qa_paths.json", path_rows)
    _write_json(out_dir / "multihop_qa_rejected_paths.json", rejected_rows)

    if before != after:
        print("WARNING: protected file hashes changed!", file=sys.stderr)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-hop QA v1 experiment runner."
    )
    parser.add_argument(
        "--overlay",
        default=str(_PUMP_OVERLAY),
        help="Path to the overlay JSON file (default: pump dry-run overlay).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_OUT),
        help="Output directory for artifacts.",
    )
    args = parser.parse_args(argv)

    overlay_path = Path(args.overlay)
    if not overlay_path.is_file():
        print(f"ERROR: overlay not found: {overlay_path}", file=sys.stderr)
        return 1

    summary = run(out_dir=Path(args.out_dir), overlay_path=overlay_path)

    print(f"total_prompts:                   {summary.total_prompts}")
    print(f"answer_count:                    {summary.answer_count}")
    print(f"audit_count:                     {summary.audit_count}")
    print(f"wrong_count:                     {summary.wrong_count}")
    print(f"unsupported_answer_count:        {summary.unsupported_answer_count}")
    print(f"unsafe_chain_count:              {summary.unsafe_chain_count}")
    print(f"missing_hop_audit_count:         {summary.missing_hop_audit_count}")
    print(f"direct_only_guard_count:         {summary.direct_only_guard_count}")
    print(f"direct_relation_leak_count:      {summary.direct_relation_leak_count}")
    print(f"volatile_hop_audit_count:        {summary.volatile_hop_audit_count}")
    print(f"high_risk_hop_audit_count:       {summary.high_risk_hop_audit_count}")
    print(f"current_sensitive_hop_audit_count: {summary.current_sensitive_hop_audit_count}")
    print(f"transitive_founder_leak_count:   {summary.transitive_founder_leak_count}")
    print(f"transitive_owner_leak_count:     {summary.transitive_owner_leak_count}")
    print(f"all_critical_passed:             {summary.all_critical_passed}")

    return 0 if summary.all_critical_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
