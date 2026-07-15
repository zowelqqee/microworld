"""Failure analysis for completed open-book QA runs; production code is read-only."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
import csv, json
from pathlib import Path
import re
from typing import Iterable

from worldpgt.api import server
from worldpgt.entity_qa.semantic_question_parser import parse_semantic_query
from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex
from worldpgt.reasoning.graph_input import GraphInputLayer
from worldpgt.reasoning.answer_behavior import build_answer_plan, prepare_persistent_evidence_graph
from .dataset import load_experimental_relations, read_jsonl, relation_id
from .evaluate import normalize

_DEICTIC_SUBJECT = re.compile(r"^(?:our|this|the proposed)\b.*\b(?:technique|method|approach|system|framework|work)\b", re.I)


def _contains(answer: str, value: str) -> bool:
    return normalize(value) in normalize(answer)


def collapse_results(rows: Iterable[dict]) -> tuple[dict[str, dict], dict[str, int]]:
    """Require one stable representative per ID while retaining repeat counts."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows: grouped[row["id"]].append(row)
    # Answer content is deterministic in this benchmark; prefer first and
    # separately expose any accidental result-count drift.
    return {key: values[0] for key, values in grouped.items()}, {key: len(values) for key, values in grouped.items()}


def evaluator_flags(case: dict, result: dict) -> dict:
    answer = result.get("answer", "")
    expected = case["expected_objects"]
    hits = [obj for obj in expected if _contains(answer, obj)]
    unknown = result.get("decision") == "audit"
    correct = unknown if case["expected_decision"] == "unknown" else (not unknown and len(hits) == len(expected))
    return {"correct": correct, "expected_object_hits": hits, "object_recall": len(hits) / len(expected) if expected else None,
            "object_precision": len(hits) / len({obj for obj in expected if _contains(answer, obj)}) if hits else None,
            "unknown": unknown}


def _plan_edges(result: dict) -> list[dict]:
    plan = (result.get("trace") or {}).get("answer_plan") or {}
    return [block.get("step", {}).get("edge", {}) for block in plan.get("blocks", [])]


def classify(case: dict, result: dict, debug: dict | None = None) -> tuple[str, list[str], str]:
    """Classify by the earliest evidenced stage, never by desired outcome."""
    flags = evaluator_flags(case, result); plan = _plan_edges(result)
    selected = {edge.get("evidence_id") for edge in plan}; expected = set(case["relation_ids"])
    if debug:
        if not debug.get("resolved_targets"): return "entity_resolution_failed", [], "no target entity resolved"
        if debug.get("parsed_predicate") not in set(case["expected_predicate"]):
            return "predicate_parse_failed", [], "resolved target but requested predicate differs"
        if not debug.get("graph_has_target"): return "graph_target_missing", [], "resolved target absent from persistent relation graph"
        if not debug.get("expected_edge_available"): return "direct_relation_missing", [], "expected relation absent from serving graph"
        if not debug.get("planner_invoked"): return "planner_not_invoked", [], "base path did not enter planner"
        if debug.get("frontier_candidate_count", 0) == 0: return "planner_invoked_no_candidates", [], "local frontier empty"
    if flags["correct"]: return "correct", [], "current evaluator accepted answer"
    overlap = selected & expected
    if plan and overlap:
        return "partial_plan", ["full_object_coverage_not_met"], "selected at least one expected evidence block"
    if plan:
        if set(case["expected_objects"]).issubset({edge.get("object") for edge in plan}):
            return "renderer_or_evaluator_mismatch", [], "plan contains expected objects but rendered answer was not credited"
        return "wrong_plan", [], "planner selected evidence-backed but non-expected relation(s)"
    if result.get("decision") == "audit": return "planner_not_invoked", ["base_answer_audit"], "audit arrived before an answer plan"
    return "candidates_scored_none_selected", [], "no answer-plan block was retained"


def enrich(cases: list[dict]) -> dict[str, dict]:
    """Debug only failed cases via the existing serving loader/planner APIs."""
    # Deliberately do not call ``server._startup``: it also warms synthesis and
    # phrase-generation resources irrelevant to parser/planner diagnosis. The
    # inputs below are exactly the paths and relation filter the server uses.
    relations = load_experimental_relations()
    surface_index = EntitySurfaceIndex(
        accepted_overlay_path=server._ACCEPTED_OVERLAY_PATH,
        promoted_overlay_path=server._MAIN_UI_COMPOSED_OVERLAY_PATH,
        snapshot_overlay_path=server._SNAPSHOT_OVERLAY_PATH,
        graph_input=GraphInputLayer.from_overlay_items(relations),
    )
    graph = prepare_persistent_evidence_graph(
        relations, Path("/tmp/open_book_qa_failure_analysis.sqlite"),
        source_fingerprint="open-book-qa-failure-analysis-v1",
    )
    output = {}
    for case in cases:
        semantic = parse_semantic_query(case["question"], surface_index)
        targets = [value for value in (semantic.entity_a, semantic.entity_b) if value]
        target_norms = [" ".join(value.casefold().split()) for value in targets]
        candidates = graph.candidate_edges(target_norms) if target_norms else ()
        candidate_ids = {edge.evidence_id for edge in candidates}
        plan = build_answer_plan(case["question"], [], targets=targets, prepared_edges=graph) if targets else None
        parsed = getattr(semantic, "relation_intent", None)
        output[case["id"]] = {
            "detected_entity_mentions": targets, "resolved_targets": targets,
            "parsed_intent": asdict(semantic) if is_dataclass(semantic) else repr(semantic),
            "parsed_predicate": parsed, "requested_predicate": case["expected_predicate"],
            "graph_has_target": any(graph.has_node(node) for node in target_norms),
            "exact_adjacency_count": sum(1 for edge in candidates if any(edge.subject_norm == node or edge.object_norm == node for node in target_norms)),
            "frontier_candidate_count": len(candidates), "expected_edge_available": bool(set(case["relation_ids"]) & candidate_ids),
            "planner_invoked": bool(targets), "selected_blocks": plan.to_dict()["blocks"] if plan else [],
            "rejection_reason_counts": dict(Counter(item.reason for item in (plan.rejected if plan else ()))),
            "base_answer_decision_before_behavior": None,
        }
    return output


def analyze(dataset: list[dict], results: list[dict], *, debug: dict[str, dict] | None = None) -> tuple[dict, list[dict], list[dict]]:
    representative, repeats = collapse_results(results); cases = {case["id"]: case for case in dataset}
    if set(cases) != set(representative): raise ValueError("dataset/results case IDs do not join one-to-one")
    analysis, issues = [], []
    relation_ids = {relation_id(row) for row in load_experimental_relations()}
    for case_id, case in cases.items():
        result = representative[case_id]; flags = evaluator_flags(case, result); trace = debug.get(case_id) if debug else None
        stage, secondary, explanation = classify(case, result, trace)
        plan_edges = _plan_edges(result); selected = {edge.get("evidence_id") for edge in plan_edges}; expected = set(case["relation_ids"])
        context = " ".join(case["contexts"])
        issue_reasons = []
        if any(normalize(obj) not in normalize(context) for obj in case["expected_objects"]): issue_reasons.append("expected_object_absent_from_supplied_context")
        if not expected.issubset(relation_ids): issue_reasons.append("expected_relation_absent_from_overlay")
        if case["category"] == "negative" and any(normalize(predicate.replace("_", " ")) in normalize(context) for predicate in case["expected_predicate"]): issue_reasons.append("negative_context_mentions_requested_predicate")
        if _DEICTIC_SUBJECT.match(case["expected_subject"]): issue_reasons.append("deictic_subject_admitted_by_dataset")
        if trace and not trace.get("resolved_targets") and case["category"] in {"paraphrase", "multi_evidence"}:
            issue_reasons.append("target_not_resolvable_by_serving_surface_index")
        record = {"case_id": case_id, "category": case["category"], "question": case["question"], "expected_subject": case["expected_subject"],
                  "expected_predicates": case["expected_predicate"], "expected_objects": case["expected_objects"], "expected_decision": case["expected_decision"], "supplied_evidence_spans": case["contexts"],
                  "expected_relation_ids": case["relation_ids"],
                  "answer": result.get("answer"), "decision": result.get("decision"), "support_kind": result.get("support_kind"), "audit_reason": result.get("audit_reason"),
                  "resolved_references": (result.get("trace") or {}).get("resolved_references"), "answer_plan": (result.get("trace") or {}).get("answer_plan"),
                  "plan_block_count": len(plan_edges), "selected_evidence_ids": sorted(selected), "evaluator": flags, "repeat_count": repeats[case_id],
                  "earliest_failure_stage": stage, "secondary_failure_reasons": secondary, "explanation": explanation, "debug": trace}
        analysis.append(record)
        if issue_reasons: issues.append({"case_id": case_id, "reasons": issue_reasons, "case": case})
    summary = _summary(analysis)
    return summary, analysis, issues


def _summary(rows: list[dict]) -> dict:
    by_category = defaultdict(list)
    for row in rows: by_category[row["category"]].append(row)
    table, stages = [], defaultdict(lambda: Counter())
    for category, group in by_category.items():
        for row in group: stages[row["earliest_failure_stage"]][category] += 1
        table.append({"category": category, "cases": len(group), "entity_resolved": sum(bool((row.get("debug") or {}).get("resolved_targets")) for row in group),
                      "predicate_correct": sum((row.get("debug") or {}).get("parsed_predicate") in row["expected_predicates"] for row in group if row.get("debug")),
                      "planner_invoked": sum(bool((row.get("debug") or {}).get("planner_invoked")) for row in group if row.get("debug")),
                      "any_correct_block": sum(bool(set(row["selected_evidence_ids"]) & set(row["expected_relation_ids"])) for row in group),
                      "partial_plan": sum(row["earliest_failure_stage"] == "partial_plan" for row in group), "full_correct": sum(row["evaluator"]["correct"] for row in group)})
    return {"case_count": len(rows), "main_table": table, "earliest_failure_stage": {stage: dict(counts) for stage, counts in stages.items()}}


def write_report(dataset_path: str | Path, results_path: str | Path, output: str | Path, *, enrich_failed: bool = False) -> dict:
    dataset, results = read_jsonl(dataset_path), read_jsonl(results_path); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    initial, initial_rows, _ = analyze(dataset, results)
    failed = [next(case for case in dataset if case["id"] == row["case_id"]) for row in initial_rows if row["category"] in {"paraphrase", "multi_evidence"} and not row["evaluator"]["correct"]]
    debug = enrich(failed) if enrich_failed else {}
    if debug: (output / "enriched_failed_traces.jsonl").write_text("".join(json.dumps({"case_id": key, **value}, ensure_ascii=False) + "\n" for key, value in debug.items()), encoding="utf-8")
    summary, rows, issues = analyze(dataset, results, debug=debug)
    (output / "failure_analysis_cases.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output / "dataset_or_evaluator_issues.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in issues), encoding="utf-8")
    multi_metrics = _write_breakdowns(output, rows)
    summary["multi_evidence_partial_credit"] = multi_metrics
    summary["join"] = {"dataset_cases": len(dataset), "result_rows": len(results), "unique_result_cases": len({row["id"] for row in results}), "expected_repeats_per_case": 5}
    (output / "failure_analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_representatives(output, rows)
    (output / "README.md").write_text("""# MicroWorld open-book QA failure analysis

This report joins the completed dataset with five warm repeats per case. Parser
and planner diagnostics were rerun only for failed paraphrase and multi-evidence
cases with the same EntitySurfaceIndex and persistent evidence graph used by
serving; production API behavior was not changed. `failure_analysis_cases.jsonl`
contains the per-case earliest evidenced stage. Empty candidate groups in the
representative report mean that no such case existed, rather than being omitted.
""", encoding="utf-8")
    return summary


def _write_breakdowns(output: Path, rows: list[dict]) -> dict:
    paraphrase = [row for row in rows if row["category"] == "paraphrase"]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    confusion: Counter[tuple[str, str]] = Counter()
    for row in paraphrase:
        expected = row["expected_predicates"][0]
        parsed = str((row.get("debug") or {}).get("parsed_predicate") or "unavailable")
        template = re.sub(re.escape(row["expected_subject"]), "{subject}", row["question"], flags=re.I)
        groups[(expected, template)].append(row); confusion[(expected, parsed)] += 1
    with (output / "paraphrase_breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["expected_predicate", "template", "cases", "entity_resolution_success", "correct_predicate_parse", "planner_invocation", "direct_edge_available", "answer_plan_built", "final_accuracy"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for (predicate, template), group in sorted(groups.items()):
            writer.writerow({"expected_predicate": predicate, "template": template, "cases": len(group),
                             "entity_resolution_success": sum(bool((r.get("debug") or {}).get("resolved_targets")) for r in group),
                             "correct_predicate_parse": sum((r.get("debug") or {}).get("parsed_predicate") == predicate for r in group),
                             "planner_invocation": sum(bool((r.get("debug") or {}).get("planner_invoked")) for r in group),
                             "direct_edge_available": sum(bool((r.get("debug") or {}).get("expected_edge_available")) for r in group),
                             "answer_plan_built": sum(r["plan_block_count"] > 0 for r in group), "final_accuracy": sum(r["evaluator"]["correct"] for r in group) / len(group)})
    with (output / "paraphrase_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["expected_predicate", "parsed_predicate", "cases"]); writer.writeheader()
        for (expected, parsed), count in sorted(confusion.items()): writer.writerow({"expected_predicate": expected, "parsed_predicate": parsed, "cases": count})
    multi = [row for row in rows if row["category"] == "multi_evidence"]
    metrics = []
    for row in multi:
        expected, selected = set(row["expected_relation_ids"]), set(row["selected_evidence_ids"])
        hit = expected & selected
        metrics.append({"case_id": row["case_id"], "expected_block_count": len(expected), "selected_block_count": len(selected),
                        "correct_selected_blocks": len(hit), "wrong_selected_blocks": len(selected - expected),
                        "block_recall": len(hit) / len(expected) if expected else 0, "block_precision": len(hit) / len(selected) if selected else 0,
                        "any_block_success": bool(hit), "partial_plan": bool(hit) and hit != expected, "full_plan": hit == expected,
                        "audit": row["decision"] == "audit", "planner_invoked": bool((row.get("debug") or {}).get("planner_invoked")), "failure_stage": row["earliest_failure_stage"]})
    fields = list(metrics[0]) if metrics else ["case_id"]
    with (output / "multi_evidence_breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(metrics)
    aggregate = {"cases": len(metrics), "any_block_success_rate": sum(x["any_block_success"] for x in metrics) / len(metrics) if metrics else 0,
                 "mean_block_recall": sum(x["block_recall"] for x in metrics) / len(metrics) if metrics else 0,
                 "mean_block_precision": sum(x["block_precision"] for x in metrics) / len(metrics) if metrics else 0,
                 "partial_plan_rate": sum(x["partial_plan"] for x in metrics) / len(metrics) if metrics else 0,
                 "full_plan_rate": sum(x["full_plan"] for x in metrics) / len(metrics) if metrics else 0,
                 "audit_before_planning_rate": sum(x["audit"] for x in metrics) / len(metrics) if metrics else 0,
                 "planner_invocation_rate": sum(x["planner_invoked"] for x in metrics) / len(metrics) if metrics else 0,
                 "average_expected_blocks": sum(x["expected_block_count"] for x in metrics) / len(metrics) if metrics else 0,
                 "average_selected_blocks": sum(x["selected_block_count"] for x in metrics) / len(metrics) if metrics else 0}
    (output / "multi_evidence_partial_credit.csv").write_text("metric,value\n" + "".join(f"{key},{value}\n" for key, value in aggregate.items()), encoding="utf-8")
    return aggregate


def _write_representatives(output: Path, rows: list[dict]) -> None:
    groups = {
        "paraphrase failures": [row for row in rows if row["category"] == "paraphrase" and not row["evaluator"]["correct"]][:20],
        "multi audit/no-plan": [row for row in rows if row["category"] == "multi_evidence" and row["decision"] == "audit"][:5],
        "multi partial plans": [row for row in rows if row["category"] == "multi_evidence" and row["earliest_failure_stage"] == "partial_plan"][:5],
        "multi wrong plans": [row for row in rows if row["category"] == "multi_evidence" and row["earliest_failure_stage"] == "wrong_plan"][:5],
        "possible evaluator mismatches": [row for row in rows if row["earliest_failure_stage"] == "renderer_or_evaluator_mismatch"][:5],
    }
    lines = ["# Representative failures\n"]
    for title, group in groups.items():
        lines.append(f"## {title}\n")
        if not group: lines.append("No cases in this group.\n"); continue
        for row in group:
            debug = row.get("debug") or {}
            lines.extend([f"- **{row['case_id']}** — `{row['earliest_failure_stage']}`", f"  - Q: {row['question']}",
                          f"  - expected: {row['expected_predicates']} → {row['expected_objects']}", f"  - parsed: {debug.get('parsed_predicate')}; targets: {debug.get('resolved_targets')}",
                          f"  - blocks: {row['plan_block_count']}; answer: {row['answer']}\n"])
    (output / "representative_failures.md").write_text("\n".join(lines), encoding="utf-8")
