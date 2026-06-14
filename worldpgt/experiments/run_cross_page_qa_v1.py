"""Run Cross-page Entity QA v1 benchmark over the wiki memory overlay.

Reads cross_page_qa_v1.csv, runs the cross-page QA pipeline for each row,
writes output CSV and summary JSON.

SAFETY CONTRACT:
- accepted_knowledge_memory_v1.json is NOT modified.
- sense_memory.py is NOT modified.
- Wiki ingestion / overlay builder semantics are NOT modified.
- Overlay provider gains only read-only bulk accessors (no semantic change).
- Validators are NOT weakened.
- No generic fallback. No neural/GPT/training/embedding code. No network.
- nanogpt/ is untouched.
- safe_for_general_runtime is always False.

Usage::

    python3 -m worldpgt.experiments.run_cross_page_qa_v1 \\
      --qa-input worldpgt/experiments/cross_page_qa_v1.csv \\
      --overlay-json worldpgt/experiments/accepted_wiki_memory_overlay_v1.json \\
      --output-csv worldpgt/experiments/cross_page_qa_v1_outputs.csv \\
      --output-json worldpgt/experiments/cross_page_qa_v1_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from worldpgt.cross_page_qa.cross_page_answer_planner import CrossPageAnswerPlanner
from worldpgt.cross_page_qa.cross_page_answer_renderer import render
from worldpgt.cross_page_qa.cross_page_answer_validator import validate
from worldpgt.cross_page_qa.cross_page_question_analyzer import analyze
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider

_OUTPUT_FIELDS = [
    "row_id",
    "question",
    "expected_decision",
    "expected_intent",
    "detected_intent",
    "source",
    "target",
    "decision",
    "answer",
    "audit_reason",
    "confidence",
    "relation_edges_used",
    "weak_links_used",
    "source_facts_used",
    "is_correct",
    "quality_flagged",
    "quality_reason",
]


def run(qa_input_path: str, overlay_json_path: str,
        output_csv_path: str, output_json_path: str) -> dict:
    provider = WikiMemoryOverlayProvider(overlay_json_path)
    planner = CrossPageAnswerPlanner(provider=provider)

    rows = list(csv.DictReader(Path(qa_input_path).read_text(encoding="utf-8").splitlines()))

    by_intent: dict[str, dict] = defaultdict(
        lambda: {"answer": 0, "audit": 0, "correct": 0, "wrong": 0})
    qa_total = len(rows)
    correct_count = wrong_count = quality_flagged_count = 0
    source_facts_total = weak_links_total = relation_edges_total = 0
    output_rows = []

    for row in rows:
        question = row["question"]
        q = analyze(question)
        plan = planner.plan(q)
        answer = render(plan)
        val = validate(row, plan.decision, answer, q.intent)

        if val.is_correct:
            correct_count += 1
        else:
            wrong_count += 1
        if val.quality_flagged:
            quality_flagged_count += 1

        by_intent[q.intent]["answer" if plan.decision == "answer" else "audit"] += 1
        by_intent[q.intent]["correct" if val.is_correct else "wrong"] += 1

        relation_edges_total += len(plan.evidence.relation_edges_used)
        weak_links_total += len(plan.evidence.weak_links_used)
        source_facts_total += len(plan.evidence.source_facts_used)

        output_rows.append({
            "row_id": row["row_id"],
            "question": question,
            "expected_decision": row.get("expected_decision", ""),
            "expected_intent": row.get("expected_intent", ""),
            "detected_intent": q.intent,
            "source": q.source or "",
            "target": q.target or "",
            "decision": plan.decision,
            "answer": answer,
            "audit_reason": plan.audit_reason or "",
            "confidence": plan.confidence,
            "relation_edges_used": ";".join(plan.evidence.relation_edges_used),
            "weak_links_used": ";".join(plan.evidence.weak_links_used),
            "source_facts_used": ";".join(plan.evidence.source_facts_used),
            "is_correct": val.is_correct,
            "quality_flagged": val.quality_flagged,
            "quality_reason": val.quality_reason,
        })

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    answer_count = sum(1 for r in output_rows if r["decision"] == "answer")
    audit_count = sum(1 for r in output_rows if r["decision"] == "audit")
    accuracy = correct_count / qa_total if qa_total else 0.0
    answer_precision = (
        sum(1 for r in output_rows if r["decision"] == "answer" and r["is_correct"]) / answer_count
        if answer_count else 1.0
    )

    summary = {
        "qa_total": qa_total,
        "answer_count": answer_count,
        "audit_count": audit_count,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "accuracy": round(accuracy, 4),
        "answer_precision": round(answer_precision, 4),
        "quality_flagged": quality_flagged_count,
        "by_intent": {k: dict(v) for k, v in sorted(by_intent.items())},
        "relation_edges_used": relation_edges_total,
        "weak_context_links_used": weak_links_total,
        "source_facts_used": source_facts_total,
        "safe_for_general_runtime": False,
    }

    Path(output_json_path).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Cross-page Entity QA v1 benchmark")
    parser.add_argument("--qa-input", required=True)
    parser.add_argument("--overlay-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)

    s = run(args.qa_input, args.overlay_json, args.output_csv, args.output_json)
    print(f"[cross_page_qa_v1] qa_total:         {s['qa_total']}")
    print(f"[cross_page_qa_v1] correct_count:    {s['correct_count']}")
    print(f"[cross_page_qa_v1] wrong_count:      {s['wrong_count']}")
    print(f"[cross_page_qa_v1] accuracy:         {s['accuracy']}")
    print(f"[cross_page_qa_v1] answer_precision: {s['answer_precision']}")
    print(f"[cross_page_qa_v1] quality_flagged:  {s['quality_flagged']}")
    print(f"[cross_page_qa_v1] answer_count:     {s['answer_count']}")
    print(f"[cross_page_qa_v1] audit_count:      {s['audit_count']}")
    print(f"[cross_page_qa_v1] by_intent:        {s['by_intent']}")
    print(f"[cross_page_qa_v1] safe_for_general_runtime: {s['safe_for_general_runtime']}")

    print("\n[safety] accepted_knowledge_memory_v1.json: NOT modified")
    print("[safety] sense_memory.py: NOT modified")
    print("[safety] ingestion/overlay builder semantics: NOT modified")
    print("[safety] validators: NOT weakened")
    print("[safety] generic fallback: NOT added")
    print("[safety] neural/GPT/training/embedding: NOT used")
    print("[safety] network access: NONE")
    print("[safety] nanogpt/: NOT touched")


if __name__ == "__main__":
    main()
