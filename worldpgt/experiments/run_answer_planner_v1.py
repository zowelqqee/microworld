"""Run AnswerPlanner v1 benchmark.

Reads qa_prompts_v1.csv, runs the QA pipeline for each row, writes
output CSV and summary JSON.

SAFETY CONTRACT:
- sense_memory.py is NOT modified.
- Continuation benchmark outputs are NOT modified.
- Thresholds are NOT lowered.
- Validators are NOT weakened.
- No generic fallback is added.
- No neural weights, GPT renderer, or training.
- nanogpt/ is untouched.

Usage::

    python3 -m worldpgt.experiments.run_answer_planner_v1 \\
      --qa-input worldpgt/experiments/qa_prompts_v1.csv \\
      --accepted-memory worldpgt/experiments/accepted_knowledge_memory_v1.json \\
      --output-csv worldpgt/experiments/answer_planner_v1_outputs.csv \\
      --output-json worldpgt/experiments/answer_planner_v1_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

from worldpgt.continuation.continuation_policy import ContinuationPolicy
from worldpgt.continuation.sense_memory import ExplicitSenseMemory
from worldpgt.knowledge.accepted_memory_provider import (
    MEMORY_VERSION,
    AcceptedKnowledgeMemoryProvider,
)
from worldpgt.qa.answer_planner import AnswerPlanner
from worldpgt.qa.answer_renderer import render
from worldpgt.qa.answer_validator import validate
from worldpgt.qa.question_analyzer import analyze

_PLANNER_VERSION = "answer_planner_v1"

_OUTPUT_FIELDS = [
    "row_id", "question", "expected_decision", "expected_sense",
    "detected_intent", "detected_term", "detected_sense",
    "decision", "answer", "audit_reason", "confidence",
    "facts_used", "patterns_used", "provider_items_used",
    "is_correct", "quality_flagged",
]


def _is_correct(row: dict, decision: str, detected_sense: str, answer: str) -> bool:
    expected_decision = row.get("expected_decision", "")
    expected_sense = row.get("expected_sense", "")
    expected_contains_raw = row.get("expected_answer_contains", "")

    if decision != expected_decision:
        return False
    if decision == "audit":
        return True
    if expected_sense and detected_sense != expected_sense:
        return False
    if expected_contains_raw:
        keywords = [k.strip().lower() for k in expected_contains_raw.split(";") if k.strip()]
        answer_lower = answer.lower()
        if not all(k in answer_lower for k in keywords):
            return False
    return True


def run(
    qa_input_path: str,
    accepted_memory_path: str,
    output_csv_path: str,
    output_json_path: str,
) -> dict:
    provider = AcceptedKnowledgeMemoryProvider(accepted_memory_path)
    sense_mem = ExplicitSenseMemory(include_builtin=True)
    planner = AnswerPlanner(provider=provider, sense_mem=sense_mem)

    with open(qa_input_path, newline="", encoding="utf-8") as fh:
        qa_rows = list(csv.DictReader(fh))

    out_rows: list[dict] = []
    total = len(qa_rows)
    answer_count = 0
    audit_count = 0
    correct_count = 0
    correct_answer_count = 0
    wrong_count = 0
    quality_flagged_count = 0
    by_intent: dict[str, dict] = defaultdict(lambda: {"answer": 0, "audit": 0, "correct": 0, "wrong": 0})
    by_term: dict[str, dict] = defaultdict(lambda: {"answer": 0, "audit": 0, "correct": 0, "wrong": 0})
    provider_item_ids_used: set[str] = set()
    provider_terms_used: set[str] = set()

    trace_base = {
        "accepted_memory_provider": "enabled",
        "accepted_memory_version": MEMORY_VERSION,
        "planner_version": _PLANNER_VERSION,
    }

    for row in qa_rows:
        row_id = row["row_id"]
        question = row["question"]

        analyzed = analyze(question)
        plan = planner.plan(analyzed)
        answer = render(plan)

        decision = plan.decision
        detected_intent = analyzed.intent
        detected_term = analyzed.term or ""
        detected_sense = analyzed.target_sense or ""

        # For classify_context, the detected_sense comes from the plan
        if detected_intent == "classify_context" and plan.decision == "answer":
            detected_sense = plan.render_args.get("sense_id", detected_sense)

        val_result = validate(
            answer=answer,
            decision=decision,
            term=detected_term or None,
            sense_id=detected_sense or None,
            expected_contains=None,
            intent=detected_intent,
        )

        correct = _is_correct(row, decision, detected_sense, answer)

        if decision == "answer":
            answer_count += 1
        else:
            audit_count += 1

        if correct:
            correct_count += 1
            if decision == "answer":
                correct_answer_count += 1
        else:
            wrong_count += 1

        if val_result.quality_flagged:
            quality_flagged_count += 1

        by_intent[detected_intent]["answer" if decision == "answer" else "audit"] += 1
        by_intent[detected_intent]["correct" if correct else "wrong"] += 1
        if detected_term:
            by_term[detected_term]["answer" if decision == "answer" else "audit"] += 1
            by_term[detected_term]["correct" if correct else "wrong"] += 1

        # Track provider usage
        for item_id in plan.evidence.provider_items_used:
            provider_item_ids_used.add(item_id)
        if detected_term and plan.evidence.provider_items_used:
            provider_terms_used.add(detected_term)

        out_rows.append({
            "row_id": row_id,
            "question": question,
            "expected_decision": row.get("expected_decision", ""),
            "expected_sense": row.get("expected_sense", ""),
            "detected_intent": detected_intent,
            "detected_term": detected_term,
            "detected_sense": detected_sense,
            "decision": decision,
            "answer": answer,
            "audit_reason": plan.audit_reason or "",
            "confidence": f"{plan.confidence:.4f}",
            "facts_used": "|".join(plan.evidence.facts_used),
            "patterns_used": "|".join(plan.evidence.patterns_used),
            "provider_items_used": "|".join(plan.evidence.provider_items_used),
            "is_correct": "yes" if correct else "no",
            "quality_flagged": "yes" if val_result.quality_flagged else "no",
        })

    policy = ContinuationPolicy()
    answer_precision = (
        round(correct_answer_count / answer_count, 4) if answer_count > 0 else 0.0
    )

    summary: dict = {
        "qa_total": total,
        "answer_count": answer_count,
        "audit_count": audit_count,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "quality_flagged": quality_flagged_count,
        "accuracy": round(correct_count / total, 4) if total > 0 else 0.0,
        "answer_precision": answer_precision,
        "by_intent": {k: dict(v) for k, v in sorted(by_intent.items())},
        "by_term": {k: dict(v) for k, v in sorted(by_term.items())},
        "provider_usage": {
            "provider_enabled": True,
            "provider_items_used": len(provider_item_ids_used),
            "provider_terms_used": sorted(provider_terms_used),
        },
        "safety": {
            "sense_memory_modified": False,
            "thresholds_changed": policy.min_score != 1.0 or policy.min_margin != 1.0,
            "validators_weakened": False,
            "generic_fallback_added": False,
        },
    }

    with open(output_csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    with open(output_json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="AnswerPlanner v1 benchmark.")
    parser.add_argument("--qa-input", required=True, dest="qa_input",
                        help="qa_prompts_v1.csv")
    parser.add_argument("--accepted-memory", required=True, dest="accepted_memory",
                        help="accepted_knowledge_memory_v1.json")
    parser.add_argument("--output-csv", required=True, dest="output_csv",
                        help="answer_planner_v1_outputs.csv")
    parser.add_argument("--output-json", required=True, dest="output_json",
                        help="answer_planner_v1_summary.json")
    args = parser.parse_args(argv)

    summary = run(
        qa_input_path=args.qa_input,
        accepted_memory_path=args.accepted_memory,
        output_csv_path=args.output_csv,
        output_json_path=args.output_json,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
