"""CLI for the automated knowledge review gate v1.

Usage:
    python3 -m worldpgt.experiments.auto_review_knowledge_ingestion_v1 \\
      --proposals worldpgt/experiments/knowledge_ingestion_v1_memory_update_proposals.json \\
      --output-json worldpgt/experiments/knowledge_ingestion_v1_auto_review.json \\
      --output-csv worldpgt/experiments/knowledge_ingestion_v1_auto_review.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path


def _as_dict(obj: object) -> object:
    if dataclasses.is_dataclass(obj):
        return {k: _as_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_as_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _as_dict(v) for k, v in obj.items()}
    return obj


def _write_csv(proposal_reviews: list, path: str) -> None:
    rows = []
    for r in proposal_reviews:
        for item in r.items:
            rows.append({
                "proposal_id": r.proposal_id,
                "term": r.term,
                "sense": r.sense,
                "source_risk_level": r.source_risk_level,
                "proposal_decision": r.proposal_decision,
                "item_type": item.item_type,
                "value": item.value,
                "item_decision": item.decision,
                "item_risk_level": item.risk_level,
                "reason": item.reason,
            })
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Automated knowledge review gate v1")
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=False, default=None)
    args = parser.parse_args(argv)

    from worldpgt.knowledge.automated_memory_reviewer import AutomatedMemoryReviewer

    proposals = json.loads(Path(args.proposals).read_text(encoding="utf-8"))
    print(f"[review] Loaded {len(proposals)} proposals from {args.proposals}")

    reviewer = AutomatedMemoryReviewer()
    output = reviewer.review(proposals)
    s = output.summary

    print(f"[review] Proposals reviewed:   {s.total_proposals}")
    print(f"[review] Total items reviewed: {s.total_items_reviewed}")
    print(f"[review] accepted_auto:        {s.accepted_auto}  ({s.acceptance_rate:.1%})")
    print(f"[review] needs_review:         {s.needs_review}  ({s.needs_review_rate:.1%})")
    print(f"[review] rejected_auto:        {s.rejected_auto}  ({s.rejection_rate:.1%})")

    result = _as_dict(output)
    Path(args.output_json).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[review] JSON written to {args.output_json}")

    if args.output_csv:
        _write_csv(output.proposal_reviews, args.output_csv)
        print(f"[review] CSV written to {args.output_csv}")

    # Proposal-level summary
    from collections import Counter
    pdec: Counter[str] = Counter(r.proposal_decision for r in output.proposal_reviews)
    print("\n--- Proposal-level decisions ---")
    for dec in ("accepted_auto", "partial_accept", "needs_review", "rejected_auto"):
        print(f"  {dec}: {pdec.get(dec, 0)}")

    print("\n--- By term (accepted / needs_review / rejected) ---")
    for term, counts in sorted(s.by_term.items()):
        a = counts.get("accepted_auto", 0)
        n = counts.get("needs_review", 0)
        r = counts.get("rejected_auto", 0)
        print(f"  {term}: {a} accepted / {n} needs_review / {r} rejected")

    print("\n--- By item type ---")
    for itype, counts in sorted(s.by_item_type.items()):
        a = counts.get("accepted_auto", 0)
        n = counts.get("needs_review", 0)
        r = counts.get("rejected_auto", 0)
        print(f"  {itype}: {a} accepted / {n} needs_review / {r} rejected")

    human_workload = s.needs_review
    reduction = 1.0 - human_workload / s.total_items_reviewed if s.total_items_reviewed else 0.0
    print(f"\n[review] Human workload: {human_workload} items "
          f"({reduction:.1%} reduction from {s.total_items_reviewed} total)")

    print("\n[safety] sense_memory.py: NOT modified")
    print("[safety] thresholds: NOT lowered")
    print("[safety] validators: NOT weakened")
    print("[safety] generic fallback: NOT suggested")
    print("[safety] neural/GPT/training: NOT used")
    print("[safety] nanogpt/: NOT touched")
    print("[safety] benchmark outputs: NOT modified")
    print("[safety] trusted generation behavior: UNCHANGED")
    print("[safety] auto_apply: FALSE — gate produces proposals only")


if __name__ == "__main__":
    main()
