"""Aggregate resumable proposal-only overnight batch artifacts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from worldpgt.benchmarks.open_book_qa.dataset import _norm


def _read(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-subjects", type=int, required=True)
    parser.add_argument("--stop-reason", default="completed")
    args = parser.parse_args()
    root = Path(args.root)
    batches = sorted(path for path in root.glob("wikidata_batch_*") if path.is_dir())
    manifests = [row for batch in batches for row in _read(batch / "resolution_manifest.json")]
    proposals = [row for batch in batches for row in _read(batch / "proposal_overlay.json")]
    status = Counter(str(row.get("canonical_resolution_status") or "unknown") for row in manifests)
    prior_candidates = {
        _norm(str(row.get("surface_subject") or row.get("subject") or ""))
        for row in _read(Path("artifacts/open_book_qa/wikidata_resolver_fix_v1/resolved_candidates.json"))
    }
    automatic_resolved = [row for row in manifests if row.get("canonical_qid")]
    newly_resolved = [row for row in automatic_resolved if _norm(str(row.get("surface_subject") or row.get("subject") or "")) not in prior_candidates]
    groups: dict[str, set[str]] = defaultdict(set)
    for row in proposals:
        groups[_norm(str(row.get("subject") or ""))].add(str(row.get("predicate") or ""))
    processed = len(manifests)
    complete = processed == args.expected_subjects and args.stop_reason == "completed"
    summary = {
        "version": "schema_expansion_overnight_run_20jul_v1",
        "proposal_only": True,
        "accepted_memory_modified": False,
        "serving_overlay_modified": False,
        "ready_for_human_review": True,
        "run_status": "complete" if complete else "partial",
        "stop_reason": args.stop_reason,
        "wikidata": {
            "input_pool": "wikidata_density_recon resolution manifest entries without canonical_qid",
            "expected_subjects": args.expected_subjects,
            "processed_subjects": processed,
            "unprocessed_subjects": max(0, args.expected_subjects - processed),
            "automatic_resolved_subjects_this_run": len(automatic_resolved),
            "previous_resolver_candidates_reconfirmed": len(automatic_resolved) - len(newly_resolved),
            "newly_resolved_subjects_beyond_prior_resolver_candidates": len(newly_resolved),
            "resolution_status_counts": dict(sorted(status.items())),
            "gate_passed_relations": len(proposals),
            "new_subjects_with_gate_passed_relations": len(groups),
            "new_multi_predicate_subjects": sum(len(predicates) >= 2 for predicates in groups.values()),
            "batch_outputs": [str(path.relative_to(root)) for path in batches],
        },
        "crossref": {
            "run_status": "not_run",
            "reason": "no unprocessed DOI candidates found in current retained quarantine/frontier artifacts",
            "gate_passed_relations": 0,
            "new_multi_predicate_subjects": 0,
        },
        "awaiting_manual_decision": {
            "promotion_called": False,
            "proposal_paths": [str((batch / "proposal_overlay.json").relative_to(root)) for batch in batches],
            "review_instruction": "Review proposal overlays and summaries; promotion remains a separate explicit human-authorized step.",
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
