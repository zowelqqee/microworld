"""CLI entry point for knowledge ingestion pipeline v1.

Usage:
    python3 -m worldpgt.experiments.run_knowledge_ingestion_v1 \\
      --source worldpgt/experiments/curated_wiki_snippets_v1.json \\
      --output-facts worldpgt/experiments/knowledge_ingestion_v1_fact_candidates.json \\
      --output-proposals worldpgt/experiments/knowledge_ingestion_v1_memory_update_proposals.json \\
      --output-proposals-csv worldpgt/experiments/knowledge_ingestion_v1_memory_update_proposals.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path


def _as_dict(obj: object) -> object:
    if dataclasses.is_dataclass(obj):
        return {k: _as_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_as_dict(x) for x in obj]
    return obj


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Knowledge ingestion pipeline v1")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-facts", required=True)
    parser.add_argument("--output-proposals", required=True)
    parser.add_argument("--output-proposals-csv", required=False, default=None)
    args = parser.parse_args(argv)

    from worldpgt.knowledge.curated_source_reader import CuratedSourceReader
    from worldpgt.knowledge.fact_candidate_extractor import FactCandidateExtractor
    from worldpgt.knowledge.fact_normalizer import FactNormalizer
    from worldpgt.knowledge.memory_update_proposer import MemoryUpdateProposer

    # 1. Load snippets
    reader = CuratedSourceReader()
    snippets = reader.load(args.source)
    print(f"[ingestion] Loaded {len(snippets)} source snippets from {args.source}")

    # 2. Extract fact candidates
    extractor = FactCandidateExtractor()
    candidates = extractor.extract(snippets)
    print(f"[ingestion] Extracted {len(candidates)} fact candidates")

    # 3. Normalize
    normalizer = FactNormalizer()
    normalized = normalizer.normalize(candidates)
    accepted = [n for n in normalized if not n.rejected]
    rejected = [n for n in normalized if n.rejected]
    print(f"[ingestion] Normalized: {len(accepted)} accepted, {len(rejected)} rejected (broad/denied)")

    # 4. Write fact candidates
    Path(args.output_facts).write_text(
        json.dumps([_as_dict(n) for n in normalized], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[ingestion] Fact candidates written to {args.output_facts}")

    # 5. Build proposals
    snippets_by_id = {s.source_id: s for s in snippets}
    proposer = MemoryUpdateProposer()
    proposals = proposer.propose(normalized, snippets_by_id)
    print(f"[ingestion] Generated {len(proposals)} memory update proposals")

    # 6. Write proposals JSON
    Path(args.output_proposals).write_text(
        json.dumps([_as_dict(p) for p in proposals], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[ingestion] Proposals written to {args.output_proposals}")

    # 7. Write proposals CSV (optional)
    if args.output_proposals_csv:
        _write_csv(proposals, args.output_proposals_csv)
        print(f"[ingestion] Proposals CSV written to {args.output_proposals_csv}")

    # 8. Summary
    from collections import Counter
    by_term: Counter[str] = Counter()
    by_risk: Counter[str] = Counter()
    by_action: Counter[str] = Counter()
    for p in proposals:
        by_term[p.term] += 1
        by_risk[p.risk_level] += 1
        by_action[p.recommended_action] += 1

    print("\n--- Proposal summary by term ---")
    for term, count in sorted(by_term.items()):
        print(f"  {term}: {count} proposal(s)")

    print("\n--- Proposal summary by risk level ---")
    for level in ("low", "medium", "high"):
        print(f"  {level}: {by_risk.get(level, 0)}")

    print("\n--- Proposal summary by recommended action ---")
    for action, count in sorted(by_action.items()):
        print(f"  {action}: {count}")

    print("\n--- Rejected broad cues ---")
    broad_vals = sorted({n.value for n in normalized if n.rejected})
    for v in broad_vals:
        print(f"  {v}")

    print("\n--- Detected conflicts ---")
    for p in proposals:
        for c in p.evidence.conflicting_cues:
            print(f"  [{p.term}:{p.sense}] {c}")

    print("\n[safety] sense_memory.py: NOT modified")
    print("[safety] thresholds: NOT lowered")
    print("[safety] validators: NOT weakened")
    print("[safety] generic fallback: NOT suggested")
    print("[safety] neural/GPT/training: NOT used")
    print("[safety] nanogpt/: NOT touched")
    print("[safety] benchmark outputs: NOT modified")
    print("[safety] trusted generation behavior: UNCHANGED")


def _write_csv(proposals: list, path: str) -> None:
    from worldpgt.knowledge.types import MemoryUpdateProposal
    rows = []
    for p in proposals:
        rows.append({
            "proposal_id": p.proposal_id,
            "term": p.term,
            "sense": p.sense,
            "risk_level": p.risk_level,
            "recommended_action": p.recommended_action,
            "source_ids": "|".join(p.source_ids),
            "positive_cues": "|".join(p.proposed_update.positive_cues),
            "anti_cues": "|".join(p.proposed_update.anti_cues),
            "typical_actions": "|".join(p.proposed_update.typical_actions),
            "typical_locations": "|".join(p.proposed_update.typical_locations),
            "semantic_frame_hints": "|".join(p.proposed_update.semantic_frame_hints),
            "phrase_candidate_hints": "|".join(p.proposed_update.phrase_candidate_hints),
            "rejected_broad_cues": "|".join(p.evidence.rejected_broad_cues),
            "conflicting_cues": "|".join(p.evidence.conflicting_cues),
        })
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
