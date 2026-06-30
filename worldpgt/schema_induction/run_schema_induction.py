"""Schema induction pipeline + CLI.

Usage:

    python3 -m worldpgt.schema_induction.run_schema_induction \
        --input-docs path/to/docs.jsonl \
        --output-dir worldpgt/experiments/schema_induction_v1/demo_run \
        --min-evidence 2 \
        --min-sources 1

Input docs JSONL lines: {"doc_id":"d1","title":"...","url":"...","text":"..."}
"""

from __future__ import annotations

import argparse

from worldpgt.schema_induction import schema_store
from worldpgt.schema_induction.entity_discovery import discover_entities
from worldpgt.schema_induction.frame_builder import build_frames
from worldpgt.schema_induction.local_type_inducer import induce_local_types
from worldpgt.schema_induction.promotion_gates import (
    GateConfig,
    apply_decisions,
    evaluate_local_type,
    evaluate_relation_family,
)
from worldpgt.schema_induction.raw_claim_extractor import extract_claims
from worldpgt.schema_induction.relation_family_builder import build_relation_families
from worldpgt.schema_induction.types import (
    DocumentRecord,
    SchemaInductionResult,
)


def _coerce_docs(raw_docs: list[dict]) -> list[DocumentRecord]:
    docs: list[DocumentRecord] = []
    for i, d in enumerate(raw_docs):
        docs.append(
            DocumentRecord(
                doc_id=str(d.get("doc_id") or f"d{i}"),
                title=str(d.get("title") or ""),
                url=str(d.get("url") or ""),
                text=str(d.get("text") or ""),
            )
        )
    return docs


def run_induction(
    raw_docs: list[dict],
    config: GateConfig | None = None,
) -> SchemaInductionResult:
    """Run the full deterministic schema induction pipeline."""

    cfg = config or GateConfig()
    docs = _coerce_docs(raw_docs)

    sentences, claims = extract_claims(docs)
    entities = discover_entities(claims, sentences)
    frames = build_frames(claims, entities)

    claim_doc_map = {c.claim_id: c.source_doc_id for c in claims}
    claim_subject_map = {c.claim_id: c.subject for c in claims}

    families = build_relation_families(frames, claim_doc_map)
    local_types = induce_local_types(entities, frames)

    # Run promotion gates.
    decisions = []
    family_decisions = {}
    for fam in families:
        decision = evaluate_relation_family(fam, claim_subject_map, cfg)
        decisions.append(decision)
        family_decisions[fam.family_id] = decision
    for lt in local_types:
        decisions.append(evaluate_local_type(lt, cfg))

    families = apply_decisions(families, family_decisions)

    promoted = [f for f in families if f.promotion_status == "promoted"]
    rejected = [d for d in decisions if d.status == "rejected"]
    generated_only = [f for f in families if f.promotion_status == "generated"]

    summary = {
        "documents_read": len(docs),
        "sentences_parsed": len(sentences),
        "entities_discovered": len(entities),
        "raw_claims_extracted": len(claims),
        "frames_built": len(frames),
        "relation_families_generated": len(families),
        "relation_families_promoted": len(promoted),
        "relation_families_generated_not_promoted": len(generated_only),
        "local_types_generated": len(local_types),
        "rejected_count": len(rejected),
        "rejected_reasons": [
            {"target_id": d.target_id, "kind": d.target_kind, "reason": d.reason}
            for d in rejected
        ],
        "gate_config": {
            "min_evidence": cfg.min_evidence,
            "min_sources": cfg.min_sources,
            "min_confidence": cfg.min_confidence,
        },
    }

    return SchemaInductionResult(
        documents=tuple(docs),
        sentences=tuple(sentences),
        entities=tuple(entities),
        claims=tuple(claims),
        frames=tuple(frames),
        families=tuple(families),
        local_types=tuple(local_types),
        decisions=tuple(decisions),
        summary=summary,
    )


def _print_summary(result: SchemaInductionResult) -> None:
    s = result.summary
    print("Schema induction summary")
    print("------------------------")
    print(f"documents read:                 {s['documents_read']}")
    print(f"sentences parsed:               {s['sentences_parsed']}")
    print(f"entities discovered:            {s['entities_discovered']}")
    print(f"raw claims extracted:           {s['raw_claims_extracted']}")
    print(f"frames built:                   {s['frames_built']}")
    print(f"relation families generated:    {s['relation_families_generated']}")
    print(f"relation families promoted:     {s['relation_families_promoted']}")
    print(f"local types generated:          {s['local_types_generated']}")
    print(f"rejected:                       {s['rejected_count']}")
    for r in s["rejected_reasons"]:
        print(f"  - {r['kind']} {r['target_id']}: {r['reason']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run schema induction over a corpus.")
    parser.add_argument("--input-docs", required=True, help="Path to docs JSONL.")
    parser.add_argument("--output-dir", required=True, help="Artifact output dir.")
    parser.add_argument("--min-evidence", type=int, default=2)
    parser.add_argument("--min-sources", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    args = parser.parse_args(argv)

    cfg = GateConfig(
        min_evidence=args.min_evidence,
        min_sources=args.min_sources,
        min_confidence=args.min_confidence,
    )
    raw_docs = schema_store.read_docs_jsonl(args.input_docs)
    result = run_induction(raw_docs, cfg)
    paths = schema_store.write_result(result, args.output_dir)

    _print_summary(result)
    print()
    print(f"Artifacts written to: {args.output_dir}")
    for name in (
        schema_store.RELATION_FAMILIES_GENERATED,
        schema_store.RELATION_FAMILIES_PROMOTED,
        schema_store.LOCAL_TYPES_GENERATED,
        schema_store.SUMMARY,
    ):
        print(f"  - {paths[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
