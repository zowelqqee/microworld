"""Run the unchanged precision gates on bounded OpenAlex proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldpgt.knowledge_pump.openalex_topic_reference_gate import validate_openalex_topic_reference_proposals


def _final_report(seed: dict, gate: dict) -> str:
    multi = gate["entities_with_two_or_more_predicate_groups_after_gate"]
    composition = gate["predicate_group_compositions"] or {"none": 0}
    composition_lines = "\n".join(f"- `{name}`: {count}" for name, count in composition.items())
    entity_lines = "\n".join(
        f"- `{row['title']}` (`{row['canonical_openalex_id']}`): `{' + '.join(row['predicate_groups'])}`"
        for row in gate["accepted_multi_predicate_entities"]
    ) or "- None"
    if multi >= 5:
        recommendation = (
            "This lane is structurally suitable for a targeted generalization held-out stratum: "
            "it has at least five independently identified entities with a non-Crossref predicate pair. "
            "It remains proposal-only; promotion and held-out construction require a separate decision."
        )
    elif multi:
        recommendation = (
            "The lane demonstrates a structurally distinct predicate pair but is too small for a meaningful "
            "generalization held-out stratum. Keep it proposal-only and seek more OpenAlex work records or a "
            "citation-graph source before evaluating."
        )
    else:
        recommendation = (
            "This bounded lane did not yield a precision-accepted multi-predicate entity. It is not suitable for "
            "a generalization held-out test. A next candidate is a bounded OpenCitations or Semantic Scholar "
            "citation-graph lane, subject to the same source and precision review."
        )
    return f"""# OpenAlex topic/citation lane — final report

## Scope and boundary

This run used the official OpenAlex Works API only. It began from {seed['starting_unique_quarantine_relations']} unique quarantined OpenAlex relations and {seed['starting_unique_seed_dois']} DOI seeds, then fetched each seed work and at most one named cited work per seed. The run made {seed['fetch']['total_queries']} API requests. All outputs remain proposal-only: accepted memory and the serving overlay were not modified.

## Gate result

| Stage | Count |
|---|---:|
| Raw candidates | {gate['input_relation_count']} |
| Passed source gate | {gate['passed_source_gate']} |
| Passed v1 + v2 precision gates | {gate['passed_precision_gate']} |
| Entities with >=1 accepted relation | {gate['entities_with_any_relation_after_gate']} |
| Entities with >=2 accepted predicate groups | {multi} |

Rejection/quarantine reasons: `{json.dumps(gate['rejected_or_quarantined_by_reason'], ensure_ascii=False, sort_keys=True)}`.

## Predicate-type composition

The precision-accepted multi-predicate composition is:

{composition_lines}

Accepted entities:

{entity_lines}

This is structurally different from Crossref's `created_by + published_by`: it combines an OpenAlex topic classification (`has_topic`) with a citation-graph relation (`references_work`). It is therefore not a second source for the same author/publisher fact pattern.

## Overlap checks

- Pre-Crossref main target-subject overlap: {seed['main_subject_overlap_count']} / {seed['main_target_subject_count']}.
- Main relation-ID overlap: {seed['main_edge_overlap_count']}.
- Promoted Crossref DOI entity overlap: {seed['crossref_promoted_entity_overlap_count']}.

## Recommendation

{recommendation}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate OpenAlex topic/citation proposal relations")
    parser.add_argument("--input", default="artifacts/open_book_qa/openalex_seed_v1/proposal_relations.json")
    parser.add_argument("--seed-summary", default="artifacts/open_book_qa/openalex_seed_v1/summary.json")
    parser.add_argument("--output-dir", default="artifacts/open_book_qa/openalex_seed_v1/precision_gate")
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    seed = json.loads(Path(args.seed_summary).read_text(encoding="utf-8"))
    report = validate_openalex_topic_reference_proposals(rows)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for key, filename in (
        ("accepted_proposal_overlay", "accepted_proposal_overlay.json"),
        ("rejected", "rejected.json"),
        ("quarantine", "quarantine.json"),
    ):
        root.joinpath(filename).write_text(
            json.dumps(report.pop(key), ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    root.joinpath("summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.seed_summary).parent.joinpath("final_report.md").write_text(_final_report(seed, report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
