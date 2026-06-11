"""
Export relation-label proposals to a CSV for manual review.

Usage:
    python examples/relation_proposal_audit_export.py
    python examples/relation_proposal_audit_export.py --limit 200
    python examples/relation_proposal_audit_export.py \\
        --input data/conceptnet_sample.csv \\
        --output data/relation_proposal_audit.csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.datasets import load_relations_csv, build_world_from_relations
from core.relation_proposal import RelationProposalEngine
from core.relation_trust import DEFAULT_RELATION_TRUST

_HERE = os.path.dirname(__file__)
_DEFAULT_INPUT = os.path.normpath(os.path.join(_HERE, "..", "data", "conceptnet_sample.csv"))
_DEFAULT_OUTPUT = os.path.normpath(os.path.join(_HERE, "..", "data", "relation_proposal_audit.csv"))

COLUMNS = [
    "source",
    "target",
    "proposed_relation",
    "path_length",
    "confidence",
    "reason",
    "evidence",
    "manual_label",
    "notes",
]


def build_audit_rows(
    input_path: str,
    limit: int = 100,
    min_count: int = 3,
    min_confidence: float = 0.4,
    min_rule_total: int = 10,
    rule_alpha: float = 1.0,
    rule_beta: float = 5.0,
    max_intermediate_degree: int | None = None,
    max_intermediate_relation_fanout: int | None = None,
    use_relation_trust: bool = False,
    use_node_quality: bool = False,
    min_node_quality: float = 0.3,
    include_disabled_relations: bool = False,
) -> list[dict]:
    rows = load_relations_csv(input_path)
    w = build_world_from_relations(rows)
    trust_table = DEFAULT_RELATION_TRUST if use_relation_trust else None
    proposals = RelationProposalEngine(w.get_relations()).propose_relations(
        min_count=min_count,
        min_confidence=min_confidence,
        min_rule_total=min_rule_total,
        rule_alpha=rule_alpha,
        rule_beta=rule_beta,
        max_intermediate_degree=max_intermediate_degree,
        max_intermediate_relation_fanout=max_intermediate_relation_fanout,
        relation_trust=trust_table,
        use_node_quality=use_node_quality,
        min_node_quality=min_node_quality,
        include_disabled_relations=include_disabled_relations,
    )
    proposals = proposals[:limit]
    return [
        {
            "source": p.source,
            "target": p.target,
            "proposed_relation": p.proposed_relation,
            "path_length": str(_path_length(p.evidence)),
            "confidence": f"{p.confidence:.6f}",
            "reason": p.reason,
            "evidence": "|".join(p.evidence),
            "manual_label": "",
            "notes": "",
        }
        for p in proposals
    ]


def _path_length(evidence: list[str]) -> int:
    return 2 if evidence else 1


def write_audit_csv(rows: list[dict], output_path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export learned relation-label proposals for manual audit."
    )
    ap.add_argument("--input", default=_DEFAULT_INPUT,
                    help="Path to the relations CSV (default: data/conceptnet_sample.csv)")
    ap.add_argument("--output", default=_DEFAULT_OUTPUT,
                    help="Path for the audit output CSV")
    ap.add_argument("--limit", type=int, default=100,
                    help="Maximum proposals to export (default: 100)")
    ap.add_argument("--min-count", type=int, default=3,
                    help="Minimum learned rule count (default: 3)")
    ap.add_argument("--min-confidence", type=float, default=0.4,
                    help="Minimum final proposal confidence (default: 0.4)")
    ap.add_argument("--min-rule-total", type=int, default=10,
                    dest="min_rule_total",
                    help="Minimum total closure examples for a learned rule (default: 10)")
    ap.add_argument("--rule-alpha", type=float, default=1.0,
                    dest="rule_alpha",
                    help="Additive numerator smoothing for learned rule confidence (default: 1.0)")
    ap.add_argument("--rule-beta", type=float, default=5.0,
                    dest="rule_beta",
                    help="Additive denominator smoothing for learned rule confidence (default: 5.0)")
    ap.add_argument("--max-intermediate-degree", type=int, default=None,
                    dest="max_intermediate_degree",
                    help="Skip chains through nodes with total degree above this threshold")
    ap.add_argument("--max-intermediate-relation-fanout", type=int, default=None,
                    dest="max_intermediate_relation_fanout",
                    help="Skip B when it has more than this many outgoing r2 edges")
    ap.add_argument("--use-relation-trust", action="store_true", default=False,
                    dest="use_relation_trust",
                    help="Scale confidence by relation trust priors")
    ap.add_argument("--use-node-quality", action="store_true", default=False,
                    dest="use_node_quality",
                    help="Filter and penalise low-quality nodes")
    ap.add_argument("--min-node-quality", type=float, default=0.3,
                    dest="min_node_quality",
                    help="Hard threshold for node quality (default 0.3)")
    ap.add_argument("--include-disabled-relations", action="store_true", default=False,
                    dest="include_disabled_relations",
                    help="Include blacklisted/noisy relations such as at_location")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        print("Generate it with:  python scripts/build_conceptnet_sample.py",
              file=sys.stderr)
        sys.exit(1)

    rows = build_audit_rows(
        input_path=args.input,
        limit=args.limit,
        min_count=args.min_count,
        min_confidence=args.min_confidence,
        min_rule_total=args.min_rule_total,
        rule_alpha=args.rule_alpha,
        rule_beta=args.rule_beta,
        max_intermediate_degree=args.max_intermediate_degree,
        max_intermediate_relation_fanout=args.max_intermediate_relation_fanout,
        use_relation_trust=args.use_relation_trust,
        use_node_quality=args.use_node_quality,
        min_node_quality=args.min_node_quality,
        include_disabled_relations=args.include_disabled_relations,
    )
    n = write_audit_csv(rows, args.output)
    print(f"Wrote {n} rows -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
