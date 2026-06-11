"""
Export top pattern predictions to a CSV for manual review.

Automatic precision is misleading on an incomplete ConceptNet sample — many
predictions counted as false positives may be plausible missing edges.  This
script exports predictions with empty manual_label / notes columns so a human
can label them offline.

Usage:
    python examples/pattern_audit_export.py
    python examples/pattern_audit_export.py --limit 200
    python examples/pattern_audit_export.py --mode mixed
    python examples/pattern_audit_export.py --relation part_of,is_a
    python examples/pattern_audit_export.py \\
        --input  data/conceptnet_sample.csv \\
        --output data/pattern_audit.csv \\
        --limit  100
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.datasets import load_relations_csv, build_world_from_relations
from core.pattern_prediction import PatternBasedPredictor, PatternPrediction
from core.relation_trust import DEFAULT_RELATION_TRUST

_HERE = os.path.dirname(__file__)
_DEFAULT_INPUT  = os.path.normpath(os.path.join(_HERE, "..", "data", "conceptnet_sample.csv"))
_DEFAULT_OUTPUT = os.path.normpath(os.path.join(_HERE, "..", "data", "pattern_audit.csv"))

COLUMNS = [
    "source",
    "relation_type",
    "target",
    "path_length",
    "confidence",
    "reason",
    "evidence",
    "manual_label",
    "notes",
]


def build_audit_rows(
    input_path: str,
    relation_filter: set[str] | None = None,
    limit: int = 100,
    min_count: int = 5,
    min_confidence: float = 0.5,
    mode: str = "transitive",
    max_intermediate_degree: int | None = None,
    hub_penalty: bool = True,
    use_relation_trust: bool = False,
    use_node_quality: bool = False,
    min_node_quality: float = 0.3,
    include_disabled_relations: bool = False,
    use_relation_drift: bool = False,
) -> list[dict]:
    """
    Load the CSV, run the predictor, and return audit rows ready to write.

    Sorting: confidence desc, then relation_type asc, then source asc.
    manual_label and notes are always empty strings.
    """
    rows = load_relations_csv(input_path)
    w = build_world_from_relations(rows)
    predictor = PatternBasedPredictor(w.get_relations())
    trust_table = DEFAULT_RELATION_TRUST if use_relation_trust else None

    if mode == "transitive":
        preds = predictor.predict_from_bigrams(
            min_count=min_count,
            min_confidence=min_confidence,
            max_intermediate_degree=max_intermediate_degree,
            hub_penalty=hub_penalty,
            relation_trust=trust_table,
            use_node_quality=use_node_quality,
            min_node_quality=min_node_quality,
            include_disabled_relations=include_disabled_relations,
            use_relation_drift=use_relation_drift,
        )
    elif mode == "mixed":
        preds = predictor.predict_from_mixed_bigrams(
            min_count=min_count,
            min_confidence=min_confidence,
            max_intermediate_degree=max_intermediate_degree,
            hub_penalty=hub_penalty,
            relation_trust=trust_table,
            use_node_quality=use_node_quality,
            min_node_quality=min_node_quality,
            include_disabled_relations=include_disabled_relations,
        )
    elif mode == "all":
        transitive = predictor.predict_from_bigrams(
            min_count=min_count,
            min_confidence=min_confidence,
            max_intermediate_degree=max_intermediate_degree,
            hub_penalty=hub_penalty,
            relation_trust=trust_table,
            use_node_quality=use_node_quality,
            min_node_quality=min_node_quality,
            include_disabled_relations=include_disabled_relations,
            use_relation_drift=use_relation_drift,
        )
        mixed = predictor.predict_from_mixed_bigrams(
            min_count=min_count,
            min_confidence=min_confidence,
            max_intermediate_degree=max_intermediate_degree,
            hub_penalty=hub_penalty,
            relation_trust=trust_table,
            use_node_quality=use_node_quality,
            min_node_quality=min_node_quality,
            include_disabled_relations=include_disabled_relations,
        )
        preds = _dedupe_predictions(transitive + mixed)
    else:
        raise ValueError("mode must be one of: transitive, mixed, all")

    if relation_filter:
        preds = [p for p in preds if p.relation_type in relation_filter]

    preds.sort(key=lambda p: (-p.confidence, p.relation_type, p.source, p.target))
    preds = preds[:limit]

    return [
        {
            "source":        p.source,
            "relation_type": p.relation_type,
            "target":        p.target,
            "path_length":   str(_path_length(p.evidence)),
            "confidence":    f"{p.confidence:.6f}",
            "reason":        p.reason,
            "evidence":      "|".join(p.evidence),
            "manual_label":  "",
            "notes":         "",
        }
        for p in preds
    ]


def _dedupe_predictions(preds: list[PatternPrediction]) -> list[PatternPrediction]:
    """Keep the highest-confidence prediction for each output triple."""
    best: dict[tuple[str, str, str], PatternPrediction] = {}
    for p in preds:
        key = (p.source, p.relation_type, p.target)
        if key not in best or p.confidence > best[key].confidence:
            best[key] = p
    return list(best.values())


def _path_length(evidence: list[str]) -> int:
    return 2 if evidence else 1


def write_audit_csv(rows: list[dict], output_path: str) -> int:
    """Write audit rows to *output_path*.  Returns number of data rows written."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export pattern predictions for manual audit."
    )
    ap.add_argument("--input",    default=_DEFAULT_INPUT,
                    help="Path to the relations CSV (default: data/conceptnet_sample.csv)")
    ap.add_argument("--output",   default=_DEFAULT_OUTPUT,
                    help="Path for the audit output CSV (default: data/pattern_audit.csv)")
    ap.add_argument("--limit",    type=int, default=100,
                    help="Maximum predictions to export (default: 100)")
    ap.add_argument("--min-confidence", type=float, default=0.5,
                    dest="min_confidence",
                    help="Minimum final prediction confidence (default: 0.5)")
    ap.add_argument("--mode", choices=("transitive", "mixed", "all"),
                    default="transitive",
                    help="Prediction mode: transitive, mixed, or all (default: transitive)")
    ap.add_argument("--relation", default=None,
                    help="Comma-separated relation types to include, e.g. part_of,made_of")
    ap.add_argument("--max-intermediate-degree", type=int, default=None,
                    dest="max_intermediate_degree",
                    help="Skip chains through nodes with total degree above this threshold")
    ap.add_argument("--no-hub-penalty", action="store_true", default=False,
                    dest="no_hub_penalty",
                    help="Disable hub-degree confidence penalty (restores pre-v1.2 behaviour)")
    ap.add_argument("--use-relation-trust", action="store_true", default=False,
                    dest="use_relation_trust",
                    help="Scale confidence by human-audit relation trust priors")
    ap.add_argument("--use-node-quality", action="store_true", default=False,
                    dest="use_node_quality",
                    help="Filter and penalise low-quality nodes (profanity, noise, etc.)")
    ap.add_argument("--min-node-quality", type=float, default=0.3,
                    dest="min_node_quality",
                    help="Hard threshold: skip chains with any node quality below this (default 0.3)")
    ap.add_argument("--include-disabled-relations", action="store_true", default=False,
                    dest="include_disabled_relations",
                    help="Include blacklisted/noisy relations such as at_location")
    ap.add_argument("--use-relation-drift", action="store_true", default=False,
                    dest="use_relation_drift",
                    help="Apply semantic-level drift penalties to made_of transitive predictions")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        print("Generate it with:  python scripts/build_conceptnet_sample.py",
              file=sys.stderr)
        sys.exit(1)

    relation_filter = None
    if args.relation:
        relation_filter = {r.strip() for r in args.relation.split(",") if r.strip()}

    rows = build_audit_rows(
        input_path=args.input,
        relation_filter=relation_filter,
        limit=args.limit,
        min_confidence=args.min_confidence,
        mode=args.mode,
        max_intermediate_degree=args.max_intermediate_degree,
        hub_penalty=not args.no_hub_penalty,
        use_relation_trust=args.use_relation_trust,
        use_node_quality=args.use_node_quality,
        min_node_quality=args.min_node_quality,
        include_disabled_relations=args.include_disabled_relations,
        use_relation_drift=args.use_relation_drift,
    )
    n = write_audit_csv(rows, args.output)
    print(f"Wrote {n} rows → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
