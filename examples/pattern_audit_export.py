"""
Export top pattern predictions to a CSV for manual review.

Automatic precision is misleading on an incomplete ConceptNet sample — many
predictions counted as false positives may be plausible missing edges.  This
script exports predictions with empty manual_label / notes columns so a human
can label them offline.

Usage:
    python examples/pattern_audit_export.py
    python examples/pattern_audit_export.py --limit 200
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
from core.pattern_prediction import PatternBasedPredictor
from core.relation_trust import DEFAULT_RELATION_TRUST

_HERE = os.path.dirname(__file__)
_DEFAULT_INPUT  = os.path.normpath(os.path.join(_HERE, "..", "data", "conceptnet_sample.csv"))
_DEFAULT_OUTPUT = os.path.normpath(os.path.join(_HERE, "..", "data", "pattern_audit.csv"))

COLUMNS = [
    "source",
    "relation_type",
    "target",
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
    max_intermediate_degree: int | None = None,
    hub_penalty: bool = True,
    use_relation_trust: bool = False,
) -> list[dict]:
    """
    Load the CSV, run the predictor, and return audit rows ready to write.

    Sorting: confidence desc, then relation_type asc, then source asc.
    manual_label and notes are always empty strings.
    """
    rows = load_relations_csv(input_path)
    w = build_world_from_relations(rows)
    preds = PatternBasedPredictor(w.get_relations()).predict_from_bigrams(
        min_count=min_count,
        max_intermediate_degree=max_intermediate_degree,
        hub_penalty=hub_penalty,
        relation_trust=DEFAULT_RELATION_TRUST if use_relation_trust else None,
    )

    if relation_filter:
        preds = [p for p in preds if p.relation_type in relation_filter]

    preds.sort(key=lambda p: (-p.confidence, p.relation_type, p.source, p.target))
    preds = preds[:limit]

    return [
        {
            "source":        p.source,
            "relation_type": p.relation_type,
            "target":        p.target,
            "confidence":    f"{p.confidence:.6f}",
            "reason":        p.reason,
            "evidence":      "|".join(p.evidence),
            "manual_label":  "",
            "notes":         "",
        }
        for p in preds
    ]


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
        max_intermediate_degree=args.max_intermediate_degree,
        hub_penalty=not args.no_hub_penalty,
        use_relation_trust=args.use_relation_trust,
    )
    n = write_audit_csv(rows, args.output)
    print(f"Wrote {n} rows → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
