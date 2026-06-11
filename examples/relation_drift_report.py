"""
Report likely relation drift in transitive reasoning chains.

Usage:
    python examples/relation_drift_report.py
    python examples/relation_drift_report.py --audit data/audit_made_of.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.datasets import load_relations_csv, build_world_from_relations
from core.relation_drift import RelationDriftEngine, read_audit_rows
from core.reasoning_relations import DEFAULT_DISABLED_RELATIONS, DEFAULT_REASONING_RELATIONS

_HERE = os.path.dirname(__file__)
_DEFAULT_INPUT = os.path.normpath(os.path.join(_HERE, "..", "data", "conceptnet_sample.csv"))
_DEFAULT_AUDIT = os.path.normpath(os.path.join(_HERE, "..", "data", "audit_made_of.csv"))


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:5.1f}%"


def format_report(reports) -> str:
    lines: list[str] = []
    lines.append("Relation drift report")
    lines.append("")
    header = (
        f"{'relation':16s} {'support':>8} {'drift':>8} "
        f"{'reviewed':>8} {'wrong':>6} {'accuracy':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for report in reports:
        lines.append(
            f"{report.relation_type:16s} {report.support:>8d} "
            f"{report.drift_support:>8d} {report.reviewed:>8d} "
            f"{report.wrong:>6d} {_pct(report.audit_accuracy):>8}"
        )
        for example in report.examples[:5]:
            path = " -> ".join(example.path)
            categories = " -> ".join(example.categories)
            label = f", audit={example.audit_label}" if example.audit_label else ""
            lines.append(
                f"  depth={example.path_length} drift={example.drift}{label}: {path}"
            )
            lines.append(f"    categories: {categories}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Discover relation drift in 2/3-hop chains.")
    ap.add_argument("--input", default=_DEFAULT_INPUT,
                    help="Relations CSV (default: data/conceptnet_sample.csv)")
    ap.add_argument("--audit", default=_DEFAULT_AUDIT,
                    help="Optional labelled audit CSV (default: data/audit_made_of.csv)")
    ap.add_argument("--max-depth", type=int, default=3,
                    help="Maximum same-relation path depth to inspect (default: 3)")
    ap.add_argument("--include-disabled-relations", action="store_true", default=False,
                    dest="include_disabled_relations",
                    help="Include blacklisted/noisy relations such as at_location")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    rows = load_relations_csv(args.input)
    world = build_world_from_relations(rows)
    audit_rows = read_audit_rows(args.audit) if args.audit and os.path.exists(args.audit) else []

    print(f"Loaded {len(world.get_relations())} relations")
    print(f"Enabled core relations : {', '.join(sorted(DEFAULT_REASONING_RELATIONS))}")
    print(f"Disabled by default    : {', '.join(sorted(DEFAULT_DISABLED_RELATIONS))}")
    if audit_rows:
        print(f"Joined audit rows      : {len(audit_rows)}")
    print()

    reports = RelationDriftEngine(world.get_relations()).build_report(
        max_depth=args.max_depth,
        include_disabled_relations=args.include_disabled_relations,
        audit_rows=audit_rows,
    )
    print(format_report(reports))


if __name__ == "__main__":
    main()
