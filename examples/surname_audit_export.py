"""
Export generated names for manual audit.

A thin wrapper over examples/surname_generate.py: it produces exactly the same
CSV (with empty ``manual_label`` / ``notes`` columns), framed as the audit step
of the pipeline.  Reviewers fill ``manual_label`` with one of:

    good | bad | unclear

The input dataset may contain given names, surnames, or any mix of personal
name types — one name per line.

Usage:
    python examples/surname_audit_export.py --input data/names.txt --count 100
    python examples/surname_audit_export.py --output data/surname_audit.csv --order 2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from examples.surname_generate import _parse_bool, run

_HERE = os.path.dirname(__file__)
_DEFAULT_INPUT = os.path.normpath(os.path.join(_HERE, "..", "data", "surnames.txt"))
_DEFAULT_OUTPUT = os.path.normpath(
    os.path.join(_HERE, "..", "data", "surname_audit.csv")
)

VALID_AUDIT_LABELS = ("good", "bad", "unclear")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Export generated personal names for manual audit (good/bad/unclear). "
            "Input may be given names, surnames, or any mixed personal-name list."
        )
    )
    ap.add_argument("--input", default=_DEFAULT_INPUT,
                    help="Path to the name list — one name per line (default: data/surnames.txt)")
    ap.add_argument("--output", default=_DEFAULT_OUTPUT,
                    help="Path for the audit CSV (default: data/surname_audit.csv)")
    ap.add_argument("--count", type=int, default=100,
                    help="Number of names to export (default: 100)")
    ap.add_argument("--order", type=int, default=2,
                    help="N-gram context order (default: 2)")
    ap.add_argument("--max-length", type=int, default=20, dest="max_length")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--column", type=int, default=None)
    ap.add_argument("--trust-profile", default=None, dest="trust_profile")
    ap.add_argument("--avoid-duplicates", default="false", dest="avoid_duplicates")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    summary = run(
        args.input,
        args.output,
        count=args.count,
        order=args.order,
        max_length=args.max_length,
        seed=args.seed,
        trust_profile_path=args.trust_profile,
        avoid_duplicates=_parse_bool(args.avoid_duplicates),
        column=args.column,
    )
    print(
        f"Exported {summary['generated']} names for audit → {summary['output']}\n"
        f"Label the manual_label column with: {', '.join(VALID_AUDIT_LABELS)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
