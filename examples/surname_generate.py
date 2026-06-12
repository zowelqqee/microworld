"""
Generate personal names (given names, surnames, or mixed) from an explicit
character-transition graph and export them for manual audit.

The output CSV has empty ``manual_label`` / ``notes`` columns for offline
review (good | bad | unclear).  The input dataset may be a plain one-name-per-line
text file containing any mix of personal name types.

Usage:
    python examples/surname_generate.py
    python examples/surname_generate.py --input data/names.txt --count 100 --order 2
    python examples/surname_generate.py --trust-profile data/surname_trust_profile.json
    python examples/surname_generate.py --avoid-duplicates true --seed 42
"""
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.surname_generator import (
    SurnameTransitionGraph,
    load_surnames,
    load_trust_profile,
)
from core.surname_policy import explain_surname_quality, surname_quality_score

_HERE = os.path.dirname(__file__)
_DEFAULT_INPUT = os.path.normpath(os.path.join(_HERE, "..", "data", "surnames.txt"))
_DEFAULT_OUTPUT = os.path.normpath(
    os.path.join(_HERE, "..", "data", "generated_surnames.csv")
)

COLUMNS = [
    "name",
    "quality_score",
    "quality_reasons",
    "duplicate",
    "source",
    "manual_label",
    "notes",
]


def generate_names(
    graph: SurnameTransitionGraph,
    count: int,
    *,
    rng: random.Random,
    source_set: set[str],
    avoid_duplicates: bool = False,
    max_length: int = 20,
    trust_profile=None,
    max_attempts: int | None = None,
) -> list[str]:
    """Generate up to *count* distinct names via weighted graph walk.

    Names are de-duplicated within the generated batch.  When
    ``avoid_duplicates`` is set, names already present in ``source_set`` are
    skipped too.  Attempts are capped to avoid spinning on a small graph.
    """
    if max_attempts is None:
        max_attempts = count * 50 + 100
    names: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(names) < count and attempts < max_attempts:
        attempts += 1
        name = graph.generate(
            max_length=max_length, rng=rng, trust_profile=trust_profile
        )
        if not name or name in seen:
            continue
        if avoid_duplicates and name in source_set:
            continue
        seen.add(name)
        names.append(name)
    return names


def build_rows(names: list[str], source_set: set[str], source_label: str) -> list[dict]:
    """Build audit rows for *names*, scoring quality and flagging duplicates."""
    rows: list[dict] = []
    for name in names:
        score = surname_quality_score(name)
        reasons = explain_surname_quality(name)
        rows.append(
            {
                "name": name,
                "quality_score": f"{score:.4f}",
                "quality_reasons": "|".join(reasons),
                "duplicate": "true" if name in source_set else "false",
                "source": source_label,
                "manual_label": "",
                "notes": "",
            }
        )
    return rows


def write_csv(rows: list[dict], output_path: str) -> int:
    """Write audit rows to *output_path*; returns number of data rows written."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def run(
    input_path: str,
    output_path: str,
    *,
    count: int = 100,
    order: int = 2,
    max_length: int = 20,
    seed: int = 42,
    trust_profile_path: str | None = None,
    avoid_duplicates: bool = False,
    column: int | None = None,
) -> dict:
    """Load, build the graph, generate, and write the audit CSV.

    Returns a small summary dict (names generated, output path, label).
    """
    surnames = load_surnames(input_path, column=column)
    source_set = set(surnames)
    graph = SurnameTransitionGraph(order=order).build(surnames)

    trust_profile = None
    source_label = "baseline"
    if trust_profile_path:
        trust_profile = load_trust_profile(trust_profile_path)
        source_label = "learned"

    rng = random.Random(seed)
    names = generate_names(
        graph,
        count,
        rng=rng,
        source_set=source_set,
        avoid_duplicates=avoid_duplicates,
        max_length=max_length,
        trust_profile=trust_profile,
    )
    rows = build_rows(names, source_set, source_label)
    n = write_csv(rows, output_path)
    return {"generated": n, "output": output_path, "source": source_label}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate personal names (given names, surnames, or mixed) "
            "from a character-transition graph."
        )
    )
    ap.add_argument("--input", default=_DEFAULT_INPUT,
                    help="Path to the name list — one name per line, given names/surnames/mixed (default: data/surnames.txt)")
    ap.add_argument("--output", default=_DEFAULT_OUTPUT,
                    help="Path for the generated audit CSV")
    ap.add_argument("--count", type=int, default=100,
                    help="Number of names to generate (default: 100)")
    ap.add_argument("--order", type=int, default=2,
                    help="N-gram context order (default: 2)")
    ap.add_argument("--max-length", type=int, default=20, dest="max_length",
                    help="Maximum generated name length (default: 20)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducible generation (default: 42)")
    ap.add_argument("--column", type=int, default=None,
                    help="0-based CSV column to read (default: first non-empty)")
    ap.add_argument("--trust-profile", default=None, dest="trust_profile",
                    help="Optional learned trust profile JSON to bias generation")
    ap.add_argument("--avoid-duplicates", default="false", dest="avoid_duplicates",
                    help="Skip names already present in the source list (true/false)")
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
        f"Generated {summary['generated']} names "
        f"({summary['source']}) → {summary['output']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
