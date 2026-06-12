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
    python examples/surname_generate.py --soft-max-length 10 --length-end-bias 1.5
    python examples/surname_generate.py --min-adjusted-quality 0.7
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
from core.name_pattern_mining import extract_name_patterns
from core.surname_policy import explain_surname_quality, surname_quality_score
from examples.surname_trust_learn import (
    DIAGNOSTIC_SHAPE_REASONS,
    canonical_quality_reason,
)

_HERE = os.path.dirname(__file__)
_DEFAULT_INPUT = os.path.normpath(os.path.join(_HERE, "..", "data", "surnames.txt"))
_DEFAULT_OUTPUT = os.path.normpath(
    os.path.join(_HERE, "..", "data", "generated_surnames.csv")
)

COLUMNS = [
    "name",
    "quality_score",
    "adjusted_quality_score",
    "quality_reasons",
    "pattern_reasons",
    "duplicate",
    "source",
    "manual_label",
    "notes",
]


def _get_shape_trust(trust_profile) -> dict[str, float]:
    """Extract shape_trust from a dict or object, or {}."""
    if trust_profile is None:
        return {}
    if isinstance(trust_profile, dict):
        return trust_profile.get("shape_trust", {}) or {}
    return getattr(trust_profile, "shape_trust", {}) or {}


def _get_pattern_trust(trust_profile) -> dict[str, float]:
    """Extract pattern_trust from a dict or object, or {}."""
    if trust_profile is None:
        return {}
    if isinstance(trust_profile, dict):
        return trust_profile.get("pattern_trust", {}) or {}
    return getattr(trust_profile, "pattern_trust", {}) or {}


def pattern_reasons(name: str, trust_profile=None) -> list[str]:
    """Return bad-pattern reasons matched by *name* under the trust profile."""
    trust = _get_pattern_trust(trust_profile)
    if not trust:
        return []
    patterns = extract_name_patterns(name)
    return [f"bad_pattern:{pattern}" for pattern in sorted(patterns & set(trust))]


def adjusted_quality_score(
    quality_score: float,
    quality_reasons: list[str],
    trust_profile=None,
    *,
    name: str | None = None,
) -> float:
    """Apply learned shape trust to the explicit quality score."""
    score = quality_score
    shape_trust = _get_shape_trust(trust_profile)
    for reason in quality_reasons:
        key = canonical_quality_reason(reason)
        if key in DIAGNOSTIC_SHAPE_REASONS:
            score *= float(shape_trust.get(key, 1.0))
    if name:
        pattern_trust = _get_pattern_trust(trust_profile)
        for pattern in extract_name_patterns(name):
            score *= float(pattern_trust.get(pattern, 1.0))
    return max(0.0, min(1.0, score))


def generate_names(
    graph: SurnameTransitionGraph,
    count: int,
    *,
    rng: random.Random,
    source_set: set[str],
    avoid_duplicates: bool = False,
    max_length: int = 20,
    min_length: int = 3,
    soft_max_length: int = 12,
    end_bias: float = 1.0,
    length_end_bias: float = 1.35,
    trust_profile=None,
    max_attempts: int | None = None,
    min_adjusted_quality: float | None = None,
    max_attempts_per_name: int = 50,
) -> list[str]:
    """Generate up to *count* distinct names via weighted graph walk.

    Names are de-duplicated within the generated batch.  When
    ``avoid_duplicates`` is set, names already present in ``source_set`` are
    skipped too.  Attempts are capped to avoid spinning on a small graph.
    """
    if max_attempts is None:
        max_attempts = count * max_attempts_per_name + 100
    names: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(names) < count and attempts < max_attempts:
        attempts += 1
        name = graph.generate(
            max_length=max_length,
            rng=rng,
            trust_profile=trust_profile,
            min_length=min_length,
            soft_max_length=soft_max_length,
            end_bias=end_bias,
            length_end_bias=length_end_bias,
        )
        if not name or name in seen:
            continue
        if avoid_duplicates and name in source_set:
            continue
        if min_adjusted_quality is not None:
            score = surname_quality_score(name)
            reasons = explain_surname_quality(name)
            adjusted = adjusted_quality_score(
                score,
                reasons,
                trust_profile,
                name=name,
            )
            if adjusted < min_adjusted_quality:
                continue
        seen.add(name)
        names.append(name)
    return names


def build_rows(
    names: list[str],
    source_set: set[str],
    source_label: str,
    trust_profile=None,
) -> list[dict]:
    """Build audit rows for *names*, scoring quality and flagging duplicates."""
    rows: list[dict] = []
    for name in names:
        score = surname_quality_score(name)
        reasons = explain_surname_quality(name)
        adjusted = adjusted_quality_score(score, reasons, trust_profile, name=name)
        patterns = pattern_reasons(name, trust_profile)
        rows.append(
            {
                "name": name,
                "quality_score": f"{score:.4f}",
                "adjusted_quality_score": f"{adjusted:.4f}",
                "quality_reasons": "|".join(reasons),
                "pattern_reasons": "|".join(patterns),
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
    min_length: int = 3,
    soft_max_length: int = 12,
    end_bias: float = 1.0,
    length_end_bias: float = 1.35,
    seed: int = 42,
    trust_profile_path: str | None = None,
    avoid_duplicates: bool = False,
    column: int | None = None,
    min_adjusted_quality: float | None = None,
    max_attempts_per_name: int = 50,
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
        min_length=min_length,
        soft_max_length=soft_max_length,
        end_bias=end_bias,
        length_end_bias=length_end_bias,
        trust_profile=trust_profile,
        min_adjusted_quality=min_adjusted_quality,
        max_attempts_per_name=max_attempts_per_name,
    )
    rows = build_rows(names, source_set, source_label, trust_profile)
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
                    help="Hard maximum generated name length (default: 20)")
    ap.add_argument("--min-length", type=int, default=3, dest="min_length",
                    help="Minimum length before END is allowed (default: 3)")
    ap.add_argument("--soft-max-length", type=int, default=12, dest="soft_max_length",
                    help="Length above which END is progressively boosted (default: 12)")
    ap.add_argument("--end-bias", type=float, default=1.0, dest="end_bias",
                    help="Flat END weight multiplier in normal range (default: 1.0)")
    ap.add_argument("--length-end-bias", type=float, default=1.35, dest="length_end_bias",
                    help="Per-step END boost exponent above soft-max-length (default: 1.35)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducible generation (default: 42)")
    ap.add_argument("--column", type=int, default=None,
                    help="0-based CSV column to read (default: first non-empty)")
    ap.add_argument("--trust-profile", default=None, dest="trust_profile",
                    help="Optional learned trust profile JSON to bias generation")
    ap.add_argument("--avoid-duplicates", default="false", dest="avoid_duplicates",
                    help="Skip names already present in the source list (true/false)")
    ap.add_argument("--min-adjusted-quality", type=float, default=None,
                    dest="min_adjusted_quality",
                    help="Optional adjusted-quality threshold for resampling")
    ap.add_argument("--max-attempts-per-name", type=int, default=50,
                    dest="max_attempts_per_name",
                    help="Generation attempts per requested name when filtering (default: 50)")
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
        min_length=args.min_length,
        soft_max_length=args.soft_max_length,
        end_bias=args.end_bias,
        length_end_bias=args.length_end_bias,
        seed=args.seed,
        trust_profile_path=args.trust_profile,
        avoid_duplicates=_parse_bool(args.avoid_duplicates),
        column=args.column,
        min_adjusted_quality=args.min_adjusted_quality,
        max_attempts_per_name=args.max_attempts_per_name,
    )
    print(
        f"Generated {summary['generated']} names "
        f"({summary['source']}) → {summary['output']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
