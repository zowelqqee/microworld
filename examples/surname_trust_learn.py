"""
Compress audit feedback into a compact transition trust profile.

Reads a manually-labelled name/surname audit CSV, reconstructs the character
transitions each name went through (at the generation ``--order``), and nudges
transition and shape trust multipliers:

    good    : multiplier *= 1.04
    unclear : multiplier *= 0.97
    bad     : multiplier *= 0.80

Multipliers start at 1.0 and are bounded to [0.1, 2.0].  The result is a small
JSON profile that examples/surname_generate.py can replay to bias generation —
feedback learning without any weights or backpropagation.

Output schema:
    {
      "order": 2,
      "transition_trust": {"ab->r": 1.05, "xq->z": 0.65},
      "shape_trust": {"long_name": 0.80, "very_short": 0.97},
      "pattern_trust": {"qwe": 0.70, "ynn$": 0.85},
      "pattern_stats": {"qwe": {"good": 0, "bad": 2, "unclear": 0}},
      "stats": {"reviewed": 100, "good": 70, "bad": 25, "unclear": 5}
    }

Usage:
    python examples/surname_trust_learn.py --input data/surname_audit.csv \\
        --order 2 --output data/surname_trust_profile.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.name_pattern_mining import mine_pattern_trust
from core.surname_generator import iter_transitions, transition_key
from core.surname_policy import explain_surname_quality
from examples.surname_audit_summary import normalize_label, read_labeled_rows

GOOD_FACTOR = 1.04
UNCLEAR_FACTOR = 0.97
BAD_FACTOR = 0.80
TRUST_MIN = 0.1
TRUST_MAX = 2.0

GENERIC_POSITIVE_REASONS = frozenset(
    {
        "looks_like_a_plausible_name",
        "reasonable_length",
        "balanced_vowels",
        "common_name_ending",
    }
)

DIAGNOSTIC_SHAPE_REASONS = frozenset(
    {
        "long_name",
        "too_long",
        "extremely_long",
        "too_many_syllable_chunks",
        "doubled_vowel",
        "very_short",
        "weird_consonant_cluster",
        "no_vowels",
        "too_many_vowels",
        "duplicate_source_name",
        "common_word_like",
        "brand_like",
        "too_fragmentary",
        "awkward_short_form",
        "medium_glued_name",
        "poor_readability",
        "weird_q_usage",
        "nickname_like",
        "distorted_known_name",
    }
)


def canonical_quality_reason(raw: str) -> str:
    """Return the stable shape-trust key for a quality reason."""
    return "_".join((raw or "").strip().lower().split())


def parse_quality_reasons(raw: str) -> list[str]:
    """Parse pipe-separated quality reasons into stable shape-trust keys."""
    reasons: list[str] = []
    for part in (raw or "").split("|"):
        reason = canonical_quality_reason(part)
        if reason:
            reasons.append(reason)
    return reasons


def row_quality_reasons(row: dict, name: str) -> list[str]:
    """Return explicit quality reasons, or recompute them for compact audits."""
    raw = (row.get("quality_reasons") or "").strip()
    if raw:
        return parse_quality_reasons(raw)
    return [canonical_quality_reason(reason) for reason in explain_surname_quality(name)]


def _clamp(value: float, min_trust: float, max_trust: float) -> float:
    return max(min_trust, min(max_trust, value))


def _label_multiplier(
    label: str,
    *,
    good_multiplier: float,
    unclear_multiplier: float,
    bad_multiplier: float,
) -> float:
    if label == "good":
        return good_multiplier
    if label == "bad":
        return bad_multiplier
    return unclear_multiplier


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def learn_trust(
    rows: list[dict],
    order: int,
    *,
    good_multiplier: float = GOOD_FACTOR,
    unclear_multiplier: float = UNCLEAR_FACTOR,
    bad_multiplier: float = BAD_FACTOR,
    min_trust: float = TRUST_MIN,
    max_trust: float = TRUST_MAX,
    mine_patterns: bool = True,
    pattern_min_support: int = 2,
    pattern_min_bad_count: int = 2,
    pattern_bad_ratio: float = 0.60,
) -> dict:
    """Build a transition trust profile from labelled audit *rows*.

    Returns the full profile dict (order, transition_trust, shape_trust, stats).
    Unclear rows count in stats and apply a weak negative multiplier by default;
    pass ``unclear_multiplier=1.0`` to restore historical no-op behavior.
    """
    transition_multipliers: dict[str, float] = {}
    shape_multipliers: dict[str, float] = {}
    stats = {"reviewed": 0, "good": 0, "bad": 0, "unclear": 0}

    for row in rows:
        name = (row.get("name") or "").strip()
        if not name:
            continue
        label = normalize_label(row.get("manual_label", ""))
        stats["reviewed"] += 1
        stats[label] += 1

        factor = _label_multiplier(
            label,
            good_multiplier=good_multiplier,
            unclear_multiplier=unclear_multiplier,
            bad_multiplier=bad_multiplier,
        )

        for context, ch in iter_transitions(name, order):
            key = transition_key(context, ch)
            current = transition_multipliers.get(key, 1.0)
            transition_multipliers[key] = _clamp(current * factor, min_trust, max_trust)

        for reason in row_quality_reasons(row, name):
            if reason in GENERIC_POSITIVE_REASONS:
                continue
            if label == "good" and reason in DIAGNOSTIC_SHAPE_REASONS:
                continue
            if label in {"unclear", "bad"} and reason not in DIAGNOSTIC_SHAPE_REASONS:
                continue

            current = shape_multipliers.get(reason, 1.0)
            shape_multipliers[reason] = _clamp(current * factor, min_trust, max_trust)

    pattern_trust: dict[str, float] = {}
    pattern_stats: dict[str, dict[str, int]] = {}
    if mine_patterns:
        pattern_trust, pattern_stats = mine_pattern_trust(
            rows,
            min_support=pattern_min_support,
            min_bad_count=pattern_min_bad_count,
            bad_ratio=pattern_bad_ratio,
        )

    return {
        "order": order,
        "transition_trust": {
            k: transition_multipliers[k] for k in sorted(transition_multipliers)
        },
        "shape_trust": {
            k: shape_multipliers[k] for k in sorted(shape_multipliers)
        },
        "pattern_trust": pattern_trust,
        "pattern_stats": pattern_stats,
        "stats": stats,
    }


def write_profile(profile: dict, output_path: str) -> None:
    """Write *profile* to *output_path* as stable, sorted JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compress surname audit feedback into a transition trust profile."
    )
    ap.add_argument("--input", required=True,
                    help="Path to the labelled audit CSV (with name + manual_label)")
    ap.add_argument("--order", type=int, default=2,
                    help="N-gram order used for generation (default: 2)")
    ap.add_argument("--output", required=True,
                    help="Path for the output trust profile JSON")
    ap.add_argument("--good-multiplier", type=float, default=GOOD_FACTOR,
                    dest="good_multiplier",
                    help="Trust multiplier for good rows (default: 1.04)")
    ap.add_argument("--unclear-multiplier", type=float, default=UNCLEAR_FACTOR,
                    dest="unclear_multiplier",
                    help="Trust multiplier for unclear rows (default: 0.97; use 1.0 for no-op)")
    ap.add_argument("--bad-multiplier", type=float, default=BAD_FACTOR,
                    dest="bad_multiplier",
                    help="Trust multiplier for bad rows (default: 0.80)")
    ap.add_argument("--min-trust", type=float, default=TRUST_MIN, dest="min_trust",
                    help="Minimum trust multiplier (default: 0.1)")
    ap.add_argument("--max-trust", type=float, default=TRUST_MAX, dest="max_trust",
                    help="Maximum trust multiplier (default: 2.0)")
    ap.add_argument("--mine-patterns", default="true", dest="mine_patterns",
                    help="Mine bad-heavy character patterns (true/false, default: true)")
    ap.add_argument("--pattern-min-support", type=int, default=2,
                    dest="pattern_min_support",
                    help="Minimum good+bad pattern support (default: 2)")
    ap.add_argument("--pattern-min-bad-count", type=int, default=2,
                    dest="pattern_min_bad_count",
                    help="Minimum bad rows for a pattern penalty (default: 2)")
    ap.add_argument("--pattern-bad-ratio", type=float, default=0.60,
                    dest="pattern_bad_ratio",
                    help="Minimum bad/(good+bad) ratio for pattern penalty (default: 0.60)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        rows = read_labeled_rows(args.input)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    profile = learn_trust(
        rows,
        args.order,
        good_multiplier=args.good_multiplier,
        unclear_multiplier=args.unclear_multiplier,
        bad_multiplier=args.bad_multiplier,
        min_trust=args.min_trust,
        max_trust=args.max_trust,
        mine_patterns=_parse_bool(args.mine_patterns),
        pattern_min_support=args.pattern_min_support,
        pattern_min_bad_count=args.pattern_min_bad_count,
        pattern_bad_ratio=args.pattern_bad_ratio,
    )
    write_profile(profile, args.output)

    s = profile["stats"]
    print(
        f"Reviewed {s['reviewed']} "
        f"(good={s['good']} bad={s['bad']} unclear={s['unclear']}); "
        f"{len(profile['transition_trust'])} transitions, "
        f"{len(profile['shape_trust'])} shape rules, "
        f"{len(profile['pattern_trust'])} bad patterns → {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
