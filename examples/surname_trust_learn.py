"""
Compress audit feedback into a compact transition trust profile.

Reads a manually-labelled name/surname audit CSV, reconstructs the character
transitions each name went through (at the generation ``--order``), and nudges a
per-transition trust multiplier:

    good    : multiplier *= 1.05
    bad     : multiplier *= 0.85
    unclear : ignored (no-op)

Multipliers start at 1.0 and are bounded to [0.1, 2.0].  The result is a small
JSON profile that examples/surname_generate.py can replay to bias generation —
feedback learning without any weights or backpropagation.

Output schema:
    {
      "order": 2,
      "transition_trust": {"ab->r": 1.05, "xq->z": 0.65},
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

from core.surname_generator import iter_transitions, transition_key
from examples.surname_audit_summary import normalize_label, read_labeled_rows

GOOD_FACTOR = 1.05
BAD_FACTOR = 0.85
TRUST_MIN = 0.1
TRUST_MAX = 2.0


def learn_trust(rows: list[dict], order: int) -> dict:
    """Build a transition trust profile from labelled audit *rows*.

    Returns the full profile dict (order, transition_trust, stats).  Transitions
    that only ever appear under ``unclear`` (or never appear) stay implicitly at
    1.0 and are omitted from the output.
    """
    multipliers: dict[str, float] = {}
    stats = {"reviewed": 0, "good": 0, "bad": 0, "unclear": 0}

    for row in rows:
        name = (row.get("name") or "").strip()
        if not name:
            continue
        label = normalize_label(row.get("manual_label", ""))
        stats["reviewed"] += 1
        stats[label] += 1

        if label == "good":
            factor = GOOD_FACTOR
        elif label == "bad":
            factor = BAD_FACTOR
        else:  # unclear → no-op
            continue

        for context, ch in iter_transitions(name, order):
            key = transition_key(context, ch)
            current = multipliers.get(key, 1.0)
            updated = max(TRUST_MIN, min(TRUST_MAX, current * factor))
            multipliers[key] = updated

    return {
        "order": order,
        "transition_trust": {k: multipliers[k] for k in sorted(multipliers)},
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
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    rows = read_labeled_rows(args.input)
    profile = learn_trust(rows, args.order)
    write_profile(profile, args.output)

    s = profile["stats"]
    print(
        f"Reviewed {s['reviewed']} "
        f"(good={s['good']} bad={s['bad']} unclear={s['unclear']}); "
        f"{len(profile['transition_trust'])} transitions → {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
