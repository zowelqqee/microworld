"""
End-to-end name/surname generation experiment: baseline vs. trust-learned generation.

Supports any one-name-per-line dataset — given names, family names, or a mix.

Pipeline:
  1. load + normalize names
  2. deterministic train/test split (the graph only sees train)
  3. build the transition graph from the train split
  4. generate baseline names and export a baseline audit CSV
  5. if a trust profile is supplied (and exists), generate trust-biased names and
     export a learned audit CSV
  6. print a side-by-side quality comparison

The comparison uses the explicit quality policy (avg quality score, valid
fraction) plus simple novelty stats — it is an interpretability check, not a
claim of beating neural generators.  If no trust profile is given, only the
baseline runs.

Usage:
    python examples/surname_generation_experiment.py --input data/names.txt
    python examples/surname_generation_experiment.py \\
        --trust-profile data/surname_trust_profile.json --order 2 --count 100
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.surname_generator import (
    SurnameTransitionGraph,
    load_surnames,
    load_trust_profile,
)
from core.surname_policy import is_valid_generated_surname, surname_quality_score
from examples.surname_generate import build_rows, generate_names, write_csv

_HERE = os.path.dirname(__file__)
_DEFAULT_INPUT = os.path.normpath(os.path.join(_HERE, "..", "data", "surnames.txt"))
_DEFAULT_BASELINE = os.path.normpath(
    os.path.join(_HERE, "..", "data", "surname_audit_baseline.csv")
)
_DEFAULT_LEARNED = os.path.normpath(
    os.path.join(_HERE, "..", "data", "surname_audit_learned.csv")
)


def split_train_test(names: list[str], test_ratio: float = 0.1, seed: int = 0):
    """Deterministically split *names* into (train, test)."""
    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    n_test = int(len(shuffled) * test_ratio)
    return shuffled[n_test:], shuffled[:n_test]


def evaluate_names(names: list[str], train_set: set[str], test_set: set[str]) -> dict:
    """Compute quality + novelty stats for a batch of generated names."""
    if not names:
        return {
            "count": 0,
            "avg_quality": 0.0,
            "valid_fraction": 0.0,
            "novel_fraction": 0.0,
            "in_test_fraction": 0.0,
        }
    scores = [surname_quality_score(n) for n in names]
    valid = sum(1 for n in names if is_valid_generated_surname(n))
    novel = sum(1 for n in names if n not in train_set)
    in_test = sum(1 for n in names if n in test_set)
    total = len(names)
    return {
        "count": total,
        "avg_quality": sum(scores) / total,
        "valid_fraction": valid / total,
        "novel_fraction": novel / total,
        "in_test_fraction": in_test / total,
    }


def run_experiment(
    input_path: str,
    *,
    order: int = 2,
    count: int = 100,
    seed: int = 42,
    baseline_output: str = _DEFAULT_BASELINE,
    learned_output: str = _DEFAULT_LEARNED,
    trust_profile_path: str | None = None,
    max_length: int = 20,
    column: int | None = None,
) -> dict:
    """Run baseline (and optionally learned) generation; return a result dict."""
    surnames = load_surnames(input_path, column=column)
    train, test = split_train_test(surnames, seed=seed)
    train_set, test_set = set(train), set(test)

    graph = SurnameTransitionGraph(order=order).build(train)

    baseline_names = generate_names(
        graph, count,
        rng=random.Random(seed),
        source_set=train_set,
        max_length=max_length,
    )
    write_csv(build_rows(baseline_names, train_set, "baseline"), baseline_output)
    baseline_eval = evaluate_names(baseline_names, train_set, test_set)

    result = {
        "train_size": len(train),
        "test_size": len(test),
        "order": order,
        "baseline": baseline_eval,
        "baseline_output": baseline_output,
        "learned": None,
        "learned_output": None,
    }

    if trust_profile_path and os.path.exists(trust_profile_path):
        trust_profile = load_trust_profile(trust_profile_path)
        learned_names = generate_names(
            graph, count,
            rng=random.Random(seed),
            source_set=train_set,
            max_length=max_length,
            trust_profile=trust_profile,
        )
        write_csv(build_rows(learned_names, train_set, "learned"), learned_output)
        result["learned"] = evaluate_names(learned_names, train_set, test_set)
        result["learned_output"] = learned_output

    return result


def _fmt_eval(label: str, e: dict) -> str:
    return (
        f"  {label:9s} | n={e['count']:>4d} | "
        f"avg_quality={e['avg_quality']:.4f} | "
        f"valid={e['valid_fraction']:.3f} | "
        f"novel={e['novel_fraction']:.3f} | "
        f"in_test={e['in_test_fraction']:.3f}"
    )


def format_result(result: dict) -> str:
    lines = [
        "Microworld-style name/surname generation experiment",
        f"  train/test : {result['train_size']} / {result['test_size']}"
        f"   order={result['order']}",
        "",
        _fmt_eval("baseline", result["baseline"]),
    ]
    if result["learned"] is not None:
        lines.append(_fmt_eval("learned", result["learned"]))
        d = result["learned"]["avg_quality"] - result["baseline"]["avg_quality"]
        lines.append("")
        lines.append(f"  avg_quality delta (learned - baseline): {d:+.4f}")
    else:
        lines.append("")
        lines.append("  (no trust profile supplied — baseline only)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Baseline vs trust-learned name/surname generation experiment. "
            "Input may be given names, surnames, or any mixed personal-name list."
        )
    )
    ap.add_argument("--input", default=_DEFAULT_INPUT,
                    help="Path to the name list — one name per line (default: data/surnames.txt)")
    ap.add_argument("--order", type=int, default=2,
                    help="N-gram context order (default: 2)")
    ap.add_argument("--count", type=int, default=100,
                    help="Names to generate per condition (default: 100)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--max-length", type=int, default=20, dest="max_length")
    ap.add_argument("--column", type=int, default=None)
    ap.add_argument("--baseline-output", default=_DEFAULT_BASELINE,
                    dest="baseline_output",
                    help="Path for the baseline audit CSV")
    ap.add_argument("--learned-output", default=_DEFAULT_LEARNED,
                    dest="learned_output",
                    help="Path for the learned audit CSV")
    ap.add_argument("--trust-profile", default=None, dest="trust_profile",
                    help="Optional trust profile JSON; if missing, baseline only")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    result = run_experiment(
        args.input,
        order=args.order,
        count=args.count,
        seed=args.seed,
        baseline_output=args.baseline_output,
        learned_output=args.learned_output,
        trust_profile_path=args.trust_profile,
        max_length=args.max_length,
        column=args.column,
    )
    print(format_result(result))


if __name__ == "__main__":
    main()
