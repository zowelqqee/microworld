"""
Export predictions that baseline trust accepted but learned trust suppressed
on the TEST split (last 30%) of the ConceptNet sample.

These rows are candidates for manual review.  For each row, fill in
manual_label with one of: should_suppress | should_keep | unclear

Usage:
    python examples/suppression_audit_export.py
    python examples/suppression_audit_export.py --limit 50
    python examples/suppression_audit_export.py \\
        --input  data/conceptnet_sample.csv \\
        --trust-profile data/trust_profile.json \\
        --output data/suppression_audit.csv \\
        --threshold 0.40
    python examples/suppression_audit_export.py \\
        --limit 50 \\
        --min-negative-delta 0.15 \\
        --max-learned-confidence 0.30 \\
        --output data/suppression_audit_calibrated.csv
    python examples/suppression_audit_export.py \\
        --policy quality_aware \\
        --output data/suppression_audit_quality_aware.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.datasets import build_world_from_relations, load_relations_csv
from core.pattern_prediction import PatternBasedPredictor, PatternPrediction
from core.relation_trust import DEFAULT_RELATION_TRUST
from core.suppression_policy import apply_suppression_policy
from core.trust_learning import TrustProfile

_HERE = os.path.dirname(__file__)
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_DEFAULT_INPUT = os.path.join(_ROOT, "data", "conceptnet_sample.csv")
_DEFAULT_TRUST = os.path.join(_ROOT, "data", "trust_profile.json")
_DEFAULT_OUTPUT = os.path.join(_ROOT, "data", "suppression_audit.csv")

COLUMNS = [
    "source",
    "relation_type",
    "target",
    "baseline_confidence",
    "learned_confidence",
    "delta",
    "evidence",
    "reason",
    "manual_label",
    "notes",
]


def find_suppressed_rows(
    baseline_preds: list[PatternPrediction],
    learned_preds: list[PatternPrediction],
    threshold: float,
) -> list[dict]:
    """Return rows where baseline accepted but learned suppressed (naive filter).

    Rows are sorted by delta ascending (most suppressed predictions first).
    manual_label and notes are always empty strings.
    """
    learned_by_triple: dict[tuple[str, str, str], PatternPrediction] = {
        (p.source, p.relation_type, p.target): p for p in learned_preds
    }

    result: list[dict] = []
    for bp in baseline_preds:
        if bp.confidence < threshold:
            continue
        key = (bp.source, bp.relation_type, bp.target)
        lp = learned_by_triple.get(key)
        if lp is None or lp.confidence >= threshold:
            continue
        delta = lp.confidence - bp.confidence
        result.append({
            "source": bp.source,
            "relation_type": bp.relation_type,
            "target": bp.target,
            "baseline_confidence": f"{bp.confidence:.6f}",
            "learned_confidence": f"{lp.confidence:.6f}",
            "delta": f"{delta:.6f}",
            "evidence": "|".join(bp.evidence),
            "reason": bp.reason,
            "manual_label": "",
            "notes": "",
        })

    result.sort(key=lambda r: float(r["delta"]))
    return result


def apply_calibrated_filter(
    rows: list[dict],
    min_negative_delta: float = 0.0,
    max_learned_confidence: float | None = None,
) -> list[dict]:
    """Apply stricter calibrated criteria on top of the naive suppressed rows.

    Keeps only rows where:
      delta <= -min_negative_delta
      learned_confidence <= max_learned_confidence  (if provided)

    With default arguments the output equals the input, preserving backward
    compatibility.
    """
    result = []
    for row in rows:
        if float(row["delta"]) > -min_negative_delta:
            continue
        if max_learned_confidence is not None:
            if float(row["learned_confidence"]) > max_learned_confidence:
                continue
        result.append(row)
    return result


def _split_rows(
    rows: list[dict[str, str]],
    train_fraction: float = 0.7,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    split_at = int(len(rows) * train_fraction)
    return rows[:split_at], rows[split_at:]


def _merge_trust(
    baseline: dict[str, float],
    learned: dict[str, float],
) -> dict[str, float]:
    merged = dict(baseline)
    merged.update(learned)
    return merged


def _run_predictions(
    relations,
    relation_trust: dict[str, float],
) -> list[PatternPrediction]:
    return PatternBasedPredictor(relations).predict_from_bigrams(
        min_count=5,
        min_confidence=0.0,
        hub_penalty=True,
        relation_trust=relation_trust,
        use_relation_drift=True,
    )


def build_suppressed_rows(
    input_path: str,
    trust_profile_path: str,
    threshold: float = 0.40,
    train_fraction: float = 0.7,
    limit: int | None = None,
    min_negative_delta: float = 0.0,
    max_learned_confidence: float | None = None,
    policy: str = "naive",
    min_token_quality: float = 0.55,
) -> list[dict]:
    """Load data, run both trust profiles on the test split, return suppressed rows."""
    if not os.path.exists(trust_profile_path):
        raise FileNotFoundError(f"Trust profile not found: {trust_profile_path}")

    rows = load_relations_csv(input_path)
    _, test_rows = _split_rows(rows, train_fraction)

    world = build_world_from_relations(test_rows)
    relations = world.get_relations()

    profile = TrustProfile.from_json(trust_profile_path)
    learned_trust = _merge_trust(DEFAULT_RELATION_TRUST, profile.relation_trust)

    baseline_preds = _run_predictions(relations, DEFAULT_RELATION_TRUST)
    learned_preds = _run_predictions(relations, learned_trust)

    result = find_suppressed_rows(baseline_preds, learned_preds, threshold)
    result = apply_calibrated_filter(result, min_negative_delta, max_learned_confidence)
    result = apply_suppression_policy(result, policy, min_token_quality)
    if limit is not None:
        result = result[:limit]
    return result


def write_audit_csv(rows: list[dict], output_path: str) -> int:
    """Write audit rows to *output_path*. Returns number of data rows written."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export suppressed predictions for manual audit."
    )
    ap.add_argument("--input", default=_DEFAULT_INPUT,
                    help="Relations CSV (default: data/conceptnet_sample.csv)")
    ap.add_argument("--trust-profile", default=_DEFAULT_TRUST, dest="trust_profile",
                    help="Learned trust profile JSON (default: data/trust_profile.json)")
    ap.add_argument("--output", default=_DEFAULT_OUTPUT,
                    help="Output CSV (default: data/suppression_audit.csv)")
    ap.add_argument("--threshold", type=float, default=0.40,
                    help="Confidence threshold (default: 0.40)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Maximum rows to export (default: all)")
    ap.add_argument("--min-negative-delta", type=float, default=0.0,
                    dest="min_negative_delta",
                    help="Minimum absolute negative delta to include (default: 0.0)")
    ap.add_argument("--max-learned-confidence", type=float, default=None,
                    dest="max_learned_confidence",
                    help="Upper bound on learned_confidence to include (default: no limit)")
    ap.add_argument("--policy", default="naive", choices=["naive", "quality_aware"],
                    help="Suppression policy: naive (default) or quality_aware")
    ap.add_argument("--min-token-quality", type=float, default=0.55,
                    dest="min_token_quality",
                    help="Min token quality for quality_aware policy (default: 0.55)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.trust_profile):
        print(f"Trust profile not found: {args.trust_profile}", file=sys.stderr)
        sys.exit(1)

    rows = build_suppressed_rows(
        input_path=args.input,
        trust_profile_path=args.trust_profile,
        threshold=args.threshold,
        limit=args.limit,
        min_negative_delta=args.min_negative_delta,
        max_learned_confidence=args.max_learned_confidence,
        policy=args.policy,
        min_token_quality=args.min_token_quality,
    )
    n = write_audit_csv(rows, args.output)
    print(f"Wrote {n} rows → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
