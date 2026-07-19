#!/usr/bin/env python3
"""Informed-reflection classification GATE PILOT — throwaway prototype, not production.

Deterministic, rule-based, NO neural inference. Uses ONLY the structural markers the
design doc (artifacts/informed_reflection_v1/design.md, section 2) proposed, plus their
direct morphological variants. The point is to test whether that proposed approach is
reliable ENOUGH to justify the full ~8-10 day build — not to build a good classifier.

Decision rule (intentionally minimal):
    if any speculative marker is present -> SPECULATIVE
    else                                 -> FACTUAL   (conservative default:
                                                        an unmarked declarative is
                                                        treated as an asserted claim)

Run:  python3 pilot_classifier.py
Writes pilot_results.json next to this file and prints a summary table.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- marker set: EXACTLY the families design.md section 2 named, plus close variants ---
# Single-word markers matched on word boundaries (so "different" does not match "if").
WORD_MARKERS = [
    "if", "might", "may", "could", "perhaps", "possibly", "probably", "maybe",
    "would", "imagine", "suppose", "arguably", "hadn't", "wouldn't",
]
# Multi-word / phrase markers matched as lowercase substrings.
PHRASE_MARKERS = [
    "what if", "one might argue", "it is interesting to consider", "i suspect",
    "in my view", "could have", "would have", "might have", "were to",
    "it seems", "seems that",
]

_WORD_RES = {m: re.compile(rf"\b{re.escape(m)}\b") for m in WORD_MARKERS}


def classify(text: str) -> tuple[str, list[str]]:
    """Return (label, hit_markers). label in {FACTUAL, SPECULATIVE}."""
    low = text.lower()
    hits: list[str] = []
    for phrase in PHRASE_MARKERS:
        if phrase in low:
            hits.append(phrase)
    for marker, rgx in _WORD_RES.items():
        if rgx.search(low):
            hits.append(marker)
    return ("SPECULATIVE" if hits else "FACTUAL"), hits


def main() -> None:
    data = json.loads((HERE / "pilot_sentences.json").read_text(encoding="utf-8"))
    rows = data["sentences"]

    results = []
    for row in rows:
        pred, hits = classify(row["text"])
        gold = row["gold"]
        # A binary rule can never be "correct" on a MIXED gold; record separately.
        if gold == "MIXED":
            correct = None
        else:
            correct = (pred == gold)
        results.append(
            {
                "id": row["id"],
                "category": row["category"],
                "text": row["text"],
                "gold": gold,
                "pred": pred,
                "markers": hits,
                "correct": correct,
            }
        )

    non_mixed = [r for r in results if r["gold"] != "MIXED"]
    mixed = [r for r in results if r["gold"] == "MIXED"]
    clean = [r for r in results if r["category"] in ("clean_factual", "clean_speculative")]

    # Positive class = FACTUAL (the class that triggers verification), per design.md sec 3.
    #   FP = gold SPECULATIVE, predicted FACTUAL  (unverified content passes as verified)
    #   FN = gold FACTUAL,     predicted SPECULATIVE (real claim escapes verification)
    gold_fact = [r for r in non_mixed if r["gold"] == "FACTUAL"]
    gold_spec = [r for r in non_mixed if r["gold"] == "SPECULATIVE"]
    fp = [r for r in gold_spec if r["pred"] == "FACTUAL"]
    fn = [r for r in gold_fact if r["pred"] == "SPECULATIVE"]
    clean_spec = [r for r in clean if r["gold"] == "SPECULATIVE"]
    clean_fp = [r for r in clean_spec if r["pred"] == "FACTUAL"]

    def rate(n: int, d: int) -> float:
        return round(100.0 * n / d, 1) if d else 0.0

    metrics = {
        "n_total": len(results),
        "n_non_mixed": len(non_mixed),
        "n_mixed": len(mixed),
        "mixed_frequency_pct": rate(len(mixed), len(results)),
        "overall_accuracy_non_mixed_pct": rate(sum(1 for r in non_mixed if r["correct"]), len(non_mixed)),
        "clean_only_accuracy_pct": rate(sum(1 for r in clean if r["correct"]), len(clean)),
        "false_positive_pct_overall": rate(len(fp), len(gold_spec)),
        "false_positive_pct_clean_only": rate(len(clean_fp), len(clean_spec)),
        "false_negative_pct": rate(len(fn), len(gold_fact)),
        "false_positive_ids": [r["id"] for r in fp],
        "false_negative_ids": [r["id"] for r in fn],
        "mixed_predictions": {str(r["id"]): r["pred"] for r in mixed},
    }

    out = {"results": results, "metrics": metrics}
    (HERE / "pilot_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # --- console summary ---
    print(f"{'id':>3} {'category':<22} {'gold':<12} {'pred':<12} {'ok':<3} markers")
    print("-" * 90)
    for r in results:
        ok = "-" if r["correct"] is None else ("Y" if r["correct"] else "N")
        print(f"{r['id']:>3} {r['category']:<22} {r['gold']:<12} {r['pred']:<12} {ok:<3} {','.join(r['markers'])}")
    print("-" * 90)
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
