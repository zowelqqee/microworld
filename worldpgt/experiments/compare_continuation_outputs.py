"""Compare Microworld Controlled Continuation v0 against a Fable5 external LLM baseline.

Usage:
    python -m worldpgt.experiments.compare_continuation_outputs \
        --prompts worldpgt/experiments/continuation_prompts.csv \
        --microworld worldpgt/experiments/microworld_continuation_outputs.csv \
        --fable5 worldpgt/experiments/fable5_baseline_continuation.jsonl \
        --output worldpgt/experiments/continuation_comparison_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional


def _norm(v) -> Optional[str]:
    """Normalise empty strings and JSON null to None for consistent comparison."""
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _classify(expected: Optional[str], selected: Optional[str]) -> str:
    """Classify one row as 'correct', 'wrong', or 'no_sense'.

    Rules:
      both None              -> 'correct'  (no-sense handling case)
      expected non-None, selected None -> 'no_sense'
      expected non-None, selected != expected -> 'wrong'
      expected non-None, selected == expected -> 'correct'
      expected None, selected non-None -> 'wrong' (spurious sense)
    """
    if expected is None and selected is None:
        return "correct"
    if expected is not None and selected is None:
        return "no_sense"
    if expected == selected:
        return "correct"
    return "wrong"


def _load_prompts(path: str) -> dict[str, dict]:
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return {row["id"]: row for row in csv.DictReader(fh)}


def _load_microworld(path: str) -> dict[str, dict]:
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return {row["id"]: row for row in csv.DictReader(fh)}


def _load_fable5(path: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows[str(obj["id"])] = obj
    return rows


def compare(
    prompts_path: str,
    microworld_path: str,
    fable5_path: str,
) -> dict:
    """Return the full comparison summary dict."""
    prompts = _load_prompts(prompts_path)
    microworld = _load_microworld(microworld_path)
    fable5 = _load_fable5(fable5_path)

    ids = list(prompts.keys())

    # Microworld counters
    mw_total = 0
    mw_continue = 0
    mw_audit = 0
    mw_suppress = 0
    mw_correct = 0
    mw_wrong = 0
    mw_no_sense = 0
    mw_conf_values: list[float] = []

    # Fable5 counters
    f5_total = 0
    f5_correct = 0
    f5_wrong = 0
    f5_no_sense = 0
    f5_conf_values: list[float] = []

    # Comparison counters
    both_correct = 0
    mw_only_correct = 0
    f5_only_correct = 0
    both_wrong = 0
    disagreements: list[dict] = []

    for pid in ids:
        expected = _norm(prompts[pid].get("expected_sense", ""))

        # --- Microworld row ---
        mw_row = microworld.get(pid, {})
        mw_selected = _norm(mw_row.get("selected_sense", ""))
        mw_decision = mw_row.get("decision", "")
        try:
            mw_conf = float(mw_row.get("confidence") or 0.0)
        except (ValueError, TypeError):
            mw_conf = 0.0

        mw_total += 1
        if mw_decision == "continue":
            mw_continue += 1
        elif mw_decision == "audit":
            mw_audit += 1
        elif mw_decision == "suppress":
            mw_suppress += 1
        mw_conf_values.append(mw_conf)

        mw_cls = _classify(expected, mw_selected)
        if mw_cls == "correct":
            mw_correct += 1
        elif mw_cls == "wrong":
            mw_wrong += 1
        else:
            mw_no_sense += 1

        # --- Fable5 row ---
        f5_row = fable5.get(pid, {})
        f5_selected = _norm(f5_row.get("selected_sense"))
        f5_conf_raw = f5_row.get("confidence")

        f5_total += 1
        if f5_conf_raw is not None:
            try:
                f5_conf_values.append(float(f5_conf_raw))
            except (ValueError, TypeError):
                pass

        f5_cls = _classify(expected, f5_selected)
        if f5_cls == "correct":
            f5_correct += 1
        elif f5_cls == "wrong":
            f5_wrong += 1
        else:
            f5_no_sense += 1

        # --- Head-to-head ---
        mw_ok = mw_cls == "correct"
        f5_ok = f5_cls == "correct"
        if mw_ok and f5_ok:
            both_correct += 1
        elif mw_ok and not f5_ok:
            mw_only_correct += 1
        elif not mw_ok and f5_ok:
            f5_only_correct += 1
        else:
            both_wrong += 1

        if mw_selected != f5_selected:
            disagreements.append(
                {
                    "id": pid,
                    "prompt": prompts[pid].get("prompt", ""),
                    "expected_sense": expected,
                    "microworld_selected_sense": mw_selected,
                    "fable5_selected_sense": f5_selected,
                    "microworld_decision": mw_decision,
                    "fable5_confidence": f5_conf_raw,
                }
            )

    def _rate(n: int, total: int) -> float:
        return round(n / total, 4) if total > 0 else 0.0

    def _avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "microworld": {
            "total": mw_total,
            "continue_count": mw_continue,
            "audit_count": mw_audit,
            "suppress_count": mw_suppress,
            "correct_sense_count": mw_correct,
            "wrong_sense_count": mw_wrong,
            "no_sense_count": mw_no_sense,
            "correct_sense_rate": _rate(mw_correct, mw_total),
            "wrong_sense_rate": _rate(mw_wrong, mw_total),
            "audit_rate": _rate(mw_audit, mw_total),
            "suppress_rate": _rate(mw_suppress, mw_total),
            "avg_confidence": _avg(mw_conf_values),
        },
        "fable5": {
            "total": f5_total,
            "correct_sense_count": f5_correct,
            "wrong_sense_count": f5_wrong,
            "no_sense_count": f5_no_sense,
            "correct_sense_rate": _rate(f5_correct, f5_total),
            "wrong_sense_rate": _rate(f5_wrong, f5_total),
            "avg_confidence": _avg(f5_conf_values),
        },
        "comparison": {
            "both_correct": both_correct,
            "microworld_only_correct": mw_only_correct,
            "fable5_only_correct": f5_only_correct,
            "both_wrong": both_wrong,
            "disagreement_count": len(disagreements),
            "disagreements": disagreements,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Microworld Controlled Continuation v0 against Fable5 external LLM baseline."
    )
    parser.add_argument("--prompts", required=True, help="Prompt CSV with id,prompt,ambiguous_term,expected_sense")
    parser.add_argument("--microworld", required=True, help="Microworld output CSV")
    parser.add_argument("--fable5", required=True, help="Fable5 baseline JSONL")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    summary = compare(args.prompts, args.microworld, args.fable5)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
