"""Promote Overlay Delta v1 runner.

Validates the self-ingestion overlay delta proposal, promotes only safe items
into a separate promoted overlay, runs the QA / adversarial / cross-page
regressions against that PROMOTED overlay (not the accepted overlay), and writes
the promotion artifacts.

SAFETY CONTRACT:
- accepted_knowledge_memory_v1.json / sense_memory.py are NOT modified.
- accepted_wiki_memory_overlay_v1.json is NOT overwritten (read-only base).
- Ingestion / overlay-builder / validator / threshold semantics are NOT changed.
- No neural/GPT/training/embedding code. No network. nanogpt/ untouched.
- safe_for_general_runtime is always False.

Usage::

    python3 worldpgt/experiments/promote_wiki_overlay_delta_v1.py
    # or
    python3 -m worldpgt.experiments.promote_wiki_overlay_delta_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Allow running directly as a script (python3 worldpgt/experiments/...py).
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.experiments import run_cross_page_qa_v1, run_entity_qa_v1
from worldpgt.self_ingestion.overlay_delta_promoter import promote
from worldpgt.self_ingestion.promotion_report import build_promotion_report

_EXPERIMENTS = Path(__file__).resolve().parent
_REPO = _EXPERIMENTS.parent.parent  # repo root (.../worldpgt -> root)

_DEFAULT_BASE = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_DEFAULT_DELTA = _EXPERIMENTS / "self_ingestion_v1" / "overlay_delta_proposal.json"
_DEFAULT_OUTDIR = _EXPERIMENTS / "self_ingestion_v1" / "promotion"

_QA_BENCHES = {
    "entity_qa_v1": ("entity", "entity_qa_prompts_v1.csv", 28),
    "entity_qa_expansion_v1": ("entity", "entity_qa_expansion_v1.csv", 111),
    "entity_qa_adversarial_v1": ("entity", "entity_qa_adversarial_v1.csv", 68),
    "cross_page_qa_v1": ("cross", "cross_page_qa_v1.csv", 71),
}


def _run_qa(kind: str, qa_csv: str, overlay_path: str, tmpdir: Path) -> dict:
    qa_input = str(_EXPERIMENTS / qa_csv)
    out_csv = str(tmpdir / "out.csv")
    out_json = str(tmpdir / "summary.json")
    if kind == "entity":
        return run_entity_qa_v1.run(qa_input, overlay_path, out_csv, out_json)
    return run_cross_page_qa_v1.run(qa_input, overlay_path, out_csv, out_json)


def run(base_overlay: str, delta_proposal: str, output_dir: str) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    promoted_path = out / "promoted_wiki_memory_overlay_v1.json"
    meta_path = out / "promoted_wiki_memory_overlay_v1.meta.json"

    promotion = promote(base_overlay, delta_proposal, str(promoted_path), str(meta_path))

    # Regressions against the PROMOTED overlay.
    regression: dict = {}
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for name, (kind, csv_name, expected) in _QA_BENCHES.items():
            summary = _run_qa(kind, csv_name, str(promoted_path), tmpdir)
            ok = (summary["qa_total"] == expected
                  and summary["correct_count"] == expected
                  and summary["wrong_count"] == 0
                  and summary.get("quality_flagged", 0) == 0
                  and float(summary.get("answer_precision", 1.0)) == 1.0)
            regression[name] = {
                "status": "PASS" if ok else "FAIL",
                "qa_total": summary["qa_total"],
                "correct_count": summary["correct_count"],
                "wrong_count": summary["wrong_count"],
                "quality_flagged": summary.get("quality_flagged", 0),
                "answer_precision": summary.get("answer_precision", 1.0),
                "expected": expected,
            }

    report = build_promotion_report(promotion, regression)

    # Write artifacts.
    (out / "promotion_validation.json").write_text(
        json.dumps(promotion.validation.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8")
    (out / "promotion_regression_summary.json").write_text(
        json.dumps(regression, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "promotion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Promote Overlay Delta v1")
    parser.add_argument("--base-overlay", default=str(_DEFAULT_BASE))
    parser.add_argument("--delta-proposal", default=str(_DEFAULT_DELTA))
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTDIR))
    args = parser.parse_args(argv)

    report = run(args.base_overlay, args.delta_proposal, args.output_dir)

    print(f"[promote] base_overlay_items:     {report['base_overlay_items']}")
    print(f"[promote] delta_proposal_items:   {report['delta_proposal_items']}")
    print(f"[promote] accepted_delta_items:   {report['accepted_delta_items']}")
    print(f"[promote] rejected_delta_items:   {report['rejected_delta_items']}")
    print(f"[promote] blocked_delta_items:    {report['blocked_delta_items']}")
    print(f"[promote] promoted_overlay_items: {report['promoted_overlay_items']}")
    print(f"[promote] safe_to_promote:        {report['safe_to_promote']}")
    print(f"[promote] safe_for_general_runtime: {report['safe_for_general_runtime']}")
    print(f"[promote] weak_links_remain_weak: {report['weak_links_remain_weak']}")
    print(f"[promote] volatile_not_promoted:  {report['volatile_items_not_promoted']}")
    print(f"[promote] inverted_not_promoted:  {report['inverted_relations_not_promoted']}")
    print(f"[promote] private_not_promoted:   {report['private_items_not_promoted']}")
    for name, st in report["regression_results"].items():
        print(f"[promote] QA {name}: {st['status']} ({st['correct_count']}/{st['qa_total']})")

    print("\n[safety] accepted_knowledge_memory_v1.json: NOT modified")
    print("[safety] sense_memory.py: NOT modified")
    print("[safety] accepted_wiki_memory_overlay_v1.json: NOT overwritten")
    print("[safety] ingestion/overlay/validator/threshold semantics: NOT changed")
    print("[safety] neural/GPT/training/embedding: NOT used")
    print("[safety] network access: NONE")
    print("[safety] nanogpt/: NOT touched")


if __name__ == "__main__":
    main()
