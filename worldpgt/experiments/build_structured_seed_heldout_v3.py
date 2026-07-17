"""Freeze a held-out benchmark from the already-frozen structured seed cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldpgt.benchmarks.open_book_qa.heldout_v1 import _main_relation_ids, build_heldout_cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Build held-out v3 from structured seed proposals")
    parser.add_argument("--input", default="artifacts/open_book_qa/structured_entity_seed_v1/proposal_relations.json")
    parser.add_argument("--output-dir", default="artifacts/open_book_qa/heldout_v3")
    args = parser.parse_args()
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    main_ids = _main_relation_ids("artifacts/open_book_qa/dataset.jsonl")
    cases, summary = build_heldout_cases(rows, main_ids)
    root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=True)
    root.joinpath("frozen_relations.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    root.joinpath("dataset.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    summary.update({
        "version": "heldout_v3",
        "source": "structured_entity_seed_v1",
        "frozen_before_question_generation": True,
    })
    root.joinpath("dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
