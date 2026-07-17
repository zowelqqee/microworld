"""Freeze Direct/Negative validation from the structured v3 source cohort.

The builder reads no prior benchmark questions, answers, or failures.  It
uses only opaque main-dataset relation/evidence IDs to enforce leakage-free
selection before deterministic question generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldpgt.benchmarks.open_book_qa.heldout_v1 import (
    _main_relation_ids,
    build_direct_negative_heldout_cases,
)
from worldpgt.benchmarks.open_book_qa.dataset import relation_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Build frozen v3 Direct/Negative held-out cases")
    parser.add_argument("--input", default="artifacts/open_book_qa/structured_entity_seed_v1/proposal_relations.json")
    parser.add_argument("--output-dir", default="artifacts/open_book_qa/heldout_v3_direct_negative")
    parser.add_argument("--direct-count", type=int, default=20)
    parser.add_argument("--negative-count", type=int, default=20)
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    main_ids = _main_relation_ids("artifacts/open_book_qa/dataset.jsonl")
    cases, summary = build_direct_negative_heldout_cases(
        rows,
        main_ids,
        direct_count=args.direct_count,
        negative_count=args.negative_count,
    )
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    selected_ids = {value for case in cases for value in case["relation_ids"]}
    frozen_rows = [row for row in rows if relation_id(row) in selected_ids]
    root.joinpath("frozen_relations.json").write_text(
        json.dumps(frozen_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    root.joinpath("dataset.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    summary.update({
        "version": "heldout_v3_direct_negative",
        "source": "structured_entity_seed_v1",
        "frozen_relation_count": len(frozen_rows),
        "frozen_before_question_generation": True,
    })
    root.joinpath("dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
