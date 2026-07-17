"""Run the existing precision gates over Crossref DOI proposal relations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldpgt.knowledge_pump.crossref_doi_gate import validate_crossref_doi_proposals


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Crossref DOI proposals through precision gates")
    parser.add_argument("--input", default="artifacts/open_book_qa/crossref_doi_seed_v1/proposal_relations.json")
    parser.add_argument("--output-dir", default="artifacts/open_book_qa/crossref_doi_seed_v1/precision_gate")
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    report = validate_crossref_doi_proposals(rows)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("accepted_proposal_overlay.json").write_text(
        json.dumps(report.pop("accepted_proposal_overlay"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    root.joinpath("rejected.json").write_text(
        json.dumps(report.pop("rejected"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    root.joinpath("quarantine.json").write_text(
        json.dumps(report.pop("quarantine"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    root.joinpath("summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
