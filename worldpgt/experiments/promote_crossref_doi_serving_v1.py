"""Publish precision-accepted Crossref DOI relations to the serving campaign graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldpgt.knowledge_pump.crossref_doi_serving_promotion import build_crossref_doi_serving_overlay


_FILENAME = "open_web_campaign_evidence_grounded_graph_overlay.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="User-authorized Crossref DOI serving promotion")
    parser.add_argument(
        "--input",
        default="artifacts/open_book_qa/crossref_doi_seed_v1/precision_gate/accepted_proposal_overlay.json",
    )
    parser.add_argument(
        "--campaign-dir",
        default="worldpgt/experiments/open_web_pump_v1/campaign_crossref_doi_v1",
    )
    args = parser.parse_args()

    source = Path(args.input)
    rows = json.loads(source.read_text(encoding="utf-8"))
    overlay, summary = build_crossref_doi_serving_overlay(rows)
    root = Path(args.campaign_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / _FILENAME
    target.write_text(json.dumps(overlay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary.update({
        "input": str(source),
        "serving_overlay_path": str(target),
        "promotion_authorization": "explicit_user_request",
    })
    root.joinpath("crossref_doi_serving_promotion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
