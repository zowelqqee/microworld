"""Universal domain bootstrap CLI.

Takes a JSONL corpus on ANY topic and produces a QA-ready overlay with no prior
entity list and no hand-written domain predicates:

    python3 worldpgt/experiments/run_domain_bootstrap_v1.py \
        --input worldpgt/experiments/domain_bootstrap_v1/o1a_docs.jsonl \
        --domain "o1a_visa" \
        --output worldpgt/experiments/domain_overlays/o1a_v1.json

Then serve it:

    python3 -m worldpgt.api.server \
        --overlay domain:worldpgt/experiments/domain_overlays/o1a_v1.json \
        --port 8000

Input JSONL lines: {"doc_id":"d1","title":"...","url":"...","text":"..."}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.schema_induction.domain_overlay_builder import build_domain_overlay


def _read_docs(path: str) -> list[dict]:
    docs: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            docs.append(json.loads(line))
    return docs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Universal domain bootstrap.")
    parser.add_argument("--input", required=True, help="Docs JSONL path.")
    parser.add_argument("--domain", default="domain", help="Domain label.")
    parser.add_argument("--output", required=True, help="Output overlay JSON path.")
    parser.add_argument("--min-evidence", type=int, default=1)
    parser.add_argument("--min-sources", type=int, default=1)
    args = parser.parse_args(argv)

    docs = _read_docs(args.input)
    print(f"[bootstrap] read {len(docs)} docs from {args.input}", flush=True)

    result = build_domain_overlay(
        docs,
        domain=args.domain,
        min_evidence=args.min_evidence,
        min_sources=args.min_sources,
    )
    overlay = result["overlay"]
    entities = result["entities"]
    stats = result["stats"]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(overlay, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Also write a sidecar with the bootstrapped entity report.
    sidecar = out_path.with_suffix(".entities.json")
    sidecar.write_text(
        json.dumps(
            [
                {
                    "canonical_label": e.canonical_label,
                    "aliases": list(e.aliases),
                    "entity_type": e.entity_type,
                    "spacy_labels": list(e.spacy_labels),
                    "occurrences": e.occurrences,
                }
                for e in entities
            ],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Domain bootstrap complete")
    print("-------------------------")
    print(f"  domain:           {args.domain}")
    print(f"  documents:        {len(docs)}")
    print(f"  entities:         {stats['entities']}")
    print(f"  definitions:      {stats['definitions']}")
    print(f"  relations:        {stats['relations']}")
    print(f"  overlay items:    {stats['overlay_items']}")
    print(f"  overlay written:  {out_path}")
    print(f"  entities report:  {sidecar}")
    print()
    print("Serve with:")
    print(f"  python3 -m worldpgt.api.server --overlay domain:{out_path} --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
