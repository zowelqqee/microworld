"""Candidate Knowledge Request v1 runner.

Turns Self-Learning Loop v1 ``missing_stable_knowledge`` opportunities (confirmed
auditing-safely by the Curriculum Regression Gate v1) into proposal-only
candidate knowledge requests and local source document requests under
``worldpgt/experiments/knowledge_requests_v1/``.

SAFETY CONTRACT (enforced by construction — this runner only reads inputs and
writes under ``knowledge_requests_v1/``):

- No trusted memory write. accepted_knowledge_memory_v1.json untouched.
- No overlay write. accepted / promoted overlays untouched.
- No wiki page creation. No self-ingestion. No promotion. No network.
- Proposed analyzer rules and extraction patterns are NOT activated.
- No neural / GPT / training / embedding / network code. nanogpt/ untouched.
- safe_for_general_runtime is always False; everything is request-only.

Usage::

    python3 worldpgt/experiments/run_candidate_knowledge_requests_v1.py
    # or
    python3 -m worldpgt.experiments.run_candidate_knowledge_requests_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

# Allow running directly as a script.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.knowledge_requests.missing_knowledge_loader import (
    MissingKnowledgeLoader,
    normalize_missing_inputs,
)
from worldpgt.knowledge_requests.request_classifier import classify_requests
from worldpgt.knowledge_requests.request_report import build_report, build_summary
from worldpgt.knowledge_requests.source_request_builder import (
    build_source_document_requests,
)

_EXPERIMENTS = Path(__file__).resolve().parent
_DEFAULT_OUTDIR = _EXPERIMENTS / "knowledge_requests_v1"

_REQUEST_FIELDS = [
    "id",
    "category",
    "question",
    "current_decision",
    "missing_knowledge_type",
    "needed_evidence_type",
    "candidate_subject",
    "candidate_relation",
    "candidate_object",
    "suggested_source_topic",
    "risk_level",
    "activation_status",
    "must_remain_audit_until_supported",
]

_SOURCE_DOC_FIELDS = [
    "id",
    "topic",
    "suggested_title",
    "related_request_count",
    "entities_to_cover",
    "relations_to_verify",
    "next_pipeline_step",
]


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
        fh.write("\n")


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_candidate_knowledge_requests(
    experiments_dir: Path = _EXPERIMENTS,
    out_dir: Path = _DEFAULT_OUTDIR,
    write: bool = True,
) -> Dict[str, Any]:
    experiments_dir = Path(experiments_dir)
    out_dir = Path(out_dir)

    loaded = MissingKnowledgeLoader(experiments_dir).load()
    missing_inputs = normalize_missing_inputs(loaded)
    requests, skipped = classify_requests(missing_inputs)
    source_docs = build_source_document_requests(requests)

    # Deduplicated = missing inputs that neither produced a request nor a skip.
    deduplicated_count = max(
        0, len(missing_inputs) - len(requests) - _missing_skips(skipped)
    )

    summary = build_summary(
        requests,
        source_docs,
        skipped,
        missing_inputs_total=len(missing_inputs),
        deduplicated_count=deduplicated_count,
    )
    report = build_report(summary, loaded, source_docs, skipped)

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            out_dir / "candidate_knowledge_requests.json",
            [asdict(r) for r in requests],
        )
        _write_csv(
            out_dir / "candidate_knowledge_requests.csv",
            _REQUEST_FIELDS,
            [
                {
                    "id": r.id,
                    "category": r.category,
                    "question": r.question,
                    "current_decision": r.current_decision,
                    "missing_knowledge_type": r.missing_knowledge_type,
                    "needed_evidence_type": r.needed_evidence_type,
                    "candidate_subject": r.candidate_subject,
                    "candidate_relation": r.candidate_relation,
                    "candidate_object": r.candidate_object,
                    "suggested_source_topic": r.suggested_source_topic,
                    "risk_level": r.risk_level,
                    "activation_status": r.activation_status,
                    "must_remain_audit_until_supported": r.must_remain_audit_until_supported,
                }
                for r in requests
            ],
        )
        _write_json(
            out_dir / "source_document_requests.json",
            [asdict(d) for d in source_docs],
        )
        _write_csv(
            out_dir / "source_document_requests.csv",
            _SOURCE_DOC_FIELDS,
            [
                {
                    "id": d.id,
                    "topic": d.topic,
                    "suggested_title": d.suggested_title,
                    "related_request_count": len(d.related_request_ids),
                    "entities_to_cover": "|".join(d.entities_to_cover),
                    "relations_to_verify": "|".join(d.relations_to_verify),
                    "next_pipeline_step": d.next_pipeline_step,
                }
                for d in source_docs
            ],
        )
        _write_json(out_dir / "knowledge_request_summary.json", summary)
        _write_json(out_dir / "knowledge_request_report.json", asdict(report))

    return {
        "loaded": loaded,
        "missing_inputs": missing_inputs,
        "requests": requests,
        "skipped": skipped,
        "source_docs": source_docs,
        "summary": summary,
        "report": report,
    }


def _missing_skips(skipped) -> int:
    # Skips that originate from missing_stable_knowledge content screening
    # (private/inversion/universal) — i.e. not the wholesale dangerous-category
    # skips. Both count as "skipped"; this helper only supports dedup math.
    return sum(
        1
        for s in skipped
        if s.reason
        in {"private_or_sensitive_data", "relation_inversion", "unsupported_universal"}
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", default=str(_EXPERIMENTS))
    parser.add_argument("--out-dir", default=str(_DEFAULT_OUTDIR))
    args = parser.parse_args(argv)

    result = run_candidate_knowledge_requests(
        experiments_dir=Path(args.experiments_dir),
        out_dir=Path(args.out_dir),
        write=True,
    )
    s = result["summary"]
    print("Candidate Knowledge Request v1 (request layer only)")
    print(f"  missing inputs total         : {s['missing_inputs_total']}")
    print(f"  requests total               : {s['requests_total']}")
    print(f"  source document requests     : {s['source_document_requests_total']}")
    print(f"  requests_by_category         : {s['requests_by_category']}")
    print(f"  requests_by_risk             : {s['requests_by_risk']}")
    print(f"  skipped_dangerous_inputs     : {s['skipped_dangerous_inputs_total']}")
    print(f"  deduplicated_count           : {s['deduplicated_count']}")
    print(f"  request_only={s['request_only']} auto_ingest={s['auto_ingest']} "
          f"auto_promote={s['auto_promote']}")
    print(f"  trusted_memory_modified={s['trusted_memory_modified']} "
          f"accepted_overlay_modified={s['accepted_overlay_modified']} "
          f"promoted_overlay_modified={s['promoted_overlay_modified']}")
    print(f"  network_calls={s['network_calls']} "
          f"safe_for_general_runtime={s['safe_for_general_runtime']}")
    print(f"  artifacts written to         : {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
