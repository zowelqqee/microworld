"""Wikipedia Self-Ingestion v1 runner (safe self-feeding, dry-run only).

Pipeline:
  manifest -> local document reader -> page candidate builder -> ingestion v2
  -> overlay items -> ingestion delta classification -> conflict/risk detection
  -> quarantine -> overlay delta proposal -> temporary dry-run overlay
  -> QA / adversarial / cross-page regression -> deterministic report.

SAFETY CONTRACT:
- Raw documents NEVER flow directly into accepted memory.
- accepted_knowledge_memory_v1.json / sense_memory.py are NOT modified.
- accepted_wiki_memory_overlay_v1.json is NOT overwritten (dry-run is separate).
- Ingestion v2 extraction and overlay-builder semantics are NOT modified.
- Validators are NOT weakened. No generic fallback. No neural/GPT/training/
  embedding code. No network. nanogpt/ untouched.
- safe_for_general_runtime is always False.

Usage::

    python3 -m worldpgt.experiments.run_wikipedia_self_ingestion_v1 \\
      --manifest worldpgt/experiments/self_ingestion_v1/sources_manifest.json \\
      --existing-overlay worldpgt/experiments/accepted_wiki_memory_overlay_v1.json \\
      --output-dir worldpgt/experiments/self_ingestion_v1
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from worldpgt.experiments import run_cross_page_qa_v1, run_entity_qa_v1
from worldpgt.knowledge.wiki_candidate_overlay_builder import (
    WikiCandidateOverlayBuilder,
    as_dict as overlay_as_dict,
)
from worldpgt.knowledge.wiki_ingestion_v2 import WikiIngestionV2, as_dict as cand_as_dict
from worldpgt.knowledge.wiki_page_reader import WikiPageReader
from worldpgt.self_ingestion.conflict_detector import detect_raw_claim
from worldpgt.self_ingestion.ingestion_delta import classify
from worldpgt.self_ingestion.local_document_reader import read_documents
from worldpgt.self_ingestion.overlay_delta_builder import build_delta
from worldpgt.self_ingestion.quarantine import make_quarantine_item
from worldpgt.self_ingestion.self_ingestion_report import build_report
from worldpgt.self_ingestion.source_manifest import load_manifest
from worldpgt.self_ingestion.types import (
    ACTION_REJECT,
    CONFLICT_EXISTING,
    DUPLICATE_EXISTING,
    NEW_CANDIDATE,
    VOLATILE_REQUIRES_SOURCE,
)
from worldpgt.self_ingestion.wiki_page_candidate_builder import build_pages

_EXPERIMENTS = Path(__file__).parent
_QA_BENCHES = {
    "entity_qa_v1": ("entity", "entity_qa_prompts_v1.csv", 28),
    "entity_qa_expansion_v1": ("entity", "entity_qa_expansion_v1.csv", 111),
    "entity_qa_adversarial_v1": ("entity", "entity_qa_adversarial_v1.csv", 68),
    "cross_page_qa_v1": ("cross", "cross_page_qa_v1.csv", 71),
}


def _page_to_overlay_items(source_id: str, page: dict) -> list[dict]:
    record = WikiPageReader()._parse_page(page)
    ingestion = WikiIngestionV2().run([record])
    candidate_dicts = [cand_as_dict(c) for c in ingestion.candidates]
    built = WikiCandidateOverlayBuilder().build(candidate_dicts)
    return [overlay_as_dict(i) for i in built.items]


def _run_qa(kind: str, qa_csv: str, overlay_path: str, tmpdir: Path) -> dict:
    out_csv = str(tmpdir / "out.csv")
    out_json = str(tmpdir / "summary.json")
    qa_input = str(_EXPERIMENTS / qa_csv)
    if kind == "entity":
        return run_entity_qa_v1.run(qa_input, overlay_path, out_csv, out_json)
    return run_cross_page_qa_v1.run(qa_input, overlay_path, out_csv, out_json)


def run(manifest_path: str, existing_overlay_path: str, output_dir: str) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Manifest (local sources only).
    manifest = load_manifest(manifest_path)
    sources = manifest.sources

    # 2. Read local documents.
    documents = read_documents(sources)
    read_errors = sum(len(d.read_errors) for d in documents)

    # 3. Build structured pages + raw claims.
    pages, raw_claims = build_pages(documents)

    # 4. Ingestion v2 -> overlay items (per page, tagged by source_id).
    tagged_items: list[tuple[str, dict]] = []
    for source_id, page in pages:
        for item in _page_to_overlay_items(source_id, page):
            tagged_items.append((source_id, item))

    # 5. Ingestion delta classification vs existing overlay.
    existing_items = json.loads(Path(existing_overlay_path).read_text(encoding="utf-8"))
    classified, item_reasons = classify(tagged_items, existing_items)

    # 6. Quarantine: raw risky claims + flagged structured items.
    quarantine = []
    qc_idx = 0
    for rc in raw_claims:
        reason = detect_raw_claim(rc.text)
        if reason is not None:
            qc_idx += 1
            quarantine.append(make_quarantine_item(
                f"raw-{qc_idx:03d}", rc.text, reason, rc.source_id))
    for idx, c in enumerate(classified):
        reason = item_reasons.get(idx)
        if reason is not None:
            qc_idx += 1
            label = (c.overlay_item.get("label") or c.overlay_item.get("subject")
                     or c.overlay_item.get("source_page") or "")
            text = f"{c.overlay_item.get('overlay_type')}::{label}"
            quarantine.append(make_quarantine_item(
                f"item-{qc_idx:03d}", text, reason, c.source_id))

    rejected_total = sum(1 for q in quarantine if q.suggested_action == ACTION_REJECT)

    # 7. Overlay delta (safe, additive, benchmark-neutral).
    delta, tainted = build_delta(classified)

    # 8. Dry-run overlay = existing + delta (existing overlay is NOT overwritten).
    dry_run_overlay = list(existing_items) + delta
    dry_run_path = out / "dry_run_overlay.json"
    dry_run_path.write_text(json.dumps(dry_run_overlay, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    # 9. QA / adversarial / cross-page regression against the dry-run overlay.
    qa_status: dict = {}
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for name, (kind, csv_name, expected) in _QA_BENCHES.items():
            summary = _run_qa(kind, csv_name, str(dry_run_path), tmpdir)
            ok = (summary["qa_total"] == expected
                  and summary["correct_count"] == expected
                  and summary["wrong_count"] == 0)
            qa_status[name] = {
                "status": "PASS" if ok else "FAIL",
                "qa_total": summary["qa_total"],
                "correct_count": summary["correct_count"],
                "wrong_count": summary["wrong_count"],
                "expected": expected,
            }

    # 10. Counts + report.
    cls_counts = Counter(c.classification for c in classified)
    high_risk_in_delta = any(d.get("risk") == "high" for d in delta)
    reason_breakdown = dict(Counter(q.reason for q in quarantine))

    notes = [
        "Raw documents never written to accepted memory; dry-run overlay is separate.",
        "Well-formed but volatile source-qualified facts are held for human review "
        "(volatile_requires_source) rather than auto-applied, to avoid changing "
        "runtime aggregates without sign-off.",
        f"Tainted (conflicting) sources excluded entirely from delta: {tainted}.",
    ]

    report = build_report(
        sources_total=len(sources),
        documents_read=len(documents),
        read_errors=read_errors,
        candidates_total=len(tagged_items),
        new_candidates=cls_counts.get(NEW_CANDIDATE, 0),
        duplicate_existing=cls_counts.get(DUPLICATE_EXISTING, 0),
        conflicts=cls_counts.get(CONFLICT_EXISTING, 0),
        quarantined_total=len(quarantine),
        rejected_total=rejected_total,
        overlay_delta_items=len(delta),
        dry_run_overlay_items=len(dry_run_overlay),
        qa_regression_status=qa_status,
        quarantine_reasons=reason_breakdown,
        high_risk_in_delta=high_risk_in_delta,
        notes=notes,
    )

    # 11. Write artifacts.
    (out / "quarantine.json").write_text(
        json.dumps([q.to_dict() for q in quarantine], indent=2, ensure_ascii=False),
        encoding="utf-8")
    (out / "overlay_delta_proposal.json").write_text(
        json.dumps(delta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "self_ingestion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "manifest_errors.json").write_text(
        json.dumps(manifest.errors, indent=2, ensure_ascii=False), encoding="utf-8")

    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Wikipedia Self-Ingestion v1 (dry-run)")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--existing-overlay", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    report = run(args.manifest, args.existing_overlay, args.output_dir)

    print(f"[self_ingest] sources_total:        {report['sources_total']}")
    print(f"[self_ingest] documents_read:       {report['documents_read']}")
    print(f"[self_ingest] candidates_total:     {report['candidates_total']}")
    print(f"[self_ingest] new_candidates:       {report['new_candidates']}")
    print(f"[self_ingest] duplicate_existing:   {report['duplicate_existing']}")
    print(f"[self_ingest] conflicts:            {report['conflicts']}")
    print(f"[self_ingest] quarantined_total:    {report['quarantined_total']}")
    print(f"[self_ingest] rejected_total:       {report['rejected_total']}")
    print(f"[self_ingest] overlay_delta_items:  {report['overlay_delta_items']}")
    print(f"[self_ingest] dry_run_overlay_items:{report['dry_run_overlay_items']}")
    print(f"[self_ingest] quarantine_reasons:   {report['quarantine_reasons']}")
    for name, st in report["qa_regression_status"].items():
        print(f"[self_ingest] QA {name}: {st['status']} "
              f"({st['correct_count']}/{st['qa_total']})")
    print(f"[self_ingest] safe_to_apply_overlay_delta: {report['safe_to_apply_overlay_delta']}")
    print(f"[self_ingest] safe_for_general_runtime:    {report['safe_for_general_runtime']}")

    print("\n[safety] accepted_knowledge_memory_v1.json: NOT modified")
    print("[safety] sense_memory.py: NOT modified")
    print("[safety] accepted_wiki_memory_overlay_v1.json: NOT overwritten")
    print("[safety] ingestion/overlay semantics: NOT modified")
    print("[safety] validators: NOT weakened")
    print("[safety] neural/GPT/training/embedding: NOT used")
    print("[safety] network access: NONE (local fixtures only)")
    print("[safety] nanogpt/: NOT touched")


if __name__ == "__main__":
    main()
