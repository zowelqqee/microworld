"""Working Context Pack v1 probe runner.

Builds read-only working context packs for a fixed probe set over the wiki
overlay (accepted by default; promoted optionally) and writes probe artifacts
under ``worldpgt/experiments/context_pack_v1/``.

SAFETY CONTRACT (enforced by construction — read-only over the overlay; writes
only under ``context_pack_v1/``):

- No trusted memory write. accepted_knowledge_memory_v1.json untouched.
- No overlay write. accepted / promoted overlays untouched (read-only).
- No planner / validator / threshold / ingestion / overlay-builder change.
- No analyzer-rule / extraction-pattern activation. No auto-ingest/promote/apply.
- No neural / GPT / training / embedding / network code. nanogpt/ untouched.
- safe_for_general_runtime is always False; QA behavior is unchanged.

Usage::

    python3 worldpgt/experiments/run_context_pack_probe_v1.py
    python3 worldpgt/experiments/run_context_pack_probe_v1.py --overlay-mode promoted
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

from worldpgt.context_pack.context_pack_builder import (
    ContextPackBuilder,
    load_knowledge_requests,
)
from worldpgt.context_pack.context_pack_renderer import render
from worldpgt.context_pack.context_pack_validator import validate_pack
from worldpgt.context_pack.types import OVERLAY_ACCEPTED, OVERLAY_PROMOTED

_EXPERIMENTS = Path(__file__).resolve().parent
_DEFAULT_OUTDIR = _EXPERIMENTS / "context_pack_v1"
_ACCEPTED_OVERLAY = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_PROMOTED_OVERLAY = (
    _EXPERIMENTS / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json"
)
_KNOWLEDGE_REQUESTS = (
    _EXPERIMENTS / "knowledge_requests_v1" / "candidate_knowledge_requests.json"
)

PROBE_QUESTIONS = [
    "How is Elon Musk connected to rockets?",
    "How is Elon Musk connected to Starlink?",
    "Why is Forbes linked to Elon Musk?",
    "What does Forbes estimate about Elon Musk?",
    "What is Tesla's current stock price?",
    "Did SpaceX found Elon Musk?",
    "Does a weak link prove that Forbes owns Elon Musk?",
    "All rockets are SpaceX products.",
    "What does SpaceX develop?",
    "Who founded SpaceX?",
]

_CSV_FIELDS = [
    "question",
    "overlay_mode",
    "matched_entities_count",
    "definitions_count",
    "direct_relations_count",
    "one_hop_relations_count",
    "two_hop_relations_count",
    "weak_links_count",
    "source_facts_count",
    "candidate_paths_count",
    "blocked_paths_count",
    "missing_knowledge_hints_count",
    "safe_for_answering",
    "safe_for_general_runtime",
    "validation_passed",
    "rendered_summary",
]

_CURRENT_LIVE_WORDS = ("current", "latest", "today", "right now", "stock price", "net worth")


def _overlay_path(mode: str) -> Path:
    return _PROMOTED_OVERLAY if mode == OVERLAY_PROMOTED else _ACCEPTED_OVERLAY


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
        fh.write("\n")


def run_probe(
    overlay_mode: str = OVERLAY_ACCEPTED,
    out_dir: Path = _DEFAULT_OUTDIR,
    questions: List[str] = None,
    write: bool = True,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    questions = questions if questions is not None else PROBE_QUESTIONS

    knowledge_requests = load_knowledge_requests(_KNOWLEDGE_REQUESTS)
    builder = ContextPackBuilder(
        str(_overlay_path(overlay_mode)),
        overlay_mode=overlay_mode,
        knowledge_requests=knowledge_requests,
    )

    packs = []
    csv_rows = []
    json_rows = []
    validation_passed = 0
    for q in questions:
        pack = builder.build(q)
        validation = validate_pack(pack)
        rendered = render(pack)
        if validation.passed:
            validation_passed += 1

        packs.append((pack, validation, rendered))
        json_rows.append(
            {
                "pack": asdict(pack),
                "validation": asdict(validation),
                "rendered_summary": rendered,
            }
        )
        csv_rows.append(
            {
                "question": q,
                "overlay_mode": overlay_mode,
                "matched_entities_count": len(pack.matched_entities),
                "definitions_count": len(pack.definitions),
                "direct_relations_count": len(pack.direct_relations),
                "one_hop_relations_count": len(pack.one_hop_relations),
                "two_hop_relations_count": len(pack.two_hop_relations),
                "weak_links_count": len(pack.weak_links),
                "source_facts_count": len(pack.source_facts),
                "candidate_paths_count": len(pack.candidate_paths),
                "blocked_paths_count": len(pack.blocked_paths),
                "missing_knowledge_hints_count": len(pack.missing_knowledge_hints),
                "safe_for_answering": pack.safe_for_answering,
                "safe_for_general_runtime": pack.safe_for_general_runtime,
                "validation_passed": validation.passed,
                "rendered_summary": rendered,
            }
        )

    summary = _build_summary(overlay_mode, packs, validation_passed)

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "context_pack_probe_outputs.json", json_rows)
        with (out_dir / "context_pack_probe_outputs.csv").open(
            "w", encoding="utf-8", newline=""
        ) as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_rows)
        _write_json(out_dir / "context_pack_probe_summary.json", summary)

    return {"packs": packs, "summary": summary, "csv_rows": csv_rows}


def _build_summary(overlay_mode, packs, validation_passed) -> Dict[str, Any]:
    total = len(packs)
    blocked = sum(1 for p, _, _ in packs if p.blocked_paths)
    safe = sum(1 for p, _, _ in packs if p.safe_for_answering)
    weak_cases = sum(1 for p, _, _ in packs if p.weak_links)
    source_cases = sum(1 for p, _, _ in packs if p.source_facts)
    current_cases = sum(
        1
        for p, _, _ in packs
        if any(w in p.question.lower() for w in _CURRENT_LIVE_WORDS)
    )
    inversion_cases = sum(
        1
        for p, _, _ in packs
        if any(b.reason == "relation_inversion" for b in p.blocked_paths)
    )
    hint_cases = sum(1 for p, _, _ in packs if p.missing_knowledge_hints)
    return {
        "probe_total": total,
        "validation_passed_count": validation_passed,
        "validation_failed_count": total - validation_passed,
        "safe_for_answering_count": safe,
        "blocked_count": blocked,
        "weak_link_cases": weak_cases,
        "source_fact_cases": source_cases,
        "current_live_cases": current_cases,
        "inversion_cases": inversion_cases,
        "missing_knowledge_hint_cases": hint_cases,
        "overlay_mode": overlay_mode,
        "safe_for_general_runtime": False,
        "trusted_memory_modified": False,
        "accepted_overlay_modified": False,
        "promoted_overlay_modified": False,
        "runtime_behavior_modified": False,
        "network_calls": False,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlay-mode",
        choices=[OVERLAY_ACCEPTED, OVERLAY_PROMOTED],
        default=OVERLAY_ACCEPTED,
    )
    parser.add_argument("--out-dir", default=str(_DEFAULT_OUTDIR))
    args = parser.parse_args(argv)

    result = run_probe(overlay_mode=args.overlay_mode, out_dir=Path(args.out_dir))
    s = result["summary"]
    print("Working Context Pack v1 probe")
    print(f"  overlay_mode                 : {s['overlay_mode']}")
    print(f"  probe_total                  : {s['probe_total']}")
    print(f"  validation_passed_count      : {s['validation_passed_count']}")
    print(f"  safe_for_answering_count     : {s['safe_for_answering_count']}")
    print(f"  blocked_count                : {s['blocked_count']}")
    print(f"  weak_link_cases              : {s['weak_link_cases']}")
    print(f"  source_fact_cases            : {s['source_fact_cases']}")
    print(f"  current_live_cases           : {s['current_live_cases']}")
    print(f"  inversion_cases              : {s['inversion_cases']}")
    print(f"  missing_knowledge_hint_cases : {s['missing_knowledge_hint_cases']}")
    print(f"  safe_for_general_runtime={s['safe_for_general_runtime']} "
          f"network_calls={s['network_calls']}")
    print(f"  trusted_memory_modified={s['trusted_memory_modified']} "
          f"accepted_overlay_modified={s['accepted_overlay_modified']} "
          f"promoted_overlay_modified={s['promoted_overlay_modified']}")
    print(f"  artifacts written to         : {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
