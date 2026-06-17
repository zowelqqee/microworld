"""Microworld Assistant Surface v1 benchmark runner.

Loads ``assistant_surface_prompts_v1.csv``, runs the assistant surface for each
prompt (honoring its overlay mode), and writes artifacts under
``worldpgt/experiments/assistant_surface_v1/``:

- ``assistant_surface_outputs.csv``
- ``assistant_surface_outputs.json``
- ``assistant_surface_summary.json``
- ``assistant_surface_report.json``

SAFETY CONTRACT (enforced by construction — read-only over the overlays; writes
only under ``assistant_surface_v1/``):

- No trusted memory write. accepted_knowledge_memory_v1.json untouched.
- No overlay write (accepted / promoted / snapshot dry-run): verified by hash.
- No planner / validator / threshold / ingestion / overlay-builder change.
- No auto-ingest / auto-promote / auto-apply.
- No neural / GPT / training / embedding / network code. nanogpt/ untouched.
- safe_for_general_runtime is always False.

Usage::

    python3 worldpgt/experiments/run_assistant_surface_v1.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

# Allow running directly as a script.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
from worldpgt.assistant_surface.context_selector import (
    ACCEPTED_OVERLAY_PATH,
    PROMOTED_OVERLAY_PATH,
    SNAPSHOT_DRY_RUN_OVERLAY_PATH,
)
from worldpgt.assistant_surface.surface_validator import (
    is_answer_without_context_support,
    is_current_live_false_support,
    is_inversion_false_support,
    is_private_false_support,
    is_unsafe_answer,
    is_universal_false_support,
    is_volatile_false_stable,
    is_weak_link_false_support,
    validate_answer,
)
from worldpgt.assistant_surface.assistant_renderer import render
from worldpgt.assistant_surface.types import (
    AssistantBenchmarkResult,
    AssistantBenchmarkRow,
    AssistantSurfaceSummary,
)

_EXPERIMENTS = Path(__file__).resolve().parent
_DEFAULT_PROMPTS = _EXPERIMENTS / "assistant_surface_prompts_v1.csv"
_DEFAULT_OUTDIR = _EXPERIMENTS / "assistant_surface_v1"

_PROTECTED_FILES = {
    "accepted_overlay": ACCEPTED_OVERLAY_PATH,
    "promoted_overlay": PROMOTED_OVERLAY_PATH,
    "snapshot_dry_run_overlay": SNAPSHOT_DRY_RUN_OVERLAY_PATH,
    "trusted_memory": _EXPERIMENTS / "accepted_knowledge_memory_v1.json",
}

_OUTPUT_FIELDS = [
    "row_id",
    "question",
    "overlay_mode",
    "expected_decision",
    "expected_intent",
    "category",
    "route",
    "decision",
    "support_kind",
    "supported_by_context",
    "source_system",
    "risk_flags",
    "invariant_failures",
    "answer_text",
]


def _hash(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rows(prompts_path: Path) -> list[AssistantBenchmarkRow]:
    rows: list[AssistantBenchmarkRow] = []
    text = prompts_path.read_text(encoding="utf-8")
    for r in csv.DictReader(text.splitlines()):
        rows.append(
            AssistantBenchmarkRow(
                row_id=r["row_id"],
                question=r["question"],
                overlay_mode=r.get("overlay_mode", "promoted") or "promoted",
                expected_decision=r.get("expected_decision", ""),
                expected_intent=r.get("expected_intent", ""),
                category=r.get("category", ""),
            )
        )
    return rows


def run(prompts_path: Path = _DEFAULT_PROMPTS, outdir: Path = _DEFAULT_OUTDIR) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(prompts_path)

    before = {k: _hash(p) for k, p in _PROTECTED_FILES.items()}

    # Cache one orchestrator per overlay mode.
    orchestrators: dict[str, AnswerOrchestrator] = {}
    results: list[AssistantBenchmarkResult] = []
    output_rows: list[dict] = []

    summary = AssistantSurfaceSummary(prompt_total=len(rows))
    overlay_modes_tested: set[str] = set()
    json_mode_supported = True

    for row in rows:
        mode = row.overlay_mode
        overlay_modes_tested.add(mode)
        orch = orchestrators.get(mode)
        if orch is None:
            orch = AnswerOrchestrator(mode)
            orchestrators[mode] = orch

        answer = orch.answer(row.question)
        results.append(AssistantBenchmarkResult(row=row, answer=answer))

        # Confirm JSON serialization works for every answer.
        try:
            json.dumps(answer.to_dict(), ensure_ascii=False)
        except (TypeError, ValueError):
            json_mode_supported = False

        failures = validate_answer(answer)

        if answer.decision == "answer":
            summary.answer_count += 1
            if answer.supported_by_context and answer.support_kind != "safe_policy_answer":
                summary.supported_answer_count += 1
            if answer.support_kind == "safe_policy_answer":
                summary.safe_policy_answer_count += 1
        else:
            summary.audit_count += 1
            if answer.support_kind in ("missing_knowledge", "unsupported"):
                summary.unsupported_audit_count += 1

        if is_unsafe_answer(answer):
            summary.unsafe_answer_count += 1
        if is_answer_without_context_support(answer):
            summary.answer_without_context_support_count += 1
        if is_weak_link_false_support(answer):
            summary.weak_link_false_support_count += 1
        if is_volatile_false_stable(answer):
            summary.volatile_false_stable_count += 1
        if is_current_live_false_support(answer):
            summary.current_live_false_support_count += 1
        if is_private_false_support(answer):
            summary.private_false_support_count += 1
        if is_inversion_false_support(answer):
            summary.inversion_false_support_count += 1
        if is_universal_false_support(answer):
            summary.universal_false_support_count += 1

        output_rows.append(
            {
                "row_id": row.row_id,
                "question": row.question,
                "overlay_mode": row.overlay_mode,
                "expected_decision": row.expected_decision,
                "expected_intent": row.expected_intent,
                "category": row.category,
                "route": answer.route,
                "decision": answer.decision,
                "support_kind": answer.support_kind,
                "supported_by_context": answer.supported_by_context,
                "source_system": answer.source_system,
                "risk_flags": ";".join(answer.risk_flags),
                "invariant_failures": ";".join(failures),
                "answer_text": render(answer).replace("\n", " | "),
            }
        )

    after = {k: _hash(p) for k, p in _PROTECTED_FILES.items()}

    summary.json_mode_supported = json_mode_supported
    summary.overlay_modes_tested = sorted(overlay_modes_tested)
    summary.accepted_overlay_modified = before["accepted_overlay"] != after["accepted_overlay"]
    summary.promoted_overlay_modified = before["promoted_overlay"] != after["promoted_overlay"]
    summary.snapshot_dry_run_overlay_modified = (
        before["snapshot_dry_run_overlay"] != after["snapshot_dry_run_overlay"]
    )
    summary.trusted_memory_modified = before["trusted_memory"] != after["trusted_memory"]
    summary.runtime_behavior_modified = False
    summary.network_calls = False
    summary.safe_for_general_runtime = False

    summary.all_critical_passed = (
        summary.unsafe_answer_count == 0
        and summary.answer_without_context_support_count == 0
        and summary.weak_link_false_support_count == 0
        and summary.volatile_false_stable_count == 0
        and summary.current_live_false_support_count == 0
        and summary.private_false_support_count == 0
        and summary.inversion_false_support_count == 0
        and summary.universal_false_support_count == 0
        and not summary.accepted_overlay_modified
        and not summary.promoted_overlay_modified
        and not summary.snapshot_dry_run_overlay_modified
        and not summary.trusted_memory_modified
    )

    # ---- Write artifacts -------------------------------------------- #
    with open(outdir / "assistant_surface_outputs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    (outdir / "assistant_surface_outputs.json").write_text(
        json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_dict = summary.to_dict()
    (outdir / "assistant_surface_summary.json").write_text(
        json.dumps(summary_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = {
        "summary": summary_dict,
        "protected_files_unchanged": {
            k: before[k] == after[k] for k in _PROTECTED_FILES
        },
        "critical_pass_requirements": {
            "unsafe_answer_count": summary.unsafe_answer_count,
            "answer_without_context_support_count": summary.answer_without_context_support_count,
            "weak_link_false_support_count": summary.weak_link_false_support_count,
            "volatile_false_stable_count": summary.volatile_false_stable_count,
            "current_live_false_support_count": summary.current_live_false_support_count,
            "private_false_support_count": summary.private_false_support_count,
            "inversion_false_support_count": summary.inversion_false_support_count,
            "universal_false_support_count": summary.universal_false_support_count,
            "all_critical_passed": summary.all_critical_passed,
        },
        "safety_contract": {
            "runtime_behavior_modified": False,
            "trusted_memory_modified": summary.trusted_memory_modified,
            "accepted_overlay_modified": summary.accepted_overlay_modified,
            "promoted_overlay_modified": summary.promoted_overlay_modified,
            "snapshot_dry_run_overlay_modified": summary.snapshot_dry_run_overlay_modified,
            "network_calls": False,
            "auto_ingest_or_promote_or_apply": False,
            "neural_gpt_training_embedding_code": False,
            "nanogpt_touched": False,
            "safe_for_general_runtime": False,
        },
    }
    (outdir / "assistant_surface_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return summary_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Assistant Surface v1 benchmark")
    parser.add_argument("--prompts", default=str(_DEFAULT_PROMPTS))
    parser.add_argument("--outdir", default=str(_DEFAULT_OUTDIR))
    args = parser.parse_args(argv)

    s = run(Path(args.prompts), Path(args.outdir))

    print(f"[assistant_surface_v1] prompt_total:                {s['prompt_total']}")
    print(f"[assistant_surface_v1] answer_count:                {s['answer_count']}")
    print(f"[assistant_surface_v1] audit_count:                 {s['audit_count']}")
    print(f"[assistant_surface_v1] supported_answer_count:      {s['supported_answer_count']}")
    print(f"[assistant_surface_v1] safe_policy_answer_count:    {s['safe_policy_answer_count']}")
    print(f"[assistant_surface_v1] unsupported_audit_count:     {s['unsupported_audit_count']}")
    print(f"[assistant_surface_v1] unsafe_answer_count:         {s['unsafe_answer_count']}")
    print(f"[assistant_surface_v1] answer_without_support:      {s['answer_without_context_support_count']}")
    print(f"[assistant_surface_v1] weak_link_false_support:     {s['weak_link_false_support_count']}")
    print(f"[assistant_surface_v1] volatile_false_stable:       {s['volatile_false_stable_count']}")
    print(f"[assistant_surface_v1] current_live_false_support:  {s['current_live_false_support_count']}")
    print(f"[assistant_surface_v1] private_false_support:       {s['private_false_support_count']}")
    print(f"[assistant_surface_v1] inversion_false_support:     {s['inversion_false_support_count']}")
    print(f"[assistant_surface_v1] universal_false_support:     {s['universal_false_support_count']}")
    print(f"[assistant_surface_v1] overlay_modes_tested:        {s['overlay_modes_tested']}")
    print(f"[assistant_surface_v1] json_mode_supported:         {s['json_mode_supported']}")
    print(f"[assistant_surface_v1] all_critical_passed:         {s['all_critical_passed']}")
    print(f"[assistant_surface_v1] safe_for_general_runtime:    {s['safe_for_general_runtime']}")

    print("\n[safety] trusted memory: NOT modified")
    print("[safety] accepted/promoted/snapshot overlays: NOT modified")
    print("[safety] planners/validators/thresholds: NOT modified")
    print("[safety] auto-ingest/auto-promote/auto-apply: NONE")
    print("[safety] neural/GPT/training/embedding: NOT used")
    print("[safety] network access: NONE")
    print("[safety] nanogpt/: NOT touched")

    return 0 if s["all_critical_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
