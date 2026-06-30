"""Schema induction + QA demo over two unrelated domains.

Proves the pipeline needs NO predefined domain predicates: it induces relation
families and answers questions for both visas/immigration and animal movement
from the same generic mechanics.

Run:

    python3 worldpgt/experiments/run_schema_pump_demo_v1.py
"""

from __future__ import annotations

from pathlib import Path

from worldpgt.schema_induction import schema_store
from worldpgt.schema_induction.promotion_gates import GateConfig
from worldpgt.schema_induction.run_schema_induction import run_induction
from worldpgt.schema_induction.schema_qa_adapter import SchemaQAAdapter

_HERE = Path(__file__).resolve().parent
_DEMO_DIR = _HERE / "schema_induction_v1"
_DOCS = _DEMO_DIR / "demo_docs.jsonl"
_OUT = _DEMO_DIR / "demo_run"

_QUESTIONS = [
    "What does Portugal D7 visa require?",
    "Что нужно для Portugal D7 visa?",
    "What does Spain non-lucrative visa prohibit?",
    "Why do giraffes move seasonally?",
    "Куда мигрируют wildebeest?",
    "Tell me about Digital nomad visa.",
    "Расскажи про giraffes.",
    "Who founded Portugal D7 visa?",  # should audit
]


def _hr(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def main() -> int:
    raw_docs = schema_store.read_docs_jsonl(_DOCS)
    cfg = GateConfig(min_evidence=2, min_sources=1)
    result = run_induction(raw_docs, cfg)
    schema_store.write_result(result, _OUT)

    _hr("GENERATED RELATION FAMILIES")
    for fam in result.families:
        print(
            f"- [{fam.promotion_status:9s}] {fam.canonical_label!r} "
            f"roles={list(fam.roles)} surface_forms={list(fam.surface_forms)} "
            f"evidence={fam.evidence_count} sources={fam.source_doc_count}"
        )

    _hr("GENERATED LOCAL TYPES")
    for lt in result.local_types:
        print(f"- {lt.label!r} members={list(lt.members)} "
              f"context={list(lt.context_terms)} conf={lt.confidence}")

    _hr("PROMOTION DECISIONS")
    for d in result.decisions:
        flag = "PROMOTED" if d.status == "promoted" else d.status.upper()
        extra = f" reason={d.reason}" if d.reason else ""
        print(f"- {d.target_kind} {d.target_id}: {flag}{extra}")

    # QA over generated schema (allow_generated so generated-only families
    # answer too; promoted families always answer).
    adapter = SchemaQAAdapter.from_result(result, allow_generated=True)

    _hr("QA EXAMPLES")
    for q in _QUESTIONS:
        ans = adapter.answer(q)
        print(f"\nQ: {q}")
        print(f"   decision={ans.decision} tier={ans.tier}")
        for line in ans.text.splitlines():
            print(f"   {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
