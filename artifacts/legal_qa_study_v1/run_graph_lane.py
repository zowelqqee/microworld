"""Run the pre-registered question set through the legal QA lane (system G2)."""

from __future__ import annotations

import json
from pathlib import Path

from worldpgt.legal_qa.legal_answer_planner import plan
from worldpgt.legal_qa.legal_question_analyzer import analyze

HERE = Path(__file__).resolve().parent


def main() -> int:
    questions = json.loads((HERE / "questions.json").read_text())
    items = json.loads((HERE / "legal_overlay.json").read_text())
    rows = []
    for q in questions:
        analyzed = analyze(q["question"])
        result = plan(analyzed, items)
        rows.append({
            **q, "system": "G2",
            "shape": analyzed.shape,
            "focus": analyzed.focus,
            "decision": result.decision,
            "answer": result.text,
            "audit_reason": result.audit_reason,
            "citations": result.citations,
            "guards_rendered": result.guards_rendered,
        })
        print(f"{q['id']} [{q['stratum']}] {result.decision:6s} {(result.text or result.audit_reason)[:88]}",
              flush=True)

    (HERE / "results_graph_lane.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    answered = sum(1 for r in rows if r["decision"] != "audit")
    print(f"\nanswered={answered}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
