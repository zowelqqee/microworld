"""Run the pre-registered question set through the real MicroWorld serving path."""

from __future__ import annotations

import json
from pathlib import Path

from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator

HERE = Path(__file__).resolve().parent


def main() -> int:
    questions = json.loads((HERE / "questions.json").read_text())
    orchestrator = AnswerOrchestrator(
        "promoted", overlay_path=str(HERE / "legal_overlay.json")
    )
    rows = []
    for q in questions:
        answer = orchestrator.answer(
            q["question"],
            web_search_enabled=False,
            community_context_enabled=False,
            cognitive_patterns_enabled=False,
        )
        text = (
            getattr(answer, "answer_text", None)
            or getattr(answer, "text", "")
            or getattr(answer, "answer", "")
        )
        trace = getattr(answer, "trace", None)
        rows.append({
            **q,
            "system": "G",
            "decision": answer.decision,
            "answer": str(text),
            "route_intent": str(getattr(trace, "route_intent", "") or ""),
            "support_kind": str(getattr(answer, "support_kind", "") or ""),
        })
        print(f"{q['id']} [{q['stratum']}] {answer.decision:7s} {str(text)[:90]}", flush=True)

    (HERE / "results_graph.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    answered = sum(1 for r in rows if r["decision"] != "audit")
    print(f"\nanswered={answered}/{len(rows)}  audited={len(rows) - answered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
