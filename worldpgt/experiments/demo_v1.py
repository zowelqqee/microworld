"""
Microworld Interactive Demo
Показывает все возможности системы с think-aloud по умолчанию.
Запуск: python3 worldpgt/experiments/demo_v1.py
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
from worldpgt.assistant_surface.think_aloud import build_think_aloud
from worldpgt.assistant_surface.types import OVERLAY_MODE_CUSTOM_PATH, OVERLAY_MODES
from worldpgt.dialogue.conversation_context import ConversationContext
from worldpgt.dialogue.coreference_resolver import resolve_coreferences
from worldpgt.dialogue.followup_rewriter import rewrite_followup
from worldpgt.entity_qa.semantic_question_parser import parse_semantic_query
from worldpgt.experiments.ask_microworld_v1 import (
    _load_overlay_items,
    _record_turn,
    _surface_index_for_dialogue,
    _unresolved_reference_answer,
)
from worldpgt.multihop_qa.assistant_adapter import try_answer_multihop


DEMO_QUESTIONS = [
    ("What is SpaceX?", "Direct definition lookup"),
    ("Who founded it?", "Coreference: 'it' -> SpaceX from context"),
    ("Is he a businessman?", "Ontology traversal: Musk -> businessman -> worker"),
    ("What firms did he kick off?", "Embedding: 'kick off' -> founded_by"),
    ("How is Starlink connected to Falcon 9?", "2-hop chain traversal"),
    ("How many companies did Elon Musk found?", "Count aggregation"),
    ("Who founded more - Musk or Bezos?", "Comparative reasoning"),
    ("Is SpaceX a person?", "Type contradiction -> Decision: no"),
    ("Tell me about Tesla Energy.", "Open synthesis with VERIFIED/INFERRED tiers"),
    ("What is the current CEO of SpaceX?", "Volatile data -> Decision: audit"),
]


@dataclass
class DemoTurn:
    question: str
    effective_question: str
    label: str
    thinking: str
    answer: str
    decision: str
    support_kind: str


def _gray(text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[90m{text}\033[0m"


def _run_question(
    *,
    question: str,
    label: str,
    orchestrator: AnswerOrchestrator,
    context: ConversationContext,
    overlay_mode: str,
    overlay_path: str | None,
    overlay_items: list[dict],
):
    index = _surface_index_for_dialogue(overlay_mode, overlay_path)
    effective_overlay_mode = OVERLAY_MODE_CUSTOM_PATH if overlay_path is not None else overlay_mode

    resolution = resolve_coreferences(question, context, index)
    followup = rewrite_followup(resolution.resolved_question, context, index)
    effective_question = followup.resolved_question
    semantic_query = parse_semantic_query(effective_question, index)

    if (
        resolution.unresolved_reference is not None
        and semantic_query.entity_a is None
        and semantic_query.entity_b is None
    ):
        answer = _unresolved_reference_answer(
            question,
            effective_overlay_mode,
            resolution.unresolved_reference,
        )
    else:
        answer = orchestrator.answer(
            effective_question,
            answer_style=followup.answer_style,
        )

    multihop_result = None
    if answer.decision == "audit" or answer.support_kind == "explicit_connection_path":
        candidate = try_answer_multihop(effective_question, overlay_items)
        if candidate.decision == "answer":
            multihop_result = candidate

    subject = semantic_query.entity_a
    if subject is None and answer.trace and answer.trace.context_summary:
        matched = answer.trace.context_summary.get("matched_entities") or []
        subject = matched[0] if matched else None
    think = build_think_aloud(
        answer,
        question=effective_question,
        subject=subject,
        multihop_result=multihop_result,
    )

    _record_turn(
        context,
        question=effective_question,
        semantic_query=semantic_query,
        answer=answer,
        index=index,
    )

    return DemoTurn(
        question=question,
        effective_question=effective_question,
        label=label,
        thinking=think.thinking,
        answer=think.answer,
        decision=multihop_result.decision if multihop_result is not None else answer.decision,
        support_kind=(
            multihop_result.support_kind if multihop_result is not None else answer.support_kind
        ),
    )


def _print_turn(index: int, turn: DemoTurn) -> None:
    print("=" * 78)
    print(f"{index}. {turn.label}")
    print(f"Q: {turn.question}")
    if turn.effective_question != turn.question:
        print(f"Resolved Q: {turn.effective_question}")
    print("")
    print(_gray("THINKING\n" + turn.thinking))
    print("")
    print("ANSWER")
    print(turn.answer)
    print("")
    print(f"Decision: {turn.decision}. Support: {turn.support_kind}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Microworld interactive demo v1.")
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Run without the default 2 second pause between questions.",
    )
    parser.add_argument(
        "--overlay",
        choices=list(OVERLAY_MODES),
        default="pump-dry-run",
        help="Overlay mode to use for the demo.",
    )
    parser.add_argument(
        "--overlay-path",
        default=None,
        help="Advanced: read one explicit overlay JSON path as a custom overlay.",
    )
    parser.add_argument(
        "--ontology-layer",
        default=None,
        help="Optional read-only ontology layer JSON path for is_a traversal.",
    )
    args = parser.parse_args(argv)

    if args.overlay_path is not None:
        path = Path(args.overlay_path)
        if not path.is_file():
            parser.error(f"--overlay-path does not exist or is not a file: {path}")
        overlay_path = str(path)
    else:
        overlay_path = None

    if args.ontology_layer is not None and not Path(args.ontology_layer).is_file():
        parser.error(f"--ontology-layer does not exist or is not a file: {args.ontology_layer}")

    orchestrator = AnswerOrchestrator(
        args.overlay,
        overlay_path=overlay_path,
        ontology_layer_path=args.ontology_layer,
    )
    context = ConversationContext()
    overlay_items = _load_overlay_items(args.overlay, overlay_path)

    print("Microworld Interactive Demo")
    print(f"Overlay: {OVERLAY_MODE_CUSTOM_PATH if overlay_path is not None else args.overlay}")
    if overlay_path is not None:
        print("Overlay mode: custom overlay path, not accepted memory.")
    elif args.overlay == "pump-dry-run":
        print("Overlay mode: pump-dry-run proposal, not accepted memory.")
    elif args.overlay == "snapshot-dry-run":
        print("Overlay mode: snapshot-dry-run proposal, not accepted memory.")
    print("")

    for idx, (question, label) in enumerate(DEMO_QUESTIONS, start=1):
        turn = _run_question(
            question=question,
            label=label,
            orchestrator=orchestrator,
            context=context,
            overlay_mode=args.overlay,
            overlay_path=overlay_path,
            overlay_items=overlay_items,
        )
        _print_turn(idx, turn)
        if not args.no_pause and idx != len(DEMO_QUESTIONS):
            time.sleep(2.0)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
