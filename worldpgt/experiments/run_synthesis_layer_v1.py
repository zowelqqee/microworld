"""Synthesis Layer (layer 3) demonstration over the wiki memory overlay.

Layer 1/2 find individual facts. Layer 3 synthesizes an answer to an *open*
question ("Tell me about SpaceX") from every relevant fact in the graph, grouped
by type, with three confidence tiers:

    VERIFIED — stable / semi_stable relations, definitions, is_a chains.
    SNAPSHOT — dated, source-qualified estimates (net worth as_of a date).
    UNKNOWN  — parts of the question the graph cannot answer.

The system never invents facts: anything not in the graph is reported as
UNKNOWN, explicitly.

SAFETY CONTRACT (identical to run_entity_qa_v1):
- accepted_knowledge_memory_v1.json is NOT modified.
- sense_memory.py is NOT modified.
- No neural/GPT/training/embedding code. No network access.

Usage::

    python3 -m worldpgt.experiments.run_synthesis_layer_v1
    python3 -m worldpgt.experiments.run_synthesis_layer_v1 --question "Tell me about Tesla."
"""

from __future__ import annotations

import argparse
from pathlib import Path

from worldpgt.entity_qa.entity_answer_planner import EntityAnswerPlanner
from worldpgt.entity_qa.entity_answer_renderer import render
from worldpgt.entity_qa.entity_question_analyzer import analyze
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider

_EXPERIMENTS = Path(__file__).resolve().parent
_DEFAULT_OVERLAY = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"

_DEMO_QUESTIONS = [
    "Tell me about SpaceX.",
    "Tell me about Elon Musk.",
    "What do you know about Blue Origin?",
    "How does Starlink work?",
]


def _answer_one(planner: EntityAnswerPlanner, question: str) -> None:
    analyzed = analyze(question)
    plan = planner.plan(analyzed)
    answer = render(plan)
    print(f"Q: {question}")
    print(f"   [intent={analyzed.intent} decision={plan.decision}]")
    print()
    for line in answer.splitlines():
        print(f"A: {line}" if line and not line.startswith("[") else f"   {line}")
    print()
    print("=" * 72)


def run(overlay_path: str, questions: list[str]) -> None:
    provider = WikiMemoryOverlayProvider(overlay_path)
    planner = EntityAnswerPlanner(provider=provider)
    for question in questions:
        _answer_one(planner, question)
    print("[safety] accepted_knowledge_memory_v1.json: NOT modified")
    print("[safety] sense_memory.py: NOT modified")
    print("[safety] neural/GPT/training/embedding: NOT used")
    print("[safety] network access: NONE")
    print("[safety] facts synthesized only from verified overlay items")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Synthesis layer v1 demo")
    parser.add_argument("--overlay-json", default=str(_DEFAULT_OVERLAY))
    parser.add_argument(
        "--question",
        action="append",
        help="Ask a specific open question (repeatable). Defaults to the demo set.",
    )
    args = parser.parse_args(argv)
    run(args.overlay_json, args.question or _DEMO_QUESTIONS)


if __name__ == "__main__":
    main()
