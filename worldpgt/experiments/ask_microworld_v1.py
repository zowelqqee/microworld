"""Microworld Assistant Surface v1 — user-facing CLI entrypoint.

A single controlled assistant surface over Microworld's existing explicit
memory, overlays, context packs, and safety gates. It routes a question to the
safest existing deterministic QA path and returns a normal user-facing answer,
or audits honestly when no supported answer exists.

This is NOT GPT, NOT a free-form generator, NOT runtime web search, NOT neural
inference. No training, embeddings, backprop, torch, or network access. It never
writes accepted memory, accepted overlay, promoted overlay, or the snapshot
dry-run overlay, and never changes runtime behavior of existing QA planners.

Usage::

    python3 worldpgt/experiments/ask_microworld_v1.py "How is Elon Musk connected to rockets?"
    python3 worldpgt/experiments/ask_microworld_v1.py "What does SpaceX develop?" --overlay promoted
    python3 worldpgt/experiments/ask_microworld_v1.py "What does SpaceX develop?" --overlay snapshot-dry-run
    python3 worldpgt/experiments/ask_microworld_v1.py "What is Starlink?" --overlay pump-dry-run
    python3 worldpgt/experiments/ask_microworld_v1.py "What is Starlink?" --overlay-path worldpgt/experiments/knowledge_pump_v1/pump_dry_run_overlay.json
    python3 worldpgt/experiments/ask_microworld_v1.py "What is Tesla's current stock price?"
    python3 worldpgt/experiments/ask_microworld_v1.py "How is Starlink connected to Elon Musk?" --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running directly as a script.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
from worldpgt.assistant_surface.context_selector import resolve_overlay
from worldpgt.assistant_surface.assistant_renderer import render
from worldpgt.assistant_surface.think_aloud import build_think_aloud, select_inferred_facts
from worldpgt.cognition.inference_engine import run_inference
from worldpgt.assistant_surface.types import (
    AssistantAnswer,
    OVERLAY_MODE_CUSTOM_PATH,
    OVERLAY_MODE_PUMP_DRY_RUN,
    OVERLAY_MODE_SNAPSHOT_DRY_RUN,
    OVERLAY_MODES,
)
from worldpgt.dialogue.conversation_context import ConversationContext, ConversationTurn
from worldpgt.dialogue.coreference_resolver import (
    CoreferenceResolution,
    resolve_coreferences,
)
from worldpgt.dialogue.followup_rewriter import rewrite_followup
from worldpgt.entity_qa.semantic_question_parser import parse_semantic_query
from worldpgt.entity_qa.types import SemanticQuery
from worldpgt.multihop_qa.assistant_adapter import (
    render_cli_multihop_result,
    try_answer_multihop,
)
from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex

_SNAPSHOT_MARKER = "Overlay mode: snapshot-dry-run proposal, not accepted memory."
_PUMP_MARKER = "Overlay mode: pump-dry-run proposal, not accepted memory."
_CUSTOM_MARKER = "Overlay mode: custom overlay path, not accepted memory."

_ACCEPTED_OVERLAY_PATH = _ROOT / "worldpgt" / "experiments" / "accepted_wiki_memory_overlay_v1.json"
_PROMOTED_OVERLAY_PATH = (
    _ROOT
    / "worldpgt"
    / "experiments"
    / "self_ingestion_v1"
    / "promotion"
    / "promoted_wiki_memory_overlay_v1.json"
)
_SNAPSHOT_OVERLAY_PATH = (
    _ROOT / "worldpgt" / "experiments" / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json"
)


def ask(
    question: str,
    overlay_mode: str = "promoted",
    overlay_path: str | None = None,
    ontology_layer_path: str | None = None,
):
    """Run the assistant surface for one question; returns an AssistantAnswer."""

    orchestrator = AnswerOrchestrator(
        overlay_mode,
        overlay_path=overlay_path,
        ontology_layer_path=ontology_layer_path,
    )
    return orchestrator.answer(question)


def _surface_index_for_dialogue(
    overlay_mode: str = "promoted",
    overlay_path: str | None = None,
) -> EntitySurfaceIndex:
    active_path = Path(overlay_path or resolve_overlay(overlay_mode)[0])
    return EntitySurfaceIndex(
        accepted_overlay_path=_ACCEPTED_OVERLAY_PATH,
        promoted_overlay_path=active_path,
        snapshot_overlay_path=_SNAPSHOT_OVERLAY_PATH,
    )


def _load_overlay_items(overlay_mode: str, overlay_path: str | None = None) -> list[dict[str, Any]]:
    path = Path(overlay_path) if overlay_path is not None else Path(resolve_overlay(overlay_mode)[0])
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _overlay_marker(overlay_mode: str) -> str:
    return {
        OVERLAY_MODE_SNAPSHOT_DRY_RUN: _SNAPSHOT_MARKER,
        OVERLAY_MODE_PUMP_DRY_RUN: _PUMP_MARKER,
        OVERLAY_MODE_CUSTOM_PATH: _CUSTOM_MARKER,
    }.get(overlay_mode, "")


def _dedupe_entities(entities: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for entity in entities:
        if not entity or entity in seen:
            continue
        seen.add(entity)
        out.append(entity)
    return out


def _mentioned_entities(
    answer: AssistantAnswer,
    semantic_query: SemanticQuery,
    index: EntitySurfaceIndex,
) -> list[str]:
    entities = [
        canonical
        for _surface, canonical, _start, _end in index.find_in_text(answer.answer_text)
    ]
    if answer.trace and answer.trace.context_summary:
        entities.extend(answer.trace.context_summary.get("matched_entities", []))
    entities.extend([semantic_query.entity_a or "", semantic_query.entity_b or ""])
    return _dedupe_entities(entities)


def _primary_entity(
    answer: AssistantAnswer,
    semantic_query: SemanticQuery,
    mentioned_entities: list[str],
) -> str | None:
    if answer.decision == "audit":
        return None
    if semantic_query.entity_a:
        return semantic_query.entity_a
    return mentioned_entities[0] if mentioned_entities else None


def _record_turn(
    context: ConversationContext,
    *,
    question: str,
    semantic_query: SemanticQuery,
    answer: AssistantAnswer,
    index: EntitySurfaceIndex,
) -> None:
    mentioned = [] if answer.decision == "audit" else _mentioned_entities(answer, semantic_query, index)
    primary = _primary_entity(answer, semantic_query, mentioned)
    if answer.decision == "no" and primary:
        mentioned = _dedupe_entities([primary, *mentioned])
    context.turns.append(
        ConversationTurn(
            question=question,
            semantic_query=semantic_query,
            decision=answer.decision,
            primary_entity=primary,
            mentioned_entities=mentioned,
            relation_type=semantic_query.relation_intent,
        )
    )


def _unresolved_reference_answer(
    question: str,
    overlay_mode: str,
    reference: str,
) -> AssistantAnswer:
    return AssistantAnswer(
        question=question,
        decision="audit",
        route="unknown_or_unsupported",
        answer_text=(
            f"unresolved_reference: could not determine what {reference!r} refers to"
        ),
        overlay_mode=overlay_mode,
        supported_by_context=False,
        support_kind="missing_knowledge",
        source_system="dialogue_context",
        safe_for_general_runtime=False,
    )


def _print_resolution(resolution: CoreferenceResolution) -> None:
    for replacement in resolution.replacements:
        print(f"[Resolved {replacement.reference!r} → {replacement.display_target}]")


def _run_interactive(
    *,
    overlay_mode: str,
    overlay_path: str | None,
    ontology_layer_path: str | None,
    json_mode: bool,
    show_semantic_query: bool,
) -> int:
    orchestrator = AnswerOrchestrator(
        overlay_mode,
        overlay_path=overlay_path,
        ontology_layer_path=ontology_layer_path,
    )
    effective_overlay_mode = OVERLAY_MODE_CUSTOM_PATH if overlay_path is not None else overlay_mode
    index = _surface_index_for_dialogue(overlay_mode, overlay_path)
    context = ConversationContext()

    while True:
        try:
            question = input("Q: ").strip()
        except EOFError:
            print("")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", ":q"}:
            break

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

        if json_mode:
            payload = answer.to_dict()
            payload["dialogue"] = {
                "original_question": question,
                "resolved_question": effective_question,
                "resolved_references": [
                    {
                        "reference": item.reference,
                        "entities": list(item.entities),
                    }
                    for item in resolution.replacements
                ],
                "unresolved_reference": resolution.unresolved_reference,
                "followup_rewrite": {
                    "resolved_question": followup.resolved_question,
                    "answer_style": followup.answer_style,
                    "reason": followup.reason,
                },
                "semantic_query": semantic_query.to_dict(),
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            _print_resolution(resolution)
            if show_semantic_query:
                print("SemanticQuery:")
                print(json.dumps(semantic_query.to_dict(), indent=2, ensure_ascii=False))
                print("")
            print(render(answer))

        _record_turn(
            context,
            question=question,
            semantic_query=semantic_query,
            answer=answer,
            index=index,
        )
    return 0


def _think_aloud_for(
    question: str,
    answer: AssistantAnswer,
    overlay_mode: str,
    overlay_path: str | None,
    multihop_result=None,
):
    """Build the think-aloud surface for one answered question (read-only)."""
    items = _load_overlay_items(overlay_mode, overlay_path)
    workspace = run_inference(items)
    cs = (answer.trace.context_summary if answer.trace else None) or {}
    matched = cs.get("matched_entities") or []
    subject = matched[0] if matched else (answer.question or question)
    inferred = (
        select_inferred_facts(workspace, subject, question)
        if answer.decision == "audit"
        else []
    )
    return build_think_aloud(
        answer,
        question=question,
        subject=subject,
        multihop_result=multihop_result,
        inferred_facts=inferred,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Microworld Assistant Surface v1 (controlled, deterministic)."
    )
    parser.add_argument("question", nargs="?", help="The user question to answer.")
    parser.add_argument(
        "--overlay",
        choices=list(OVERLAY_MODES),
        default=None,
        help="Overlay mode: accepted (283), promoted (310, default), "
        "snapshot-dry-run (461 proposal only), or pump-dry-run (proposal only).",
    )
    parser.add_argument(
        "--overlay-path",
        default=None,
        help="Advanced: read one explicit overlay JSON path as a proposal/custom overlay.",
    )
    parser.add_argument(
        "--ontology-layer",
        default=None,
        help=(
            "Optional read-only ontology layer JSON path, e.g. a Wikidata P279 "
            "layer. Used only for is_a traversal."
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_mode",
        action="store_true",
        help="Print the full AssistantAnswer JSON object instead of text.",
    )
    parser.add_argument(
        "--enable-multihop",
        action="store_true",
        help="Enable experimental read-only multi-hop QA over explicit supported relation chains.",
    )
    parser.add_argument(
        "--think-aloud",
        dest="think_aloud",
        action="store_true",
        help="Show a [THINKING] block (how the answer was reached) above the [ANSWER].",
    )
    parser.add_argument(
        "--show-semantic-query",
        action="store_true",
        help="Print the parsed SemanticQuery before the answer.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start a REPL and keep in-memory dialogue context for this session.",
    )
    args = parser.parse_args(argv)

    if not args.interactive and args.question is None:
        parser.error("question is required unless --interactive is used")

    if args.overlay is not None and args.overlay_path is not None:
        parser.error("--overlay and --overlay-path cannot be used together")

    overlay_mode = args.overlay or "promoted"
    overlay_path = None
    if args.overlay_path is not None:
        path = Path(args.overlay_path)
        if not path.is_file():
            parser.error(f"--overlay-path does not exist or is not a file: {path}")
        overlay_path = str(path)

    ontology_layer_path = None
    if args.ontology_layer is not None:
        path = Path(args.ontology_layer)
        if not path.is_file():
            parser.error(f"--ontology-layer does not exist or is not a file: {path}")
        ontology_layer_path = str(path)

    if args.interactive:
        return _run_interactive(
            overlay_mode=overlay_mode,
            overlay_path=overlay_path,
            ontology_layer_path=ontology_layer_path,
            json_mode=args.json_mode,
            show_semantic_query=args.show_semantic_query,
        )

    answer = ask(
        args.question or "",
        overlay_mode,
        overlay_path=overlay_path,
        ontology_layer_path=ontology_layer_path,
    )
    semantic_query = None
    if args.show_semantic_query:
        semantic_index = _surface_index_for_dialogue(overlay_mode, overlay_path)
        semantic_query = parse_semantic_query(args.question or "", semantic_index)
    rendered_multihop = None
    multihop_result = None

    # Preserve normal single-hop behavior. The only existing answer path that
    # the flag may refine is the older cross-page connection answer, which is
    # not a direct single-hop fact and lacks the stricter multi-hop validator
    # used for current-sensitive predicates such as leader_of.
    should_try_multihop = (
        args.enable_multihop
        and (
            answer.decision == "audit"
            or answer.support_kind == "explicit_connection_path"
        )
    )
    if should_try_multihop:
        items = _load_overlay_items(overlay_mode, overlay_path)
        multihop_result = try_answer_multihop(args.question or "", items)
        if (
            multihop_result.decision == "answer"
            or answer.support_kind == "explicit_connection_path"
        ):
            effective_overlay_mode = (
                OVERLAY_MODE_CUSTOM_PATH if overlay_path is not None else overlay_mode
            )
            rendered_multihop = render_cli_multihop_result(
                multihop_result,
                overlay_mode=effective_overlay_mode,
                overlay_marker=_overlay_marker(effective_overlay_mode),
            )

    think_aloud = (
        _think_aloud_for(
            args.question or "",
            answer,
            overlay_mode,
            overlay_path,
            multihop_result=multihop_result,
        )
        if args.think_aloud
        else None
    )

    if args.json_mode:
        semantic_payload = (
            {"semantic_query": semantic_query.to_dict()} if semantic_query is not None else {}
        )
        think_payload = (
            {"think_aloud": {"thinking": think_aloud.thinking, "answer": think_aloud.answer}}
            if think_aloud is not None
            else {}
        )
        if rendered_multihop is not None and multihop_result is not None:
            print(json.dumps({
                "mode": "multihop",
                "multihop": multihop_result.to_dict(),
                "single_hop_first_pass": answer.to_dict(),
                **semantic_payload,
                **think_payload,
            }, indent=2, ensure_ascii=False))
        else:
            payload = answer.to_dict()
            payload.update(semantic_payload)
            payload.update(think_payload)
            print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if semantic_query is not None:
            print("SemanticQuery:")
            print(json.dumps(semantic_query.to_dict(), indent=2, ensure_ascii=False))
            print("")
        if think_aloud is not None:
            print(think_aloud.render())
        else:
            print(rendered_multihop if rendered_multihop is not None else render(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
