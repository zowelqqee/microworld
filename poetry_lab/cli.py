#!/usr/bin/env python3
"""poetry_lab CLI — drive the experimental runtime from the terminal.

    python cli.py ingest
    python cli.py write "write a poem about autumn"
    python cli.py write "write in the style of Pushkin" --stanzas 3
    python cli.py continue "Мороз и солнце; день чудесный!"
    python cli.py write "write a poem about space using classical Russian imagery" --trace

``--trace`` prints the reasoning artifact (activated concepts, per-line plan)
so the layer boundary is visible, the same inspectability the production
runtime exposes through its ReasoningTrace.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from poemcore.engine import PoetryEngine, Result  # noqa: E402
from poemcore.ingest import write_artifacts, write_narrative_artifacts  # noqa: E402
from poemcore.narrative import NarrativeEngine, NarrativeResult  # noqa: E402


def _print_result(result: Result, *, trace: bool, as_json: bool) -> None:
    if as_json:
        print(json.dumps(
            {
                "request": result.request.to_dict(),
                "plan": result.plan.to_dict(),
                "reasoning": result.reasoning.to_dict() if result.reasoning else None,
                "poem_intent": result.poem_intent.to_dict() if result.poem_intent else None,
                "line_intents": [intent.to_dict() for intent in result.line_intents],
                "surface_realization": [trace.to_dict() for trace in result.poem.realization],
                "poem": result.poem.lines,
                "novelty": {
                    "ratio": result.novelty.novelty_ratio,
                    "echoed_lines": result.novelty.echoed_lines,
                },
            },
            ensure_ascii=False,
            indent=2,
        ))
        return

    print(f"\n  {result.poem.title}\n")
    for line in result.poem.lines:
        print(f"  {line}" if line else "")
    print()
    print(f"  [novelty {result.novelty.novelty_ratio:.2f} · "
          f"echoed {result.novelty.echoed_lines}/{result.novelty.total_lines} lines · "
          f"mode {result.request.mode}]")

    if trace:
        print("\n  --- reasoning trace ---")
        print(f"  seeds: {result.request.seed_concepts}")
        print("  activated concepts:")
        for concept, act in result.plan.activated[:10]:
            print(f"    {act:6.2f}  {concept}")
        print("  line plan:")
        for ln in result.plan.lines:
            print(f"    {ln.index:>2} {ln.move:<10} focus={ln.focus:<14} "
                  f"rhyme={ln.rhyme_label} syll={ln.target_syllables}")
        if result.reasoning:
            goal = result.reasoning.goal
            print("  poem goal:")
            print(f"    theme={goal.theme or '(implicit)'} speaker={goal.speaker} "
                  f"mood={goal.mood} setting={goal.setting}")
            print("  stanza plan:")
            for stanza in result.reasoning.stanzas:
                print(f"    {stanza.index:>2} {stanza.purpose:<18} anchor={stanza.anchor:<14} "
                      f"progression={list(stanza.progression)}")
            print("  line decisions:")
            for decision in result.reasoning.lines:
                print(f"    {decision.index:>2} {decision.relation:<11} "
                      f"{decision.subject} -> {decision.action or '?'} -> {decision.object} "
                      f"[{decision.modifier}]")
        if result.poem.realization:
            print("  surface realization:")
            for surface in result.poem.realization:
                subject, action, object_ = surface.planned
                realized = ", ".join(surface.realized) or "none"
                print(f"    {surface.index:>2} {surface.strategy:<22} "
                      f"{subject} -> {action or '?'} -> {object_} | hit={realized}")


def _print_narrative(result: NarrativeResult, *, trace: bool, as_json: bool) -> None:
    if as_json:
        print(json.dumps({
            "request": result.request.to_dict(), "scene_plan": result.plan.to_dict(),
            "paragraph": result.paragraph.sentences,
            "surface_realization": [item.to_dict() for item in result.paragraph.realization],
            "reasoning_plan": result.reasoning_plan.to_dict() if result.reasoning_plan else None,
            "world_state": result.world_state.to_dict() if result.world_state else None,
            "scene_state": result.scene_state.to_dict() if result.scene_state else None,
            "novelty": {"ratio": result.novelty.novelty_ratio, "echoed_sentences": result.novelty.echoed_lines},
            "corpus": result.meta,
        }, ensure_ascii=False, indent=2))
        return
    print("\n  " + result.paragraph.text() + "\n")
    print(f"  [novelty {result.novelty.novelty_ratio:.2f} · "
          f"echoed {result.novelty.echoed_lines}/{result.novelty.total_lines} sentences]")
    if trace:
        goal = result.plan.goal
        print("\n  --- narrative trace ---")
        print(f"  scene goal: topic={goal.topic} mode={goal.mode} speaker={goal.speaker} location={goal.location}")
        print(f"  characters={list(goal.characters)} seeds={list(goal.seeds)}")
        if result.scene_state:
            state = result.scene_state
            print(f"  scene state: focus={state.narrative_focus} objective={state.scene_objective} "
                  f"field={list(state.active_field)}")
        print("  sentence plan:")
        for item in result.plan.sentences:
            print(f"    {item.index:>2} {item.purpose:<18} {item.subject} -> {item.action or '?'} -> {item.object} "
                  f"relation={item.relation} fragment={' '.join(item.fragment) or '-'} "
                  f"detail={' '.join(item.detail_fragment) or '-'} "
                  f"transition={item.transition or '-'} dialogue={item.dialogue}")
        print("  surface realization:")
        for item in result.paragraph.realization:
            print(f"    {item.index:>2} hit={','.join(item.realized) or 'none'} dialogue={item.dialogue}")
        if result.reasoning_plan:
            print(f"  reasoning plan: {len(result.reasoning_plan.steps)} steps, score={result.reasoning_plan.score:.1f}")
        if result.world_state:
            print(f"  committed world state: t={result.world_state.t} facts={len(result.world_state.facts)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="poetry_lab experimental runtime")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="(re)build poetry artifacts from the verse corpus")
    p_ingest_narrative = sub.add_parser(
        "ingest-narrative", help="build prose artifacts from one file or an uploaded corpus directory"
    )
    p_ingest_narrative.add_argument("--source", default=None, help=".txt file or directory of prose files")
    p_ingest_narrative.add_argument("--output", default=None, help="artifact JSON destination")
    p_ingest_narrative.add_argument(
        "--max-sentences-per-source", type=int, default=None,
        help="deterministically balance a mixed corpus by limiting each source",
    )

    p_slim = sub.add_parser(
        "slim-narrative",
        help="write a memory-lean phone copy of the narrative artifact (drops verse/novelty "
             "tables + trims the concept graph; ~370 MB resident vs ~1 GB)",
    )
    p_slim.add_argument("--source", default=None, help="full artifact JSON (default: artifacts/narrative_model.json)")
    p_slim.add_argument("--output", default=None, help="slim artifact destination (default: artifacts/narrative_model.phone.json)")
    p_slim.add_argument("--edges-per-node", type=int, default=12, help="strongest concept-graph edges kept per node")

    p_write = sub.add_parser("write", help="generate a poem from a prompt")
    p_write.add_argument("prompt")
    p_write.add_argument("--stanzas", type=int, default=2)
    p_write.add_argument("--seed", default=None, help="override render seed for a variant")
    p_write.add_argument("--trace", action="store_true")
    p_write.add_argument("--json", action="store_true")

    p_cont = sub.add_parser("continue", help="continue a given verse (read from arg or stdin)")
    p_cont.add_argument("verse", nargs="?", default="")
    p_cont.add_argument("--stanzas", type=int, default=2)
    p_cont.add_argument("--trace", action="store_true")
    p_cont.add_argument("--json", action="store_true")

    p_narrate = sub.add_parser("narrate", help="generate one planned prose paragraph")
    p_narrate.add_argument("prompt")
    p_narrate.add_argument("--after", default="", help="preceding paragraph or dialogue context")
    p_narrate.add_argument("--sentences", type=int, default=3)
    p_narrate.add_argument(
        "--seed", default=None,
        help="replayable route seed; omit for a fresh graph walk on every run",
    )
    p_narrate.add_argument("--trace", action="store_true")
    p_narrate.add_argument("--json", action="store_true")
    p_narrate.add_argument("--reasoning", action="store_true", help="plan sentence transitions with WorldState + beam search")

    args = parser.parse_args(argv)

    if args.command == "ingest":
        path = write_artifacts()
        print(f"wrote {path}")
        return 0
    if args.command == "ingest-narrative":
        # `write_narrative_artifacts` takes the output path first and the
        # corpus source second.  Passing `--source` positionally previously
        # treated a directory such as `corpus/` as the output file and failed
        # before any mixed-corpus model could be produced.
        path = write_narrative_artifacts(
            target=Path(args.output) if args.output else None,
            source=Path(args.source) if args.source else None,
            max_sentences_per_source=args.max_sentences_per_source,
        )
        print(f"wrote {path}")
        return 0
    if args.command == "slim-narrative":
        from poemcore.ingest import slim_narrative_artifact  # noqa: E402
        path = slim_narrative_artifact(
            source=Path(args.source) if args.source else None,
            target=Path(args.output) if args.output else None,
            edges_per_node=args.edges_per_node,
        )
        print(f"wrote {path}")
        return 0

    if args.command == "narrate":
        result = NarrativeEngine().run(args.prompt, context=args.after, sentences=args.sentences, seed=args.seed, reasoning=args.reasoning)
        _print_narrative(result, trace=args.trace, as_json=args.json)
        return 0

    engine = PoetryEngine()

    if args.command == "write":
        result = engine.run(args.prompt, stanzas=args.stanzas, seed=args.seed)
        _print_result(result, trace=args.trace, as_json=args.json)
        return 0

    if args.command == "continue":
        verse = args.verse or sys.stdin.read()
        result = engine.run("continue this verse", given_verse=verse, stanzas=args.stanzas)
        _print_result(result, trace=args.trace, as_json=args.json)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
