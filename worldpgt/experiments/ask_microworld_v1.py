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
from worldpgt.assistant_surface.types import (
    OVERLAY_MODE_CUSTOM_PATH,
    OVERLAY_MODE_PUMP_DRY_RUN,
    OVERLAY_MODE_SNAPSHOT_DRY_RUN,
    OVERLAY_MODES,
)
from worldpgt.multihop_qa.assistant_adapter import (
    render_cli_multihop_result,
    try_answer_multihop,
)

_SNAPSHOT_MARKER = "Overlay mode: snapshot-dry-run proposal, not accepted memory."
_PUMP_MARKER = "Overlay mode: pump-dry-run proposal, not accepted memory."
_CUSTOM_MARKER = "Overlay mode: custom overlay path, not accepted memory."


def ask(question: str, overlay_mode: str = "promoted", overlay_path: str | None = None):
    """Run the assistant surface for one question; returns an AssistantAnswer."""

    orchestrator = AnswerOrchestrator(overlay_mode, overlay_path=overlay_path)
    return orchestrator.answer(question)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Microworld Assistant Surface v1 (controlled, deterministic)."
    )
    parser.add_argument("question", help="The user question to answer.")
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
    args = parser.parse_args(argv)

    if args.overlay is not None and args.overlay_path is not None:
        parser.error("--overlay and --overlay-path cannot be used together")

    overlay_mode = args.overlay or "promoted"
    overlay_path = None
    if args.overlay_path is not None:
        path = Path(args.overlay_path)
        if not path.is_file():
            parser.error(f"--overlay-path does not exist or is not a file: {path}")
        overlay_path = str(path)

    answer = ask(args.question, overlay_mode, overlay_path=overlay_path)
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
        multihop_result = try_answer_multihop(args.question, items)
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

    if args.json_mode:
        if rendered_multihop is not None and multihop_result is not None:
            print(json.dumps({
                "mode": "multihop",
                "multihop": multihop_result.to_dict(),
                "single_hop_first_pass": answer.to_dict(),
            }, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(rendered_multihop if rendered_multihop is not None else render(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
