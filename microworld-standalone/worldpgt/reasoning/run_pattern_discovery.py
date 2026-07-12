"""Nightly pattern-discovery runner.

Analyzes the promoted overlay's fact graph and writes discovered
``GraphPattern`` observations to a JSON artifact. Intended for a scheduled
(nightly) cycle; safe to run at any time — it only reads the overlay and
rewrites the artifact deterministically.

Usage:

    python3 -m worldpgt.reasoning.run_pattern_discovery \
        --overlay promoted \
        --output worldpgt/artifacts/graph_patterns.json \
        --min-support 2 \
        --min-confidence 0.5
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib

from worldpgt.reasoning.pattern_discovery import discover_patterns
from worldpgt.reasoning.pattern_store import DEFAULT_PATTERNS_PATH, save_patterns


def _load_overlay_items(overlay_mode: str, overlay_path: str | None) -> list[dict]:
    if overlay_path:
        return json.loads(pathlib.Path(overlay_path).read_text())
    from worldpgt.assistant_surface.context_selector import resolve_overlay

    path, _ = resolve_overlay(overlay_mode)
    return json.loads(pathlib.Path(path).read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", default="promoted", help="overlay mode to analyze")
    parser.add_argument("--overlay-path", default=None, help="explicit overlay JSON path")
    parser.add_argument(
        "--output", default=str(DEFAULT_PATTERNS_PATH), help="output artifact path"
    )
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-patterns", type=int, default=200)
    args = parser.parse_args(argv)

    items = _load_overlay_items(args.overlay, args.overlay_path)
    as_of = _dt.date.today().isoformat()
    patterns = discover_patterns(
        items,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        max_patterns=args.max_patterns,
        as_of=as_of,
    )
    target = save_patterns(
        patterns,
        path=args.output,
        metadata={
            "overlay": args.overlay if not args.overlay_path else args.overlay_path,
            "as_of": as_of,
            "min_support": args.min_support,
            "min_confidence": args.min_confidence,
            "pattern_count": len(patterns),
        },
    )
    print(f"Discovered {len(patterns)} pattern(s) → {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
