"""Remove non-lead definition fragments from the pump dry-run overlay.

This is a proposal-overlay cleanup for facts that were accepted before the
definition precision gate learned to reject discourse-marker continuations such
as "also ..." and "previously ...".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.knowledge_pump.precision_firewall import _BAD_DEFINITION_LEADS

_DEFAULT_OVERLAY = (
    _ROOT
    / "worldpgt"
    / "experiments"
    / "knowledge_pump_v1"
    / "pump_dry_run_overlay.json"
)


def _read_json(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected a JSON list in {path}")
    return [row for row in rows if isinstance(row, dict)]


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def definition_lead(item: dict[str, Any]) -> str:
    definition = str(item.get("definition") or "").strip().lower()
    return definition.split()[0] if definition.split() else ""


def is_bad_definition(item: dict[str, Any]) -> bool:
    if item.get("overlay_type") != "overlay_definition":
        return False
    return definition_lead(item) in _BAD_DEFINITION_LEADS


def cleanup_items(
    rows: list[dict[str, Any]],
    *,
    max_examples: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for item in rows:
        if is_bad_definition(item):
            removed.append(item)
            continue
        kept.append(item)
    return kept, removed[:max_examples]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay-path", default=str(_DEFAULT_OVERLAY))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args(argv)

    overlay_path = Path(args.overlay_path)
    rows = _read_json(overlay_path)
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for item in rows:
        if is_bad_definition(item):
            removed.append(item)
        else:
            kept.append(item)

    if not args.dry_run:
        _write_json(overlay_path, kept)

    print(f"overlay_path: {overlay_path}")
    print(f"dry_run: {args.dry_run}")
    print(f"items_before: {len(rows)}")
    print(f"items_after: {len(kept)}")
    print(f"removed_count: {len(removed)}")
    print("removed_examples:")
    for item in removed[: max(args.examples, 0)]:
        subject = str(item.get("subject") or "")
        definition = str(item.get("definition") or "")
        evidence = str(item.get("evidence_text") or "")
        print(f"- {subject} => {definition} | {evidence[:140]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
