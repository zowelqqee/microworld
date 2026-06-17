"""Load precision-filtered pump facts for the QA benchmark.

Only ``overlay_relation`` and ``overlay_definition`` items become answerable
facts. ``overlay_entity`` cards (page-existence cards) are intentionally *not*
used to generate answerable-fact questions, and weak ``overlay_context_link``
items are never treated as facts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldpgt.pump_fact_qa.types import FACT_KIND_DEFINITION, FACT_KIND_RELATION, PumpFact


def _read_json(path: Path) -> Any:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _is_weak_context(item: dict[str, Any]) -> bool:
    return (
        item.get("overlay_type") == "overlay_context_link"
        and item.get("trust") == "weak_context_only"
    )


def load_pump_facts(
    precision_path: str | Path,
    fallback_path: str | Path | None = None,
) -> list[PumpFact]:
    """Read precision-filtered answerable facts.

    Prefers ``precision_path`` (``pump_precision_answerable_delta.json``); if it
    is missing or empty, falls back to ``fallback_path``
    (``pump_answerable_delta.json``). Entity cards and weak context are excluded.
    Output is sorted deterministically.
    """

    items = _read_json(Path(precision_path))
    if not items and fallback_path is not None:
        items = _read_json(Path(fallback_path))

    facts: list[PumpFact] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if _is_weak_context(item):
            continue
        otype = item.get("overlay_type")
        if otype == "overlay_relation":
            facts.append(
                PumpFact(
                    kind=FACT_KIND_RELATION,
                    subject=str(item.get("subject", "")),
                    predicate=str(item.get("predicate", "")),
                    obj=str(item.get("object", "")),
                    source_page=str(item.get("source_page", "")),
                )
            )
        elif otype == "overlay_definition":
            facts.append(
                PumpFact(
                    kind=FACT_KIND_DEFINITION,
                    subject=str(item.get("subject", "")),
                    predicate=str(item.get("predicate", "is_a")),
                    obj=str(item.get("definition", "")),
                    source_page=str(item.get("source_page", "")),
                )
            )
        # overlay_entity and all other types are not answerable-fact sources.

    facts.sort(key=lambda f: (f.kind, f.subject.lower(), f.predicate, f.obj.lower()))
    return facts
