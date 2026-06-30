"""Build a QA-ready overlay from re-extracted snapshot facts.

The re-extraction artifact intentionally contains answerable facts only:
``overlay_relation`` and ``overlay_definition``.  The assistant surface can use
that directly for slot questions, but open/entity questions work better when the
overlay also has lightweight ``overlay_entity`` cards for entity resolution.

This script is proposal-only.  It writes a derived QA overlay beside a
re-extraction run and never mutates accepted/promoted memory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.knowledge.entity_type_classifier import classify_entity_type

_ANSWERABLE_TYPES = {"overlay_relation", "overlay_definition"}
_DEFAULT_REEXTRACT_DIR = (
    _ROOT / "worldpgt" / "experiments" / "knowledge_pump_v1" / "reextract_existing_v1"
)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _entity_id(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", label.strip()).strip("_")
    return f"reextract:{safe or 'entity'}"


def _looks_like_bad_relation_object(predicate: str, obj: str) -> bool:
    low = obj.lower().strip()
    if not low:
        return True
    if re.fullmatch(r"[\d,.\s]+", low):
        return True
    if re.match(r"^\d[\d,.\s]*(?:\w+)?(?:\s|$)", low):
        return True
    if obj.endswith(","):
        return True
    if re.match(r"^[A-Z][A-Za-z]+,\s+the\b", obj):
        return True
    if low in {"few officers", "competitive for college graduates"}:
        return True
    predicate_stems = {
        "produces": ("produce", "produced", "produces"),
        "develops": ("develop", "developed", "develops"),
        "provides": ("provide", "provided", "provides"),
        "uses": ("use", "used", "uses"),
        "supports": ("support", "supported", "supports"),
        "publishes": ("publish", "published", "publishes"),
    }
    if low.startswith(predicate_stems.get(predicate, ())):
        return True
    return False


def keep_answerable_item(item: dict[str, Any]) -> tuple[bool, str]:
    """Return whether a re-extracted answerable item is useful enough for QA."""

    overlay_type = item.get("overlay_type")
    if overlay_type not in _ANSWERABLE_TYPES:
        return False, "not_answerable_type"

    subject = _norm_text(item.get("subject"))
    if not subject:
        return False, "missing_subject"

    if overlay_type == "overlay_definition":
        definition = _norm_text(item.get("definition"))
        if not definition:
            return False, "missing_definition"
        if len(definition.split()) > 32:
            return False, "definition_too_long"
        return True, "kept"

    predicate = _norm_text(item.get("predicate"))
    obj = _norm_text(item.get("object"))
    if not predicate or not obj:
        return False, "missing_relation_field"
    if _looks_like_bad_relation_object(predicate, obj):
        return False, "bad_relation_object"
    return True, "kept"


def _definition_type_by_subject(items: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if item.get("overlay_type") != "overlay_definition":
            continue
        subject = _norm_text(item.get("subject"))
        definition = _norm_text(item.get("definition"))
        if subject and definition:
            out.setdefault(subject, classify_entity_type(definition) or "other")
    return out


def _should_card_for_object(label: str) -> bool:
    if not label or len(label) < 4:
        return False
    if len(label.split()) > 8:
        return False
    if label[0].islower() and not any(ch.isupper() for ch in label):
        return False
    return True


def build_qa_overlay(answerable_items: list[dict[str, Any]]) -> dict[str, Any]:
    kept: list[dict[str, Any]] = []
    rejected_by_reason: Counter[str] = Counter()
    for item in answerable_items:
        if not isinstance(item, dict):
            rejected_by_reason["not_object"] += 1
            continue
        keep, reason = keep_answerable_item(item)
        if keep:
            kept.append(item)
        else:
            rejected_by_reason[reason] += 1

    type_by_subject = _definition_type_by_subject(kept)
    labels: dict[str, str] = {}
    for item in kept:
        subject = _norm_text(item.get("subject"))
        if subject:
            labels.setdefault(subject, type_by_subject.get(subject, "other"))
        if item.get("overlay_type") == "overlay_relation":
            obj = _norm_text(item.get("object"))
            if _should_card_for_object(obj):
                labels.setdefault(obj, "other")

    entity_cards = [
        {
            "overlay_type": "overlay_entity",
            "entity_id": _entity_id(label),
            "label": label,
            "aliases": [],
            "entity_type": entity_type,
            "source_page": label,
            "source_candidate_type": "reextract_entity_card",
            "trust": "overlay_candidate",
            "risk": "low",
            "safe_for_general_runtime": False,
        }
        for label, entity_type in sorted(labels.items(), key=lambda kv: kv[0].lower())
    ]

    overlay = entity_cards + kept
    return {
        "overlay": overlay,
        "summary": {
            "status": "complete",
            "input_answerable_count": len(answerable_items),
            "qa_overlay_items_count": len(overlay),
            "entity_card_count": len(entity_cards),
            "answerable_kept_count": len(kept),
            "answerable_rejected_count": sum(rejected_by_reason.values()),
            "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
            "relation_count": sum(1 for item in kept if item.get("overlay_type") == "overlay_relation"),
            "definition_count": sum(1 for item in kept if item.get("overlay_type") == "overlay_definition"),
        },
    }


def build_from_paths(input_path: Path, output_path: Path, summary_path: Path) -> dict[str, Any]:
    answerable = _read_json(input_path, [])
    if not isinstance(answerable, list):
        raise ValueError(f"expected list in {input_path}")
    result = build_qa_overlay(answerable)
    _write_json(output_path, result["overlay"])
    _write_json(summary_path, result["summary"])
    return result["summary"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(_DEFAULT_REEXTRACT_DIR / "reextract_answerable_delta.json"),
        help="Path to reextract_answerable_delta.json",
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_REEXTRACT_DIR / "qa_overlay.json"),
        help="Output QA-ready overlay path",
    )
    parser.add_argument(
        "--summary",
        default=str(_DEFAULT_REEXTRACT_DIR / "qa_overlay_summary.json"),
        help="Output summary path",
    )
    args = parser.parse_args(argv)

    summary = build_from_paths(Path(args.input), Path(args.output), Path(args.summary))
    for key in (
        "status",
        "input_answerable_count",
        "qa_overlay_items_count",
        "entity_card_count",
        "answerable_kept_count",
        "answerable_rejected_count",
        "relation_count",
        "definition_count",
    ):
        print(f"{key}: {summary[key]}")
    print(f"output: {args.output}")
    print(f"summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
