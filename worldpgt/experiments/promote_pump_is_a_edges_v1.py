"""Promote Knowledge Pump ``is_a`` readiness candidates into a QA overlay.

This runner is intentionally narrower than accepted-memory ingestion: it reads
the pump promotion-readiness audit, selects only ``is_a`` candidates that passed
the existing readiness/QA gates, re-validates them with the overlay delta
validator, and writes a separate promoted overlay artifact for QA/traversal.

It never overwrites accepted memory, the accepted overlay, the existing
self-ingestion promoted overlay, or the snapshot dry-run overlay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.experiments import run_pump_promotion_readiness_audit_v1 as readiness
from worldpgt.self_ingestion.overlay_delta_validator import validate_delta

_EXPERIMENTS = Path(__file__).resolve().parent
_PUMP_DIR = _EXPERIMENTS / "knowledge_pump_v1"
_OUT_DIR = _PUMP_DIR / "is_a_promotion_v1"
_BASE_OVERLAY = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING"


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _fact_key(item: dict[str, Any]) -> tuple[str, str, str]:
    if item.get("overlay_type") == "overlay_definition":
        return (_norm(item.get("subject")), "is_a", _norm(item.get("definition")))
    return (_norm(item.get("subject")), _norm(item.get("predicate")), _norm(item.get("object")))


def _candidate_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (_norm(row.get("subject")), _norm(row.get("predicate") or "is_a"), _norm(row.get("object")))


def _is_is_a_item(item: dict[str, Any]) -> bool:
    if item.get("overlay_type") == "overlay_relation":
        return item.get("predicate") == "is_a"
    if item.get("overlay_type") == "overlay_definition":
        return item.get("predicate") in (None, "", "is_a")
    return False


def _protected_hashes(experiments_dir: Path, root: Path) -> dict[str, str]:
    return {
        "trusted_memory": _sha(experiments_dir / "accepted_knowledge_memory_v1.json"),
        "accepted_overlay": _sha(experiments_dir / "accepted_wiki_memory_overlay_v1.json"),
        "promoted_overlay": _sha(
            experiments_dir / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json"
        ),
        "snapshot_dry_run_overlay": _sha(
            experiments_dir / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json"
        ),
        "sense_memory": _sha(root / "worldpgt" / "continuation" / "sense_memory.py"),
    }


def _dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = _fact_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def run(
    *,
    pump_dir: Path = _PUMP_DIR,
    out_dir: Path = _OUT_DIR,
    base_overlay_path: Path = _BASE_OVERLAY,
    experiments_dir: Path = _EXPERIMENTS,
    root: Path = _ROOT,
) -> dict[str, Any]:
    before = _protected_hashes(experiments_dir, root)

    audit_out = out_dir / "promotion_readiness_audit"
    audit_summary = readiness.run(pump_dir=pump_dir, out_dir=audit_out)
    candidates = _read_json(audit_out / "promotion_readiness_candidates.json", [])
    precision_items = _read_json(pump_dir / "pump_precision_answerable_delta.json", [])
    base_items = _read_json(base_overlay_path, [])
    if not isinstance(candidates, list):
        candidates = []
    if not isinstance(precision_items, list):
        precision_items = []
    if not isinstance(base_items, list):
        base_items = []

    precision_index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    audited_is_a_count = 0
    for item in precision_items:
        if not isinstance(item, dict) or not _is_is_a_item(item):
            continue
        audited_is_a_count += 1
        precision_index.setdefault(_fact_key(item), []).append(item)

    selected_rows = [
        row for row in candidates
        if isinstance(row, dict) and row.get("predicate") == "is_a"
    ]
    selected_items: list[dict[str, Any]] = []
    missing_items: list[dict[str, Any]] = []
    for row in selected_rows:
        matches = precision_index.get(_candidate_key(row), [])
        if matches:
            selected_items.append(matches[0])
        else:
            missing_items.append(row)

    delta_items = _dedupe_items(selected_items)
    validation = validate_delta(delta_items, base_items)
    promoted_items = list(base_items) + list(validation.accepted_items)

    delta_path = out_dir / "pump_is_a_promotion_delta.json"
    overlay_path = out_dir / "pump_is_a_promoted_overlay.json"
    validation_path = out_dir / "pump_is_a_promotion_validation.json"
    report_path = out_dir / "pump_is_a_promotion_report.json"
    _write_json(delta_path, validation.accepted_items)
    _write_json(overlay_path, promoted_items)
    _write_json(validation_path, validation.to_dict())

    after = _protected_hashes(experiments_dir, root)
    confirmations = {
        "accepted_overlay_modified": before["accepted_overlay"] != after["accepted_overlay"],
        "promoted_overlay_modified": before["promoted_overlay"] != after["promoted_overlay"],
        "snapshot_dry_run_overlay_modified": before["snapshot_dry_run_overlay"] != after["snapshot_dry_run_overlay"],
        "trusted_memory_modified": before["trusted_memory"] != after["trusted_memory"],
        "sense_memory_modified": before["sense_memory"] != after["sense_memory"],
        "auto_promote_to_accepted_memory": False,
    }

    report = {
        "promotion_name": "pump_is_a_edges_v1",
        "source_pump_dir": str(pump_dir),
        "base_overlay": str(base_overlay_path),
        "audit_summary": audit_summary,
        "audited_is_a_count": audited_is_a_count,
        "readiness_is_a_candidate_count": len(selected_rows),
        "selected_is_a_delta_count": len(delta_items),
        "missing_precision_items_count": len(missing_items),
        "validation_accepted_count": validation.accepted_count,
        "validation_rejected_count": validation.rejected_count,
        "validation_blocked_count": validation.blocked_count,
        "base_overlay_items": len(base_items),
        "promoted_overlay_items": len(promoted_items),
        "delta_path": str(delta_path),
        "promoted_overlay_path": str(overlay_path),
        "confirmations": confirmations,
    }
    _write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote pump is_a readiness candidates into a QA overlay.")
    parser.add_argument("--pump-dir", default=str(_PUMP_DIR))
    parser.add_argument("--out-dir", default=str(_OUT_DIR))
    parser.add_argument("--base-overlay", default=str(_BASE_OVERLAY))
    args = parser.parse_args(argv)

    report = run(
        pump_dir=Path(args.pump_dir),
        out_dir=Path(args.out_dir),
        base_overlay_path=Path(args.base_overlay),
    )
    print("Pump is_a Promotion v1")
    for key in (
        "audited_is_a_count",
        "readiness_is_a_candidate_count",
        "selected_is_a_delta_count",
        "validation_accepted_count",
        "validation_rejected_count",
        "validation_blocked_count",
        "base_overlay_items",
        "promoted_overlay_items",
        "promoted_overlay_path",
    ):
        print(f"  {key}: {report.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
