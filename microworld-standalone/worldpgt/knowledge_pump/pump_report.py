"""Report writers for Knowledge Pump v1."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from worldpgt.knowledge_pump.types import FrontierTitle


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_frontier(frontier: list[FrontierTitle], json_path: str | Path, csv_path: str | Path) -> None:
    payload = [f.to_dict() for f in frontier]
    write_json(json_path, payload)
    with Path(csv_path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["title", "source", "reason", "weight"])
        writer.writeheader()
        writer.writerows(payload)


def build_summary(**kwargs) -> dict[str, Any]:
    base = {
        "auto_ingest": False,
        "auto_promote": False,
        "trusted_memory_modified": False,
        "accepted_overlay_modified": False,
        "promoted_overlay_modified": False,
        "snapshot_dry_run_overlay_modified": False,
        "runtime_behavior_modified": False,
        "safe_for_general_runtime": False,
    }
    base.update(kwargs)
    return base

