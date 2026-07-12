"""Checkpoint read/write helpers for Knowledge Pump v1."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from worldpgt.knowledge_pump.types import PumpBatchRecord, PumpCheckpoint


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_checkpoint(path: str | Path) -> PumpCheckpoint | None:
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return PumpCheckpoint(**data)


def write_checkpoint(path: str | Path, checkpoint: PumpCheckpoint) -> None:
    checkpoint.last_run_at = utc_now()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(checkpoint.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_history(path: str | Path) -> list[PumpBatchRecord]:
    p = Path(path)
    if not p.exists():
        return []
    return [PumpBatchRecord(**row) for row in json.loads(p.read_text(encoding="utf-8"))]


def append_history(path: str | Path, record: PumpBatchRecord) -> list[PumpBatchRecord]:
    history = load_history(path)
    history.append(record)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([r.to_dict() for r in history], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return history

