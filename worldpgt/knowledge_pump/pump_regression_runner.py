"""Regression helpers for Knowledge Pump v1."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from worldpgt.assistant_surface import context_selector
from worldpgt.experiments import run_assistant_surface_v1


def run_assistant_regression(out_dir: str | Path, overlay_path: str | Path | None = None) -> dict:
    out = Path(out_dir)
    assistant_out = out / "assistant"
    old_context_overlay = context_selector.SNAPSHOT_DRY_RUN_OVERLAY_PATH
    old_runner_overlay = run_assistant_surface_v1.SNAPSHOT_DRY_RUN_OVERLAY_PATH
    old_protected_overlay = run_assistant_surface_v1._PROTECTED_FILES["snapshot_dry_run_overlay"]
    if overlay_path is not None:
        pump_overlay = Path(overlay_path)
        context_selector.SNAPSHOT_DRY_RUN_OVERLAY_PATH = pump_overlay
        run_assistant_surface_v1.SNAPSHOT_DRY_RUN_OVERLAY_PATH = pump_overlay
        run_assistant_surface_v1._PROTECTED_FILES["snapshot_dry_run_overlay"] = pump_overlay
    try:
        summary = run_assistant_surface_v1.run(outdir=assistant_out)
    finally:
        context_selector.SNAPSHOT_DRY_RUN_OVERLAY_PATH = old_context_overlay
        run_assistant_surface_v1.SNAPSHOT_DRY_RUN_OVERLAY_PATH = old_runner_overlay
        run_assistant_surface_v1._PROTECTED_FILES["snapshot_dry_run_overlay"] = old_protected_overlay
    summary["benchmark_overlay_path"] = str(Path(overlay_path)) if overlay_path is not None else str(old_context_overlay)
    for name in ("assistant_surface_outputs.csv", "assistant_surface_outputs.json"):
        src = assistant_out / name
        if src.exists():
            dst = out / name.replace("assistant_surface", "pump_assistant")
            shutil.copyfile(src, dst)
    return summary


def write_not_run(out_dir: str | Path, reason: str) -> dict:
    summary = {"status": "not_run_requires_adapter", "reason": reason}
    Path(out_dir, "pump_assistant_outputs.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
