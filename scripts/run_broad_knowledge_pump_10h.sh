#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PUMP_SECONDS="${PUMP_SECONDS:-36000}"
TARGET_TOTAL="${TARGET_TOTAL:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-60}"
MAX_BATCHES_PER_CYCLE="${MAX_BATCHES_PER_CYCLE:-10}"
DELAY_SEC="${DELAY_SEC:-0.1}"
FRONTIER_WEIGHT="${FRONTIER_WEIGHT:-120}"
PYTHON_BIN="${PYTHON_BIN:-/Library/Frameworks/Python.framework/Versions/3.13/bin/python3}"

FRONTIER_JSON="worldpgt/experiments/knowledge_pump_v1/dynamic_frontier_titles.json"
FRONTIER_CSV="worldpgt/experiments/knowledge_pump_v1/dynamic_frontier_titles.csv"
OVERLAY_JSON="worldpgt/experiments/knowledge_pump_v1/pump_dry_run_overlay.json"
DOCS_DIR="worldpgt/experiments/knowledge_pump_v1/batch_snapshots/normalized_docs"
BACKFILL_REPORT="worldpgt/experiments/knowledge_pump_v1/lead_definition_backfill_v1/report.json"
SUMMARY_JSON="worldpgt/experiments/knowledge_pump_v1/pump_summary.json"

started_at="$(date +%s)"
deadline=$((started_at + PUMP_SECONDS))
cycle=0

echo "[10h-pump] start $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "[10h-pump] budget_seconds=$PUMP_SECONDS target_total=$TARGET_TOTAL batch_size=$BATCH_SIZE max_batches_per_cycle=$MAX_BATCHES_PER_CYCLE"

while [ "$(date +%s)" -lt "$deadline" ]; do
  cycle=$((cycle + 1))
  now="$(date +%s)"
  remaining=$((deadline - now))
  echo "[10h-pump] cycle=$cycle remaining_seconds=$remaining"

  "$PYTHON_BIN" -m worldpgt.knowledge_pump.broad_frontier_seeder \
    --frontier-json "$FRONTIER_JSON" \
    --frontier-csv "$FRONTIER_CSV" \
    --weight "$FRONTIER_WEIGHT"

  "$PYTHON_BIN" worldpgt/experiments/run_knowledge_pump_v1.py \
    --allow-network \
    --resume \
    --target-total "$TARGET_TOTAL" \
    --batch-size "$BATCH_SIZE" \
    --max-batches "$MAX_BATCHES_PER_CYCLE" \
    --delay-sec "$DELAY_SEC" \
    --frontier-policy default \
    --force-low-yield-fetch \
    --enable-schema-induction

  "$PYTHON_BIN" -m worldpgt.knowledge_pump.lead_definition_backfill \
    --overlay-json "$OVERLAY_JSON" \
    --docs-dir "$DOCS_DIR" \
    --report-json "$BACKFILL_REPORT"

  "$PYTHON_BIN" - "$SUMMARY_JSON" "$BACKFILL_REPORT" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
backfill = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

print(
    "[10h-pump] summary "
    f"batches_completed={summary.get('batches_completed')} "
    f"fetch_success={summary.get('fetch_success_count_total')} "
    f"ready={summary.get('ready_for_ingestion_count_total')} "
    f"answerable={summary.get('pump_answerable_fact_delta_count')} "
    f"definitions={summary.get('pump_definition_delta_count')} "
    f"relations={summary.get('pump_relation_delta_count')} "
    f"overlay_items={backfill.get('overlay_items_count')} "
    f"backfill_added={backfill.get('added_count')} "
    f"critical={summary.get('all_critical_passed')}"
)

protected = [
    "trusted_memory_modified",
    "accepted_overlay_modified",
    "promoted_overlay_modified",
    "snapshot_dry_run_overlay_modified",
]
if summary.get("all_critical_passed") is not True:
    raise SystemExit("[10h-pump] STOP: all_critical_passed is not true")
if any(summary.get(key) for key in protected):
    raise SystemExit("[10h-pump] STOP: protected memory artifact was modified")
if summary.get("unsafe_answer_count", 0) or summary.get("answer_without_context_support_count", 0):
    raise SystemExit("[10h-pump] STOP: assistant safety counters are non-zero")
PY

  echo "[10h-pump] cycle=$cycle done $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
done

echo "[10h-pump] completed budget $(date -u '+%Y-%m-%dT%H:%M:%SZ') cycles=$cycle"
