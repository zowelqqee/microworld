#!/bin/sh
# Sequential, resumable, proposal-only Wikidata campaign.  Each 24-subject
# batch persists independently; the EXIT trap always writes an honest summary.
set -u
ROOT="artifacts/schema_expansion_v1/overnight_run_20jul"
EXPECTED=336
PYTHON="${PYTHON_BIN:-python3}"
DEADLINE=$(( $(date +%s) + 28800 ))
STOP_REASON="completed"
mkdir -p "$ROOT"
finish() {
  "$PYTHON" -m worldpgt.knowledge_pump.overnight_summary_v1 \
    --root "$ROOT" --expected-subjects "$EXPECTED" --stop-reason "$STOP_REASON" \
    > "$ROOT/summary_stdout.json" || true
}
trap finish EXIT INT TERM

offset=0
while [ "$offset" -lt "$EXPECTED" ]; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    STOP_REASON="time_limit_reached"
    exit 0
  fi
  batch="$ROOT/wikidata_batch_$(printf '%03d' "$offset")"
  if [ -f "$batch/summary.json" ]; then
    offset=$((offset + 24))
    continue
  fi
  "$PYTHON" -m worldpgt.knowledge_pump.wikidata_pipeline_v1 \
    --subjects-source unresolved-pool \
    --skip-subjects "$offset" --max-subjects 24 \
    --property-whitelist first-round-plus-top5 --delay-seconds 0.5 \
    --output "$batch" --allow-network || { STOP_REASON="batch_failure_at_$offset"; exit 1; }
  offset=$((offset + 24))
done
